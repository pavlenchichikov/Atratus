"""Analyst agent CLI: run / score / backfill.

run      one judgment per eligible asset, written to analyst_log
score    the current standing against the baselines
backfill fill outcomes for judgments whose horizon has elapsed

Eligibility is the throttle. 208 assets times a daily LLM call with a full
dossier is a bill every day, so the agent runs the watchlist plus anything with
an event dated today, and skips any asset whose dossier is byte-identical to
the one already judged.
"""

import argparse
import datetime as _dt
import json
import os
import sqlite3
import sys

import train_payoff
from config import radar_category
from core.analyst import agent, calibrate, dossier, payoff, store
from core.llm_proposer import ProviderUnavailable


def _load_table():
    """payoff_stats.json, or None if train_payoff.py has never been run.

    A separate function so a test can monkeypatch it instead of reaching the
    real fitted artifact at the worktree root.
    """
    if not os.path.exists(train_payoff.STATS_PATH):
        return None
    with open(train_payoff.STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _provider_call():
    """An f(prompt) -> str bound to the configured provider.

    core/llm_proposer.py:332 _backend() already is exactly this: it resolves
    GTRADE_AR_LLM to anthropic, openai or ollama at call time so tests can
    monkeypatch, and wraps the result in a console trace. Writing a second
    provider layer beside it would be a second thing to keep in sync with the
    env flags.
    """
    from core.llm_proposer import _backend
    return _backend("analyst")


def _eligible(today=None):
    """Assets worth paying for a judgment on today.

    The watchlist, because that is what the owner is actually watching, plus
    anything reporting earnings today, because that is when an independent
    read is worth most.

    The earnings half costs one uncached network fetch per ticker across the
    whole asset map, which is slow but free; the calls this throttle exists to
    bound are the paid LLM ones further down. If that scan fails for any
    reason the watchlist still stands on its own, because a degraded
    eligibility list is a smaller loss than a run that produces nothing.

    watchlist._load() is the only reader of watchlist.json and it carries the
    fallback to DEFAULT_LISTS. Reaching past it to json.load would duplicate
    that fallback and drift from it.
    """
    import watchlist
    from config import FULL_ASSET_MAP
    from core import events

    lists = watchlist._load()
    chosen = {a for assets in lists.values() for a in assets}

    day = today or _dt.date.today().isoformat()
    try:
        due = events.earnings_for(dict(FULL_ASSET_MAP))
        chosen |= {a for a, e in due.items() if e.get("date") == day}
    except Exception:
        pass
    return sorted(chosen)


def _judge_one(d, asset, h, horizon, call, depth, cells, table,
               written, refused):
    """One judgment, for one asset over one horizon. Returns the two counters.

    Split out of cmd_run when the horizon became a loop: the same dossier over
    one day and over five is two questions, so it is two model calls and two
    rows, which the log's (date, asset, horizon) primary key has always allowed.
    """
    j = agent.judge(d, call=call, depth=depth, horizon=horizon)
    if j is None:
        return written, refused + 1

    fc = calibrate.forecast(j, cells, asset, radar_category(asset),
                            d["atr"], d["close"], table)
    store.write_judgment({
        "date": d["date"], "asset": asset, "horizon": horizon,
        "direction": j["direction"], "conviction": j["conviction"],
        "vol_regime": j["vol_regime"], "key_risk": j["key_risk"],
        "thesis": j["thesis"],
        "evidence_json": json.dumps(j["evidence"]),
        "dossier_hash": h, "llm_model": os.getenv("GTRADE_AR_LLM", "default"),
        "forecast_pct": fc["pct"], "lo_pct": fc["lo"], "hi_pct": fc["hi"],
        "atr_at_signal": d["atr"], "close_at_signal": d["close"],
    })
    _print_judgment(asset, j, fc, horizon=horizon)
    return written + 1, refused


def cmd_run(args):
    if (os.getenv("GTRADE_ANALYST") or "1").strip() == "0":
        print("[analyst] GTRADE_ANALYST=0, nothing to do.")
        return 0

    table = _load_table()
    if table is None:
        print("[analyst] no payoff_stats.json - run train_payoff.py first.")
        return 1

    # An explicit --llm/--model wins over .env for this run only, so trying a
    # different model never edits the file the next run reads.
    if getattr(args, "llm", None):
        os.environ["GTRADE_AR_LLM"] = args.llm
    if getattr(args, "model", None):
        os.environ["GTRADE_AR_LLM_MODEL"] = args.model

    call = _provider_call()
    store.ensure_table()
    cells = calibrate.fit(store.scored_rows(), table, radar_category)

    # An explicit list is judged as given: no watchlist, no earnings scan, and
    # no dossier-hash skip, because asking for one asset by name means asking
    # for it now rather than being told it was already judged today.
    named = _named_assets(getattr(args, "assets", None))
    targets = named or _eligible()

    # Naming an asset means a person wants to read the reasoning, so it gets
    # the full structured prompt. A sweep wants throughput and gets the brief
    # one. Measured on the local model: 2189s per asset full against 637s
    # brief, which turns a 28-asset watchlist pass from five hours into
    # seventeen. --depth overrides when the default guesses wrong.
    depth = getattr(args, "depth", None) or ("full" if named else "brief")
    print("[analyst] %d asset(s) via %s, depth %s%s" % (
        len(targets), os.getenv("GTRADE_AR_LLM", "anthropic"), depth,
        " (named)" if named else ""))

    try:
        horizons = [int(h) for h in
                    (getattr(args, "horizons", "1") or "1").split(",") if h.strip()]
    except ValueError:
        print("[analyst] --horizons takes whole numbers of trading days.")
        return 1
    as_of = getattr(args, "as_of", None)
    if as_of:
        print("[analyst] as-of %s: the dossier is rewound, so fundamentals, "
              "headlines, the guru verdict and the market classifiers are "
              "blank. They cannot be fetched for a past date and faking them "
              "would be look-ahead." % as_of)

    written = skipped = refused = 0
    # Every network field in the dossier is wrapped in _safe, so a dead source
    # reads as None and the run continues on a quietly thinner dossier. Counting
    # them turns "no internet" from something you notice weeks later in the
    # judgments into one line at the end of the run.
    seen = {k: [0, 0] for k in dossier.NETWORK_BLOCKS}   # [arrived, applicable]
    for asset in targets:
        d = dossier.build(asset, today=as_of)
        for name, filled in dossier.filled_blocks(d).items():
            if filled is None:       # not applicable to this asset, not a miss
                continue
            seen[name][0] += filled
            seen[name][1] += 1
        if d["close"] is None or d["atr"] is None:
            skipped += 1
            continue
        h = dossier.dossier_hash(d)
        for horizon in horizons:
            if not named and store.judged_with_hash(asset, h, horizon=horizon):
                skipped += 1
                continue
            try:
                written, refused = _judge_one(d, asset, h, horizon, call, depth,
                                              cells, table, written, refused)
            except ProviderUnavailable as exc:
                # One line, then stop. A sweep of 28 assets would otherwise
                # print the same missing-package error 28 times and finish
                # claiming 28 refusals.
                print("[analyst] %s" % exc)
                print("[analyst] nothing was asked, so nothing was judged. "
                      "Provider ollama needs no key and no Anthropic package.")
                return 1
    print(f"[analyst] written={written} skipped={skipped} refused={refused}")
    print("[analyst] sources: " + ", ".join(
        "%s %d/%d" % (k, got, n) if n else "%s n/a" % k
        for k, (got, n) in ((k, seen[k]) for k in dossier.NETWORK_BLOCKS)))
    # Only a source that COULD have answered and did not: an index has no
    # earnings and gold no P/E, and calling those a dead connection is how this
    # line first cried wolf on a run where nothing was wrong.
    dead = [k for k in dossier.NETWORK_BLOCKS if seen[k][1] and not seen[k][0]]
    if dead:
        print("[analyst] NOTHING came back from: %s, though every asset judged "
              "could have had it. Check the connection and the VPN route "
              "before trusting these calls." % ", ".join(dead))
    macro = dossier.macro_status()
    if macro:
        print("[analyst] %s" % macro)
    return 0


def _named_assets(raw):
    """Assets named on the command line, validated against the map."""
    if not raw:
        return []
    from config import FULL_ASSET_MAP
    out, unknown = [], []
    for a in (x.strip().upper() for x in raw.replace(";", ",").split(",")):
        if not a:
            continue
        (out if a in FULL_ASSET_MAP else unknown).append(a)
    if unknown:
        print("[analyst] not in the asset map, ignored: " + ", ".join(unknown))
    return out


def _print_judgment(asset, j, fc, horizon=1):
    """The whole opinion, not just a counter.

    A run that prints three totals gives no way to see WHY the analyst said
    what it said, which is the only part of a judgment a person can argue
    with. The number is the cheap half; the reasoning is the half worth
    reading.
    """
    arrow = {"up": "LONG", "down": "SHORT", "flat": "FLAT"}[j["direction"]]
    pct = fc.get("pct")
    lo, hi = fc.get("lo"), fc.get("hi")
    print()
    print("  %-10s %-5s %dd   conviction %d/5   vol %s" % (
        asset, arrow, horizon, j["conviction"], j["vol_regime"]))
    if pct is not None:
        band = ""
        if lo is not None and hi is not None:
            band = "   80%% band %+.2f%% .. %+.2f%%" % (lo * 100, hi * 100)
        # The direction is the model's, the payoff is the empirical table's for
        # that side. They disagreed on 10 of the first 33 judgments and the card
        # printed "LONG ... expected payoff -0.32%" with nothing marking it. On
        # the log so far the agreeing half scores MAE 0.416 against 0.517.
        flag = "" if pct > 0 else "   [the payoff table disagrees]"
        print("             expected payoff %+.2f%%%s%s"
              % (pct * 100, band, flag))
        print("             basis: %s, %d observation(s)" % (
            fc.get("source", "?"), fc.get("n", 0)))
    if j.get("key_risk"):
        print("             risk:  %s" % j["key_risk"])
    for line in _wrap(j.get("thesis") or "", 68):
        print("             %s" % line)
    if j.get("evidence"):
        print("             read:  %s" % ", ".join(j["evidence"]))


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(" ".join(text.split()), width) or []


def cmd_backfill(args):
    print(f"[analyst] filled {store.backfill_outcomes()} outcomes, "
          f"{store.pending_count()} still pending")
    return 0


def _ensemble_signal(con, asset, date):
    """The prediction_log signal for this asset and date, or None.

    None means no matching row - the ensemble baseline has nothing to say
    for this judgment, so the caller excludes the row rather than guessing.
    """
    try:
        row = con.execute(
            "SELECT signal FROM prediction_log WHERE asset=? AND date=? "
            "AND signal IN ('BUY','SELL') LIMIT 1", (asset, date)).fetchone()
    except sqlite3.OperationalError:
        row = None
    return row[0] if row else None


def _score_baselines(table):
    """rows and the three aligned baseline lists (in ATR units), plus the two
    skip counters.

    A row with no forecast is skipped entirely - there is nothing to score.
    A row with a forecast but no matching prediction_log signal still counts
    toward the agent, the zero baseline, the empirical baseline and coverage:
    it carries a None in `ensemble` instead, and mae_atr's own "not None"
    filter is what drops it from that one baseline. Dropping the row itself
    would have shrunk len(rows) and skewed coverage - both of which gate the
    verdict - over nothing but a same-day ensemble gap.
    """
    con = sqlite3.connect(store.DB_PATH)
    rows, zero, empirical, ensemble = [], [], [], []
    no_forecast = no_ensemble_match = 0
    for r in store.scored_rows():
        if r.get("forecast_pct") is None:
            no_forecast += 1
            continue
        r = dict(r)
        atr_sig, close_sig = r["atr_at_signal"], r["close_at_signal"]
        r["forecast_atr"] = payoff.ret_atr(r["forecast_pct"], atr_sig, close_sig)

        r["agent_direction"] = r["direction"]
        asset_class = radar_category(r["asset"])
        judgment = {"direction": r["direction"], "conviction": r["conviction"],
                    "vol_regime": r["vol_regime"]}
        emp = calibrate.forecast(judgment, {}, r["asset"], asset_class,
                                 atr_sig, close_sig, table)

        signal = _ensemble_signal(con, r["asset"], r["date"])
        if signal is None:
            no_ensemble_match += 1
            ens_atr = None
        else:
            ens_direction = "up" if signal == "BUY" else "down"
            r["ensemble_direction"] = ens_direction
            ens_judgment = {**judgment, "direction": ens_direction}
            ens = calibrate.forecast(ens_judgment, {}, r["asset"], asset_class,
                                     atr_sig, close_sig, table)
            ens_atr = payoff.ret_atr(ens["pct"], atr_sig, close_sig)

        rows.append(r)
        zero.append(0.0)
        empirical.append(payoff.ret_atr(emp["pct"], atr_sig, close_sig))
        ensemble.append(ens_atr)
    con.close()
    return (rows, {"zero": zero, "empirical": empirical, "ensemble": ensemble},
            no_forecast, no_ensemble_match)


def cmd_score(args):
    from core.analyst.score import field_usage, standings
    from core.analyst.score import verdict as analyst_verdict

    if not os.path.exists(store.DB_PATH):
        print("[analyst] no market.db - run the data pipeline "
              "(data_engine.py / predict.py) first.")
        return 1

    table = _load_table() or {"asset": {}, "class": {}}
    rows, baselines, no_forecast, no_ensemble_match = _score_baselines(table)

    s = standings(rows, baselines)
    print(f"[analyst] scored rows: {len(rows)} "
          f"({no_forecast} skipped without a forecast; "
          f"{no_ensemble_match} of the {len(rows)} counted still lack a "
          f"matching prediction_log signal and are absent only from the "
          f"ensemble comparison)")
    print(f"[analyst] standings: {s}")

    v = analyst_verdict(s, len(rows))
    print(f"[analyst] verdict: {v['verdict']} on {len(rows)} scored judgments")
    for c in v["checks"]:
        if not c["ok"]:
            print("            missing: %-16s %s" % (c["name"], c["want"]))

    if getattr(args, "fields", False):
        fu = field_usage(rows)
        print(f"[analyst] fields: {len(fu['fields'])} ever cited over "
              f"{fu['n']} directional judgments, {fu['measurable']} with "
              f"enough of both sides to test")
        for name, e in sorted(fu["fields"].items(),
                              key=lambda kv: -kv[1]["cited"]):
            hw = "-" if e["hit_with"] is None else "%.2f" % e["hit_with"]
            ho = "-" if e["hit_without"] is None else "%.2f" % e["hit_without"]
            print("    %-26s cited %3d   hit %s vs %s   %s"
                  % (name, e["cited"], hw, ho, e["verdict"]))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="analyst agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--assets", help="comma-separated assets to judge now, "
                                      "instead of the watchlist")
    run.add_argument("--llm", choices=("anthropic", "openai", "ollama"),
                     help="provider for this run only")
    run.add_argument("--model", help="model name for this run only")
    run.add_argument("--depth", choices=("brief", "full"),
                     help="how much reasoning to ask for. Default: full for a "
                          "named asset, brief for a sweep")
    run.add_argument("--horizons", default="1",
                     help="comma-separated horizons in trading days (default 1). "
                          "Each is its own question and its own LLM call; the "
                          "log's primary key has carried a horizon column since "
                          "it was written and nothing ever used it")
    run.add_argument("--as-of", dest="as_of",
                     help="judge a PAST date (YYYY-MM-DD) instead of today. The "
                          "dossier is rewound: fundamentals, headlines, the guru "
                          "verdict and the three market classifiers cannot be, "
                          "so they come back blank rather than carrying data "
                          "from after the date being judged")
    run.set_defaults(fn=cmd_run)
    sc = sub.add_parser("score")
    sc.add_argument("--fields", action="store_true",
                    help="per-field usage: how often the model cites each "
                         "dossier field and whether citing it pays")
    sc.set_defaults(fn=cmd_score)
    sub.add_parser("backfill").set_defaults(fn=cmd_backfill)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
