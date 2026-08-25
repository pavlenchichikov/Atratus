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


def cmd_run(args):
    if (os.getenv("GTRADE_ANALYST") or "1").strip() == "0":
        print("[analyst] GTRADE_ANALYST=0, nothing to do.")
        return 0

    table = _load_table()
    if table is None:
        print("[analyst] no payoff_stats.json - run train_payoff.py first.")
        return 1

    call = _provider_call()
    store.ensure_table()
    cells = calibrate.fit(store.scored_rows(), table, radar_category)
    written = skipped = refused = 0
    for asset in _eligible():
        d = dossier.build(asset)
        if d["close"] is None or d["atr"] is None:
            skipped += 1
            continue
        h = dossier.dossier_hash(d)
        if store.judged_with_hash(asset, h):
            skipped += 1
            continue

        j = agent.judge(d, call=call)
        if j is None:
            refused += 1
            continue

        fc = calibrate.forecast(j, cells, asset, radar_category(asset),
                                d["atr"], d["close"], table)

        store.write_judgment({
            "date": d["date"], "asset": asset, "horizon": 1,
            "direction": j["direction"], "conviction": j["conviction"],
            "vol_regime": j["vol_regime"], "key_risk": j["key_risk"],
            "thesis": j["thesis"],
            "evidence_json": json.dumps(j["evidence"]),
            "dossier_hash": h, "llm_model": os.getenv("GTRADE_AR_LLM", "default"),
            "forecast_pct": fc["pct"], "lo_pct": fc["lo"], "hi_pct": fc["hi"],
            "atr_at_signal": d["atr"], "close_at_signal": d["close"],
        })
        written += 1
    print(f"[analyst] written={written} skipped={skipped} refused={refused}")
    return 0


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
    from core.analyst.score import standings

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

    verdict = "SHIP" if (
        s["control"]["survives_shuffle"] is False
        and s["coverage"]["rate"] is not None
        and 0.75 <= s["coverage"]["rate"] <= 0.85
        and "zero" in s["agent"]["beats"]
        and "empirical" in s["agent"]["beats"]
        and len(rows) >= 500
    ) else "HOLD"
    print(f"[analyst] verdict: {verdict} on {len(rows)} scored judgments")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="analyst agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    sub.add_parser("backfill").set_defaults(fn=cmd_backfill)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
