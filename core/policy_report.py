"""What the policy layers decided, and what that was worth on live signals.

Two different questions live here and they must not be confused.

`reports()` reads what each offline fit last concluded: the timing rules, the
fitted-Q challenger, the online generation stack and the sizing rule. Those are
BACKTEST verdicts over reconstructed history.

`reconcile()` answers the other one: on the signals production actually emitted
and the returns that actually happened, what was each layer worth? It reads
`prediction_log`, which stores the emitted signal, the shadow timing decision
and `actual_next_ret` per asset per day, and it charges the same commission and
slippage the backtest charges by running the same simulator.

Coverage is reported, never assumed. A layer with no rows reads as "no data",
which is a different answer from "no effect", and the two were confused once
already in this project.
"""
import json
import os

import numpy as np

from core.backtesting import (
    COMMISSION,
    FOREX_COMMISSION,
    FOREX_SLIPPAGE,
    SLIPPAGE,
    max_drawdown_from_returns,
    sharpe_from_returns,
    simulate_positions,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (file, what it is)
REPORT_FILES = (
    ("timing_rules", "timing_report.json",
     "Stage A: the interpretable timing rules, fitted by ES"),
    ("timing_fqi", "timing_fqi_report.json",
     "Stage B: the fitted-Q challenger to those rules"),
    ("timing_online", "timing_online.json",
     "Stage C: the online generation stack and its trust region"),
    ("sizing", "sizing_report.json",
     "SP-3a: how big a position is, at matched exposure"),
    ("direction", "direction_report.json",
     "SP-3b: whether to follow the ensemble at all, fitted on LIVE outcomes"),
)

_SIDE = {"BUY": 1, "SELL": -1, "WAIT": 0}


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def reports(base=None):
    """One row per policy: its verdict, its headline numbers, when it ran.

    A missing file is reported as missing rather than skipped: a policy nobody
    has fitted yet and a policy that failed are different states.
    """
    base = base or BASE
    out = []
    for name, fname, what in REPORT_FILES:
        blob = _read(os.path.join(base, fname))
        row = {"name": name, "what": what, "file": fname,
               "present": blob is not None}
        if blob:
            row["verdict"] = blob.get("verdict") or (
                "generation %s" % blob.get("generation")
                if "generation" in blob else None)
            row["fitted"] = blob.get("fitted") or blob.get("ts")
            row["params"] = blob.get("params")
            row["n"] = blob.get("n") or blob.get("assets")
            row["p"] = blob.get("p")
            row["mean_d"] = blob.get("mean_d")
            per = blob.get("per_asset") or {}
            if per:
                vals = sorted(per.values())
                row["median_d"] = vals[len(vals) // 2]
                row["up"] = sum(1 for v in vals if v > 0)
                row["down"] = sum(1 for v in vals if v < 0)
        out.append(row)
    return out


def _costs(asset, forex):
    if asset in forex:
        return FOREX_COMMISSION, FOREX_SLIPPAGE
    return COMMISSION, SLIPPAGE


def _timing_sides(signals, actions):
    """Positions the SHADOW timing decision held, from what was logged.

    The logged action is the decision the serve path actually took that day, so
    nothing is recomputed here and nothing can be improved with hindsight.
    """
    pos, out = 0, []
    for sig, action in zip(signals, actions):
        if action and action.startswith("ENTER"):
            # The log writes the SIDE into the action, "ENTER:+1"/"ENTER:-1",
            # and only that form has ever been written. Matching a bare "ENTER"
            # never fired, so entries fell through to "the raw signal stands"
            # and a policy entering while the signal was flat recorded no
            # position at all: 24 of 637 logged entries as of 2026-08-23.
            pos = (1 if action.endswith("+1")
                   else -1 if action.endswith("-1") else sig)
        elif action in ("EXIT", "STAY_OUT"):
            pos = 0
        elif action != "HOLD":
            pos = sig          # no decision logged: the raw signal stands
        out.append(pos)
    return np.asarray(out, dtype=int)


def timing_hits(raw, sides, next_ret):
    """Was each timing decision right? THE definition, shared with the fit.

    A timing policy does not predict a direction, so "hit" has to be defined
    for what it DOES decide:

      in a position   the next bar moved the way the position is facing
      flat, signal on staying out was right, because the position the raw
                      signal wanted would not have made money

    Bars where the raw signal is flat and the policy is flat are not a decision
    either way and are left out, so the rate is over bars where something was
    actually chosen. `train_timing.hit_stats` computes `raw` from the
    probabilities and thresholds and then calls this, so the number the live
    journal shows and the number a fit reports cannot drift apart.

    `verdicts` is the per-bar answer, True/False/None, which is what lets a row
    in the trade log say correct or missed the way a signal's row does.
    """
    raw = np.asarray(raw, dtype=int)
    sides = np.asarray(sides, dtype=int)
    nr = np.asarray(next_ret, dtype=float)

    live = np.isfinite(nr)
    held = live & (sides != 0)
    flat_with_signal = live & (sides == 0) & (raw != 0)

    verdicts = np.full(len(nr), None, dtype=object)
    verdicts[held] = (sides[held] * nr[held] > 0)
    verdicts[flat_with_signal] = (raw[flat_with_signal] * nr[flat_with_signal] <= 0)

    held_hits = int((sides[held] * nr[held] > 0).sum())
    avoided = int((raw[flat_with_signal] * nr[flat_with_signal] <= 0).sum())
    n_held, n_flat = int(held.sum()), int(flat_with_signal.sum())
    decided = n_held + n_flat
    earned = float((sides[live] * nr[live]).sum())
    return {
        "bars": int(live.sum()), "held": n_held, "held_hits": held_hits,
        "flat_with_signal": n_flat, "avoided_losses": avoided,
        "decided": decided, "hits": held_hits + avoided,
        "accuracy": (held_hits + avoided) / decided if decided else float("nan"),
        "held_accuracy": held_hits / n_held if n_held else float("nan"),
        "avoid_accuracy": avoided / n_flat if n_flat else float("nan"),
        "ret_per_bar": earned / int(live.sum()) if live.any() else 0.0,
        "verdicts": verdicts,
    }


def live_timing_hits(rows, column="timing_action"):
    """The same reading, over journal rows instead of a reconstructed series.

    `rows` are prediction_log dicts, oldest first, each with `signal`,
    `actual_next_ret` and the action column. `column` picks whose decisions:
    the served ones, or a watched challenger's in `shadow_action`.

    A row with no action logged, or no realised return yet, gets a verdict of
    None: not wrong, not right, not yet a decision to score.
    """
    raw = np.asarray([_SIDE.get((r.get("signal") or "").upper(), 0)
                      for r in rows], dtype=int)
    actions = [r.get(column) for r in rows]
    nr = np.asarray([float(r["actual_next_ret"])
                     if r.get("actual_next_ret") is not None else np.nan
                     for r in rows], dtype=float)
    out = timing_hits(raw, _timing_sides(raw, actions), nr)
    # A bar the policy never spoke on is not a decision, whatever the position
    # carried into it happened to be.
    for i, a in enumerate(actions):
        if not a:
            out["verdicts"][i] = None
    return out


def _stats(sides, rets, comm, slip, sizes=None):
    profit, trades, winrate, daily = simulate_positions(
        sides, rets, commission=comm, slippage=slip, sizes=sizes)
    return {"profit": profit, "trades": trades, "winrate": winrate,
            "max_dd": max_drawdown_from_returns(daily),
            "sharpe": sharpe_from_returns(daily)}


def per_asset_profit(rows, forex=()):
    """{asset: profit} on the emitted signals, charged the same as the backtest.

    Per asset rather than pooled because every gate in this project is a PAIRED
    test over assets: one asset that moved a lot must not be able to carry a
    verdict on its own.
    """
    by_asset = {}
    for r in rows:
        if r.get("actual_next_ret") is None:
            continue
        by_asset.setdefault(r["asset"], []).append(r)
    out = {}
    for asset, rs in by_asset.items():
        rs.sort(key=lambda r: r["date"])
        sig = np.asarray([_SIDE.get((r.get("signal") or "").upper(), 0)
                          for r in rs], dtype=int)
        ret = np.asarray([float(r["actual_next_ret"]) for r in rs], dtype=float)
        if not len(ret):
            continue
        # A book that never takes a position earns exactly zero, which is a
        # number and a comparable one. Dropping it would make "stand aside"
        # vanish from a paired comparison instead of scoring, and a gate that
        # cannot see an arm cannot reject it either.
        if not sig.any():
            out[asset] = 0.0
            continue
        comm, slip = _costs(asset, forex)
        out[asset] = _stats(sig, ret, comm, slip)["profit"]
    return out


def reconcile(rows, forex=(), sizing=None):
    """What each layer was worth on the signals production actually emitted.

    `rows` is an iterable of dicts with asset, date, signal, probability,
    actual_next_ret and timing_action, i.e. prediction_log. Returns per-arm
    totals plus the coverage each arm actually had, because an arm with no
    logged decisions has no result rather than a neutral one.
    """
    by_asset = {}
    for r in rows:
        if r.get("actual_next_ret") is None:
            continue
        by_asset.setdefault(r["asset"], []).append(r)

    # Stage A and Stage B are separate arms, never one "timing" number. They ran
    # over different date ranges, so blending them would report a policy that
    # never existed. Rows written before the stage column arrived carry no stage
    # and are Stage A's: it is the only one that had ever served.
    keys = ("emitted", "timing A", "timing B", "timing B (watched)", "sizing")
    arms = {k: [] for k in keys}
    covered = dict.fromkeys(keys, 0)
    assets = dict.fromkeys(keys, 0)
    for asset, rs in by_asset.items():
        rs.sort(key=lambda r: r["date"])
        sig = np.asarray([_SIDE.get((r.get("signal") or "").upper(), 0)
                          for r in rs], dtype=int)
        ret = np.asarray([float(r["actual_next_ret"]) for r in rs], dtype=float)
        if not len(ret) or not sig.any():
            continue
        comm, slip = _costs(asset, forex)

        arms["emitted"].append(_stats(sig, ret, comm, slip))
        covered["emitted"] += len(ret)
        assets["emitted"] += 1

        # ("arm", column, stage wanted). The watched arm reads its OWN column:
        # under GTRADE_TIMING_STAGE=shadow the rules serve and the Q is only
        # recorded, so its decisions sit in shadow_action beside them, on the
        # very same bars. That is the one comparison the live log can make
        # without either policy having decided the other's days.
        for arm, col, want in (("timing A", "timing_action", "A"),
                               ("timing B", "timing_action", "B"),
                               ("timing B (watched)", "shadow_action", None)):
            # Each stage is scored ONLY on the rows it actually decided, not on
            # the whole history with the other stage's days filled in by the raw
            # signal. Filling them made the answer depend on where the stage's
            # window happened to start: a window opening on a HOLD has no
            # position to hold, so the same policy read -0.72 or -0.03 purely by
            # position in the log. Different days is a fact to state, not a hole
            # to paper over.
            idx = [k for k, r in enumerate(rs)
                   if r.get(col)
                   and (want is None or (r.get("timing_stage") or "A") == want)]
            if not idx:
                continue
            arm_sig = sig[idx]
            arm_ret = ret[idx]
            arm_actions = [rs[k][col] for k in idx]
            arms[arm].append(_stats(_timing_sides(arm_sig, arm_actions),
                                    arm_ret, comm, slip))
            covered[arm] += len(idx)
            assets[arm] += 1

        if sizing is not None:
            probs = np.asarray([float(r.get("probability") or 0.5)
                                for r in rs], dtype=float)
            sizes = sizing(asset, rs, probs, sig)
            if sizes is not None:
                arms["sizing"].append(_stats(sig, ret, comm, slip, sizes=sizes))
                covered["sizing"] += len(ret)
                assets["sizing"] += 1

    out = {}
    for name, stats in arms.items():
        if not stats:
            out[name] = {"rows": 0, "assets": 0, "status": "no data"}
            continue
        out[name] = {
            "rows": covered[name], "assets": assets[name], "status": "measured",
            "profit": float(np.mean([s["profit"] for s in stats])),
            "trades": int(sum(s["trades"] for s in stats)),
            "winrate": float(np.mean([s["winrate"] for s in stats])),
            "sharpe": float(np.mean([s["sharpe"] for s in stats])),
            "max_dd": float(np.mean([s["max_dd"] for s in stats])),
        }
    return out
