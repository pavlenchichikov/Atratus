"""Daily self-maintaining loop orchestrator (Windows Task Scheduler target).

Runs the safe pipeline (data_engine, predict, reconcile), scans drift, and
writes loop_state.json. Each step is isolated: a failure is recorded and the
cycle still produces a report. NEVER retrains - retraining is approved on the
/loop page and run by loop_retrain.py.

Deploy only AFTER the baseline finishes. Register with run_loop.bat / schtasks.
"""
import datetime as _dt
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import FULL_ASSET_MAP
from core import drift, loop_state, runlock
from core.analyst import store

STATE_PATH = os.path.join(BASE, "loop_state.json")
LOCK_PATH = os.path.join(BASE, "_loop.lock")
REGISTRY_PATH = os.path.join(BASE, "models", "champion_registry.json")
QUALITY_PATH = os.path.join(BASE, "models", "quality_report.json")


def run_step(name, fn, fmt=None):
    """Run one pipeline step, capturing any failure so the cycle continues.

    `fmt`, when given, turns fn()'s return value into the "ok" message - so a
    step that silently did nothing (an empty backfill, filling 0 of 500
    pending outcomes) reads differently in the report from one that did the
    work, instead of both looking like the same blank "ok".
    """
    try:
        result = fn()
        return {"step": name, "status": "ok", "msg": fmt(result) if fmt else ""}
    except Exception as exc:
        return {"step": name, "status": "failed", "msg": str(exc)[:200]}


def _refresh_macro():
    """Central bank meeting dates, merged into the calendar. Returns the total.

    Raises when NO source answered, so the cycle reports a failed step rather
    than an "ok" that quietly left the calendar as it was two months ago.
    """
    from core import macro

    fetched, failed = macro.fetch()
    if not fetched:
        raise RuntimeError("no macro source answered: %s"
                           % "; ".join("%s %s" % kv for kv in sorted(failed.items())))
    merged = macro.merge(macro.load(), fetched)
    macro.save(merged)
    return len(merged)


def scan_assets(rows):
    """Map pre-fetched per-asset rows through the pure drift classifier."""
    return [
        drift.classify_asset(
            r["asset"], r.get("acc"), r.get("n", 0), r.get("baseline_acc"),
            r.get("age_days"), r.get("is_stale", False),
            r.get("recent_outcomes", []))
        for r in rows
    ]


def _run_script(script):
    subprocess.run([sys.executable, script], cwd=BASE, check=True)


def _build_rows():
    """Fetch the inputs drift needs for every asset from the live data."""
    import json

    from core import track_record

    reg = {}
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            reg = json.load(f)

    qmap = {}
    if os.path.exists(QUALITY_PATH):
        try:
            with open(QUALITY_PATH, encoding="utf-8") as f:
                qlist = json.load(f)
            qmap = {rec["Asset"]: rec.get("CB_Acc") for rec in qlist}
        except Exception:
            qmap = {}

    stale = {r["asset"] for r in
              track_record.stale_assets(max_age_days=drift.DRIFT_CONFIG["stale_days"])}

    rows = []
    for asset in FULL_ASSET_MAP:
        acc_info = track_record.asset_accuracy(asset)
        track = track_record.asset_track(asset, limit=40)
        outcomes = [int(t["correct"]) for t in reversed(track)
                    if t.get("correct") is not None]
        entry = reg.get(asset) or {}
        baseline = qmap.get(asset)
        age_days = None
        trained = entry.get("updated_at")
        if trained:
            try:
                d = _dt.datetime.strptime(str(trained)[:10], "%Y-%m-%d")
                age_days = (_dt.datetime.utcnow() - d).days
            except Exception:
                age_days = None
        rows.append({
            "asset": asset, "acc": acc_info.get("acc"), "n": acc_info.get("n", 0),
            "baseline_acc": baseline, "age_days": age_days,
            "is_stale": asset in stale, "recent_outcomes": outcomes,
        })
    return rows


def main():
    ok, reason = runlock.acquire(LOCK_PATH, "cycle")
    if not ok:
        print(f"[loop] {reason}; skipping this cycle.")
        return
    try:
        steps = []
        steps.append(run_step("data_engine", lambda: _run_script("data_engine.py")))
        # A published schedule goes stale on its own, and a file somebody has to
        # remember to refresh is the guru_log failure again: 636 verdicts and 14
        # scored outcomes, because scoring lived in a script nobody ran. Safe to
        # repeat - a dead source contributes nothing and hand-added events
        # survive, see core/macro.py.
        steps.append(run_step("macro_calendar", _refresh_macro,
                              fmt=lambda n: f"{n} event(s) on file"))
        steps.append(run_step("predict", lambda: _run_script("predict.py")))
        steps.append(run_step("reconcile", lambda: _run_script("performance_tracker.py")))
        steps.append(run_step("analyst_backfill", store.backfill_outcomes,
                              fmt=lambda n: f"filled {n} outcomes"))

        assets = []
        drift_step = run_step("drift", lambda: assets.extend(scan_assets(_build_rows())))
        steps.append(drift_step)

        proposed = sorted(a["asset"] for a in assets if a["status"] == "propose")
        state = loop_state.load_state(STATE_PATH)
        state["last_run"] = _dt.datetime.utcnow().isoformat(timespec="seconds")
        state["steps"] = {s["step"]: {"status": s["status"], "msg": s["msg"]} for s in steps}
        state["assets"] = assets
        state["proposed"] = proposed
        # keep approvals that are still proposed; drop the rest
        state["approved"] = [a for a in state.get("approved", []) if a in proposed]
        hist = state.get("history", [])
        hist.insert(0, {"ts": state["last_run"], "proposed": len(proposed),
                        "failed_steps": [s["step"] for s in steps if s["status"] == "failed"]})
        state["history"] = hist[:30]
        loop_state.save_state(STATE_PATH, state)
        print("[loop] cycle done. proposed retrains: %d" % len(proposed))
    finally:
        runlock.release(LOCK_PATH)


if __name__ == "__main__":
    main()
