"""
Model Health Monitor - Atratus
=================================================
Checks age, quality, and drift of trained models.

CLI usage:
  python model_health.py              -- full health report
  python model_health.py --stale 7    -- show models older than 7 days
  python model_health.py --json       -- JSON output for GUI
  python model_health.py --mismatched -- assets whose files and registry disagree
"""

import argparse
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import FULL_ASSET_MAP

MODEL_DIR = os.path.join(BASE_DIR, "models")
REGISTRY_PATH = os.path.join(MODEL_DIR, "champion_registry.json")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "tuned_thresholds.json")


def _load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _table_name(asset):
    """Convert asset key to the filename prefix used for model files."""
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _file_age_days(path):
    """Return file age in fractional days, or None if file does not exist."""
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (datetime.now() - datetime.fromtimestamp(mtime)).total_seconds() / 86400


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_health_summary():
    """Return a dict with overall model health statistics."""
    registry = _load_json(REGISTRY_PATH)
    now = datetime.now()

    cb_count = 0
    lstm_count = 0
    ages = []

    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb_path = os.path.join(MODEL_DIR, f"{tbl}_cb.cbm")
        lstm_path = os.path.join(MODEL_DIR, f"{tbl}_lstm.keras")

        cb_age = _file_age_days(cb_path)
        lstm_age = _file_age_days(lstm_path)

        if cb_age is not None:
            cb_count += 1
            ages.append((asset, cb_age))
        if lstm_age is not None:
            lstm_count += 1
            ages.append((asset, lstm_age))

    avg_age = sum(a for _, a in ages) / len(ages) if ages else 0.0
    oldest = max(ages, key=lambda x: x[1]) if ages else (None, 0)

    scores = []
    for asset, entry in registry.items():
        s = entry.get("score")
        if s is not None:
            scores.append((asset, s))
    scores.sort(key=lambda x: x[1], reverse=True)

    return {
        "cb_count": cb_count,
        "lstm_count": lstm_count,
        "avg_age_days": round(avg_age, 1),
        "oldest_asset": oldest[0],
        "oldest_age_days": round(oldest[1], 1) if oldest[0] else 0,
        "registry_entries": len(registry),
        "best_score": scores[0] if scores else None,
        "worst_score": scores[-1] if scores else None,
        "timestamp": now.isoformat(),
    }


def get_stale_models(max_age_days=7):
    """Return list of dicts for assets whose models are older than max_age_days."""
    registry = _load_json(REGISTRY_PATH)
    stale = []

    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb_age = _file_age_days(os.path.join(MODEL_DIR, f"{tbl}_cb.cbm"))
        lstm_age = _file_age_days(os.path.join(MODEL_DIR, f"{tbl}_lstm.keras"))

        if cb_age is None and lstm_age is None:
            continue

        max_model_age = max(a for a in [cb_age, lstm_age] if a is not None)
        if max_model_age >= max_age_days:
            entry = registry.get(asset, {})
            stale.append({
                "asset": asset,
                "cb_age_days": round(cb_age, 1) if cb_age is not None else None,
                "lstm_age_days": round(lstm_age, 1) if lstm_age is not None else None,
                "score": entry.get("score"),
                "status": "RETRAIN",
            })

    stale.sort(key=lambda x: max(x["cb_age_days"] or 0, x["lstm_age_days"] or 0), reverse=True)
    return stale


def get_missing_models():
    """Return list of asset names that have no .cbm and no .keras file."""
    missing = []
    for asset in FULL_ASSET_MAP:
        tbl = _table_name(asset)
        cb = os.path.exists(os.path.join(MODEL_DIR, f"{tbl}_cb.cbm"))
        lstm = os.path.exists(os.path.join(MODEL_DIR, f"{tbl}_lstm.keras"))
        if not cb and not lstm:
            missing.append(asset)
    return missing


def get_quality_ranking():
    """Return list of dicts sorted by score descending from champion_registry."""
    registry = _load_json(REGISTRY_PATH)
    ranking = []

    for asset, entry in registry.items():
        score = entry.get("score")
        if score is None:
            continue
        ranking.append({
            "asset": asset,
            "score": round(score, 2),
            "policy": entry.get("policy", "UNKNOWN"),
            "updated_at": entry.get("updated_at", ""),
        })

    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def _score_color(score):
    if score >= 5.0:
        return "\033[92m"   # green
    if score >= 2.0:
        return "\033[93m"   # yellow
    if score >= 0:
        return "\033[33m"   # dark yellow
    return "\033[91m"       # red

_RST = "\033[0m"
_W   = 62


def _print_report(max_age_days=7):
    summary = get_health_summary()
    stale   = get_stale_models(max_age_days)
    missing = get_missing_models()
    ranking = get_quality_ranking()

    now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
    print()
    print("=" * _W)
    print(f"  MODEL HEALTH  |  {now}")
    print("=" * _W)

    # -- SUMMARY ---------------------------------------------------
    oldest_lbl = summary['oldest_asset'] or "N/A"
    best  = summary["best_score"]
    worst = summary["worst_score"]
    print()
    print(f"  Models  : {summary['cb_count']} CatBoost + {summary['lstm_count']} LSTM"
          f"   |   Registry: {summary['registry_entries']} entries")
    print(f"  Avg age : {summary['avg_age_days']}d"
          f"   |   Oldest: {oldest_lbl} ({summary['oldest_age_days']}d)")
    if best and worst:
        print(f"  Best    : {best[0]} ({best[1]:+.2f})"
              f"   |   Worst: {worst[0]} ({worst[1]:+.2f})")
    print()

    # -- STALE MODELS ----------------------------------------------
    tag_stale = f"-- STALE  (>{max_age_days}d) "
    print("  " + tag_stale + "-" * max(0, _W - 2 - len(tag_stale)))
    if stale:
        print(f"  {'Asset':<10}  {'CB':<7}  {'LSTM':<7}  {'Score':>6}  Status")
        print("  " + "-" * (_W - 4))
        for s in stale:
            cb_s   = f"{s['cb_age_days']}d"   if s["cb_age_days"]   is not None else "-"
            lstm_s = f"{s['lstm_age_days']}d" if s["lstm_age_days"] is not None else "-"
            sc_s   = f"{s['score']:+.2f}"     if s["score"]         is not None else "N/A"
            print(f"  {s['asset']:<10}  {cb_s:<7}  {lstm_s:<7}  {sc_s:>6}  {s['status']}")
    else:
        print("  All models are fresh.")
    print()

    # -- MISSING MODELS --------------------------------------------
    print("  -- MISSING ---------------------------------------------")
    if missing:
        for m in missing:
            print(f"  \033[91m{m}\033[0m  - no .cbm / .keras found")
    else:
        print("  All assets have models.")
    print()

    # -- QUALITY RANKING -------------------------------------------
    print("  -- QUALITY RANKING -------------------------------------")
    print(f"  {'Asset':<10}  {'Score':>7}  {'Policy':<12}  Updated")
    print("  " + "-" * (_W - 4))
    for r in ranking:
        updated_s = r["updated_at"][:10] if r["updated_at"] else "N/A"
        clr = _score_color(r["score"])
        score_s = f"{r['score']:+.2f}"
        print(f"  {r['asset']:<10}  {clr}{score_s:>7}{_RST}  {r['policy']:<12}  {updated_s}")
    print()


def _print_json(max_age_days=7):
    data = {
        "summary": get_health_summary(),
        "stale": get_stale_models(max_age_days),
        "missing": get_missing_models(),
        "ranking": get_quality_ranking(),
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def mismatched_registry(base=None):
    """Assets whose champion FILES are newer than the registry entry describing
    them, newest first. Empty is the healthy answer.

    The two used to be written at different times: model files per asset as each
    was promoted, the registry once at the end of the run. Any interruption in
    between left an asset whose .cbm on disk expects one feature count while the
    entry the serving path builds its pool from names another, and CatBoost then
    refuses with "Feature N is present in model but not in pool" - the asset
    silently vanishes from the signals. train_hybrid now writes the entry with
    the files, so this can only report damage done before that.
    """
    base = base or BASE_DIR
    path = os.path.join(base, "models", "champion_registry.json")
    try:
        with open(path, encoding="utf-8") as fh:
            registry = json.load(fh)
    except (OSError, ValueError):
        return []
    out = []
    for asset, entry in (registry or {}).items():
        model = os.path.join(base, "models", "%s_cb.cbm" % asset.lower())
        stamp = (entry or {}).get("updated_at")
        if not stamp or not os.path.exists(model):
            continue
        mtime = datetime.fromtimestamp(os.path.getmtime(model)).isoformat()
        if mtime > stamp:
            out.append({"asset": asset, "registry": stamp[:19], "model": mtime[:19]})
    return sorted(out, key=lambda r: r["model"], reverse=True)


def print_mismatched():
    """One line per damaged asset, plus the list to paste into the retrain."""
    rows = mismatched_registry()
    if not rows:
        print("  registry and model files agree on every asset.")
        return []
    print("  %-10s %-20s %s" % ("asset", "registry entry", "model file"))
    for r in rows:
        print("  %-10s %-20s %s" % (r["asset"], r["registry"], r["model"]))
    print()
    print("  retrain these with force-promote to rewrite both together:")
    print("  " + ",".join(r["asset"] for r in rows))
    return rows


def degraded_members(base=None):
    """Assets whose neural champions do not LOAD where serving loads them.

    mismatched_registry compares two timestamps, and timestamps can agree while
    the file is unreadable: training runs under keras 2.10 in the GPU
    environment and writes a legacy HDF5 under a .keras name, serving runs under
    keras 3, and of the three sequence members only the LSTM path has a rebuild
    fallback. Measured 2026-08-21: 49 assets were serving on CatBoost alone and
    every one of the 55 legacy assets had lost its TCN, while --mismatched
    reported a clean registry. A check that never opens a file cannot see this.

    Slow on purpose - it loads what serving loads, in the environment serving
    runs in, which is the only way the answer means anything.
    """
    from core.model_io import (
        get_lookback,
        load_lstm_model,
        load_tcn_model,
        load_transformer_model,
    )
    base = base or BASE_DIR
    mdir = os.path.join(base, "models")
    registry = _load_json(os.path.join(mdir, "champion_registry.json")) or {}
    out = []
    for asset, entry in registry.items():
        table = _table_name(asset)
        n_features = len((entry or {}).get("features") or [])
        lookback = get_lookback(entry or {}, asset)
        lost = []
        loaders = (
            ("lstm", lambda p: load_lstm_model(p, lookback, n_features)[0]),
            ("transformer", lambda p: load_transformer_model(p, lookback, n_features)),
            ("tcn", lambda p: load_tcn_model(p, lookback, n_features)),
        )
        for member, load in loaders:
            path = os.path.join(mdir, "%s_%s.keras" % (table, member))
            if not os.path.exists(path):
                continue
            try:
                model = load(path)
            except Exception:
                model = None
            if model is None:
                lost.append(member)
        if lost:
            out.append({"asset": asset, "lost": lost})
    return sorted(out, key=lambda r: (-len(r["lost"]), r["asset"]))


def print_degraded():
    """One line per asset serving without some of its neural members."""
    rows = degraded_members()
    if not rows:
        print("  every champion loads in this environment.")
        return []
    cb_only = [r for r in rows if len(r["lost"]) == 3]
    print("  %-10s %s" % ("asset", "members that did not load"))
    for r in rows:
        print("  %-10s %s" % (r["asset"], ", ".join(r["lost"])))
    print()
    print("  %d of %d assets are degraded; %d serve on CatBoost alone."
          % (len(rows), len(_load_json(os.path.join(BASE_DIR, "models",
                                                    "champion_registry.json")) or {}),
             len(cb_only)))
    if cb_only:
        print("  retrain these with force-promote to rewrite files and entry "
              "together:")
        print("  " + ",".join(r["asset"] for r in cb_only))
    return rows


def print_missing():
    """Assets in the map with no CatBoost champion at all.

    Not the same population as --mismatched or --degraded: those are assets
    that HAVE a champion and something is wrong with it. These have never been
    trained, so every fit that rebuilds an environment from champion
    probabilities skips them silently and reports a smaller n without saying
    which names are absent.
    """
    import os

    from config import FULL_ASSET_MAP
    from core.track_record import _table_name

    missing = [a for a in FULL_ASSET_MAP
               if not os.path.exists(os.path.join(
                   MODEL_DIR, "%s_cb.cbm" % _table_name(a)))]
    if not missing:
        print("  every asset in the map has a champion.")
        return missing
    print("  %d of %d assets have no champion yet:"
          % (len(missing), len(FULL_ASSET_MAP)))
    print("  " + ",".join(missing))
    print()
    print("  These need a plain training run, NOT force-promote: there is no")
    print("  incumbent to beat, so the first model is promoted on its own.")
    return missing


def main():
    parser = argparse.ArgumentParser(description="Model Health Monitor")
    parser.add_argument("--stale", type=int, default=7,
                        help="Flag models older than N days (default: 7)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")
    parser.add_argument("--mismatched", action="store_true",
                        help="List assets whose model files are newer than their "
                             "champion-registry entry, which drops them from the "
                             "signals with a feature-count error")
    parser.add_argument("--missing", action="store_true",
                        help="List assets in the map that have no CatBoost "
                             "champion at all, so every policy fit skips them")
    parser.add_argument("--degraded", action="store_true",
                        help="List assets whose neural champions do not load "
                             "here, so they serve on fewer members than the "
                             "registry claims. Slow: it opens every file.")
    args = parser.parse_args()

    if args.missing:
        print_missing()
        return
    if args.degraded:
        print_degraded()
        return
    if args.mismatched:
        print_mismatched()
        return
    if args.json:
        _print_json(max_age_days=args.stale)
    else:
        _print_report(max_age_days=args.stale)


if __name__ == "__main__":
    main()
