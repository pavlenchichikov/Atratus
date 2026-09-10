"""Quarantine champions that are older than the feature chain that fed them.

    python retire_stale_champions.py            # scan only
    python retire_stale_champions.py --apply    # move them to models/_retired_<stamp>

A champion trained before core/features.py last changed was fitted on features
that no longer mean the same thing. Serving one is worse than serving nothing:
the asset card looks identical to an honest one and there is no way for a person
to tell them apart.

This exists because of 2026-09-09. The weekly features carried up to four days
of future, and removing that leak cost the models most of their confidence: the
fold-admission floor asks for at least 10 trades in a fold, honest probabilities
sit closer to 0.5, fewer signals cross the threshold, and 432 of 847 assets came
back "no robust folds" and kept their OLD champion. Force-promote cannot help -
there is nothing to promote.

Nothing is deleted. The files move, so restoring one is a move back.
"""
import argparse
import datetime as dt
import os
import shutil

import config

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
FEATURES = os.path.join(BASE, "core", "features.py")
CHAMPION_FILES = ("_cb.cbm", "_lstm.keras", "_transformer.keras", "_tcn.keras",
                  "_meta.pkl", "_calib.pkl", "_scaler.pkl")


def _table(asset):
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def stale_assets(model_dir=None, since=None, universe=None):
    """[(asset, age in days)] for champions older than the feature chain.

    `since` defaults to the mtime of core/features.py: a champion older than the
    code that built its inputs cannot have been trained on them.
    """
    model_dir = model_dir or MODEL_DIR
    if since is None:
        since = os.path.getmtime(FEATURES)
    out = []
    for asset in (universe if universe is not None else config.FULL_ASSET_MAP):
        cb = os.path.join(model_dir, _table(asset) + "_cb.cbm")
        if not os.path.exists(cb):
            continue
        m = os.path.getmtime(cb)
        if m < since:
            out.append((asset, (since - m) / 86400.0))
    return sorted(out)


def retire(assets, model_dir=None, dest=None):
    """Move every champion file of these assets aside. Returns files moved."""
    model_dir = model_dir or MODEL_DIR
    dest = dest or os.path.join(
        model_dir, "_retired_" + dt.datetime.now().strftime("%Y%m%d-%H%M"))
    os.makedirs(dest, exist_ok=True)
    moved = 0
    for asset in assets:
        t = _table(asset)
        for suffix in CHAMPION_FILES:
            src = os.path.join(model_dir, t + suffix)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dest, t + suffix))
                moved += 1
    return moved, dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="move the files; without it nothing is touched")
    ap.add_argument("--since", default=None,
                    help="ISO date; default is the mtime of core/features.py")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    since = (dt.datetime.fromisoformat(args.since).timestamp() if args.since
             else os.path.getmtime(FEATURES))
    stale = stale_assets(args.model_dir, since)
    print("cutoff              : %s"
          % dt.datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M"))
    print("champions on disk   : %d" % sum(
        1 for a in config.FULL_ASSET_MAP
        if os.path.exists(os.path.join(args.model_dir, _table(a) + "_cb.cbm"))))
    print("older than the chain: %d" % len(stale))
    if stale:
        print("  oldest:")
        for a, age in sorted(stale, key=lambda x: -x[1])[:8]:
            print("     %-12s %.1f days older" % (a, age))

    if not args.apply:
        print()
        print("scan only, nothing moved. --apply quarantines them; those assets "
              "then produce no signal until they train successfully.")
        return 0

    moved, dest = retire([a for a, _ in stale], args.model_dir)
    print()
    print("moved %d file(s) for %d asset(s) to %s"
          % (moved, len(stale), os.path.relpath(dest, BASE)))
    print("Restoring one is a move back. Run [3] Predict to refresh the radar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
