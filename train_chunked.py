"""
train_chunked.py  -  local chunked trainer (gitignored)
=======================================================
Trains assets in small chunks, restarting a fresh train_hybrid.py process per
chunk so TensorFlow memory is fully released between chunks. The multi-thread
trainer cannot safely call clear_session mid-run (it would destroy the models
of the other worker threads), so a per-chunk process restart is the reliable
way to keep RAM flat over a long run.

Resumable: each completed chunk is recorded in _chunk_progress.txt; rerun to
pick up where it stopped. quality_report.json is merged across chunks so the
final report covers every asset. Adopting or reverting a genome clears both
files, because a new genome means every asset must be retrained.

Promotion is champion-challenger by default: an asset keeps its existing champion
unless the new model beats it. That is what turns a genome validated on a small
holdout into a per-asset check instead of a blanket replacement. Pass
--force-promote to replace every champion regardless, which is what registry
recovery and a baseline rebuild need.

Run:  python train_chunked.py        (close browser / Docker first for RAM)
Tune: CHUNK_SIZE=10 python train_chunked.py
      python train_chunked.py --jobs 2   (two chunk processes at once)

--jobs is the same parallelism the unattended campaign runs its training under
(GTRADE_AR_TRAIN_JOBS=2 in auto_loop.CAMPAIGN, measured 27% faster there on a
host-bound workload). It divides the box between the processes rather than
asking for more of it - see _chunk_env - so the assets training at once stay at
4 and the VRAM pools stay inside the card. The gain comes from overlapping what
one process serialises behind the GIL: feature building, the database reads,
CatBoost.

Two is the ceiling on a 4 GB card, and it is the default. Drop to `--jobs 1` if
a chunk dies out of memory; that is the one configuration that has trained this
asset list end to end at full width.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from config import FULL_ASSET_MAP

PROGRESS = os.path.join(BASE, "_chunk_progress.txt")
QPATH = os.path.join(BASE, "models", "quality_report.json")
QCUM = os.path.join(BASE, "_chunk_quality.json")
CHUNK = int(os.getenv("CHUNK_SIZE", "15"))

# RAM-safe light profile for the data-rich remainder. Defaults in train_hybrid
# reproduce the old heavy config, so these env knobs are what make it lighter.
LIGHT_ENV = {
    # GTRADE_LIGHT, not a hand-rolled clamp. NEURAL_SLOTS=4 and CB_THREADS=1
    # used to be pinned here; train_hybrid's own comment calls that "the old
    # caller-side clamp ... which on a GPU box starves CatBoost of every core
    # while risking four concurrent cuDNN contexts" - but it was only ever
    # removed on the train_hybrid side, so this file kept forcing exactly that
    # on a GPU. Those two numbers are the CPU-path values, and train_hybrid
    # already derives them per device under this flag: on CPU it still picks
    # 4 slots / 1 CB thread, on GPU 1 slot (concurrent cuDNN OOMs outside the
    # TF pool) and half the cores for CatBoost.
    "GTRADE_LIGHT": "1",
    "GTRADE_WORKERS": "4",
    "GTRADE_TF_THREADS": "2",
    "GTRADE_MAX_FOLDS": "5",
    "GTRADE_ADAPTIVE_NETS": "1",
    "GTRADE_NET_WARMSTART": "1",
    "GTRADE_NET_CAP": "80",
    "GTRADE_EPOCHS_LSTM": "90",
    "GTRADE_EPOCHS_TF": "60",
    "GTRADE_EPOCHS_TCN": "50",
    "TF_CPP_MIN_LOG_LEVEL": "2",
}


def _done():
    if not os.path.exists(PROGRESS):
        return set()
    with open(PROGRESS, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _merge_quality():
    """Fold this chunk's quality_report into the cumulative one so the final
    report covers all assets instead of only the last chunk."""
    cum = {}
    if os.path.exists(QCUM):
        try:
            with open(QCUM, encoding="utf-8") as fh:
                cum = {r["Asset"]: r for r in json.load(fh)}
        except Exception:
            cum = {}
    if os.path.exists(QPATH):
        try:
            with open(QPATH, encoding="utf-8") as fh:
                for r in json.load(fh):
                    cum[r["Asset"]] = r
        except Exception:
            pass
    recs = list(cum.values())
    for path in (QCUM, QPATH):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)


LOG_DIR = os.path.join(BASE, "_chunk_logs")


def _pool_base():
    """The VRAM pool share this box trains at, before it is divided by jobs.

    Read from the campaign profile so the chunked retrain and the unattended
    research run load the card identically. An explicit GTRADE_TF_POOL_PCT in
    the environment still wins, because a person tuning one run should not have
    to edit the campaign to do it.
    """
    env = (os.getenv("GTRADE_TF_POOL_PCT") or "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        from auto_loop import CAMPAIGN

        return float(CAMPAIGN["GTRADE_TF_POOL_PCT"])
    except Exception:
        return 0.34


def _fit_jobs(jobs):
    """Jobs the card can actually take, asked before anything is launched.

    train_chunked had no such check while the research path did, so the same
    two-process stall was still reachable from the menu: a local LLM server
    holding half the card is enough, and the symptom is not an error but a run
    that never finishes. Falls back to the requested count when there is no GPU
    or nvidia-smi cannot answer.
    """
    if jobs <= 1:
        return jobs
    try:
        from auto_research import free_vram_mb, gpu_fit_jobs

        fits = gpu_fit_jobs(jobs, pool_pct=_pool_base())
    except Exception:
        return jobs
    if fits < jobs:
        print("  [GPU] %s MiB free fits %d chunk process(es), not %d. "
              "Running %d." % (free_vram_mb(), fits, jobs, fits))
    return fits


def _chunk_env(chunk, force_promote, jobs):
    """One chunk process's environment.

    Everything sized against the WHOLE box has to be divided, or the second
    process meets a card that is already full. Measured on this machine with one
    process: 3507 of 4096 MiB used, 589 free, which is nowhere near a second
    copy of itself.

      GTRADE_TF_POOL_PCT  shrunk FIRST, from the SAME base the unattended
                          campaign uses rather than a number written down here.
                          Two launchers of one trainer disagreeing about the
                          load is how the identical arm took 27% longer purely
                          because of who started it. train_hybrid reserves that
                          share of FREE VRAM and cuDNN then allocates its
                          workspaces OUTSIDE the pool, so the pair measured
                          3956 of 4096 MiB at 0.25 each - 140 MiB of headroom,
                          which is what stalled on 2026-08-24. The campaign
                          base is now 0.34, putting the pair near 3.3 GB.
      GTRADE_WORKERS      halved, so the assets training at once stay at 4. The
                          gain is overlap of what one process serialises behind
                          the GIL, not more work on the GPU.
      GTRADE_NEURAL_SLOTS pinned to 1. The parallelism is the process count; a
                          second slot INSIDE a process is what emptied 27
                          genomes on 2026-08-17 by handing models the wrong
                          sequence length.
    """
    env = dict(os.environ)
    env.update(LIGHT_ENV)
    if force_promote:
        env["GTRADE_FORCE_PROMOTE"] = "1"
    env["GTRADE_ASSETS"] = ",".join(chunk)
    if jobs > 1:
        env["GTRADE_WORKERS"] = str(max(1, int(LIGHT_ENV["GTRADE_WORKERS"]) // jobs))
        env["GTRADE_TF_POOL_PCT"] = "%.2f" % max(0.15, _pool_base() / jobs)
        env["GTRADE_NEURAL_SLOTS"] = "1"
    return env


def _run_chunk(ci, total, chunk, force_promote, jobs):
    """Train one chunk. Returns (ci, chunk, returncode).

    Output goes to a per-chunk log only when several chunks run at once: two
    trainers interleaving into one console is unreadable, and the sequential run
    is something a human watches.
    """
    env = _chunk_env(chunk, force_promote, jobs)
    print(f"\n===== CHUNK {ci}/{total}  ({len(chunk)} assets) =====")
    print("  " + ", ".join(chunk))
    if jobs <= 1:
        rc = subprocess.run([sys.executable, "train_hybrid.py"], cwd=BASE,
                            env=env, check=False).returncode
        return ci, chunk, rc
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"chunk_{ci:02d}.log")
    with open(path, "w", encoding="utf-8") as log:
        # A header and a terminator, because the trainer's own output says
        # nothing about which chunk it was or whether the file is finished. A
        # log that just stops mid-asset and a log of a completed chunk looked
        # identical, so the only way to tell was to count the progress file.
        log.write("[chunked] chunk %d/%d  started %s\n  %s\n\n"
                  % (ci, total, datetime.now().isoformat(timespec="seconds"),
                     ", ".join(chunk)))
        log.flush()
        rc = subprocess.run([sys.executable, "train_hybrid.py"], cwd=BASE,
                            env=env, stdout=log, check=False,
                            stderr=subprocess.STDOUT).returncode
        log.write("\n[chunked] chunk %d/%d  finished %s  rc=%d\n"
                  % (ci, total, datetime.now().isoformat(timespec="seconds"), rc))
    print(f"[chunked] chunk {ci}/{total} finished rc={rc}  ->  "
          f"{os.path.relpath(path, BASE)}")
    return ci, chunk, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-promote", action="store_true",
                    help="replace every champion regardless of whether the new "
                         "model won; for registry recovery and baseline rebuilds")
    ap.add_argument("--assets-file", default=None,
                    help="train exactly the assets named in this file, one per "
                         "line. Written by `model_health.py --list KIND --out "
                         "FILE` in the environment SERVING runs in, because "
                         "which champions are broken can only be decided there "
                         "while the training itself has to happen here.")
    ap.add_argument("--jobs", type=int, default=int(os.getenv("TRAIN_JOBS", "2")),
                    help="chunk processes to run at once (default 2, the "
                         "ceiling on a 4 GB card). The workers and the VRAM pool "
                         "are divided to match, so neither the concurrent "
                         "training count nor the memory demand goes up. Use 1 "
                         "if a chunk dies out of memory.")
    args = ap.parse_args()
    # Ask the card before committing to the second process. Without this the
    # menu could still reach the 2026-08-24 stall that the research path was
    # already protected from.
    jobs = _fit_jobs(max(1, args.jobs))

    # Seed the cumulative quality report from whatever exists before chunk 1
    # overwrites it, so earlier assets are not lost mid-run.
    if not os.path.exists(QCUM) and os.path.exists(QPATH):
        shutil.copyfile(QPATH, QCUM)

    done = _done()
    if args.assets_file:
        if not os.path.exists(args.assets_file):
            print("[chunked] no such list: %s" % args.assets_file)
            return 1
        with open(args.assets_file, encoding="utf-8") as fh:
            wanted = [ln.strip().upper() for ln in fh if ln.strip()]
        unknown = [a for a in wanted if a not in FULL_ASSET_MAP]
        if unknown:
            print("[chunked] not in the asset map: %s" % ", ".join(unknown[:8]))
            return 1
        todo = [a for a in FULL_ASSET_MAP if a in set(wanted)]
        if not todo:
            print("[chunked] the list is empty - nothing to do.")
            return 0
        print("[chunked] list from %s: %d asset(s)"
              % (os.path.basename(args.assets_file), len(todo)))
    else:
        todo = [a for a in FULL_ASSET_MAP if a not in done]
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]

    mode = ("FORCE-PROMOTE (every champion replaced)" if args.force_promote
            else "champion-challenger (a champion changes only if beaten)")
    try:
        from core import adopted as _adopted
        _rec = _adopted.load()
    except Exception:
        _rec = None
    adopted_line = ("adopted: %s (%s)" % (_rec.get("label"), _rec.get("adopted"))
                    if _rec else "adopted: nothing (production defaults)")

    print("=" * 66)
    print("  ATRATUS CHUNKED TRAINER  -  fresh process per chunk (RAM-safe)")
    print(f"  done/skipped: {len(done)}   to train: {len(todo)}   chunks: {len(chunks)} x {CHUNK}")
    workers = int(LIGHT_ENV["GTRADE_WORKERS"]) // jobs if jobs > 1 else int(LIGHT_ENV["GTRADE_WORKERS"])
    print(f"  profile: {workers} workers | net cap 80 | 5 folds | epochs 90/60/50")
    print(f"  parallel: {jobs} chunk process(es) at once"
          + (f"; per-chunk logs in {os.path.basename(LOG_DIR)}" if jobs > 1 else ""))
    print("  promotion: " + mode)
    print("  " + adopted_line)
    print("=" * 66)
    if not todo:
        print("  Nothing to do - all assets recorded done. Delete _chunk_progress.txt to redo.")
        return

    # One writer for the shared bookkeeping. The trainer merges its own three
    # MODEL_DIR files under its own lock; these two are this process's.
    book = threading.Lock()
    failed = []

    def _finish(ci, chunk, rc):
        if rc != 0:
            failed.append(ci)
            print(f"\n[chunked] chunk {ci} exited with code {rc}.")
            print("[chunked] lower CHUNK_SIZE (or --jobs) if it ran out of RAM, "
                  "then rerun to resume.")
            return
        with book:
            _merge_quality()
            with open(PROGRESS, "a", encoding="utf-8") as f:
                f.writelines(a + "\n" for a in chunk)
        print(f"[chunked] chunk {ci}/{len(chunks)} complete - memory reset for the next chunk.")

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_run_chunk, ci, len(chunks), chunk,
                               args.force_promote, jobs)
                   for ci, chunk in enumerate(chunks, 1)]
        for fut in futures:
            if failed:
                # A failure is almost always the box running out of something,
                # so the chunks not started yet would fail the same way. Already
                # running ones are left to finish: their work is real and the
                # progress file only records what actually completed.
                fut.cancel()
                continue
            try:
                _finish(*fut.result())
            except Exception as exc:
                failed.append(-1)
                print(f"\n[chunked] a chunk could not be started: {exc}")

    if failed:
        print("\n[chunked] stopped with %d failed chunk(s); rerun to resume."
              % len(failed))
        return
    print("\n[chunked] ALL CHUNKS COMPLETE.  Next: python predict.py")


if __name__ == "__main__":
    raise SystemExit(main())
