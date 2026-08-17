"""Unattended research cycle: search, gate, A/B, adopt. Stops before the retrain.

Sequencing is a state machine, not a model pressing menu buttons. The phase is
DERIVED from the files each cycle rather than stored: a stored cursor and the
tree disagree the moment a phase is run by hand, and every phase here is also a
command the owner runs directly.

What a model IS allowed to choose is which experiment to run next - the axis,
the label, the budget, whether to spend the LLM proposer. That is core.ar_director,
enabled with GTRADE_AR_DIRECTOR=1. What it is never allowed to choose is how a
result is judged: the score basis and the objective are frozen on the first
cycle and re-checked on every one, because picking those after seeing a verdict
is a search for a measurement that passes rather than a measurement. The
director can only ask to start a NEW campaign, with a written reason, which
resets the freeze and sets the search archive aside.

Stopping. `python auto_loop.py --stop` asks the running loop to finish the phase
it is in and exit; the loop also checks before starting each phase. Killing it
outright is safe too, it just costs whatever the running phase had not
checkpointed. Every phase resumes on the next start:
  - the search saves the archive after EVERY genome and banks the signature
    before evaluating it, so at most one genome is lost;
  - held-out training caches per chunk under GTRADE_AR_TRAIN_CHUNK, so an
    interrupted A/B arm loses one chunk instead of all 8 to 11 hours;
  - a finished A/B arm is cached by genome signature, so a restart re-reads it.

It keeps cycling until something is adopted. A failed A/B is not an ending: the
candidate is recorded as measured against this reference, the next cycle picks
the next gate-adoptable elite, and when none is left it goes back to searching.

Run:
  python auto_loop.py                 # cycle until an adoption, a failure or --stop
  python auto_loop.py --dry-run       # the phase it would run now, touching nothing
  python auto_loop.py --status        # the stage it is in, plus recent history
  python auto_loop.py --hours 12      # add a deadline instead of running open-ended
  python auto_loop.py --stop          # ask a running loop to stop cleanly
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from core import runlock

STATE_PATH = os.path.join(BASE, "_auto_loop.json")
LOCK_PATH = os.path.join(BASE, "_auto_loop.lock")
STOP_PATH = os.path.join(BASE, "_auto_loop.stop")
REPORT_PATH = os.path.join(BASE, "_auto_loop_report.txt")
ARCHIVE_PATH = os.path.join(BASE, "_qd_archive.json")

# The campaign. These replace the auto_research.bat menu for unattended runs and
# are authoritative over .env, which is why every one of them carries a value:
# a key absent from the environment is refilled by load_dotenv inside
# auto_research, which is how a pinned 17 GB model reached a 15.7 GB machine.
CAMPAIGN = {
    "GTRADE_AR_AXES": "qd",
    "GTRADE_AR_SCORE_BASIS": "net_auc",
    "GTRADE_AR_OBJECTIVE": "mean",
    "GTRADE_AR_SCREEN": "0",
    "GTRADE_AR_ILLUM": "full",
    "GTRADE_AR_RL": "1",
    "GTRADE_AR_WIKI": "1",
    "GTRADE_LABEL_MODE": "direction",
    "GTRADE_LABEL_HORIZON": "1",
    "GTRADE_AR_PROPOSER": "evolutionary",
    "AR_BUDGET": "15",
    # Assets per held-out training chunk. The whole point of the unattended
    # loop is that it can be stopped, so an arm that only checkpoints after 11
    # hours is the wrong shape here. 7, not 5: the trainer derives 6 workers on
    # this GPU, so a chunk of 5 would leave a worker slot idle for the whole
    # run, while 7 fills them and still splits a 14-asset holdout in two.
    "GTRADE_AR_TRAIN_CHUNK": "7",
    # Load. Measured mid-run on the RTX 2050 box: CPU 25% of 12 threads, GPU
    # 39%, RAM 4.1 GB free of 15.7 - neither saturated, because net training
    # serialises behind one GPU slot while CatBoost waits.
    #   CB_THREADS   free on the CPU side, nothing else pays for it.
    #   TF_POOL_PCT  shrunk FIRST, which buys 400-600 MB outside the pool.
    #   NEURAL_SLOTS the actual lever and the only risky one: cuDNN workspaces
    #                live outside the TF pool, so a second slot can OOM even
    #                though the pool itself fits. SAFE_LOAD below is the retry.
    # GTRADE_WORKERS is deliberately left derived: the holdout arm already
    # holds about 6 GB and only 4.1 is free, so more workers is the one knob
    # here that trades a stall for a swap.
    "GTRADE_CB_THREADS": "12",
    "GTRADE_TF_POOL_PCT": "0.50",
    "GTRADE_NEURAL_SLOTS": "2",
}

# What a failed training phase is retried under, once. Two neural slots is the
# setting that buys the throughput and also the only one that can kill a run
# outright, so the fallback turns an OOM from a dead loop into a slower one.
SAFE_LOAD = {"GTRADE_NEURAL_SLOTS": "1", "GTRADE_TF_POOL_PCT": "0.60"}

# Phases where a retry at lower load can plausibly help: the ones that train.
TRAINING_PHASES = ("search", "ab_run")

# Changing one of these after a verdict is what separates a measurement from a
# search for a measurement that passes, so the campaign freezes them on the
# first cycle and refuses to continue if they move.
FROZEN = ("GTRADE_AR_SCORE_BASIS", "GTRADE_AR_OBJECTIVE")

NET_BASES = ("net_auc", "net_gain", "ens_auc")


def build_env(environ=None, budget=0):
    """The environment every phase runs under.

    setdefault, not update: CAMPAIGN is the DEFAULT campaign and an explicit
    environment value wins. update() discarded them, so every answer the
    launcher menu collected for a key that also lives here - the proposer, the
    wiki - was silently thrown away and the run went ahead on the built-in
    value instead.

    What the original update() protected is unaffected: every key still ends up
    PRESENT in the child's environment, which is what stops the child's
    load_dotenv refilling a missing one from .env. And the two keys where a
    stray shell value would actually be dangerous are the frozen ones, which
    freeze_problems checks separately.
    """
    env = dict(os.environ if environ is None else environ)
    for key, value in CAMPAIGN.items():
        env.setdefault(key, value)
    if budget:
        env["AR_BUDGET"] = str(budget)
    return env


def campaign_problems(env):
    """Settings that contradict each other, in plain words.

    These rules lived only as prose in the launcher's REM block, where a human
    reading the menu applied them. An unattended run has no such reader.
    """
    out = []
    basis = env.get("GTRADE_AR_SCORE_BASIS", "raw")
    illum = env.get("GTRADE_AR_ILLUM", "cb")
    if basis in NET_BASES:
        if env.get("GTRADE_AR_SCREEN") == "1":
            out.append(
                "basis %s with GTRADE_AR_SCREEN=1: the screen stubs every neural "
                "member to a constant 0.5, so all candidates screen identically "
                "and net levers are discarded on CatBoost's opinion" % basis)
        if illum != "full":
            out.append(
                "basis %s with GTRADE_AR_ILLUM=%s: the archive would be "
                "illuminated by CatBoost alone, so no net lever can become an "
                "elite and the basis would only re-score the final gate"
                % (basis, illum))
    if basis == "raw" and illum == "full":
        out.append(
            "GTRADE_AR_ILLUM=full on the raw Score basis: net training does not "
            "reproduce on this GPU (same seed, same config, 0.45 to 1.52 Score "
            "apart, more than the adoption floor), so the archive would rank "
            "noise. Use GTRADE_AR_SCORE_BASIS=net_auc for net work.")
    return out


def freeze_problems(frozen, env):
    """Frozen gate constants that moved since the campaign started."""
    if not frozen:
        return []
    return ["%s changed mid-campaign: %s -> %s"
            % (k, frozen[k], env.get(k)) for k in FROZEN
            if frozen.get(k) != env.get(k)]


def probe(base=None):
    """What the tree says about where the cycle stands."""
    import ab_build
    import adopt_genome

    base = base or BASE
    ref = ab_build.reference()
    pool = adopt_genome.candidates(base)
    tested = ab_build.tested_against(ref["sig"], base)
    cfg = ab_build.read_config()
    # A config is pending only while at least one of its arms has no result
    # against the live reference. Derived rather than deleted on success, so a
    # run finished by hand leaves the loop in the right state too.
    pending = bool(cfg) and cfg.get("reference_sig") == ref["sig"] and any(
        c.get("sig") not in tested for c in (cfg.get("candidates") or []))
    return {
        "reference": ref["label"],
        "ab_pending": pending,
        # Measured against the LIVE reference, not merely present in some A/B
        # file. A candidate that beat an earlier reference says nothing about
        # the one running now, and adopting it in sequence would walk the
        # adoption backwards one file at a time.
        "adoptable": [c["label"] for c in pool
                      if c.get("validated") and c.get("sig") in tested
                      and not ab_build.is_reference(c, ref)],
        "untested": [c["label"] for c in ab_build.auto_picks(
            pool, ref, ab_build._gate_by_sig(), tested)],
    }


def next_action(state):
    """The one phase to run now. Finish what is started before starting more."""
    if state["ab_pending"]:
        return "ab_run"
    if state["adoptable"]:
        return "adopt"
    if state["untested"]:
        return "ab_build"
    return "search"


def _run(action, env):
    """Run one phase. check=False: a failure is recorded and handled here, not
    raised as a traceback out of the loop."""
    cmd = command(action, env)
    print("[loop] %s" % " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=BASE, env=env, check=False).returncode


def command(action, env):
    py = sys.executable
    if action == "search":
        return [py, "auto_research.py"]
    if action == "ab_build":
        return [py, "ab_build.py", "--auto",
                "--objective", env["GTRADE_AR_OBJECTIVE"]]
    if action == "ab_run":
        return [py, "ab_build.py", "--run"]
    if action == "adopt":
        return [py, "adopt_genome.py", "--auto"]
    raise ValueError("unknown action: %s" % action)


# --- stop requests ----------------------------------------------------------

def stop_requested():
    return os.path.exists(STOP_PATH)


def request_stop(reason="stop requested"):
    with open(STOP_PATH, "w", encoding="utf-8") as fh:
        fh.write("%s %s\n" % (datetime.datetime.now().isoformat(timespec="seconds"),
                              reason))


def clear_stop():
    """Drop a previous request. An explicit start outranks an old stop file,
    which would otherwise make the next launch exit at once for no visible reason."""
    try:
        os.remove(STOP_PATH)
    except OSError:
        pass


# --- state ------------------------------------------------------------------

def _load_state():
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"campaign": None, "history": []}


def _save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


def publish(state, phase, detail="", cycle=None):
    """Record the stage the cycle is in, for the console banner and the web UI.

    Written on every transition rather than only at the end, because the whole
    point of the field is to answer "what is it doing right now" during a phase
    that can run for hours.
    """
    state["current"] = {
        "phase": phase, "detail": detail,
        "since": datetime.datetime.now().isoformat(timespec="seconds"),
        "cycle": cycle if cycle is not None else len(state.get("history") or []) + 1,
        "campaign": state.get("campaign"),
        "pid": os.getpid(),
    }
    _save_state(state)
    return state


def read_state():
    """The loop's stage for a read-only consumer (the /research page).

    Never raises and never imports the pipeline: an unreadable or missing file
    reads as "the loop is not running", which is the truth in that case.
    """
    st = _load_state()
    return {"current": st.get("current"), "campaign": st.get("campaign"),
            "campaign_reason": st.get("campaign_reason"),
            "history": (st.get("history") or [])[:10]}


def _banner(cycle, action, st, env, deadline):
    left = ("no deadline, runs until an adoption" if deadline is None
            else "%.1f h left" % max(0.0, (deadline - time.time()) / 3600.0))
    print()
    print("=" * 72)
    print("  CYCLE %d    PHASE: %s    (%s)" % (cycle, action.upper(), left))
    print("  reference  %s" % st["reference"])
    print("  campaign   basis %s | objective %s | axes %s | budget %s"
          % (env.get("GTRADE_AR_SCORE_BASIS"), env.get("GTRADE_AR_OBJECTIVE"),
             env.get("GTRADE_AR_AXES"), env.get("AR_BUDGET")))
    print("  load       %s neural slots | %s CB threads | pool %s | chunk %s"
          % (env.get("GTRADE_NEURAL_SLOTS"), env.get("GTRADE_CB_THREADS"),
             env.get("GTRADE_TF_POOL_PCT"), env.get("GTRADE_AR_TRAIN_CHUNK")))
    print("  waiting    ab_run=%s | adoptable=%s | untested=%s"
          % (st["ab_pending"], ",".join(st["adoptable"]) or "-",
             ",".join(st["untested"]) or "-"))
    print("=" * 72, flush=True)


def adoption_report(env):
    """Everything needed to decide on the retrain, in one console block.

    The loop stops here, so this is the last thing printed before a human takes
    over; it has to carry the genome itself, not a pointer to a file.
    """
    import adopt_genome
    from config import FULL_ASSET_MAP
    from core import adopted as _adopted

    lines = adopt_genome.report_lines(_adopted.load(), adopt_genome._previous())
    lines += [
        "", "RETRAIN PLAN",
        "  assets        %d (chunk progress was cleared, so all of them)"
        % len(FULL_ASSET_MAP),
        ("  promotion     champion-challenger: an asset keeps its champion "
         "unless the new model beats it"),
        "  1.            cmd /c \"run_in_env.bat python train_chunked.py\"",
        "  2.            python predict.py",
        "", "CAMPAIGN THIS CAME FROM",
        "  basis         %s" % env.get("GTRADE_AR_SCORE_BASIS"),
        "  objective     %s" % env.get("GTRADE_AR_OBJECTIVE"),
        "  axes          %s" % env.get("GTRADE_AR_AXES"),
        "  label         %s/%s" % (env.get("GTRADE_LABEL_MODE"),
                                   env.get("GTRADE_LABEL_HORIZON")),
    ]
    return lines


def start_campaign(state, env, reason=""):
    """Freeze the gate constants for a campaign and set the old archive aside.

    The archive MUST go when the basis changes: fitness on the Score scale runs
    1.5 to 8.9 and on the AUC scale about 0.01, so one surviving Score elite
    outranks every AUC elite for the rest of the run. Kept as a .bak beside it
    rather than deleted, and .bak is already ignored as local research state.
    """
    state["campaign"] = {k: env[k] for k in FROZEN}
    state["campaign_started"] = datetime.datetime.now().isoformat(timespec="seconds")
    state["campaign_reason"] = reason
    if os.path.exists(ARCHIVE_PATH):
        os.replace(ARCHIVE_PATH, ARCHIVE_PATH + ".bak")
        print("[loop] archive set aside as %s.bak" % os.path.basename(ARCHIVE_PATH))
    return state


def apply_director(env, state):
    """Let the director choose the next search settings. Returns (env, state).

    Only the search levers are copied across. A new-campaign request is applied
    here because it is the one place that can also re-freeze and clear the
    archive, which is what makes it a campaign change rather than an edit.
    """
    from core import ar_director

    if not ar_director.director_on():
        return env, state
    from core import ar_memory

    settings = ar_director.propose(
        ar_memory.findings_all(), env,
        archive_n=len(ar_memory.blob_get("_qd_archive", {}) or {}),
        cycles=len(state.get("history") or []))
    if not settings:
        return env, state
    fresh = settings.pop("new_campaign", None)
    reason = settings.pop("reason", "")
    env = dict(env)
    env.update(settings)
    print("[director] %s | %s" % (
        " ".join("%s=%s" % kv for kv in sorted(settings.items())),
        reason or "no reason given"))
    if fresh:
        why = fresh.pop("reason", "")
        env.update(fresh)
        state = start_campaign(state, env, why)
        print("[director] NEW CAMPAIGN: %s" % why)
    return env, state


def _status(env):
    st = probe()
    state = _load_state()
    cur = state.get("current") or {}
    print("stage     : %s%s" % (cur.get("phase") or "never run",
                                "  since %s (cycle %s, pid %s)"
                                % (cur.get("since"), cur.get("cycle"),
                                   cur.get("pid")) if cur.get("since") else ""))
    if cur.get("detail"):
        print("            %s" % cur["detail"])
    print("reference : %s" % st["reference"])
    print("campaign  : %s" % (state.get("campaign") or "not started"))
    if state.get("campaign_reason"):
        print("            %s" % state["campaign_reason"])
    print("pending   : ab_run=%s  adoptable=%s  untested=%s"
          % (st["ab_pending"], st["adoptable"] or "-", st["untested"] or "-"))
    print("next      : %s" % next_action(st))
    print("stop file : %s" % ("present, a start would clear it"
                              if stop_requested() else "none"))
    for h in (state.get("history") or [])[:8]:
        print("  %s  %-9s rc=%s" % (h.get("ts"), h.get("action"), h.get("rc")))


def main():
    ap = argparse.ArgumentParser(description="unattended research cycle")
    ap.add_argument("--hours", type=float, default=0.0,
                    help="optional deadline in hours; 0 (the default) keeps "
                         "cycling until an adoption, a failure or --stop")
    ap.add_argument("--budget", type=int, default=0,
                    help="search iterations per cycle (0 = the campaign value)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the phase that would run and exit")
    ap.add_argument("--status", action="store_true",
                    help="print where the cycle stands and exit")
    ap.add_argument("--stop", action="store_true",
                    help="ask a running loop to finish its phase and exit")
    args = ap.parse_args()

    env = build_env(budget=args.budget)

    if args.stop:
        request_stop()
        print("[loop] stop requested. The running loop exits after its current "
              "phase; progress is checkpointed either way.")
        return 0

    problems = campaign_problems(env)
    state = _load_state()
    problems += freeze_problems(state.get("campaign"), env)
    if problems:
        print("[loop] the campaign is not runnable:")
        for p in problems:
            print("  - %s" % p)
        return 1

    if args.status:
        _status(env)
        return 0
    if args.dry_run:
        st = probe()
        print("[loop] reference: %s" % st["reference"])
        print("[loop] ab_pending=%s  adoptable=%s  untested=%s"
              % (st["ab_pending"], st["adoptable"] or "-", st["untested"] or "-"))
        action = next_action(st)
        print("[loop] would run: %s" % " ".join(command(action, env)))
        return 0

    ok, reason = runlock.acquire(LOCK_PATH, "auto_loop")
    if not ok:
        print("[loop] %s; not starting." % reason)
        return 1
    clear_stop()
    try:
        if not state.get("campaign"):
            state = start_campaign(state, env, "first campaign")
        deadline = time.time() + args.hours * 3600.0 if args.hours > 0 else None
        cycle = 0
        while deadline is None or time.time() < deadline:
            if stop_requested():
                print("\n[loop] stop requested; exiting between phases.")
                return 0
            env, state = apply_director(env, state)
            st = probe()
            action = next_action(st)
            cycle += 1
            state = publish(state, action, "reference %s" % st["reference"], cycle)
            _banner(cycle, action, st, env, deadline)
            rc = _run(action, env)
            if rc != 0 and action in TRAINING_PHASES:
                print("[loop] %s exited with %d. Retrying once at conservative "
                      "load (%s): the aggressive setting that can fail this way "
                      "is the second neural slot, whose cuDNN workspace lives "
                      "outside the TF pool."
                      % (action, rc, ", ".join("%s=%s" % kv
                                               for kv in sorted(SAFE_LOAD.items()))))
                state = publish(state, action, "retry at conservative load", cycle)
                rc = _run(action, dict(env, **SAFE_LOAD))
            state["history"].insert(0, {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "action": action, "rc": rc, "cycle": cycle})
            state["history"] = state["history"][:60]
            _save_state(state)
            if rc != 0:
                print("[loop] %s exited with %d; stopping." % (action, rc))
                return rc
            if action == "adopt":
                state = publish(state, "adopted", "waiting for a manual retrain",
                                cycle)
                lines = adoption_report(env)
                print()
                for line in lines:
                    print(line)
                with open(REPORT_PATH, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                print("\n[loop] also written to %s" % os.path.basename(REPORT_PATH))
                print("[loop] stopping here on purpose: models/ still holds the "
                      "previous generation until the retrain runs.")
                return 0
        print("\n[loop] deadline reached; stopping cleanly.")
        return 0
    except KeyboardInterrupt:
        print("\n[loop] interrupted. Finished work is checkpointed; rerun "
              "python auto_loop.py to resume from the same phase.")
        return 130
    finally:
        # Leave a truthful stage behind: a "current" left at SEARCH after the
        # process is gone would read on the page as a run still going.
        if (state.get("current") or {}).get("phase") not in (None, "adopted"):
            state["current"]["phase"] = "stopped"
            state["current"]["detail"] = "no loop is running"
        _save_state(state)
        runlock.release(LOCK_PATH)


if __name__ == "__main__":
    sys.exit(main())
