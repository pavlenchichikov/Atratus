"""Progress state for auto_research: what it is doing and how much longer.

The agent and the trainer are SEPARATE processes, so there are two files and each
has exactly one writer:
  ar_progress.json       written only by auto_research (phase, step, history)
  ar_progress_unit.json  written only by train_hybrid (the training in flight)
Readers merge them and never write.

Everything here is fail-safe on purpose: a progress write must never break a
twelve-hour training run, so no function in this module raises into its caller.
"""

import datetime
import json
import os
import threading

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_FILE = os.path.join(BASE, "ar_progress.json")
UNIT_FILE = os.path.join(BASE, "ar_progress_unit.json")

# A heartbeat rewrites updated_at every 60 s, so 5 minutes of silence really does
# mean something stopped.
HEARTBEAT_S = 60
STALE_AFTER_S = 300

_lock = threading.Lock()


def median(values):
    """Median of the non-negative numbers in values; None when there are none."""
    nums = sorted(v for v in (values or [])
                  if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0)
    if not nums:
        return None
    mid = len(nums) // 2
    if len(nums) % 2:
        return float(nums[mid])
    return (nums[mid - 1] + nums[mid]) / 2.0


def unit_remaining(pending, asset_hist, workers, unit_kind=None, unit_hist=None):
    """Seconds left in the training now in flight, plus the basis for that number.

    pending are the assets not finished yet, asset_hist maps asset to a list of
    past SERVICE times (its own start to its own finish), workers is how many
    assets train at once.

    Two corrections make this honest. Per-asset times, never the average: the
    schedule puts the expensive assets last, so an average is biased low. And the
    assets run in parallel, so their times do not add up - the remaining wall
    clock is the work spread over the lanes, and never less than the single
    longest asset left. Unknown pending assets are priced at the population median
    of known assets (one vote per asset), not at pooled samples.
    """
    if not pending:
        return 0.0, "unit finished"
    known = {asset: median((asset_hist or {}).get(asset)) for asset in pending}
    have = [v for v in known.values() if v is not None]
    if not have:
        fallback = median((unit_hist or {}).get(unit_kind))
        if fallback is None:
            return None, "no history yet"
        return fallback, "unit-kind median, no per-asset history yet"
    # Collect per-asset medians from asset_hist for unknown assets
    asset_medians = [median(times_list) for times_list in (asset_hist or {}).values()
                     if times_list]
    typical = median(asset_medians)
    times = [known[asset] if known[asset] is not None else typical for asset in pending]
    try:
        lanes = max(1, int(workers or 1))
    except (TypeError, ValueError):
        lanes = 1
    est = max(max(times), sum(times) / lanes)
    if len(have) < len(pending):
        return est, "per-asset history (unknown assets at the median of known ones)"
    return est, "per-asset history"


def run_remaining(unit_left, pending_units, unit_hist):
    """Seconds left in the whole run: the current unit plus each unit not started.

    A future unit with no measured history is NOT invented: it is left out of the
    total and counted in the basis, so the number stays a floor rather than a
    fiction.
    """
    if unit_left is None:
        return None, "no history yet"
    try:
        total = float(unit_left)
    except (TypeError, ValueError):
        return None, "no history yet"
    unmeasured = 0
    for kind in (pending_units or []):
        typical = median((unit_hist or {}).get(kind))
        if typical is None:
            unmeasured += 1
        else:
            total += typical
    if unmeasured:
        return total, ("current unit plus unit-kind medians (%d future unit(s) "
                       "unmeasured, not counted)" % unmeasured)
    return total, "current unit plus unit-kind medians"


def _now():
    return datetime.datetime.now()


def _as_dict(value):
    """value when it is a dict, else {}. Nested structures come from a JSON file
    another process wrote and a human may have hand-edited, so a reader must
    degrade instead of assuming shape.
    """
    return value if isinstance(value, dict) else {}


def _write(path, payload):
    """Atomic and fail-safe. A progress write must never reach the caller as an
    exception, and a reader must never see half a document, so the body goes to a
    temp file in the same directory and is moved into place with os.replace. If
    the write fails partway, the temp file is removed on a best-effort basis so
    nothing leaks into the worktree.
    """
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        body = dict(payload)
        body.setdefault("pid", os.getpid())
        body["updated_at"] = _now().isoformat(timespec="seconds")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _read(path):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_agent(payload):
    with _lock:
        _write(AGENT_FILE, payload)


def write_unit(payload):
    with _lock:
        _write(UNIT_FILE, payload)


def read_agent():
    return _read(AGENT_FILE)


def read_unit():
    return _read(UNIT_FILE)


def age_seconds(record, now=None):
    """Seconds since the record was written; None when that cannot be known."""
    stamp = _as_dict(record).get("updated_at")
    if not stamp:
        return None
    try:
        then = datetime.datetime.fromisoformat(stamp)
    except Exception:
        return None
    return max(0.0, ((now or _now()) - then).total_seconds())


_hb_stop = None
_hb_thread = None


def start_heartbeat(which):
    """Rewrite updated_at every HEARTBEAT_S so silence is evidence of a stop.

    Without this, age says nothing: a single asset can hold a worker for over
    three hours, so a perfectly healthy run would look abandoned.
    """
    global _hb_stop, _hb_thread
    if _hb_thread is not None:
        return
    path = AGENT_FILE if which == "agent" else UNIT_FILE
    stop = threading.Event()

    def _beat():
        while not stop.wait(HEARTBEAT_S):
            with _lock:
                record = _read(path)
                if not record:
                    continue
                record["beats"] = int(record.get("beats") or 0) + 1
                _write(path, record)

    _hb_stop = stop
    _hb_thread = threading.Thread(target=_beat, name="ar-progress-hb", daemon=True)
    _hb_thread.start()


def stop_heartbeat():
    global _hb_stop, _hb_thread
    if _hb_stop is not None:
        _hb_stop.set()
    _hb_stop = None
    _hb_thread = None


_unit_started_at = {}


def unit_begin(order, workers):
    """Start a training unit. Called once by the trainer before the pool runs."""
    try:
        with _lock:
            _unit_started_at.clear()
            _write(UNIT_FILE, {
                "started": _now().isoformat(timespec="seconds"),
                "order": list(order or []), "assets_total": len(order or []),
                "workers": int(workers or 1), "in_flight": [], "done": [],
            })
    except Exception:
        pass


def unit_asset_start(asset):
    """One asset entered a worker lane."""
    try:
        with _lock:
            _unit_started_at[asset] = _now()
            record = _read(UNIT_FILE)
            flight = [a for a in (record.get("in_flight") or []) if a != asset]
            record["in_flight"] = flight + [asset]
            _write(UNIT_FILE, record)
    except Exception:
        pass


def unit_asset_done(asset):
    """One asset left its lane. Records ITS OWN service time, not wall clock,
    because up to `workers` assets overlap and their times must not be summed."""
    try:
        with _lock:
            began = _unit_started_at.pop(asset, None)
            took = int((_now() - began).total_seconds()) if began else 0
            record = _read(UNIT_FILE)
            record["in_flight"] = [a for a in (record.get("in_flight") or []) if a != asset]
            record["done"] = (record.get("done") or []) + [[asset, took]]
            _write(UNIT_FILE, record)
    except Exception:
        pass


def unit_end():
    """End a training unit. Trims `order` down to the assets that actually ran
    (finished or still in flight at this moment), so a run stopped early -
    Ctrl+C breaks the submit loop before every asset is submitted - does not
    leave never-started assets looking "pending" on a unit that has ended.
    """
    try:
        with _lock:
            _unit_started_at.clear()
            record = _read(UNIT_FILE)
            order = record.get("order") or []
            done_names = {pair[0] for pair in (record.get("done") or [])
                          if isinstance(pair, (list, tuple)) and pair}
            ran = done_names | set(record.get("in_flight") or [])
            record["order"] = [asset for asset in order if asset in ran]
            record["in_flight"] = []
            record["ended"] = _now().isoformat(timespec="seconds")
            _write(UNIT_FILE, record)
    except Exception:
        pass


def snapshot(now=None):
    """Merged read-only view of both files for the page and the console banner."""
    agent, unit = read_agent(), read_unit()
    if not agent and not unit:
        return {"state": "no data"}
    age = age_seconds(agent or unit, now)
    history = _as_dict(agent.get("history"))
    step = _as_dict(agent.get("step"))
    done_pairs = unit.get("done") or []
    done = [pair[0] for pair in done_pairs if isinstance(pair, (list, tuple)) and pair]
    order = unit.get("order") or []
    pending = [asset for asset in order if asset not in done]
    unit_left, unit_basis = unit_remaining(
        pending, _as_dict(history.get("assets")), unit.get("workers"),
        step.get("unit_kind"), history)
    run_left, run_basis = run_remaining(unit_left, agent.get("pending_units"), history)
    return {
        "state": "running",
        "phase": agent.get("phase"),
        "step": step,
        "results": agent.get("results") or {},
        "unit": {
            "assets_total": unit.get("assets_total"),
            "assets_done": len(done),
            "in_flight": unit.get("in_flight") or [],
            "pending": pending,
            "workers": unit.get("workers"),
            "started": unit.get("started"),
        },
        "eta": {"unit_left_s": unit_left, "run_left_s": run_left,
                "basis": unit_basis, "run_basis": run_basis},
        "age_s": age,
        "stale": bool(age is not None and age > STALE_AFTER_S),
        "pid": agent.get("pid") or unit.get("pid"),
        "run_started": agent.get("run_started"),
    }
