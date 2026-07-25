"""Progress state for auto_research: what it is doing and how much longer.

The agent and the trainer are SEPARATE processes, so there are two files and each
has exactly one writer:
  ar_progress.json       written only by auto_research (phase, step, history)
  ar_progress_unit.json  written only by train_hybrid (the training in flight)
Readers merge them and never write.

Everything here is fail-safe on purpose: a progress write must never break a
twelve-hour training run, so no function in this module raises into its caller.
"""

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
    longest asset left.
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
    # Collect all raw times from asset_hist for unknown assets
    all_times = []
    for times_list in (asset_hist or {}).values():
        if times_list:
            all_times.extend(times_list)
    typical = median(all_times) if all_times else median(have)
    times = [known[asset] if known[asset] is not None else typical for asset in pending]
    lanes = max(1, int(workers or 1))
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
    total = float(unit_left)
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
