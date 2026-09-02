"""One live status line for a run that spawns trainers.

A long run here is a parent process that starts train_hybrid children: the
chunked retrain, the research gate, a confirmation harness. The children print
their own per-asset result lines and the parent has, until now, printed a banner
per chunk and then nothing for an hour. This is the row that fills that hour:

  [####------------] 3/8 ref@12000  chunk1 1/3 EVRG  VRAM 3421/4096MB 78% 2proc  47m elapsed ~2h37m left

The field order is deliberate. A narrow console truncates from the right, so the
readout that cannot be got from anywhere else survives the cut: `2proc` is how
many processes actually hold GPU memory. The trainer falls back to the CPU on its
own when the card cannot be configured, saying so in one line at startup that
scrolls away within minutes; a run that should hold two processes and holds none
is that fallback, or a child that died, and either way it is hours of the wrong
measurement. When the count sits at zero, this says so out loud.

Only ONE object may own the console at a time. `emit()` is how everything else
prints while a bar is up - it wipes the row, prints the line, and lets the next
tick redraw - so a status line and a caller's own output never smear together.
With no bar running it is a plain print, which is why callers can use it
unconditionally.
"""

import math
import os
import shutil
import subprocess
import sys
import threading
import time

from core import ar_progress

SMI_EVERY_S = 3.0     # nvidia-smi is a subprocess; once a second is wasteful
NO_GPU_ALARM_S = 150  # how long a unit may hold no GPU process before it is called out
BAR_CELLS = 16

_active = None
_active_lock = threading.Lock()


def emit(msg):
    """Print a line without smearing the live status row."""
    with _active_lock:
        owner = _active
    if owner is None:
        print(msg, flush=True)
        return
    owner.say(msg)


def hm(seconds):
    """Seconds as 47m or 2h37m; '?' when there is nothing honest to say."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return "?"
    if math.isnan(seconds) or seconds < 0:
        return "?"
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def gpu_stats(exe=None, runner=None):
    """(used_mb, total_mb, gpu_pct, procs), or None when nvidia-smi is absent."""
    exe = exe or shutil.which("nvidia-smi")
    if not exe:
        return None

    def _run(args):
        if runner is not None:
            return runner(args)
        try:
            r = subprocess.run([exe, *args, "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=4, check=False)
            return [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
        except Exception:
            return []

    gpu = _run(["--query-gpu=memory.used,memory.total,utilization.gpu"])
    if not gpu:
        return None
    try:
        used, total, pct = (int(float(x)) for x in gpu[0].split(","))
    except ValueError:
        return None
    return used, total, pct, len(_run(["--query-compute-apps=pid,used_memory"]))


def render(unit_i, total_units, label, done, total, flight, gpu, elapsed_s,
           eta_s, inner_word="chunk1"):
    """The status row as a string. Pure, so it can be checked without a console."""
    frac = ((unit_i - 1) + (done / total if total else 0.0)) / max(1, total_units)
    filled = int(max(0.0, min(1.0, frac)) * BAR_CELLS)
    if gpu:
        used, gtotal, pct, procs = gpu
        mem = f"VRAM {used}/{gtotal}MB {pct}% {procs}proc"
    else:
        mem = "no nvidia-smi"
    return ("  [%s] %d/%d %-13s %s %d/%s %-10s %s  %s elapsed ~%s left"
            % ("#" * filled + "-" * (BAR_CELLS - filled), unit_i, total_units,
               label, inner_word, done, total or "?", ",".join(flight)[:10],
               mem, hm(elapsed_s), hm(eta_s)))


class Status:
    """The live row for a run of `total_units` child trainings.

    Per-asset progress inside the unit comes from ar_progress_unit.json, which
    the trainer writes as it goes - and only the FIRST chunk process writes the
    real file (the others are pointed at a scratch dir so two writers cannot make
    the estimate a lie), so the row says 'chunk1' rather than pretending to see
    every asset of the unit.

    The estimate is ar_progress.unit_remaining, not an average: it prices each
    pending asset at ITS OWN past service time and spreads the work over the
    worker lanes, because the schedule puts the expensive assets last and an
    average is biased low there. Units that have finished give the per-unit
    typical time for everything not started yet.
    """

    def __init__(self, total_units, plan_min=None, inner_word="chunk1"):
        self.total_units = max(1, int(total_units or 1))
        self.plan_min = plan_min
        self.inner_word = inner_word
        self.unit_i = 0
        self.unit_label = "starting"
        self.unit_started = time.time()
        # Anything the trainer wrote before this run began is a leftover, so the
        # first draws must not read it as progress.
        self.unit_started_iso = ar_progress._now().isoformat(timespec="seconds")
        self.durations = []
        self.asset_hist = {}
        self.t0 = time.time()
        self._smi_val = None
        self._smi_at = 0.0
        self._no_gpu_since = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread = None

    # -- console ---------------------------------------------------------
    def visible(self):
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    def _width(self):
        return max(60, shutil.get_terminal_size((120, 24)).columns - 1)

    def say(self, msg):
        """An event line, above the status row rather than through it."""
        with self._lock:
            if self.visible():
                sys.stdout.write("\r" + " " * self._width() + "\r")
            print(msg, flush=True)

    # -- data ------------------------------------------------------------
    def _gpu(self):
        if time.time() - self._smi_at >= SMI_EVERY_S:
            self._smi_val = gpu_stats()
            self._smi_at = time.time()
        return self._smi_val

    def _unit_view(self):
        """(done, total, in_flight, seconds_left) for the chunk being trained."""
        rec = ar_progress.read_unit() or {}
        order = rec.get("order") or []
        if not order or (rec.get("started") or "") < self.unit_started_iso:
            return 0, 0, [], None
        done = [p[0] for p in (rec.get("done") or [])
                if isinstance(p, (list, tuple)) and p]
        pending = [a for a in order if a not in done]
        left, _basis = ar_progress.unit_remaining(
            pending, self.asset_hist, rec.get("workers"))
        return len(done), len(order), rec.get("in_flight") or [], left

    def _eta_s(self, unit_left):
        remaining = self.total_units - self.unit_i
        if self.durations:
            typical = sum(self.durations) / len(self.durations)
        elif self.plan_min:
            typical = self.plan_min * 60 / self.total_units
        else:
            typical = None
        if unit_left is not None:
            here = unit_left
        elif typical is not None:
            here = max(0.0, typical - (time.time() - self.unit_started))
        else:
            return None
        if typical is None:
            return here
        return here + typical * max(0, remaining)

    # -- lifecycle -------------------------------------------------------
    def unit(self, i, label):
        """A child training is starting; `i` is 1-based."""
        self.unit_i = int(i)
        self.unit_label = str(label)[:13]
        self.unit_started = time.time()
        self.unit_started_iso = ar_progress._now().isoformat(timespec="seconds")

    def set_progress(self, done_units, label):
        """For a caller whose units run SIDE BY SIDE, where "which unit is in
        flight" has no single answer: it reports how many have finished instead,
        and the label says what is running now."""
        self.unit_i = int(done_units) + 1
        self.unit_label = str(label)[:13]

    def unit_done(self):
        """Bank the unit's wall time and its per-asset service times."""
        self.durations.append(time.time() - self.unit_started)
        rec = ar_progress.read_unit() or {}
        for pair in (rec.get("done") or []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[1]:
                self.asset_hist.setdefault(pair[0], []).append(pair[1])

    def start(self):
        """Take the console. No-op when there is no terminal to draw on."""
        global _active
        if not self.visible():
            return self
        with _active_lock:
            _active = self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        global _active
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        with _active_lock:
            if _active is self:
                _active = None
        if self.visible():
            sys.stdout.write("\r" + " " * self._width() + "\r")
            sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- drawing ---------------------------------------------------------
    def _loop(self):
        while not self._stop.wait(1.0):
            try:
                self.draw()
            except Exception:
                pass  # a status line must never take a run down

    def draw(self):
        done, total, flight, unit_left = self._unit_view()
        gpu = self._gpu()
        if gpu and gpu[3] == 0:
            self._no_gpu_since = self._no_gpu_since or time.time()
            if time.time() - self._no_gpu_since > NO_GPU_ALARM_S:
                self.say("  [!] no process has held the GPU for %d s - a child "
                         "may have fallen back to CPU, check its [GPU]/[CPU] "
                         "line above" % NO_GPU_ALARM_S)
                self._no_gpu_since = time.time()
        elif gpu:
            self._no_gpu_since = None
        line = render(self.unit_i, self.total_units, self.unit_label, done, total,
                      flight, gpu, time.time() - self.t0, self._eta_s(unit_left),
                      self.inner_word)
        with self._lock:
            w = self._width()
            sys.stdout.write("\r" + line[:w].ljust(w))
            sys.stdout.flush()


def quiet_child_env(env=None):
    """A child environment with the trainer's own bar turned off.

    The parent owns one row; a child rewriting the same row makes it flicker
    between the two of them.
    """
    out = dict(env if env is not None else os.environ)
    out["GTRADE_NO_TICKER"] = "1"
    return out
