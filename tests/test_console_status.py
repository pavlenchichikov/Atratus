"""The live status row: what it says, and what it refuses to say.

No terminal and no GPU are needed here - the rendering is a pure function and
nvidia-smi is injected - which is the point: the row is watched for hours during
a run nobody wants to repeat, so its rules have to be checkable in a second.
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from core import ar_progress, console_status


def _smi(gpu_line, procs):
    """A stand-in nvidia-smi: the two queries gpu_stats makes, in order."""
    def runner(args):
        if any("compute-apps" in a for a in args):
            return ["%d, 100" % i for i in range(procs)]
        return [gpu_line]
    return runner


def test_gpu_stats_reads_the_card_and_counts_the_processes_on_it():
    got = console_status.gpu_stats(exe="nvidia-smi",
                                   runner=_smi("3421, 4096, 78", procs=2))
    assert got == (3421, 4096, 78, 2)


def test_gpu_stats_reports_no_process_rather_than_guessing():
    # The whole reason the count is on the row: zero processes on the card during
    # a training run is a CPU fallback or a dead child, and it must be visible.
    assert console_status.gpu_stats(exe="nvidia-smi",
                                    runner=_smi("120, 4096, 0", procs=0))[3] == 0


def test_gpu_stats_is_none_when_nvidia_smi_cannot_answer():
    assert console_status.gpu_stats(exe="nvidia-smi", runner=lambda a: []) is None


def test_the_memory_readout_comes_before_the_clock():
    """A narrow console truncates from the right. The clock can be worked out
    from the start time; the process count and the VRAM cannot be got anywhere
    else once the run is under way, so they must not be the first thing cut."""
    line = console_status.render(3, 8, "ref@12000", 1, 3, ["EVRG"],
                                 (3421, 4096, 78, 2), 2820, 9420)
    assert line.index("VRAM") < line.index("elapsed")
    assert "2proc" in line
    assert "47m elapsed" in line and "2h37m left" in line


def test_the_row_says_so_when_there_is_no_nvidia_smi():
    line = console_status.render(1, 4, "chunk 1/4", 0, 0, [], None, 0, None)
    assert "no nvidia-smi" in line
    assert "~? left" in line, "an unknown estimate must not print as a number"


def test_hm_reads_as_time_or_admits_it_does_not_know():
    assert console_status.hm(0) == "0m"
    assert console_status.hm(2820) == "47m"
    assert console_status.hm(9420) == "2h37m"
    assert console_status.hm(None) == "?"
    assert console_status.hm(float("nan")) == "?"
    assert console_status.hm(-5) == "?"


def test_emit_is_a_plain_print_when_no_bar_owns_the_console(capsys):
    console_status.emit("hello")
    assert capsys.readouterr().out.strip() == "hello"


def test_a_leftover_progress_file_is_not_read_as_progress(tmp_path, monkeypatch):
    """The unit file lives on between runs. Counting yesterday's finished assets
    as today's would put the bar at 3/3 before anything had trained."""
    monkeypatch.setenv("AR_PROGRESS_DIR", str(tmp_path))
    stale = {"started": "2000-01-01T00:00:00", "order": ["A", "B"],
             "assets_total": 2, "workers": 2, "in_flight": [],
             "done": [["A", 100]]}
    (tmp_path / os.path.basename(ar_progress.UNIT_FILE)).write_text(
        json.dumps(stale), encoding="utf-8")
    st = console_status.Status(4)
    assert st._unit_view() == (0, 0, [], None)
    # positive control: the same file stamped after the unit began IS read
    st.unit_started_iso = "1999-01-01T00:00:00"
    done, total, _flight, _left = st._unit_view()
    assert (done, total) == (1, 2)


def test_progress_of_side_by_side_units_counts_what_finished(tmp_path, monkeypatch):
    monkeypatch.setenv("AR_PROGRESS_DIR", str(tmp_path))
    st = console_status.Status(6)
    st.set_progress(2, "regate full")
    line = console_status.render(st.unit_i, st.total_units, st.unit_label,
                                 0, 0, [], None, 60, 600)
    assert " 3/6 " in line, "two finished means the third is the one in flight"


def test_a_child_is_told_not_to_draw_without_touching_this_process(monkeypatch):
    monkeypatch.delenv("GTRADE_NO_TICKER", raising=False)
    env = console_status.quiet_child_env({"A": "1"})
    assert env["GTRADE_NO_TICKER"] == "1" and env["A"] == "1"
    assert "GTRADE_NO_TICKER" not in os.environ
