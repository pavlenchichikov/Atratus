"""compare() unit test for the local-only ab_foundation harness."""
import pytest

ab = pytest.importorskip("ab_foundation")


def _rows(scores, accs):
    return [{"Asset": f"A{i}", "Score": s, "LSTM_Acc": a}
            for i, (s, a) in enumerate(zip(scores, accs))]


def test_hold_on_small_or_negative():
    a = _rows([1, 1, 1, 1, 1], [0.5] * 5)
    b = _rows([1, 1, 1, 1, 1], [0.5] * 5)
    assert ab.compare(a, b)["verdict"] == "HOLD"


def test_adopt_on_clear_win():
    a = _rows([1.0] * 8, [0.50] * 8)
    b = _rows([2.5] * 8, [0.55] * 8)
    v = ab.compare(a, b)
    assert v["verdict"] == "ADOPT" and v["mean_d"] == pytest.approx(1.5)


def test_net_regression_blocks_adopt():
    a = _rows([1.0] * 8, [0.55] * 8)
    b = _rows([2.5] * 8, [0.50] * 8)  # score up but nets got worse
    assert ab.compare(a, b)["verdict"] == "HOLD"
