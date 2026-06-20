"""Tests for text reports (digest and bot replies)."""

from core import reports

SIGNALS = [
    {"asset": "BTC", "date": "2026-06-10", "signal": "BUY", "probability": 0.62,
     "acc": {"n": 20, "correct": 13, "acc": 0.65}},
    {"asset": "TSLA", "date": "2026-06-10", "signal": "WAIT", "probability": 0.51,
     "acc": {"n": 5, "correct": 2, "acc": 0.4}},
    {"asset": "GOLD", "date": "2026-06-10", "signal": "SELL", "probability": 0.59,
     "acc": {"n": 0, "correct": 0, "acc": None}},
]


def test_top_message_excludes_wait():
    msg = reports.build_top_message(SIGNALS, n=5)
    assert "BTC" in msg and "GOLD" in msg
    assert "TSLA" not in msg


def test_top_message_empty():
    msg = reports.build_top_message([], n=5)
    assert "no" in msg.lower()


def test_signal_message_shows_track():
    track = [
        {"date": "2026-06-10", "signal": "BUY", "probability": 0.62,
         "actual_next_ret": None, "correct": None},
        {"date": "2026-06-09", "signal": "SELL", "probability": 0.58,
         "actual_next_ret": 0.004, "correct": 0},
    ]
    msg = reports.build_signal_message("BTC", track, {"n": 2, "correct": 1, "acc": 0.5})
    assert "BTC" in msg
    assert "2026-06-10" in msg
    assert "50%" in msg


def test_signal_message_no_history():
    msg = reports.build_signal_message("BTC", [], {"n": 0, "correct": 0, "acc": None})
    assert "no" in msg.lower()


def test_risk_message_without_state():
    msg = reports.build_risk_message(None, {"max_daily_loss": 0.05, "max_drawdown_halt": 0.15})
    assert "5%" in msg or "5.0%" in msg


def test_risk_message_with_state():
    state = {"current_capital": 9500.0, "peak_capital": 10000.0,
             "initial_capital": 10000.0, "open_positions": {"BTC": {}}}
    msg = reports.build_risk_message(state, {"max_daily_loss": 0.05, "max_drawdown_halt": 0.15})
    assert "9" in msg and "500" in msg.replace(" ", "").replace(",", "")


def test_digest_contains_sections():
    msg = reports.build_digest(
        signals=SIGNALS,
        stale=[{"asset": "OZON", "last_date": "2026-05-01", "age_days": 42}],
        risk={"current_capital": 10000.0, "peak_capital": 10000.0,
              "initial_capital": 10000.0, "open_positions": {}},
        date_str="2026-06-12",
    )
    assert "2026-06-12" in msg
    assert "BTC" in msg
    assert "OZON" in msg


def test_digest_handles_empty():
    msg = reports.build_digest(signals=[], stale=[], risk=None, date_str="2026-06-12")
    assert "no" in msg.lower()
