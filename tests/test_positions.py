"""Unit tests for core.positions.build_positions (pure, no I/O)."""

from core.positions import build_positions


def _b(date, signal, ret=None):
    return {"date": date, "signal": signal, "ret": ret}


def test_consecutive_buys_are_one_long_position():
    bars = [
        _b("d1", "BUY", 0.02),
        _b("d2", "BUY", 0.01),
        _b("d3", "WAIT"),
    ]
    r = build_positions(bars)
    # one long segment (d1-d2), one flat segment (d3)
    longs = [s for s in r["segments"] if s["side"] == 1]
    assert len(longs) == 1
    seg = longs[0]
    assert seg["start_date"] == "d1" and seg["end_date"] == "d2" and seg["bars"] == 2
    assert not seg["open"]
    # chained return 1.02 * 1.01 - 1
    assert abs(seg["ret"] - (1.02 * 1.01 - 1)) < 1e-9
    # markers: enter at d1, exit at d2
    assert {"date": "d1", "type": "enter", "side": 1} in r["markers"]
    assert {"date": "d2", "type": "exit", "side": 1} in r["markers"]
    # one closed trade
    assert len(r["trades"]) == 1
    # current is flat (last bar WAIT)
    assert r["current"] == {"state": "FLAT"}


def test_short_return_is_inverted():
    r = build_positions([_b("d1", "SELL", 0.02), _b("d2", "WAIT")])
    short = [s for s in r["segments"] if s["side"] == -1][0]
    # short profits when price falls; ret 0.02 up -> position -0.02
    assert abs(short["ret"] - (-0.02)) < 1e-9


def test_open_position_has_no_exit_and_is_current():
    bars = [_b("d1", "WAIT"), _b("d2", "BUY", 0.03), _b("d3", "BUY")]  # d3 not reconciled
    r = build_positions(bars)
    last = r["segments"][-1]
    assert last["side"] == 1 and last["open"]
    # no exit marker for the open position
    assert not any(m["type"] == "exit" and m["side"] == 1 for m in r["markers"])
    # open position is not a closed trade
    assert r["trades"] == []
    # current is LONG since d2, 2 bars, ret from the single reconciled bar
    assert r["current"]["state"] == "LONG"
    assert r["current"]["since"] == "d2" and r["current"]["bars"] == 2
    assert abs(r["current"]["ret"] - 0.03) < 1e-9
    assert r["current"]["fresh"] is False


def test_flip_buy_to_sell_makes_two_positions():
    r = build_positions([_b("d1", "BUY", 0.01), _b("d2", "SELL", 0.01)])
    sides = [s["side"] for s in r["segments"]]
    assert sides == [1, -1]
    # the second is still open (last bar), the first is a closed long trade
    assert len(r["trades"]) == 1 and r["trades"][0]["side"] == 1
    assert r["current"]["state"] == "SHORT" and r["current"]["fresh"] is True


def test_empty_and_all_flat():
    assert build_positions([])["current"] is None
    r = build_positions([_b("d1", "WAIT"), _b("d2", "WAIT")])
    assert r["current"] == {"state": "FLAT"}
    assert r["trades"] == [] and r["markers"] == []
