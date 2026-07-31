"""Unit tests for core.alerts (pure; no database, no network)."""

from core import alerts, digest


def ev(asset, kind, frm, to, conf=None, date="2026-07-31"):
    return digest.DigestEvent(asset=asset, kind=kind, from_signal=frm,
                              to_signal=to, from_timing=None, to_timing=None,
                              confidence=conf, date=date)


def test_a_flip_an_entry_and_an_exit_encode_in_order():
    payload, n_sig, n_tim = alerts.encode_events([
        ev("SBER", digest.FLIP, "BUY", "SELL"),
        ev("AAPL", digest.ENTRY_BUY, "WAIT", "BUY", 0.62),
        ev("GAZP", digest.EXIT, "BUY", "WAIT"),
    ])
    assert payload == "v1|3|0|SBER:F:B:S:;AAPL:E:W:B:62;GAZP:X:B:W:"
    assert (n_sig, n_tim) == (3, 0)


def test_both_entry_kinds_share_one_code_because_the_side_is_already_there():
    # E plus a to_signal of B or S carries everything the phone renders, so a
    # separate code per side would only be another value to keep in step.
    buy, _, _ = alerts.encode_events([ev("AAA", digest.ENTRY_BUY, "WAIT",
                                         "BUY", 0.7)])
    sell, _, _ = alerts.encode_events([ev("AAA", digest.ENTRY_SELL, "WAIT",
                                          "SELL", 0.7)])
    assert buy.endswith("AAA:E:W:B:70")
    assert sell.endswith("AAA:E:W:S:70")


def test_timing_events_are_counted_and_never_listed():
    payload, n_sig, n_tim = alerts.encode_events([
        ev("SBER", digest.FLIP, "BUY", "SELL"),
        ev("AAPL", digest.TIMING_CHANGE, "BUY", "BUY"),
        ev("GAZP", digest.TIMING_CHANGE, "SELL", "SELL"),
    ])
    assert payload == "v1|1|2|SBER:F:B:S:"
    assert (n_sig, n_tim) == (1, 2)


def test_an_empty_event_list_still_encodes_a_readable_header():
    assert alerts.encode_events([]) == ("v1|0|0|", 0, 0)


def test_a_missing_confidence_leaves_the_field_empty_not_zero():
    # A zero would read as "0 percent confident", which is a different claim
    # from "no confidence was computed" (every exit has none).
    payload, _, _ = alerts.encode_events([ev("AAA", digest.EXIT, "BUY", "WAIT")])
    assert payload.endswith("AAA:X:B:W:")


def test_confidence_rounds_the_same_way_the_old_notification_text_did():
    # build_push_text prints %.0f, so the phone must not start disagreeing with
    # what the server used to show for the same number.
    payload, _, _ = alerts.encode_events([ev("AAA", digest.ENTRY_BUY, "WAIT",
                                             "BUY", 0.6249)])
    assert payload.endswith("AAA:E:W:B:62")


def test_an_asset_holding_a_separator_is_dropped_but_still_counted():
    # No live ticker contains one, but a ticker added later must not be able to
    # corrupt the payload silently.
    payload, n_sig, _ = alerts.encode_events([
        ev("A:B", digest.FLIP, "BUY", "SELL"),
        ev("OKAY", digest.FLIP, "BUY", "SELL"),
    ])
    assert payload == "v1|2|0|OKAY:F:B:S:"
    assert n_sig == 2


def test_alert_rows_are_shaped_for_the_table_and_include_timing_events():
    rows = alerts.alert_rows(
        [ev("SBER", digest.FLIP, "BUY", "SELL", 0.8),
         ev("AAPL", digest.TIMING_CHANGE, "BUY", "BUY")],
        "hash123", False, "2026-07-31T10:00:00")
    assert rows == [
        ("2026-07-31T10:00:00", "2026-07-31", "SBER", "flip", "BUY", "SELL",
         0.8, "hash123", 0),
        ("2026-07-31T10:00:00", "2026-07-31", "AAPL", "timing_change", "BUY",
         "BUY", None, "hash123", 0),
    ]


def test_alert_rows_record_a_delivered_push_as_one():
    rows = alerts.alert_rows([ev("SBER", digest.FLIP, "BUY", "SELL")],
                             "h", True, "2026-07-31T10:00:00")
    assert rows[0][-1] == 1
