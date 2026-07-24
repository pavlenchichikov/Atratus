"""Unit tests for core.digest (pure; mirrors the client's lib/digest.dart)."""

from core import digest


def h(asset, date, signal, prob=None, timing=None):
    return {"asset": asset, "date": date, "signal": signal, "prob": prob,
            "timing_label": timing}


def test_detects_entry_flip_and_exit():
    rows = [
        h("AAA", "2026-07-23", "WAIT"), h("AAA", "2026-07-24", "BUY", 0.7),
        h("BBB", "2026-07-23", "BUY", 0.6), h("BBB", "2026-07-24", "SELL", 0.3),
        h("CCC", "2026-07-23", "BUY", 0.6), h("CCC", "2026-07-24", "WAIT"),
    ]
    kinds = {e.asset: e.kind for e in digest.build_digest(rows)}
    assert kinds == {"AAA": digest.ENTRY_BUY, "BBB": digest.FLIP,
                     "CCC": digest.EXIT}


def test_entry_sell_confidence_is_one_minus_prob():
    rows = [h("DDD", "2026-07-23", "WAIT"), h("DDD", "2026-07-24", "SELL", 0.25)]
    ev = digest.build_digest(rows)[0]
    assert ev.kind == digest.ENTRY_SELL
    assert abs(ev.confidence - 0.75) < 1e-9
    assert (ev.from_signal, ev.to_signal, ev.date) == ("WAIT", "SELL", "2026-07-24")


def test_exit_has_no_confidence():
    rows = [h("CCC", "2026-07-23", "BUY", 0.6), h("CCC", "2026-07-24", "WAIT")]
    assert digest.build_digest(rows)[0].confidence is None


def test_timing_change_only_when_signal_unchanged():
    rows = [h("AAA", "2026-07-23", "BUY", 0.7),
            h("AAA", "2026-07-24", "BUY", 0.7, "policy: entering")]
    ev = digest.build_digest(rows)[0]
    assert ev.kind == digest.TIMING_CHANGE
    assert (ev.from_timing, ev.to_timing) == (None, "policy: entering")


def test_signal_change_wins_over_timing_change():
    rows = [h("AAA", "2026-07-23", "WAIT", None, "policy: waiting"),
            h("AAA", "2026-07-24", "BUY", 0.7, "policy: entering")]
    assert digest.build_digest(rows)[0].kind == digest.ENTRY_BUY


def test_no_event_when_nothing_changed_or_single_row():
    rows = [h("AAA", "2026-07-23", "BUY", 0.7), h("AAA", "2026-07-24", "BUY", 0.7),
            h("BBB", "2026-07-24", "BUY", 0.9)]
    assert digest.build_digest(rows) == []


def test_baseline_is_the_previous_distinct_date():
    # Two rows share the latest date: the baseline must still be 07-23.
    rows = [h("AAA", "2026-07-23", "WAIT"), h("AAA", "2026-07-24", "BUY", 0.7),
            h("AAA", "2026-07-24", "BUY", 0.7)]
    assert digest.build_digest(rows)[0].kind == digest.ENTRY_BUY


def test_missing_prob_yields_no_confidence():
    rows = [h("AAA", "2026-07-23", "WAIT"), h("AAA", "2026-07-24", "BUY", None)]
    assert digest.build_digest(rows)[0].confidence is None


def test_order_by_kind_then_confidence_then_asset():
    rows = [
        h("BBB", "2026-07-23", "WAIT"), h("BBB", "2026-07-24", "BUY", 0.8),
        h("EEE", "2026-07-23", "WAIT"), h("EEE", "2026-07-24", "BUY", 0.8),
        h("AAA", "2026-07-23", "WAIT"), h("AAA", "2026-07-24", "BUY", 0.9),
        h("MMM", "2026-07-23", "BUY", 0.6), h("MMM", "2026-07-24", "WAIT"),
        h("ZZZ", "2026-07-23", "SELL", 0.4), h("ZZZ", "2026-07-24", "WAIT"),
        h("FFF", "2026-07-23", "BUY", 0.6), h("FFF", "2026-07-24", "SELL", 0.2),
    ]
    order = [e.asset for e in digest.build_digest(rows)]
    # flip, then entries by confidence desc (AAA .9) then asset asc (BBB, EEE),
    # then exits which all tie on a None confidence and fall back to asset asc.
    assert order == ["FFF", "AAA", "BBB", "EEE", "MMM", "ZZZ"]


def test_since_date_baseline():
    rows = [h("AAA", "2026-07-20", "WAIT"), h("AAA", "2026-07-22", "BUY", 0.7),
            h("AAA", "2026-07-24", "BUY", 0.7)]
    ev = digest.build_digest(rows, since_date="2026-07-21")
    assert ev[0].kind == digest.ENTRY_BUY
    assert ev[0].from_signal == "WAIT"


def test_since_date_on_the_current_date_yields_nothing():
    rows = [h("AAA", "2026-07-20", "WAIT"), h("AAA", "2026-07-24", "BUY", 0.7)]
    assert digest.build_digest(rows, since_date="2026-07-24") == []
