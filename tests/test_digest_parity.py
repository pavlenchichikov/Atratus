"""The shared fixture drives both core/digest.py and the client's digest.dart.

The freshness test is what makes the duplication safe: it fails when the two
copies of the fixture diverge, which is the failure mode that would otherwise
pass quietly (a case added on one side only).
"""

import json
import os

import pytest

from core import digest

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "digest_cases.json")


def _cases():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _mobile_fixture():
    """The client copy: GTRADE_MOBILE_DIR wins, else a sibling mobile_app dir."""
    root = os.getenv("GTRADE_MOBILE_DIR")
    if not root:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(os.path.dirname(repo), "mobile_app")
    return os.path.join(root, "test", "fixtures", "digest_cases.json")


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_fixture_case(case):
    events = digest.build_digest(case["rows"], since_date=case["since_date"])
    assert len(events) == len(case["expected"]), case["name"]
    for got, want in zip(events, case["expected"]):
        assert got.asset == want["asset"]
        assert got.kind == want["kind"]
        assert got.from_signal == want["from_signal"]
        assert got.to_signal == want["to_signal"]
        assert got.from_timing == want["from_timing"]
        assert got.to_timing == want["to_timing"]
        assert got.date == want["date"]
        if want["confidence"] is None:
            assert got.confidence is None
        else:
            assert abs(got.confidence - want["confidence"]) < 1e-9


def test_mobile_copy_is_identical():
    path = _mobile_fixture()
    if not os.path.exists(path):
        pytest.skip(f"client tree not next to the repo and GTRADE_MOBILE_DIR "
                    f"unset: {path}")
    with open(FIXTURE, "rb") as fh:
        want = fh.read()
    with open(path, "rb") as fh:
        got = fh.read()
    assert got == want, ("the client fixture copy drifted from the source; copy "
                         f"{FIXTURE} over {path}")
