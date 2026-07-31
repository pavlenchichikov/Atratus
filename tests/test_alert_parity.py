"""The shared fixture drives both core/alerts.py and the client's
alert_filter.dart.

The freshness test is what makes the duplication safe: it fails when the two
copies diverge, which is the failure mode that would otherwise pass quietly (a
case added on one side only).
"""

import json
import os

import pytest

from core import alerts, digest

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "alert_cases.json")


def _cases():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def _mobile_fixture():
    """The client copy. Tried in order: GTRADE_MOBILE_DIR, the sibling
    mobile_app checkout (present in the main checkout, absent in a worktree -
    which is exactly where this branch's work happens), then the known absolute
    client path. Returns (path_or_None, [every path tried]) so a skip can say
    what it looked for instead of failing mysteriously."""
    roots = []
    env_root = os.getenv("GTRADE_MOBILE_DIR")
    if env_root:
        roots.append(env_root)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.join(os.path.dirname(repo), "mobile_app"))
    # Known absolute client path (ASCII source, Cyrillic value via \uXXXX so
    # the .py file itself stays ASCII-only).
    roots.append("C:/\u041d\u043e\u0432\u0430\u044f_\u043f\u0430\u043f\u043a\u0430/mobile_app")

    tried = [os.path.join(root, "test", "fixtures", "alert_cases.json")
             for root in roots]
    for path in tried:
        if os.path.exists(path):
            return path, tried
    return None, tried


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_fixture_payload(case):
    events = [digest.DigestEvent(asset=e["asset"], kind=e["kind"],
                                 from_signal=e["from_signal"],
                                 to_signal=e["to_signal"], from_timing=None,
                                 to_timing=None, confidence=e["confidence"],
                                 date=e["date"])
              for e in case["events"]]
    payload, _n_sig, _n_tim = alerts.encode_events(events)
    assert payload == case["payload"], case["name"]


def test_mobile_copy_is_identical():
    path, tried = _mobile_fixture()
    if path is None:
        pytest.skip("client tree not found - tried: " + ", ".join(tried))
    with open(FIXTURE, "rb") as fh:
        want = fh.read()
    with open(path, "rb") as fh:
        got = fh.read()
    assert got == want, ("the client fixture copy drifted from the source; copy "
                         f"{FIXTURE} over {path}")
