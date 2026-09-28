"""The barrier must be causal, signed, and honest about a missing champion."""
import pandas as pd

import lead_baseline as lb


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def _lead_and_follower(k, n=600):
    """A leader, and an asset whose NEXT bar is k times the leader's move.

    Deliberately not an alternating series: one of those is "inverted same-day"
    and "following next-day" at once, so it cannot tell the two relations apart
    - which is exactly the confusion this pair of tests exists to catch.
    """
    import random
    rnd = random.Random(7)
    lead_vals = [rnd.choice([0.01, -0.01]) for _ in range(n)]
    asset_vals = [0.0] + [k * v for v in lead_vals[:-1]]
    return _series(lead_vals), _series(asset_vals)


def test_the_sign_is_read_only_from_bars_before_train_end():
    """Reading it from the whole history is the winner's curse the barrier
    exists to guard against: the rule would be fitted on what it is scored on."""
    lead, follows = _lead_and_follower(1.0)
    flipped = follows.copy()
    flipped.iloc[400:] = -flipped.iloc[400:]      # the asset turns after the cutoff
    sign, ic = lb.lead_sign(lead, flipped, flipped.index[400])
    assert sign == 1 and ic > 0.9, (sign, ic)


def test_an_inverted_asset_gets_an_inverted_rule():
    lead, inverted = _lead_and_follower(-1.0)
    cutoff = inverted.index[400]
    sign, ic = lb.lead_sign(lead, inverted, cutoff)
    assert sign == -1 and ic < -0.9, (sign, ic)
    # and scoring uses that sign: the rule is right, not wrong, on this asset
    acc, n = lb.rule_accuracy(lead, inverted, sign, cutoff)
    assert n > 0 and acc > 0.9, (acc, n)


def test_the_prediction_is_about_the_NEXT_bar():
    """The call made on bar t is scored against t -> t+1. Scoring it against the
    bar it was made on would grade the rule on a move that had already happened."""
    lead = _series([0.01] * 100)
    # every next bar rises: a "follow the leader" call is always right
    asset = _series([0.02] * 100)
    acc, n = lb.rule_accuracy(lead, asset, 1, asset.index[40])
    assert acc == 1.0 and n == len(asset) - 41   # the last bar has no next one


def test_an_asset_with_too_little_history_is_refused_not_guessed():
    lead = _series([0.01, -0.01] * 200)
    short = _series([0.01, -0.01] * 20)
    assert lb.lead_sign(lead, short, short.index[-1]) == (None, None)


def test_a_champion_without_an_accuracy_still_reports_the_rule(tmp_path, monkeypatch):
    """A missing ens_acc must not hide the asset: the barrier is still a fact
    about it, and the champion column simply says nothing."""
    rows = [{"asset": "X", "ic": 0.4, "sign": 1, "rule_acc": 0.6, "n": 100,
             "model_acc": None, "train_end": "2025-01-01"}]
    lb.report(rows)   # must not raise on a None model_acc


def test_a_subset_run_never_rewrites_the_whole_book_report(tmp_path, monkeypatch):
    """train_sizing and train_timing rewrite their report on every run, and a
    ten-asset smoke once destroyed the 207-asset evidence that way."""
    out = tmp_path / "lead_baseline.json"
    out.write_text('{"rows": "the whole book"}', encoding="utf-8")
    monkeypatch.setattr(lb, "OUT_PATH", str(out))
    monkeypatch.setattr(lb, "REGISTRY_PATH", str(tmp_path / "registry.json"))
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lb, "evaluate", lambda *a, **k: [
        {"asset": "X", "ic": 0.4, "sign": 1, "rule_acc": 0.6, "n": 99,
         "model_acc": 0.5, "train_end": "2025-01-01"}])
    monkeypatch.setattr(lb.sqlite3, "connect", lambda *a, **k: _FakeCon())
    assert lb.main(["NIKKEI"]) == 0
    assert out.read_text(encoding="utf-8") == '{"rows": "the whole book"}'
    assert lb.main([]) == 0
    assert "the whole book" not in out.read_text(encoding="utf-8")


class _FakeCon:
    def close(self):
        pass


def _db_with(tmp_path, series):
    """market.db holding one close table per {asset: close list}, daily dates."""
    import sqlite3
    path = str(tmp_path / "market.db")
    con = sqlite3.connect(path)
    for asset, closes in series.items():
        df = pd.DataFrame({"Date": pd.date_range("2022-01-03", periods=len(closes), freq="D")
                           .strftime("%Y-%m-%d"), "close": closes})
        df.to_sql(lb.table_of(asset), con, index=False)
    return con


def _prices(rets, start=100.0):
    out = [start]
    for r in rets:
        out.append(out[-1] * (1 + r))
    return out


def test_current_calls_select_the_follower_and_call_its_next_bar(tmp_path):
    """The follower is ranked on its own recent bars, the noise asset is left
    out, and the call is the leader's LAST move, inverted for an inverse pair."""
    import random
    rnd = random.Random(3)
    lead_r = [rnd.choice([0.01, -0.01]) for _ in range(400)]
    lead_r[-1] = 0.01                                     # the US closed up today
    follow_r = [0.0] + [0.5 * v for v in lead_r[:-1]]     # next bar follows
    inverse_r = [0.0] + [-0.5 * v for v in lead_r[:-1]]
    noise_r = [rnd.choice([0.01, -0.01]) for _ in range(400)]
    con = _db_with(tmp_path, {"SP500": _prices(lead_r), "FOLLOW": _prices(follow_r),
                              "INVERSE": _prices(inverse_r), "NOISE": _prices(noise_r)})
    calls = lb.current_calls(con, ["FOLLOW", "INVERSE", "NOISE", "SP500"], top=2)
    assert set(calls) == {"FOLLOW", "INVERSE"}
    assert calls["FOLLOW"]["call"] == "BUY"
    assert calls["INVERSE"]["call"] == "SELL"
    assert calls["FOLLOW"]["rank"] in (1, 2)


def test_no_call_until_the_us_session_of_that_bar_has_closed(tmp_path):
    """Before the evening data update the asset already has today's bar and the
    leader does not. The leader's previous move is yesterday's news."""
    import random
    rnd = random.Random(5)
    lead_r = [rnd.choice([0.01, -0.01]) for _ in range(400)]
    follow_r = [0.0] + [0.5 * v for v in lead_r[:-1]]
    con = _db_with(tmp_path, {"SP500": _prices(lead_r[:-1]),   # one bar behind
                              "FOLLOW": _prices(follow_r)})
    calls = lb.current_calls(con, ["FOLLOW"], top=1)
    assert calls["FOLLOW"]["call"] is None


def test_no_call_for_an_asset_whose_last_bar_is_behind_the_leader(tmp_path):
    """TAIEX missing yesterday's bar would otherwise be 'called' for a bar that
    already traded."""
    import random
    rnd = random.Random(9)
    lead_r = [rnd.choice([0.01, -0.01]) for _ in range(400)]
    follow_r = [0.0] + [0.5 * v for v in lead_r[:-1]]
    con = _db_with(tmp_path, {"SP500": _prices(lead_r),
                              "FOLLOW": _prices(follow_r[:-1])})   # one bar behind
    calls = lb.current_calls(con, ["FOLLOW"], top=1)
    assert calls["FOLLOW"]["call"] is None
