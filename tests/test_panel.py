"""The pooled panel: normalisation that cannot leak, folds that cannot touch."""
import numpy as np
import pandas as pd

from core.panel import EMBARGO_DAYS, MIN_ASSETS_PER_DATE, RANK_FEATURES, build_panel, date_folds, rank_within_date


def _panel(rows):
    return pd.DataFrame(rows)


def test_rank_is_computed_within_a_date_not_over_the_whole_column():
    """BTC's macd_hist is four orders of magnitude above EURUSD's. Pooled raw,
    that column tells the model which asset it is looking at."""
    p = _panel([{"date": "2026-01-01", "asset": "BTC", "macd_hist": 900.0},
                {"date": "2026-01-01", "asset": "EUR", "macd_hist": 0.001},
                {"date": "2026-01-02", "asset": "BTC", "macd_hist": -900.0},
                {"date": "2026-01-02", "asset": "EUR", "macd_hist": 0.002}])
    out = rank_within_date(p, ["macd_hist"])
    assert out.loc[0, "macd_hist"] == 1.0 and out.loc[1, "macd_hist"] == 0.5
    # On the second date BTC is the LOWER of the two, and says so.
    assert out.loc[2, "macd_hist"] == 0.5 and out.loc[3, "macd_hist"] == 1.0


def test_a_later_date_cannot_change_an_earlier_rank():
    """The leak this guards is the reason ranking was chosen over z-scoring:
    a mean over the whole sample carries the future into every training row."""
    rows = [{"date": "2026-01-01", "asset": "A", "sma_20": 1.0},
            {"date": "2026-01-01", "asset": "B", "sma_20": 2.0},
            {"date": "2026-01-02", "asset": "A", "sma_20": 3.0},
            {"date": "2026-01-02", "asset": "B", "sma_20": 4.0}]
    before = rank_within_date(_panel(rows), ["sma_20"]).loc[:1, "sma_20"].tolist()
    rows[2]["sma_20"] = 1e9          # the future moves violently
    after = rank_within_date(_panel(rows), ["sma_20"]).loc[:1, "sma_20"].tolist()
    assert before == after


def test_only_the_measured_scale_dependent_features_are_ranked():
    """Ranking a return would throw away that the whole market fell 3%: every
    asset ranks the same either way. Only the three measured offenders go."""
    assert set(RANK_FEATURES) == {"macd_hist", "sma_20", "sma_50"}
    p = _panel([{"date": "2026-01-01", "asset": "A", "ret_1": 0.01},
                {"date": "2026-01-01", "asset": "B", "ret_1": 0.02}])
    assert rank_within_date(p, RANK_FEATURES)["ret_1"].tolist() == [0.01, 0.02]


def _frame(n, start="2020-01-01", value=1.0):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"ret_1": np.full(n, value), "macd_hist": np.arange(n) * 1.0,
                         "target": np.arange(n) % 2}, index=idx)


def test_build_panel_drops_an_asset_with_too_little_history():
    rows = {"A": _frame(400), "SHORT": _frame(50)}
    out = build_panel(rows, ["ret_1", "macd_hist"], min_assets=1)
    assert set(out["asset"]) == {"A"}


def test_build_panel_drops_dates_with_no_real_cross_section():
    """A rank over three names is not a cross-section, and those dates are the
    ragged start of the history where assets have not begun yet."""
    rows = {chr(65 + i): _frame(400, start="2020-01-%02d" % (1 + i)) for i in range(4)}
    out = build_panel(rows, ["ret_1"], min_assets=4)
    per_date = out.groupby("date")["asset"].size()
    assert per_date.min() >= 4
    # Positive control: the same panel keeps the thin dates when the floor drops.
    loose = build_panel(rows, ["ret_1"], min_assets=1)
    assert loose.groupby("date")["asset"].size().min() < 4


def test_folds_leave_an_embargo_between_train_and_test():
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    folds = date_folds(dates, n_folds=5, embargo=EMBARGO_DAYS)
    assert folds
    for train, test in folds:
        gap = (pd.Timestamp(test.min()) - pd.Timestamp(train.max())).days
        assert gap > EMBARGO_DAYS, "train ends %s, test starts %s" % (train.max(), test.min())
        assert not set(train) & set(test)


def test_the_embargo_is_what_creates_the_gap():
    """Positive control. Without it the last training label reads a close that
    lands in the test window, and the check above would pass for the wrong reason."""
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    folds = date_folds(dates, n_folds=5, embargo=0)
    gaps = [(pd.Timestamp(t.min()) - pd.Timestamp(tr.max())).days for tr, t in folds]
    assert min(gaps) == 1


def test_folds_move_forward_and_never_train_on_the_future():
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    folds = date_folds(dates, n_folds=4)
    for train, test in folds:
        assert train.max() < test.min()
    assert folds[0][1].min() < folds[-1][1].min()


def test_too_few_dates_yields_no_folds_rather_than_a_degenerate_one():
    assert date_folds(pd.date_range("2020-01-01", periods=10, freq="D")) == []


def test_the_cross_section_floor_is_named_once():
    assert MIN_ASSETS_PER_DATE >= 20
