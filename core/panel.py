"""One panel over every asset, for a model that learns across them.

The per-asset ensemble sees about 3760 rows per model. At that size the standard
error of a rank correlation is 1/sqrt(n) = 0.016, so any information coefficient
below roughly 0.03 is invisible BY CONSTRUCTION. That is the whole reason this
module exists: "direction is unpredictable" was measured one asset at a time, and
one asset at a time is where it could never have been measured.

Pooled, the same 852 assets carry 3.2M rows, where the same standard error is
0.00056. An IC of 0.02 reads t = 1.2 per asset and t = 36 on the panel.

Nothing here trains or serves. It builds the panel and cuts it into folds, and
it is deliberately separate from train_hybrid so a pooled experiment cannot
touch the per-asset champions.
"""
import numpy as np
import pandas as pd

# Measured 2026-09-08 across twelve assets from four classes: the ratio of the
# largest to the smallest per-asset median |value|. 29 of the 32 candidate
# features come in under 20x, because they are already returns, z-scores,
# correlations or bounded oscillators. These three do not, and pooling their raw
# values would teach the model which asset it is looking at rather than what the
# asset is doing.
#
#     sma_50     20683x        sma_20     20251x        macd_hist  69640x
#
# They are replaced by their rank ACROSS ASSETS ON THE SAME DATE, which uses no
# information from any other date and therefore cannot leak.
RANK_FEATURES = ("macd_hist", "sma_20", "sma_50")

POOL_MIN_BARS = 300         # an asset with less history is noise in the panel
MIN_ASSETS_PER_DATE = 20    # a cross-sectional rank over a handful is not one
EMBARGO_DAYS = 5            # covers the one-bar label plus a margin


def rank_within_date(panel, cols):
    """Replace `cols` with their cross-sectional rank in [0, 1] on each date.

    Ranking within a date is the one normalisation that cannot leak: it reads
    only rows stamped with that same date. Normalising by a full-sample mean, or
    by a per-asset mean computed over the whole history, would carry the future
    into every training row and is the classic way a panel model looks brilliant
    and serves nothing.
    """
    out = panel.copy()
    for c in cols:
        if c not in out.columns:
            continue
        g = out.groupby("date")[c]
        # pct=True already yields (0, 1]; a date where every value is NaN stays NaN.
        out[c] = g.rank(pct=True, na_option="keep")
    return out


def build_panel(rows_by_asset, features, min_assets=MIN_ASSETS_PER_DATE):
    """Stack per-asset frames into one panel, ranked where ranking is needed.

    `rows_by_asset` is {asset: DataFrame indexed by date} and must already carry
    the feature columns and `target`. Dates thinner than `min_assets` are dropped:
    a cross-sectional rank over five names is not a cross-section, and those
    dates are almost always the ragged start of the history.
    """
    frames = []
    for asset, df in rows_by_asset.items():
        if df is None or len(df) < POOL_MIN_BARS:
            continue
        keep = [c for c in features if c in df.columns]
        if "target" not in df.columns:
            continue
        part = df[keep + ["target"]].copy()
        part["asset"] = asset
        part["date"] = pd.to_datetime(df.index).normalize()
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["target"])
    counts = panel.groupby("date")["asset"].transform("size")
    panel = panel[counts >= min_assets]
    panel = rank_within_date(panel, RANK_FEATURES)
    return panel.sort_values(["date", "asset"]).reset_index(drop=True)


def date_folds(dates, n_folds=5, embargo=EMBARGO_DAYS):
    """Expanding-window walk-forward over DATES, with a purge gap.

    Splitting by row or by asset would put an asset's Monday in train and another
    asset's Monday in test, and every asset moves with the market on Monday. Only
    a split by date keeps the two sides genuinely apart.

    The gap matters as much as the split: the label is the NEXT bar, so the last
    training date already reads a close that lands in the test window. `embargo`
    dates are dropped between the two sides to pay for that.

    Returns [(train_dates, test_dates)], oldest fold first.
    """
    uniq = sorted(pd.unique(pd.to_datetime(dates)))
    if n_folds < 1 or len(uniq) < (n_folds + 1) * (embargo + 2):
        return []
    block = len(uniq) // (n_folds + 1)
    out = []
    for i in range(1, n_folds + 1):
        cut = block * i
        test = uniq[cut:cut + block] if i < n_folds else uniq[cut:]
        train = uniq[:max(0, cut - embargo)]
        if not train or not test:
            continue
        out.append((np.array(train), np.array(test)))
    return out
