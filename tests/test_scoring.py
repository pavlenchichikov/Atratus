"""Unit tests for core.scoring (the serve-time scoring shared by predict.py and
alert_bot.py) and the config-derived MOEX routing in the Telegram bot.
"""

import os

import pandas as pd

from core import scoring


def _df(n=60, cols=("close", "rsi", "sma_20", "trend_strength")):
    return pd.DataFrame({c: [1.0] * n for c in cols})


def test_select_features_prefers_registry():
    df = _df(cols=("close", "rsi", "sma_20", "vol_z"))
    reg = {"features": ["close", "rsi", "missing_col", "vol_z"]}
    # Registry order preserved, non-existent columns dropped.
    assert scoring.select_features(df, reg) == ["close", "rsi", "vol_z"]


def test_select_features_falls_back_to_active_set():
    df = _df(cols=("close", "rsi", "sma_20", "trend_strength"))
    feats = scoring.select_features(df, None)
    # Fallback uses the active candidate set, intersected with the df's columns.
    assert feats and set(feats).issubset(set(df.columns))


def test_score_asset_returns_none_when_champion_missing(tmp_path):
    # Empty model dir: no {table}_cb.cbm - None, no model loading attempted.
    assert scoring.score_asset(_df(), "BTC", "btc", None, {}, str(tmp_path)) is None


def test_score_asset_returns_none_on_short_history(tmp_path):
    # A cb model exists but the df is too short (<50 rows) - None.
    open(os.path.join(tmp_path, "btc_cb.cbm"), "w").close()
    assert scoring.score_asset(_df(n=10), "BTC", "btc", None, {}, str(tmp_path)) is None


def test_moex_universe_covers_previously_missing_names():
    """The bot routes MOEX assets from config.RADAR_GROUPS, so names added to
    config flow through automatically. These were dropped by the old hardcoded
    31-item list and must now be present."""
    from config import RADAR_GROUPS

    moex = set(RADAR_GROUPS["MOEX"])
    for name in ["CBOM", "HHRU", "SOFL", "ASTR", "WUSH", "TRMK", "MTLR",
                 "RASP", "NMTP", "FEES", "UPRO", "MSNG", "FIVE", "FIXP",
                 "LENT", "MVID", "SMLT", "LSRG", "PHOR", "SGZH"]:
        assert name in moex, f"{name} missing from the MOEX scan universe"
    assert len(moex) >= 50


def test_alert_bot_moex_targets_derived_from_config():
    """MOEX_TARGETS is derived, not a hand-maintained literal list."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "alert_bot.py")
    with open(src_path, encoding="utf-8") as f:
        src = f.read()
    assert 'MOEX_TARGETS = set(RADAR_GROUPS["MOEX"])' in src
