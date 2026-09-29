import sqlite3

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import finra_fetch
from core import features as F


def test_parse_keeps_only_our_symbols_and_maps_them_to_tables():
    text = ("Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
            "20260925|AAPL|6485654.2|66813|11119362.8|B,Q,N\n"
            "20260925|ZZZZ|1|0|2|Q\n"
            "20260925|BRK.B|10|0|40|N\n")
    rows = finra_fetch.parse(text, {"AAPL": "aapl", "BRK.B": "brkb"})
    assert rows == [("2026-09-25", "aapl", 6485654.2, 11119362.8),
                    ("2026-09-25", "brkb", 10.0, 40.0)]


def _engine(tmp_path, rows):
    path = str(tmp_path / "m.db")
    con = sqlite3.connect(path)
    con.execute(finra_fetch.DDL)
    con.executemany("INSERT INTO finra_shvol VALUES (?,?,?,?)", rows)
    con.commit()
    con.close()
    return create_engine("sqlite:///" + path)


def _frame(n=100):
    d = pd.date_range("2026-01-01", periods=n, freq="B")
    c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({"Date": d, "close": c, "high": c + 1, "low": c - 1})


def test_short_ratio_z_is_filled_flagged_and_uses_no_future(tmp_path):
    df = _frame()
    days = df["Date"].dt.strftime("%Y-%m-%d").tolist()
    rows = [(d, "aapl", 40.0 + (i % 7), 100.0) for i, d in enumerate(days[:80])]
    out = F.add_finra_features(df.copy(), "aapl", _engine(tmp_path, rows))
    assert not out[["short_ratio_z", "short_ratio_has"]].isna().any().any()
    assert out["short_ratio_has"].iloc[:80].sum() > 40 and out["short_ratio_has"].iloc[85:].sum() == 0
    # changing a LATER row must not move an earlier z (no look-ahead)
    rows2 = list(rows)
    rows2[70] = (rows2[70][0], "aapl", 99.0, 100.0)
    (tmp_path / "b").mkdir()
    out2 = F.add_finra_features(df.copy(), "aapl", _engine(tmp_path / "b", rows2))
    assert np.allclose(out["short_ratio_z"].iloc[:70], out2["short_ratio_z"].iloc[:70])


def test_assets_without_finra_rows_get_zeros(tmp_path):
    out = F.add_finra_features(_frame(), "btc", _engine(tmp_path, []))
    assert (out["short_ratio_z"] == 0).all() and (out["short_ratio_has"] == 0).all()


def test_har_range_is_a_positive_next_day_range_forecast_without_nans():
    out = F.add_har_features(_frame())
    assert not out["har_range"].isna().any()
    assert (out["har_range"].iloc[30:] > 0).all()
