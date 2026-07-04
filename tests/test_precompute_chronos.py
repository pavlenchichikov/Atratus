import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import precompute_chronos as pc


def _fake_forecaster(context_arr, horizon):
    last = float(context_arr[-1])
    return {0.1: last * 0.98, 0.5: last * 1.01, 0.9: last * 1.04}


def _seed_prices(engine, table, n=80):
    df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=n).astype(str),
                       "Close": np.linspace(100, 120, n)})
    df.to_sql(table, engine, index=False)


def test_precompute_caches_and_is_incremental(tmp_path):
    eng = create_engine("sqlite:///" + str(tmp_path / "m.db"))
    _seed_prices(eng, "sp500")
    n1 = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64)
    assert n1 > 0
    cached = pd.read_sql("SELECT * FROM chronos_cache WHERE asset='sp500'", eng)
    assert set(["asset", "date", "chronos_ret", "chronos_spread", "chronos_dir"]).issubset(cached.columns)
    # re-run: nothing new (incremental skip)
    n2 = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64)
    assert n2 == 0


def test_precompute_is_model_aware(tmp_path):
    """Two different base models cache the SAME dates independently - the second model
    is NOT dedup-skipped as an (asset,date) duplicate of the first."""
    eng = create_engine("sqlite:///" + str(tmp_path / "mm.db"))
    _seed_prices(eng, "sp500")
    a = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64,
                            model="amazon/chronos-t5-tiny")
    b = pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64,
                            model="amazon/chronos-t5-base")
    assert a > 0 and b == a                                  # base cached its own rows, not 0
    rows = pd.read_sql("SELECT model, COUNT(*) c FROM chronos_cache GROUP BY model", eng)
    assert set(rows["model"]) == {"amazon/chronos-t5-tiny", "amazon/chronos-t5-base"}
    # each model re-run is still incremental (no double-cache)
    assert pc.precompute_asset("SP500", eng, forecaster=_fake_forecaster, context=64,
                               model="amazon/chronos-t5-base") == 0


def test_migrate_adds_model_column_and_backfills(tmp_path):
    import sqlite3
    db = str(tmp_path / "leg.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE chronos_cache (asset TEXT, date TEXT, chronos_ret REAL, "
                "chronos_spread REAL, chronos_dir REAL)")   # legacy: no model column
    con.execute("INSERT INTO chronos_cache VALUES ('sp500','2020-01-01',0.1,0.2,1.0)")
    con.commit(); con.close()
    pc.migrate(db)
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(chronos_cache)").fetchall()]
    model = con.execute("SELECT model FROM chronos_cache").fetchone()[0]
    con.close()
    assert "model" in cols and model == "amazon/chronos-t5-tiny"   # backfilled to tiny
