"""Precompute Chronos zero-shot forecast features and cache them per (asset, date).

Opt-in and heavy (a rolling forecast per bar), so run this once when you want to A/B
Chronos features; training then reads the cache cheaply. Needs requirements-chronos.txt.

    python precompute_chronos.py [--assets SP500,NVDA|all] [--model tiny..large] [--fresh]

The base model picks the accuracy/speed trade-off: a bigger Chronos model forecasts
better but is much slower per bar. A short name (tiny/mini/small/base/large) resolves to
the matching amazon/chronos-t5-* checkpoint; a full Hugging Face id is also accepted.
--fresh wipes the cache first (start from scratch). The cache is keyed by model and the
reader (core.features.add_chronos_features) AUTO-DETECTS the cached model, so precomputing
one model is enough - training just uses whatever is cached, no GTRADE_CHRONOS_MODEL to set.
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

from core.chronos_features import (CHRONOS_MODELS, DEFAULT_CHRONOS_MODEL,
                                   forecast_features, resolve_model)
from core.features import CHRONOS_CACHE_TABLE
from core.track_record import _table_name

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
SELECTION = "SP500,NVDA,BTC,ETH,EURUSD,GBPJPY,GAS,AAPL,SBER,DAX"


def resolve_assets(spec):
    """The asset list for --assets: 'all' -> the full 181-asset universe, else the
    comma-separated names given."""
    if (spec or "").strip().lower() == "all":
        from config import FULL_ASSET_MAP
        return list(FULL_ASSET_MAP.keys())
    return [x.strip() for x in spec.split(",") if x.strip()]


def migrate(db_path=DB_PATH):
    """Make the Chronos cache MODEL-AWARE: add a nullable 'model' column (idempotent) and
    backfill pre-existing rows as the tiny default (the historical single-model era), so
    different base models cache + read independently instead of the second model's bars
    being silently discarded as (asset, date) duplicates. No-op when the table/column is
    absent/present."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in con.execute(
            "PRAGMA table_info(%s)" % CHRONOS_CACHE_TABLE).fetchall()]
        if cols and "model" not in cols:
            con.execute("ALTER TABLE %s ADD COLUMN model TEXT" % CHRONOS_CACHE_TABLE)
            con.execute("UPDATE %s SET model = ? WHERE model IS NULL" % CHRONOS_CACHE_TABLE,
                        (DEFAULT_CHRONOS_MODEL,))
            con.commit()
    except Exception:
        pass
    finally:
        con.close()


def clear_cache(db_path=DB_PATH):
    """Wipe the Chronos cache so a run starts from scratch. No-op if the table is absent."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS %s" % CHRONOS_CACHE_TABLE)
        con.commit()
    finally:
        con.close()


def _cached_dates(engine, table, model):
    """Dates already cached for (asset, model). A missing table/column -> empty set (so
    the bars are (re)computed)."""
    try:
        df = pd.read_sql(
            "SELECT date FROM %s WHERE asset = ? AND model = ?" % CHRONOS_CACHE_TABLE,
            engine, params=(table, model))
        return set(df["date"].astype(str))
    except Exception as exc:
        low = str(exc).lower()
        if "no such table" in low or "no such column" in low:
            return set()
        raise


def precompute_asset(asset, engine, forecaster=None, context=64, horizon=5,
                     model=DEFAULT_CHRONOS_MODEL):
    """Forecast + cache the uncached bars of one asset FOR THIS MODEL. Returns rows newly
    written (0 when this model already cached them all)."""
    table = _table_name(asset)
    # OHLCV column case varies (real market.db is lower-case; some tables use "Close"),
    # so read all columns and normalize to lower-case before selecting close.
    prices = pd.read_sql('SELECT * FROM "%s" ORDER BY Date' % table,
                         engine, index_col="Date")
    prices.columns = [c.lower() for c in prices.columns]
    prices.index = pd.to_datetime(prices.index)
    feats = forecast_features(prices["close"], context=context, horizon=horizon,
                              model=model, forecaster=forecaster).dropna()
    if feats.empty:
        return 0
    already = _cached_dates(engine, table, model)
    rows = []
    for dt, r in feats.iterrows():
        d = str(pd.Timestamp(dt).date())
        if d in already:
            continue
        rows.append({"asset": table, "date": d, "model": model,
                     "chronos_ret": r["chronos_ret"],
                     "chronos_spread": r["chronos_spread"], "chronos_dir": r["chronos_dir"]})
    if not rows:
        return 0
    pd.DataFrame(rows).to_sql(CHRONOS_CACHE_TABLE, engine, if_exists="append", index=False)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=SELECTION,
                    help="comma-separated asset names, or 'all' for the full 181-asset "
                         "universe (default: a 10-asset selection)")
    ap.add_argument("--model", default="tiny",
                    help="Chronos base model: %s, or a full Hugging Face id "
                         "(default: tiny)" % "/".join(CHRONOS_MODELS))
    ap.add_argument("--fresh", action="store_true",
                    help="wipe the cache first and recompute from scratch")
    args = ap.parse_args(argv)
    model = resolve_model(args.model)
    assets = resolve_assets(args.assets)
    print("[chronos] model=%s  assets=%d  fresh=%s" % (model, len(assets), args.fresh))
    if args.fresh:
        clear_cache(DB_PATH)
        print("[chronos] cache wiped - starting from scratch")
    migrate(DB_PATH)                                   # make the cache model-aware first
    engine = create_engine("sqlite:///" + DB_PATH)
    total = 0
    for a in assets:
        n = precompute_asset(a, engine, model=model)
        total += n
        print("[chronos] %s: +%d bars cached" % (a, n))
    print("[chronos] done: %d bars cached across %d assets" % (total, len(assets)))


if __name__ == "__main__":
    sys.exit(main())
