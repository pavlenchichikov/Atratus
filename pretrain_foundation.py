"""Foundation pretraining for the neural members (spec 2026-07-18).

Part 1 (this task): builds a pooled cross-asset dataset on the common feature
schema, with a leakage-safe cutoff derived from each asset's own walk-forward
folds. Part 2 (a later task) adds the training loop and CLI entry point that
trains one foundation net per architecture; train_hybrid (GTRADE_FOUNDATION=1)
will then seed each asset's fold-1 nets from these weights via the existing
warm-start store.

Leakage guard: the foundation must never see any bar inside a kept fold's
val/test window. compute_cutoff() backs each asset's cutoff date off by
`lookback + embargo` bars from the first kept fold's val-start index, and the
pooled dataset only uses rows strictly before the (global, minimum-across-
assets) cutoff.
"""
import os

import numpy as np

import config

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "market.db")
FOUNDATION_DIR = os.path.join(BASE, "models", "foundation")

# The common feature schema every asset's data is projected onto before pooling.
FOUNDATION_FEATURES = ("ret_1", "ret_5", "ret_10", "ret_20", "vol_z", "rsi",
                       "macd_hist", "bb_pos", "trend_strength", "atr")

# Coarse classes for the one-hot tail appended to every pooled sequence.
CLASS_GROUPS = ("crypto", "us", "eu", "ru", "forex", "commodity", "index")

# Every config.ASSET_TYPES group key mapped to exactly one coarse class above.
# "TOP SIGNALS" is a curated cross-cutting highlight list (it mixes crypto/us/
# ru/commodity tickers, e.g. ETH+TSLA+GOLD+VIX+PLTR+IMOEX) rather than a
# partition of the asset universe, so class_of() below skips it when looking
# up an asset's class - it still needs an entry here so the completeness
# assert (every ASSET_TYPES key covered) passes.
_GROUP_TO_CLASS = {
    "TOP SIGNALS": "us",
    "CRYPTO": "crypto",
    "COMMODITIES": "commodity",
    "INDICES & MACRO": "index",
    "US TECH": "us",
    "US HEALTHCARE": "us",
    "US FINANCE": "us",
    "US CONSUMER": "us",
    "US INDUSTRIAL": "us",
    "US SEMI": "us",
    "US SOFTWARE": "us",
    "EU INDICES": "eu",
    "EU STOCKS": "eu",
    "RUS BLUE CHIPS": "ru",
    "RUS FINANCE": "ru",
    "RUS TECH": "ru",
    "RUS METALS": "ru",
    "RUS INFRA": "ru",
    "RUS CONSUMER": "ru",
    "RUS PROPERTY": "ru",
    "FOREX MAJORS": "forex",
    "FOREX CROSSES": "forex",
    "FOREX EXOTIC": "forex",
}
assert set(_GROUP_TO_CLASS) == set(config.ASSET_TYPES), (
    "config.ASSET_TYPES groups changed - update _GROUP_TO_CLASS in pretrain_foundation.py"
)
assert set(_GROUP_TO_CLASS.values()) <= set(CLASS_GROUPS), (
    "_GROUP_TO_CLASS maps to a class outside CLASS_GROUPS"
)

# TOP SIGNALS overlaps every other group (it is a highlight reel, not a
# partition); class_of() skips it so lookups fall through to an asset's real
# (single) sector/asset-class group.
_GROUPS_SKIPPED_FOR_LOOKUP = frozenset({"TOP SIGNALS"})


def _normalize(name: str) -> str:
    """Table-name normalization used across the repo for asset matching."""
    return name.lower().replace("^", "").replace(".", "").replace("-", "")


def class_of(asset: str) -> str:
    """Coarse class (crypto/us/eu/ru/forex/commodity/index) for an asset.

    `asset` may be either a config.ASSET_TYPES key ("BTC") or a market.db table
    name (tables are asset names normalized via `_normalize`); both are matched.
    Falls back to "us" for anything unrecognized.
    """
    norm = _normalize(asset)
    for group, assets in config.ASSET_TYPES.items():
        if group in _GROUPS_SKIPPED_FOR_LOOKUP:
            continue
        for a in assets:
            if a == asset or _normalize(a) == norm:
                return _GROUP_TO_CLASS.get(group, "us")
    return "us"


def compute_cutoff(engine, tables, max_folds=None, lookback=60, embargo=10):
    """Global (minimum-across-assets) leakage-safe cutoff date, as an ISO string.

    For each table: read its row count and dates, derive walk-forward splits
    the same way train_hybrid's champion-selection objective does
    (adaptive_split_params + make_walk_forward_splits), keep only the last
    `max_folds` of them, and back the first kept fold's val-start index off by
    `lookback + embargo` bars to get a per-table cutoff date. Tables too short
    to produce any split are skipped. The returned cutoff is the minimum (i.e.
    earliest / most conservative) date across all tables that produced one.
    """
    import pandas as pd

    from core.backtesting import adaptive_split_params, make_walk_forward_splits

    if max_folds is None:
        max_folds = int(os.getenv("GTRADE_MAX_FOLDS", "10"))

    cutoffs = []
    for table in tables:
        dates = pd.read_sql(
            f'SELECT "Date" FROM "{table}" ORDER BY "Date"', engine,
        )["Date"].tolist()
        n = len(dates)
        sp = adaptive_split_params(n)
        if sp is None:
            continue
        splits = make_walk_forward_splits(n, **sp, embargo=embargo)
        if not splits:
            continue
        kept = splits[-max_folds:] if max_folds > 0 else splits
        va_start = kept[0][1].start
        idx = max(va_start - lookback - embargo, 0)
        cutoffs.append(dates[idx])

    if not cutoffs:
        raise ValueError("no table has enough rows to compute a leakage cutoff")
    return min(cutoffs)


def build_pooled_sequences(engine, tables, cutoff, lookback=60):
    """Pooled (X, y, skipped) cross-asset sliding-window dataset, cutoff-safe.

    For each table: read the raw OHLCV rows, skip the table outright if it has
    fewer than 300 raw rows strictly before `cutoff` (not enough history to be
    worth engineering). Otherwise run engineer_features on the full table
    (rolling/EWM features are causal, so this is equivalent to running it on
    the pre-cutoff slice for every row that survives the cutoff filter, without
    re-deriving warm-up windows), recompute the target with make_target on the
    resulting close series (aligned to the engineered rows), then restrict to
    rows strictly before cutoff. Each asset's FOUNDATION_FEATURES columns are
    z-scored using that asset's own pre-cutoff mean/std (in-window scaling is
    fine here - this is pretraining, not evaluation). A constant 7-dim one-hot
    for the asset's coarse class is appended, sliding windows of length
    `lookback` are built, and the result is pooled across all tables.

    Returns:
        X: float array, shape (n_seq, lookback, len(FOUNDATION_FEATURES) + 7).
        y: the target aligned to each sequence's last (most recent) row.
        skipped: list of table names that did not have >= 300 pre-cutoff rows
            (or too few engineered rows to form even one sequence).
    """
    import pandas as pd

    from core.features import engineer_features, make_target

    n_classes = len(CLASS_GROUPS)
    n_features = len(FOUNDATION_FEATURES) + n_classes
    xs, ys, skipped = [], [], []

    for table in tables:
        raw = pd.read_sql(f'SELECT * FROM "{table}"', engine)
        date_col = "Date" if "Date" in raw.columns else "date"
        n_raw_before = int((raw[date_col] < cutoff).sum())
        if n_raw_before < 300:
            skipped.append(table)
            continue

        eng = engineer_features(raw)
        target_full = make_target(eng["close"]).to_numpy(dtype=float)
        mask = (eng["date"] < cutoff).to_numpy()
        feats = eng.loc[mask, list(FOUNDATION_FEATURES)].to_numpy(dtype=float)
        target = target_full[mask]

        n_rows = feats.shape[0]
        if n_rows <= lookback:
            skipped.append(table)
            continue

        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        feats = (feats - mean) / std

        onehot_row = np.zeros(n_classes, dtype=float)
        onehot_row[CLASS_GROUPS.index(class_of(table))] = 1.0
        onehot = np.tile(onehot_row, (n_rows, 1))
        combined = np.hstack([feats, onehot])

        n_seq = n_rows - lookback + 1
        windows = np.lib.stride_tricks.sliding_window_view(
            combined, (lookback, combined.shape[1]),
        ).reshape(n_seq, lookback, combined.shape[1]).copy()
        y_asset = target[lookback - 1:]

        xs.append(windows)
        ys.append(y_asset)

    if xs:
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
    else:
        X = np.empty((0, lookback, n_features))
        y = np.empty((0,))
    return X, y, skipped


# -- Part 2: training loop, manifest, CLI ------------------------------------

# Fixed sizing for the foundation nets - deliberately NOT adaptive_units()'d off
# data size like train_hybrid's per-asset nets: the foundation always trains on
# the full pooled cross-asset dataset, so its capacity is a constant.
FOUNDATION_LOOKBACK = 60
FOUNDATION_UNITS = 64


def train_foundation(X, y, epochs=40, batch=512, val_frac=0.1, out_dir=None):
    """Train the 3 foundation nets (LSTM-multitask, transformer, TCN) on pooled X/y.

    Validation split: the last `val_frac` fraction of the (already time-ordered
    per asset) pooled array is held out for validation/early-stopping. CAVEAT:
    build_pooled_sequences concatenates assets one after another, so the pooled
    array is time-ordered only WITHIN each asset's block, not globally across
    assets - a simple tail slice of the concatenated array can therefore mix in
    a different (and not necessarily "later") asset's early rows if that asset
    happens to fall in the tail. This is acceptable here: the foundation net is
    a shared cross-asset warm start, not a per-asset walk-forward-scored model,
    so a small amount of cross-asset shuffling in the validation slice only
    affects early-stopping's stopping point, not any leakage-sensitive metric.

    Sizing: all 3 nets are pinned to FOUNDATION_UNITS (fixed capacity - the
    dataset is the full pooled cross-asset set, not a single small asset, so
    there is no need to size capacity to the data like train_hybrid does).
    input_shape is read from X.shape[1:] (NOT hardcoded), so this also works
    with the smaller lookback used by tests.

    The LSTM-multitask builder (core.architectures.build_lstm_multitask) always
    returns a direction (primary) + magnitude (auxiliary) head. The foundation
    only cares about direction, but the model must still keep the magnitude
    head (and be fed a target for it) so the saved weights stay shape-compatible
    with train_hybrid's warm-start loader (_ws_load, which does model.set_weights
    with a shape check). The magnitude target is fed as zeros with loss_weight
    0.0, so it contributes no gradient and does not affect training.

    Returns: {"lstm": {...}, "transformer": {...}, "tcn": {...}} - each a dict
    of that net's final (post restore_best_weights) validation metrics, keyed
    by the net's own model.metrics_names.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    from core.architectures import (
        build_lstm_multitask, build_tcn, build_transformer_encoder,
    )

    out_dir = out_dir or FOUNDATION_DIR
    os.makedirs(out_dir, exist_ok=True)

    n = len(X)
    n_val = max(1, int(round(n * val_frac)))
    split = max(1, n - n_val)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_tr = X_tr.astype("float32")
    X_val = X_val.astype("float32")
    y_tr = y_tr.astype("float32")
    y_val = y_val.astype("float32")

    input_shape = (X.shape[1], X.shape[2])
    u1 = FOUNDATION_UNITS
    metrics = {}

    # -- LSTM multitask (direction primary, magnitude silenced) --------------
    print("[foundation] training lstm...", flush=True)
    lstm_mt = build_lstm_multitask(
        input_shape, n_train_samples=len(X_tr),
        units1=u1, units2=max(16, u1 // 2), head_dim=max(16, u1 // 2),
    )
    # Recompile to zero the magnitude loss_weight (the builder's own default is
    # 0.2 for the auxiliary task) while keeping the same optimizer/loss/output
    # shapes, so warm-starting from these weights later needs no shape surgery.
    lstm_mt.compile(
        optimizer=lstm_mt.optimizer,
        loss={"direction": "binary_crossentropy", "magnitude": "huber"},
        loss_weights={"direction": 1.0, "magnitude": 0.0},
        metrics={"direction": "accuracy"},
    )
    y_mag_tr = np.zeros_like(y_tr, dtype="float32")
    y_mag_val = np.zeros_like(y_val, dtype="float32")
    es_lstm = EarlyStopping(monitor="val_direction_loss", mode="min", patience=5,
                            restore_best_weights=True)
    lstm_mt.fit(
        X_tr, {"direction": y_tr, "magnitude": y_mag_tr},
        validation_data=(X_val, {"direction": y_val, "magnitude": y_mag_val}),
        epochs=epochs, batch_size=batch, callbacks=[es_lstm], verbose=0,
    )
    lstm_eval = lstm_mt.evaluate(
        X_val, {"direction": y_val, "magnitude": y_mag_val},
        batch_size=batch, verbose=0,
    )
    metrics["lstm"] = dict(zip(lstm_mt.metrics_names, [float(v) for v in lstm_eval]))
    lstm_mt.save(os.path.join(out_dir, "lstm.keras"))
    print("[foundation] lstm done", flush=True)

    # -- Transformer encoder --------------------------------------------------
    print("[foundation] training transformer...", flush=True)
    tf_enc = build_transformer_encoder(
        input_shape, n_train_samples=len(X_tr), ff_dim=u1,
    )
    es_tf = EarlyStopping(monitor="val_loss", mode="min", patience=5,
                          restore_best_weights=True)
    tf_enc.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch, callbacks=[es_tf], verbose=0,
    )
    tf_eval = tf_enc.evaluate(X_val, y_val, batch_size=batch, verbose=0)
    tf_eval = tf_eval if isinstance(tf_eval, list) else [tf_eval]
    metrics["transformer"] = dict(zip(tf_enc.metrics_names, [float(v) for v in tf_eval]))
    tf_enc.save(os.path.join(out_dir, "transformer.keras"))
    print("[foundation] transformer done", flush=True)

    # -- TCN -------------------------------------------------------------------
    print("[foundation] training tcn...", flush=True)
    tcn_model = build_tcn(
        input_shape, n_train_samples=len(X_tr), n_filters=u1,
    )
    es_tcn = EarlyStopping(monitor="val_loss", mode="min", patience=5,
                           restore_best_weights=True)
    tcn_model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch, callbacks=[es_tcn], verbose=0,
    )
    tcn_eval = tcn_model.evaluate(X_val, y_val, batch_size=batch, verbose=0)
    tcn_eval = tcn_eval if isinstance(tcn_eval, list) else [tcn_eval]
    metrics["tcn"] = dict(zip(tcn_model.metrics_names, [float(v) for v in tcn_eval]))
    tcn_model.save(os.path.join(out_dir, "tcn.keras"))
    print("[foundation] tcn done", flush=True)

    return metrics


def write_manifest(out_dir, cutoff, tables_used, val_metrics, lookback=FOUNDATION_LOOKBACK,
                   max_folds=None, embargo=None):
    """Write manifest.json describing this foundation snapshot into out_dir.

    `net_kwargs` records the EXACT kwargs train_foundation passed to each
    builder (derived from FOUNDATION_UNITS - single source of truth, not a
    hardcoded duplicate), so train_hybrid's warm-start path can rebuild every
    asset's nets at the foundation's own architecture and guarantee
    set_weights shape-matches. num_heads for the transformer is recorded
    explicitly even though train_foundation relies on the builder's default
    (4) rather than passing it.

    `max_folds`/`embargo` record the leakage-cutoff parameters actually used
    by compute_cutoff() for this snapshot, so train_hybrid can refuse to
    warm-start when it is about to train with more folds than the cutoff
    protected against (see train_hybrid._load_foundation).
    """
    import json
    from datetime import datetime, timezone

    u1 = FOUNDATION_UNITS
    net_kwargs = {
        "lstm": {"units1": u1, "units2": max(16, u1 // 2), "head_dim": max(16, u1 // 2)},
        "transformer": {"ff_dim": u1, "num_heads": 4},
        "tcn": {"n_filters": u1},
    }

    manifest = {
        "version": 1,
        "features": list(FOUNDATION_FEATURES),
        "classes": list(CLASS_GROUPS),
        "lookback": lookback,
        "units": FOUNDATION_UNITS,
        "net_kwargs": net_kwargs,
        "cutoff": cutoff,
        "max_folds": max_folds,
        "embargo": embargo,
        "n_assets": len(tables_used),
        "val": val_metrics,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    """CLI entry point: build the pooled foundation dataset, train, and save it.

    Table discovery reuses db_check.get_tables (raw sqlite_master listing) +
    db_check._price_tables (keeps only tables with an OHLC column set), which
    already excludes log/meta tables without a Date column.
    """
    import argparse
    import sqlite3

    import sqlalchemy

    from db_check import _price_tables, get_tables

    parser = argparse.ArgumentParser(description="Foundation pretraining CLI")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--assets-limit", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=FOUNDATION_LOOKBACK)
    parser.add_argument("--embargo", type=int, default=10)
    args = parser.parse_args()

    # Resolved explicitly (rather than left to compute_cutoff's own default)
    # so the EXACT values used for the leakage cutoff can be recorded in the
    # manifest for train_hybrid's leakage-refusal check.
    max_folds = int(os.getenv("GTRADE_MAX_FOLDS", "10"))

    conn = sqlite3.connect(DB_PATH)
    try:
        tables = _price_tables(conn.cursor(), get_tables(conn.cursor()))
    finally:
        conn.close()
    tables = sorted(tables)
    if args.assets_limit:
        tables = tables[:args.assets_limit]
    print(f"[foundation] discovered {len(tables)} price tables", flush=True)

    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")

    print("[foundation] computing leakage-safe cutoff...", flush=True)
    cutoff = compute_cutoff(engine, tables, max_folds=max_folds,
                            lookback=args.lookback, embargo=args.embargo)
    print(f"[foundation] cutoff={cutoff}", flush=True)

    xs, ys, tables_used, tables_skipped = [], [], [], []
    for i, table in enumerate(tables, 1):
        print(f"[foundation] building pooled dataset [{i}/{len(tables)}] {table}...",
              flush=True)
        X_t, y_t, skipped_t = build_pooled_sequences(
            engine, [table], cutoff, lookback=args.lookback,
        )
        if skipped_t:
            tables_skipped.append(table)
            continue
        xs.append(X_t)
        ys.append(y_t)
        tables_used.append(table)

    if not xs:
        raise ValueError("no table produced any pooled sequence before cutoff")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    print(f"[foundation] pooled dataset: {X.shape[0]} sequences from "
          f"{len(tables_used)} assets ({len(tables_skipped)} skipped)", flush=True)

    metrics = train_foundation(X, y, epochs=args.epochs, out_dir=FOUNDATION_DIR)
    write_manifest(FOUNDATION_DIR, cutoff, tables_used, metrics, lookback=args.lookback,
                   max_folds=max_folds, embargo=args.embargo)
    print(f"[foundation] done. manifest + weights written to {FOUNDATION_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
