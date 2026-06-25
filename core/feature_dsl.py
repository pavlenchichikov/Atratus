# core/feature_dsl.py
"""Constrained feature-transform DSL for the auto-research agent. A spec is a dict
{name, op, inputs, params}. materialize() handles dataframe-only ops, so it is pure
and testable; the engine-dependent lead_lag op is applied by add_dsl_features. No
eval, no arbitrary code."""
import re

ALLOWED_OPS = {"zscore", "ratio", "lag", "diff", "rolling", "interaction", "lead_lag"}
_AGG = {"mean", "std", "sum"}
_LEADERS = {"sp500", "vix", "btc", "gold", "dxy", "tnx"}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
_ARITY = {"zscore": 1, "ratio": 2, "lag": 1, "diff": 1, "rolling": 1,
          "interaction": 2, "lead_lag": 1}
EPS = 1e-9


def validate_spec(spec, columns):
    """True if spec is a well-formed, in-bounds transform. columns is the set of
    available dataframe column names; lead_lag inputs reference a known leader."""
    if not isinstance(spec, dict):
        return False
    name, op = spec.get("name"), spec.get("op")
    inputs = spec.get("inputs") or []
    params = spec.get("params") or {}
    if op not in ALLOWED_OPS:
        return False
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return False
    if len(inputs) != _ARITY[op]:
        return False
    if op == "lead_lag":
        h = params.get("horizon", 1)
        return inputs[0] in _LEADERS and isinstance(h, int) and 1 <= h <= 20
    if any(c not in columns for c in inputs):
        return False
    if op in ("zscore", "rolling"):
        w = params.get("window", 20)
        if not (isinstance(w, int) and 2 <= w <= 200):
            return False
        if op == "rolling" and params.get("agg", "mean") not in _AGG:
            return False
    if op in ("lag", "diff"):
        k = params.get("k", 1)
        if not (isinstance(k, int) and 1 <= k <= 20):
            return False
    return True


def materialize(df, spec):
    """Compute a dataframe-only transform as a guarded Series. Raises ValueError for
    lead_lag (engine op) or an unknown op."""
    op = spec["op"]
    inputs = spec["inputs"]
    params = spec.get("params") or {}
    if op == "zscore":
        s = df[inputs[0]]
        w = params.get("window", 20)
        out = (s - s.rolling(w).mean()) / (s.rolling(w).std() + EPS)
    elif op == "ratio":
        out = df[inputs[0]] / (df[inputs[1]] + EPS)
    elif op == "lag":
        out = df[inputs[0]].shift(params.get("k", 1))
    elif op == "diff":
        out = df[inputs[0]].diff(params.get("k", 1))
    elif op == "rolling":
        w = params.get("window", 20)
        out = getattr(df[inputs[0]].rolling(w), params.get("agg", "mean"))()
    elif op == "interaction":
        out = df[inputs[0]] * df[inputs[1]]
    else:
        raise ValueError("materialize does not handle op %r" % op)
    return out.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def _lead_lag_col(df, engine, spec):
    """As-of forward-filled leader return (no leakage), mirroring add_cross_lag."""
    import pandas as pd
    table = spec["inputs"][0]
    horizon = (spec.get("params") or {}).get("horizon", 1)
    date_col = "Date" if "Date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is None:
        return pd.Series(0.0, index=df.index)
    idx = pd.to_datetime(df[date_col])
    try:
        ref = pd.read_sql(f"SELECT Date, Close FROM {table}", engine,
                          index_col="Date", parse_dates=["Date"])
        ref.index = pd.to_datetime(ref.index).normalize()
        ref.columns = [c.lower() for c in ref.columns]
        ref = ref[~ref.index.duplicated(keep="last")].sort_index()
        feat = ref["close"].pct_change(horizon)
        return feat.reindex(idx, method="ffill").fillna(0.0).values
    except Exception:
        return pd.Series(0.0, index=df.index)


def add_dsl_features(df, engine, specs):
    """Apply a list of specs as columns. Invalid or failing specs are skipped and
    returned in the skipped list. lead_lag uses the engine; the rest use materialize."""
    cols = set(df.columns)
    skipped = []
    for spec in specs or []:
        if not validate_spec(spec, cols):
            skipped.append(spec.get("name", "?") if isinstance(spec, dict) else "?")
            continue
        try:
            if spec["op"] == "lead_lag":
                df[spec["name"]] = _lead_lag_col(df, engine, spec)
            else:
                df[spec["name"]] = materialize(df, spec)
        except Exception:
            skipped.append(spec["name"])
    return df, skipped


def load_dsl_specs():
    """Read the agent's proposed specs from GTRADE_DSL_SPECS (a JSON file path).
    Returns an empty list when unset or unreadable, so training is a no-op."""
    import json
    import os
    path = os.getenv("GTRADE_DSL_SPECS")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []
