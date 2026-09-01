"""Read signals and their history from market.db for the web, bot and digest.

These interfaces compute nothing: predict.py writes to prediction_log
(performance_tracker); here we only run queries.
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "market.db")

ACC_LAST_N = 20  # how many of the most recent verified signals feed accuracy
# Below this many verified signals in the CURRENT model generation, the accuracy
# falls back to the lifetime record across all generations - otherwise every
# retrain (feature_version change) resets the panel to "no verified signals"
# even for assets with a long history.
_ACC_MIN_SCOPED = 5


def _table_name(asset: str) -> str:
    # same normalization as in predict.py
    return asset.lower().replace("^", "").replace(".", "").replace("-", "")


def _connect(db_path=None):
    return sqlite3.connect(db_path or DB_PATH)


def _plog_cols(con):
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(prediction_log)").fetchall()]
    except sqlite3.OperationalError:
        return []


def _has_model_version(con) -> bool:
    return "model_version" in _plog_cols(con)


def _hit_rate(con, asset: str, last_n: int, model_version=None):
    """(n, correct) over the most recent verified signals, optionally scoped to
    one model_version."""
    where = "asset=? AND correct IS NOT NULL"
    params = [asset]
    if model_version is not None:
        where += " AND model_version=?"
        params.append(model_version)
    rows = con.execute(
        f"SELECT correct FROM prediction_log WHERE {where} ORDER BY date DESC LIMIT ?",
        (*params, last_n),
    ).fetchall()
    return len(rows), sum(r[0] for r in rows)


def _accuracy(con, asset: str, last_n: int) -> dict:
    """Hit-rate over the last verified signals. Scoped to the current feature
    generation (so an old model's record never blends into the active model),
    but when that generation has too few verified signals - e.g. right after a
    retrain - it falls back to the lifetime record across all generations, so
    the panel is not misleadingly empty for an asset with real history. The
    `all_versions` flag says whether the returned figure is that lifetime
    fallback."""
    all_versions = False
    if _has_model_version(con):
        from core.features import feature_version
        n, correct = _hit_rate(con, asset, last_n, feature_version())
        if n < _ACC_MIN_SCOPED:
            ln, lc = _hit_rate(con, asset, last_n)  # lifetime, all generations
            if ln > n:
                n, correct, all_versions = ln, lc, True
    else:
        n, correct = _hit_rate(con, asset, last_n)
    return {"n": n, "correct": correct, "acc": (correct / n) if n else None,
            "all_versions": all_versions}


def asset_accuracy(asset: str, last_n: int = ACC_LAST_N, db_path=None) -> dict:
    with _connect(db_path) as con:
        try:
            return _accuracy(con, asset, last_n)
        except sqlite3.OperationalError:
            return {"n": 0, "correct": 0, "acc": None}


def latest_signals(db_path=None, acc_last_n: int = ACC_LAST_N) -> list:
    """Latest signal per asset plus accuracy over the most recent verified ones.

    `signal` is the DISPLAY value (the live-gated sig_shown when the gate
    suppressed the raw call); `signal_raw` keeps the model's own call and
    `gate_reason` says why they differ. Consumers that act on signals (webapp
    radar, push_signals) read `signal` and inherit the gate for free."""
    with _connect(db_path) as con:
        gated = "sig_shown" in _plog_cols(con)
        has_timing = "timing_action" in _plog_cols(con)
        extra = ", p.sig_shown, p.gate_reason" if gated else ", NULL, NULL"
        extra += ", p.timing_action, p.timing_reason" if has_timing else ", NULL, NULL"
        try:
            rows = con.execute(
                "SELECT p.asset, p.date, p.signal, p.probability" + extra + " "
                "FROM prediction_log p "
                "JOIN (SELECT asset, MAX(date) AS d FROM prediction_log GROUP BY asset) m "
                "ON p.asset = m.asset AND p.date = m.d "
                "ORDER BY p.asset"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for asset, date, signal, prob, shown, reason, t_act, t_rsn in rows:
            out.append({
                "asset": asset,
                "date": date,
                "signal": shown or signal,
                "signal_raw": signal,
                "gate_reason": reason,
                "probability": prob,
                "acc": _accuracy(con, asset, acc_last_n),
                "timing_action": t_act,
                "timing_reason": t_rsn,
            })
        return out


def latest_gated(asset: str, db_path=None) -> dict:
    """Gated display values of one asset's LATEST prediction row:
    {"signal": <sig_shown or raw>, "signal_raw": raw, "gate_reason": reason}.
    Empty dict when the asset has no rows or the table/columns are absent."""
    with _connect(db_path) as con:
        gated = "sig_shown" in _plog_cols(con)
        cols = _plog_cols(con)
        has_timing = "timing_action" in cols
        extra = ", sig_shown, gate_reason" if gated else ", NULL, NULL"
        extra += ", timing_action, timing_reason" if has_timing else ", NULL, NULL"
        # The watched challenger, under GTRADE_TIMING_STAGE=shadow. NULL when
        # the column has not been migrated in yet, which a database that has
        # only been READ since the upgrade will not have.
        extra += (", shadow_action" if "shadow_action" in cols
                  else ", NULL AS shadow_action")
        try:
            row = con.execute(
                "SELECT signal" + extra + " FROM prediction_log "
                "WHERE asset=? ORDER BY date DESC LIMIT 1",
                (asset,),
            ).fetchone()
        except sqlite3.OperationalError:
            return {}
        if not row:
            return {}
        signal, shown, reason, t_act, t_rsn, s_act = row
        return {"signal": shown or signal, "signal_raw": signal,
                "gate_reason": reason,
                "timing_action": t_act, "timing_reason": t_rsn,
                "shadow_action": s_act}


CONFIDENCE_BANDS = ((0.00, 0.05), (0.05, 0.10), (0.10, 0.15),
                    (0.15, 0.25), (0.25, 0.50))


def confidence_bands(db_path=None, source="probability", days=None):
    """Does a more confident signal land more often, and earn more?

    `source` names the column the confidence is read from, so a candidate
    definition can be measured the same way the current one is: `probability`
    is the served ensemble number, `meta_prob` the stacker's.

    Confidence is |p - 0.5|, which is what the card draws. Measured on the live
    log 2026-09-01 over 7929 verified directional predictions it carried
    nothing: spearman(confidence, correct) +0.013 at p=0.25, and against the
    PAYOFF +0.008 at p=0.50, with the most confident band of 2077 signals
    landing 0.483 against 0.495 for the band below it. A number on the card
    that orders nothing is decoration, so this is the panel that says so.

    Payoff, not just hit rate: being right and making money are different
    questions, and on this log the side answers the second one (BUY -0.146%,
    SELL +0.092%) while confidence answers neither.
    """
    if source == "agreement":
        return _agreement_bands(db_path=db_path, days=days)
    with _connect(db_path) as con:
        if source not in _plog_cols(con):
            return {"source": source, "n": 0, "bands": [], "rho": None,
                    "p": None, "hit": None, "payoff": None}
        where = "correct IS NOT NULL AND signal IN ('BUY','SELL')"
        args = []
        if days:
            where += " AND date >= date('now', ?)"
            args.append("-%d days" % int(days))
        try:
            rows = con.execute(
                "SELECT %s, correct, signal, actual_next_ret FROM prediction_log "
                "WHERE %s" % (source, where), args).fetchall()
        except sqlite3.OperationalError:
            return {"source": source, "n": 0, "bands": [], "rho": None,
                    "p": None, "hit": None, "payoff": None}

    data = []
    for p, correct, sig, ret in rows:
        if p is None or correct is None:
            continue
        pay = None if ret is None else (ret if sig == "BUY" else -ret)
        data.append((abs(float(p) - 0.5), int(correct), pay))
    if not data:
        return {"source": source, "n": 0, "bands": [], "rho": None,
                "p": None, "hit": None, "payoff": None}

    # Bucketed through _band_of, which puts the top edge INSIDE the last band.
    # The half-open loop this replaces dropped every row at exactly p=0 or p=1,
    # 682 of the live log's 7929, and dropped them silently: the panel showed a
    # smaller n and nobody could see which rows were missing.
    bands = []
    for lo, hi in CONFIDENCE_BANDS:
        sel = [(c, pay) for conf, c, pay in data if _band_of(0.5 + conf) == (lo, hi)]
        if not sel:
            continue
        pays = [pay for _c, pay in sel if pay is not None]
        bands.append({
            "lo": lo, "hi": hi, "n": len(sel),
            "hit": sum(c for c, _p in sel) / len(sel),
            "payoff": (sum(pays) / len(pays)) if pays else None,
        })

    rho = pval = None
    if len(data) >= 3:
        try:
            from scipy.stats import spearmanr

            r, pv = spearmanr([d[0] for d in data], [d[1] for d in data])
            rho, pval = round(float(r), 4), round(float(pv), 4)
        except Exception:
            pass
    allpay = [d[2] for d in data if d[2] is not None]
    return {"source": source, "n": len(data), "bands": bands,
            "rho": rho, "p": pval,
            "hit": sum(d[1] for d in data) / len(data),
            "payoff": (sum(allpay) / len(allpay)) if allpay else None,
            # An ordering claim needs the bands to disagree with each other by
            # more than their own noise; rho over the raw pairs is the honest
            # summary and is reported beside them rather than instead of them.
            "informative": ("unknown" if pval is None or pval >= 0.05
                            else ("yes" if rho > 0 else "INVERTED"))}


def _agreement_bands(db_path=None, days=None):
    """confidence_bands for the CANDIDATE definition: the members agreeing.

    Mapped onto the same 0 to 0.5 scale the probability bands use, so the two
    definitions are read off one table: confidence = 0.5 * (1 - spread), which
    is 0.5 when all four members say the same thing and 0 when they span the
    whole range.

    Returns n=0 until rows carry at least three members. Every row written
    before 2026-09-01 has two, so this measures nothing yet BY CONSTRUCTION,
    and reporting n rather than a rate is the point: the machinery is in place
    and the answer is not available, which is a different thing from a flat
    result.
    """
    with _connect(db_path) as con:
        cols = _plog_cols(con)
        if not {"tf_prob", "tcn_prob"} <= set(cols):
            return {"source": "agreement", "n": 0, "bands": [], "rho": None,
                    "p": None, "hit": None, "payoff": None,
                    "informative": "unknown"}
        where = "correct IS NOT NULL AND signal IN ('BUY','SELL')"
        args = []
        if days:
            where += " AND date >= date('now', ?)"
            args.append("-%d days" % int(days))
        rows = con.execute(
            "SELECT cb_prob, lstm_prob, tf_prob, tcn_prob, correct, signal, "
            "actual_next_ret FROM prediction_log WHERE " + where, args).fetchall()

    data = []
    for cb, lstm, tf, tcn, correct, sig, ret in rows:
        spread = member_spread({"cb_prob": cb, "lstm_prob": lstm,
                                "tf_prob": tf, "tcn_prob": tcn})
        if spread is None or correct is None:
            continue
        pay = None if ret is None else (ret if sig == "BUY" else -ret)
        data.append((0.5 * (1.0 - min(1.0, spread)), int(correct), pay))
    if not data:
        return {"source": "agreement", "n": 0, "bands": [], "rho": None,
                "p": None, "hit": None, "payoff": None,
                "informative": "unknown"}

    # Same bucketing as confidence_bands: perfect agreement maps to exactly
    # 0.50 and a half-open top band would throw those rows away, which is the
    # one case this measure most needs to see.
    bands = []
    for lo, hi in CONFIDENCE_BANDS:
        sel = [(c, pay) for conf, c, pay in data if _band_of(0.5 + conf) == (lo, hi)]
        if not sel:
            continue
        pays = [p for _c, p in sel if p is not None]
        bands.append({"lo": lo, "hi": hi, "n": len(sel),
                      "hit": sum(c for c, _p in sel) / len(sel),
                      "payoff": (sum(pays) / len(pays)) if pays else None})
    rho = pval = None
    if len(data) >= 3:
        try:
            from scipy.stats import spearmanr

            r, pv = spearmanr([d[0] for d in data], [d[1] for d in data])
            rho, pval = round(float(r), 4), round(float(pv), 4)
        except Exception:
            pass
    allpay = [d[2] for d in data if d[2] is not None]
    return {"source": "agreement", "n": len(data), "bands": bands,
            "rho": rho, "p": pval,
            "hit": sum(d[1] for d in data) / len(data),
            "payoff": (sum(allpay) / len(allpay)) if allpay else None,
            "informative": ("unknown" if pval is None or pval >= 0.05
                            else ("yes" if rho > 0 else "INVERTED"))}


def _band_of(prob):
    """Which CONFIDENCE_BANDS row a probability falls in, or None."""
    if prob is None:
        return None
    conf = abs(float(prob) - 0.5)
    for lo, hi in CONFIDENCE_BANDS:
        if lo <= conf < hi:
            return (lo, hi)
    return CONFIDENCE_BANDS[-1] if conf >= CONFIDENCE_BANDS[-1][1] else None


def confidence_evidence(asset, prob, db_path=None, k=50.0):
    """What THIS confidence has historically been worth, for this asset.

    The card printed |p - 0.5| as a confidence, and on the live log that number
    orders nothing (see confidence_bands). This replaces the claim with the
    record: of the past signals on this asset that landed in the same band, how
    many were right and what the position earned.

    Shrunk toward the same band's rate across every asset, with the same shape
    core/analyst/payoff.shrink and core/guru.py already use: an asset with nine
    observations must not print a confident number, and the honest fallback is
    what the band is worth in general rather than silence.

    Returns n as its own field on purpose. A rate without its sample size is
    the thing this function exists to stop.
    """
    band = _band_of(prob)
    if band is None:
        return {"band": None, "n": 0, "hit": None, "payoff": None,
                "hit_all": None, "n_all": 0}
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                "SELECT asset, probability, correct, signal, actual_next_ret "
                "FROM prediction_log WHERE correct IS NOT NULL "
                "AND signal IN ('BUY','SELL')").fetchall()
        except sqlite3.OperationalError:
            return {"band": band, "n": 0, "hit": None, "payoff": None,
                    "hit_all": None, "n_all": 0}

    mine, everyone = [], []
    for a, p, correct, sig, ret in rows:
        if p is None or _band_of(p) != band:
            continue
        pay = None if ret is None else (ret if sig == "BUY" else -ret)
        everyone.append((int(correct), pay))
        if a == asset:
            mine.append((int(correct), pay))
    if not everyone:
        return {"band": band, "n": 0, "hit": None, "payoff": None,
                "hit_all": None, "n_all": 0}

    hit_all = sum(c for c, _p in everyone) / len(everyone)
    hit = hit_all
    if mine:
        from core.analyst.payoff import shrink

        hit = shrink(len(mine), sum(c for c, _p in mine) / len(mine), hit_all, k)
    pays = [p for _c, p in (mine or everyone) if p is not None]
    return {"band": band, "n": len(mine), "hit": hit,
            "payoff": (sum(pays) / len(pays)) if pays else None,
            "hit_all": hit_all, "n_all": len(everyone)}


def member_spread(row):
    """How far the four ensemble members are from agreeing, 0 to 1.

    The candidate replacement for |p - 0.5|: four models agreeing is not the
    same claim as one model being sure, and until 2026-09-01 nothing could tell
    them apart because only two of the four were ever stored.

    None when fewer than three members are present, which is every row written
    before that date. Deliberately None and not zero: a missing member is not
    an agreeing one, and this number is about to be scored against outcomes.
    """
    vals = [row.get(k) for k in ("cb_prob", "lstm_prob", "tf_prob", "tcn_prob")]
    vals = [float(v) for v in vals if v is not None]
    if len(vals) < 3:
        return None
    return float(max(vals) - min(vals))


def asset_track(asset: str, limit: int = 30, db_path=None) -> list:
    """Signal history for an asset, newest first."""
    with _connect(db_path) as con:
        cols = _plog_cols(con)
        # Both timing columns, so a row can say what each policy decided that
        # day and be checked against what the bar then did. Guarded: a database
        # that has only been READ since the upgrade has not migrated them in.
        # tf_prob and tcn_prob join the same guard: core/scoring.py always
        # produced them, nothing stored them until 2026-09-01, and a database
        # that has only been READ since has not migrated them in.
        extra = "".join(
            (", " + name) if name in cols else (", NULL AS " + name)
            for name in ("timing_action", "shadow_action", "tf_prob", "tcn_prob"))
        try:
            rows = con.execute(
                "SELECT date, signal, probability, actual_next_ret, correct, "
                "cb_prob, lstm_prob" + extra +
                " FROM prediction_log WHERE asset=? ORDER BY date DESC LIMIT ?",
                (asset, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            {"date": d, "signal": s, "probability": p,
             "actual_next_ret": r, "correct": c,
             "cb_prob": cb, "lstm_prob": lstm,
             "tf_prob": tf, "tcn_prob": tcn,
             "timing_action": t_act, "shadow_action": s_act}
            for d, s, p, r, c, cb, lstm, t_act, s_act, tf, tcn in rows
        ]


def price_series(asset: str, days: int = 60, db_path=None, con=None) -> list:
    """An asset's last closes ascending by date: [{'date','close'}, ...].

    `con` lets a caller looping over many assets reuse one connection. That is
    not a micro-optimisation: the radar draws 822 sparklines, and measured on
    the live database the queries cost 0.15s while opening a connection per
    asset cost the other 11. Callers that pass nothing behave exactly as
    before, with one repair - the connection is now closed. sqlite3's context
    manager commits a transaction, it does not close, so the old `with` form
    leaked a connection per call until the collector caught up.
    """
    table = _table_name(asset)
    own = con is None
    if own:
        con = _connect(db_path)
    try:
        rows = con.execute(
            f'SELECT Date, Close FROM "{table}" ORDER BY Date DESC LIMIT ?',
            (days,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        if own:
            con.close()
    rows.reverse()
    return [{"date": str(d)[:10], "close": c} for d, c in rows if c is not None]


def price_series_many(assets, days: int = 60, db_path=None) -> dict:
    """{asset: closes} for many assets over ONE connection.

    Every per-asset price loop in the web layer goes through here. An asset
    with no table maps to an empty list, exactly as price_series returns one.
    """
    con = _connect(db_path)
    try:
        return {a: price_series(a, days, con=con) for a in assets}
    finally:
        con.close()


def volume_series(asset: str, days: int = 60, db_path=None) -> list:
    """Last `days` bars of traded volume and turnover, ascending by date.

    ohlc_series deliberately returns price only, because every caller of it so
    far wants price. Volume is a separate question and gets a separate reader
    rather than a wider row that four other callers would have to ignore.
    """
    table = _table_name(asset)
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                f'SELECT Date, Volume, Value FROM "{table}" '
                f'ORDER BY Date DESC LIMIT ?',
                (days,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    rows.reverse()
    return [{"date": str(d)[:10], "volume": v, "value": val}
            for d, v, val in rows if v is not None]


def ohlc_series(asset: str, days: int = 120, db_path=None) -> list:
    """Last `days` OHLC bars ascending by date: [{date,open,high,low,close}, ...]."""
    table = _table_name(asset)
    with _connect(db_path) as con:
        try:
            rows = con.execute(
                f'SELECT Date, Open, High, Low, Close FROM "{table}" '
                f'ORDER BY Date DESC LIMIT ?',
                (days,),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    rows.reverse()
    out = []
    for d, o, h, l, c in rows:
        if None in (o, h, l, c):
            continue
        out.append({"date": str(d)[:10], "open": o, "high": h, "low": l, "close": c})
    return out


def stale_assets(max_age_days: int = 7, assets=None, db_path=None, today=None) -> list:
    """Assets whose market.db data is older than the threshold (or missing)."""
    if assets is None:
        from config import FULL_ASSET_MAP
        assets = list(FULL_ASSET_MAP.keys())
    today_dt = datetime.strptime(today, "%Y-%m-%d") if today else datetime.now()

    out = []
    with _connect(db_path) as con:
        for asset in assets:
            table = _table_name(asset)
            try:
                row = con.execute(f'SELECT MAX(Date) FROM "{table}"').fetchone()
                last = row[0] if row else None
            except sqlite3.OperationalError:
                last = None
            if last is None:
                out.append({"asset": asset, "last_date": None, "age_days": None})
                continue
            try:
                last_dt = datetime.strptime(str(last)[:10], "%Y-%m-%d")
            except ValueError:
                continue
            age = (today_dt - last_dt).days
            if age > max_age_days:
                out.append({"asset": asset, "last_date": str(last)[:10], "age_days": age})
    return out
