"""Persistent cross-run memory for the auto-research agent.

Three small JSON files next to the other _auto_research state:
- _ar_tried.json: the permanent registry of every evaluated candidate
  signature (nothing is ever re-tested across runs);
- _ar_eval_cache.json: a cache for expensive trainings, keyed by data +
  feature-space version. Base runs key off the env (base_key); re-gate
  candidate runs key off the genome signature (genome_key), because their
  envs embed temp spec-file paths and would never repeat verbatim;
- _ar_findings.json: the cumulative findings journal (one record per run).

An unreadable file is treated as empty (same tolerance as load_state)."""

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIED_PATH = os.path.join(BASE, "_ar_tried.json")
CACHE_PATH = os.path.join(BASE, "_ar_eval_cache.json")
FINDINGS_PATH = os.path.join(BASE, "_ar_findings.json")
DB_PATH = os.path.join(BASE, "market.db")
# Chunked held-out training (GTRADE_AR_TRAIN_CHUNK) turns one arm into several
# entries: a 14-asset holdout at chunk 5 is 3 chunks, doubled by the CB-only
# train, so a 4-arm A/B alone banks ~24. At the old cap of 120 a long unattended
# run could evict a chunk it had already paid hours for and retrain it.
CACHE_CAP = 400
REPLICATION_PATH = os.path.join(BASE, "_ar_replication.json")
_RL_BLOB_DIR = BASE


def _load(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def blob_get(name, default=None):
    """Generic named JSON blob (used by core.ar_rl for scheduler state)."""
    return _load(os.path.join(_RL_BLOB_DIR, f"{name}.json"), default)


def blob_put(name, obj):
    """Store a named JSON blob."""
    _save(os.path.join(_RL_BLOB_DIR, f"{name}.json"), obj)


# Mirrors auto_research._score_basis's accepted set. Kept here rather than
# imported because auto_research imports THIS module; tests/test_ar_memory.py
# asserts the two lists still agree, so the copy cannot drift silently.
SCORE_BASES = ("raw", "neural", "net_auc", "net_gain", "ens_auc", "trade_t")


def tried_scope():
    """The registry namespace for the active basis (raw keeps the bare kind).

    "Tried" means "we already spent compute learning this candidate's value" -
    and a value only exists inside one basis. The 1624 genome signatures banked
    before 2026-08-16 were all scored by a CatBoost-only screen with the neural
    members stubbed to a constant, so under net_auc they are UNMEASURED, not
    tried. Sharing one namespace made the search refuse to revisit any of them:
    measured 2026-08-16, ~52 of 100 steps died on dedup and the two emitters
    that could have filled the empty niches (nets, novelty) booked zero children
    all run, leaving 6 niches illuminated and the refine phase never reached.
    """
    b = (os.getenv("GTRADE_AR_SCORE_BASIS") or "raw").strip().lower()
    if b not in SCORE_BASES:
        b = "raw"
    return b


def selection_scope():
    """The registry namespace for a non-default SEARCH SET, '' for the default one.

    Same argument as tried_scope: the value a candidate was measured at belongs
    to the set it was measured on. A genome scored over five assets is not the
    same evidence as one scored over ten, so it must not lock the full set out
    of ever revisiting it. The default returns '' so every signature banked
    before this existed keeps its exact bucket.
    """
    v = (os.getenv("GTRADE_AR_SELECTION") or "full").strip().lower()
    return "" if v == "full" else v


def _scoped(kind):
    scope = tried_scope()
    if scope != "raw":
        kind = "%s@%s" % (kind, scope)
    sel = selection_scope()
    return "%s#%s" % (kind, sel) if sel else kind


def tried_seen(kind, sig):
    """Whether this candidate signature was ever evaluated (any past run) ON THE
    ACTIVE BASIS - see tried_scope."""
    return sig in _load(TRIED_PATH, {}).get(_scoped(kind), [])


def tried_add(kind, sig):
    reg = _load(TRIED_PATH, {})
    bucket = reg.setdefault(_scoped(kind), [])
    if sig not in bucket:
        bucket.append(sig)
        _save(TRIED_PATH, reg)


def tried_count():
    return sum(len(v) for v in _load(TRIED_PATH, {}).values())


def tried_recent(kind, n=20):
    """The last n evaluated signatures for a kind (as stored, oldest-first). Fed to
    the LLM proposer as an 'avoid these' list so it stops re-proposing tried candidates."""
    return _load(TRIED_PATH, {}).get(_scoped(kind), [])[-n:]


def replication_seen(sig):
    """Whether this candidate signature cleared the held-out gate in any PRIOR run."""
    return bool(_load(REPLICATION_PATH, {}).get(sig))


def replication_add(sig, ts):
    """Record a held-out-gate clear for sig at ISO time ts; return the number of
    distinct runs (timestamps) that have now cleared it."""
    reg = _load(REPLICATION_PATH, {})
    stamps = reg.setdefault(sig, [])
    if ts not in stamps:
        stamps.append(ts)
        _save(REPLICATION_PATH, reg)
    return len(stamps)


def replicated_sigs():
    """Signatures that have cleared the held-out gate in two or more runs."""
    reg = _load(REPLICATION_PATH, {})
    return {sig for sig, stamps in reg.items() if len(stamps or []) >= 2}


def findings_append(record):
    journal = _load(FINDINGS_PATH, [])
    journal.append(record)
    _save(FINDINGS_PATH, journal)


def findings_summary():
    """Cumulative counters for the end-of-run print."""
    journal = _load(FINDINGS_PATH, [])
    adoptable = sum(1 for rec in journal
                    for w in rec.get("winners", []) if w.get("adoptable"))
    replicated = sum(1 for rec in journal
                     for w in rec.get("winners", []) if w.get("replicated"))
    return {"experiments": tried_count(), "adoptable": adoptable,
            "replicated": replicated}


def findings_recent(n=20):
    """The last n findings-journal records, newest first (empty on unreadable file)."""
    return list(reversed(_load(FINDINGS_PATH, [])))[:n]


def findings_all():
    """The full findings journal (oldest first)."""
    return _load(FINDINGS_PATH, [])


def _objective_suffix():
    """Cache-key marker for the selection objective. Present ONLY under
    GTRADE_OBJECTIVE_V2 so every pre-existing cache entry keeps its key;
    v1 and v2 Score generations can never hit each other's entries."""
    on = (os.getenv("GTRADE_OBJECTIVE_V2") or "").strip() in ("1", "true", "True")
    return ["objective-v2"] if on else []


def _seed_suffix():
    """Cache-key marker for the training seed. Unconditional, unlike
    _objective_suffix: every entry cached before training became deterministic
    came from an UNSEEDED run, so it must not be served to a seeded one - a
    stale base row against a fresh variant row is a delta made of the seed
    change. It also gives each GTRADE_SEED re-roll its own cache namespace, so
    re-rolling to measure seed noise does not read the previous roll's rows."""
    from core.net_hygiene import seed_base
    return ["seed-%d" % seed_base()]


def data_fingerprint(subset):
    """Newest bar date per asset table of the subset; changes when new data
    arrives. A missing table is a deterministic marker; a whole-DB failure
    returns a unique value (cache MISS, never a wrong hit)."""
    from core.track_record import _table_name
    try:
        con = sqlite3.connect(DB_PATH)
        try:
            parts = []
            for a in subset.split(","):
                t = _table_name(a.strip())
                try:
                    row = con.execute(f'SELECT MAX(Date) FROM "{t}"').fetchone()
                    parts.append(f"{t}={row[0]}")
                except sqlite3.Error:
                    parts.append(f"{t}=?")
            return "|".join(parts)
        finally:
            con.close()
    except Exception:
        return "err-" + uuid.uuid4().hex


def base_key(subset, env):
    """Cache key for a BASE training: same subset + env + feature space +
    data snapshot means the same quality rows."""
    from core.features import feature_version
    payload = json.dumps(
        [subset, env, feature_version(), data_fingerprint(subset)]
        + _objective_suffix() + _seed_suffix(),
        sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def genome_key(subset, gsig, kind=""):
    """Cache key for a CANDIDATE training: the genome signature stands in for the
    env dict (candidate envs embed temp spec-file paths, so the raw env cannot key
    the cache). kind separates the full train from the CB-only screen train. Same
    invalidation rules as base_key: feature space or new data means a MISS."""
    from core.features import feature_version
    payload = json.dumps(
        [subset, gsig, kind, feature_version(), data_fingerprint(subset)]
        + _objective_suffix() + _seed_suffix(),
        sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def cache_get(key):
    entry = _load(CACHE_PATH, {}).get(key)
    return entry["rows"] if entry else None


def cache_put(key, rows):
    cache = _load(CACHE_PATH, {})
    cache[key] = {"rows": rows, "ts": datetime.utcnow().isoformat()}
    if len(cache) > CACHE_CAP:
        oldest = sorted(cache, key=lambda k: cache[k].get("ts", ""))
        for k in oldest[:len(cache) - CACHE_CAP]:
            del cache[k]
    _save(CACHE_PATH, cache)
