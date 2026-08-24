"""Read-only joins over the research journals and the serving tables.

The agent's evidence lives in six files with six shapes and no relation
between them, so no question can be asked across them: which lever ever
produced anything, why a genome was not adopted, what was tried that resembles
this one. This module answers those by joining what is already on disk.

Nothing here writes. Every value is recomputed from its source on each call,
which is the point: a stored derivative drifts. The per-genome `clears` count
was stored, drifted to 1451 while a stuck loop re-counted the same clear, and
`ab_build.auto_picks` ranks A/B candidates by it.
"""

import dataclasses
import functools
import glob
import json
import os
import sqlite3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "market.db")


def _read_json(path, default):
    """The file's contents, or `default` when it is missing or unreadable.

    Same rule as ar_memory._load: these files are written by a long-running
    agent while a page is being read, so a half-written file must degrade to
    an empty section rather than fail the request.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _path(name, base=None):
    return os.path.join(base or BASE, name)


def _findings(base=None):
    data = _read_json(_path("_ar_findings.json", base), [])
    return data if isinstance(data, list) else []


def journalled_sigs(base=None):
    """{genome signature: [timestamps of the records that flagged it]}.

    Signature comes from auto_research so it matches every other consumer of
    the same genome; imported lazily because this module is loaded by the web
    app, which must not pay for the research stack on startup.
    """
    import auto_research as ar

    out = {}
    for rec in _findings(base):
        ts = rec.get("ts") or ""
        for w in rec.get("winners") or []:
            gd = w.get("genome")
            if not isinstance(gd, dict):
                continue
            try:
                sig = ar.genome_sig(ar.Genome(**gd))
            except (TypeError, ValueError):
                continue
            out.setdefault(sig, []).append(ts)
    return out


def funnel(base=None):
    """The five counts from tried to adopted.

    `tried` is deduplicated across buckets: the registry is namespaced
    `kind@basis`, so one genome measured under two bases appears twice and
    counting rows would double it.
    """
    tried = _read_json(_path("_ar_tried.json", base), {})
    seen = set()
    if isinstance(tried, dict):
        for v in tried.values():
            if isinstance(v, list):
                seen.update(v)
    repl = _read_json(_path("_ar_replication.json", base), {})
    if not isinstance(repl, dict):
        repl = {}
    adopted = _read_json(_path("adopted_genome.json", base), None)
    return {
        "tried": len(seen),
        "journalled": len(journalled_sigs(base)),
        "cleared_once": len(repl),
        "cleared_twice": sum(1 for s in repl.values()
                             if isinstance(s, list) and len(s) >= 2),
        "adopted": 1 if isinstance(adopted, dict) and adopted.get("genome") else 0,
    }


@functools.lru_cache(maxsize=1)
def _gene_defaults():
    """auto_research.Genome's own field defaults, so a gene sitting at its
    default value is not counted as a lever. Read from the dataclass rather
    than retyped by hand: a hand-typed copy drifts from the real default and
    starts inventing levers nobody chose, or hiding ones they did. Memoised
    because `ar` is a lazy import and the fields never change within a
    process.
    """
    import auto_research as ar

    return {f.name: f.default for f in dataclasses.fields(ar.Genome)
            if f.default is not dataclasses.MISSING}


# Explicit labels for the eleven tuning genes, in place of a generic
# key.replace("_", "") rendering.
_LEVER_LABELS = {
    "thr_margin": "thr", "regime_mode": "regimemode",
    "lookback_delta": "lookbackdelta", "cb_depth_delta": "cbdepthdelta",
    "cb_lr_mult": "cblrmult", "cb_iter_mult": "cbitermult",
    "cb_uniqueness": "cbuniqueness", "net_seeds": "netseeds",
    "net_uniqueness": "netuniqueness", "net_calibrate": "netcalibrate",
    "band_delta": "banddelta",
}


def levers_of(genome):
    """The genes one genome actually chose, as stable string keys.

    A genome is a bundle, and every summary in the project aggregates one
    level coarser than that: `ar_director.axis_yield` counts by AXIS, so a
    result belonging to one dropped feature is credited to the whole axis.
    These keys are what let two genomes be compared at all.
    """
    if not isinstance(genome, dict):
        return []
    out = ["drop:%s" % d for d in genome.get("drops") or []]
    out += ["op:%s" % e for e in genome.get("extra") or []]
    defaults = _gene_defaults()
    mode = genome.get("label_mode")
    window = genome.get("label_window")
    if mode is not None and (mode, window) != (defaults.get("label_mode"),
                                                defaults.get("label_window")):
        out.append("label:%s/%s" % (mode, window))
    for key, label in _LEVER_LABELS.items():
        val = genome.get(key)
        default = defaults.get(key)
        if val is not None and val != default:
            out.append("%s:%s" % (label, val))
    return out


def _median(values):
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def levers(base=None):
    """Per-lever yield over the whole journal, busiest lever first."""
    import auto_research as ar

    repl = _read_json(_path("_ar_replication.json", base), {})
    replicated = {s for s, stamps in (repl or {}).items()
                  if isinstance(stamps, list) and len(stamps) >= 2}
    adopted = _read_json(_path("adopted_genome.json", base), None)
    adopted_sig = None
    if isinstance(adopted, dict) and isinstance(adopted.get("genome"), dict):
        try:
            adopted_sig = ar.genome_sig(ar.Genome(**adopted["genome"]))
        except (TypeError, ValueError):
            adopted_sig = None

    seen_genome = {}          # lever -> set of signatures carrying it
    flagged = {}              # lever -> set of signatures ever flagged
    lifts = {}                # lever -> list of neural lifts
    for rec in _findings(base):
        for w in rec.get("winners") or []:
            gd = w.get("genome")
            if not isinstance(gd, dict):
                continue
            try:
                sig = ar.genome_sig(ar.Genome(**gd))
            except (TypeError, ValueError):
                continue
            for lever in levers_of(gd):
                seen_genome.setdefault(lever, set()).add(sig)
                if w.get("adoptable"):
                    flagged.setdefault(lever, set()).add(sig)
                if w.get("neural_lift") is not None:
                    lifts.setdefault(lever, []).append(w["neural_lift"])

    rows = []
    for lever, sigs in seen_genome.items():
        rows.append({
            "lever": lever,
            "genomes": len(sigs),
            "flagged": len(flagged.get(lever, ())),
            "replicated": len(sigs & replicated),
            "adopted": 1 if adopted_sig in sigs else 0,
            "neural_lift": _median(lifts.get(lever, [])),
        })
    rows.sort(key=lambda r: (-r["genomes"], r["lever"]))
    return rows


def verdicts(base=None):
    """{signature: [A/B results]}, newest file last.

    The file name is kept on each row because it is the only thing that dates
    a verdict: the payload carries no timestamp.
    """
    out = {}
    pattern = os.path.join(base or BASE, "_ab_genomes_*.json")
    for path in sorted(glob.glob(pattern)):
        data = _read_json(path, {})
        if not isinstance(data, dict):
            continue
        results = data.get("results")
        if not isinstance(results, dict):
            continue
        for label, res in results.items():
            if not isinstance(res, dict) or not res.get("sig"):
                continue
            row = dict(res)
            row["label"] = label
            row["file"] = os.path.basename(path)
            row["reference"] = data.get("reference")
            row["holdout"] = data.get("holdout")
            out.setdefault(res["sig"], []).append(row)
    return out


def _genomes_by_sig(base=None):
    """{signature: genome dict}, first sighting wins."""
    import auto_research as ar

    out = {}
    for rec in _findings(base):
        for w in rec.get("winners") or []:
            gd = w.get("genome")
            if not isinstance(gd, dict):
                continue
            try:
                sig = ar.genome_sig(ar.Genome(**gd))
            except (TypeError, ValueError):
                continue
            out.setdefault(sig, gd)
    return out


def genomes(base=None):
    """Every genome that ever produced a journalled finding, sig-sorted.

    The list a page needs to let a person reach a first genome at all: every
    other entry point (neighbours, verdicts) only produces a link once one
    genome is already selected.
    """
    by_sig = _genomes_by_sig(base)
    return [{"sig": sig, "levers": levers_of(gd)}
            for sig, gd in sorted(by_sig.items())]


def genome(sig, base=None):
    """Everything known about one genome, in one dict.

    An unknown signature returns the empty shape rather than raising: the
    page renders it as "nothing recorded", which is a true answer.
    """
    import auto_research as ar

    gd = _genomes_by_sig(base).get(sig)
    findings = []
    for rec in _findings(base):
        for w in rec.get("winners") or []:
            wd = w.get("genome")
            if not isinstance(wd, dict):
                continue
            try:
                if ar.genome_sig(ar.Genome(**wd)) != sig:
                    continue
            except (TypeError, ValueError):
                continue
            findings.append({
                "ts": rec.get("ts", ""), "mode": rec.get("mode", ""),
                "basis": rec.get("basis"), "axis": w.get("axis", ""),
                "p": w.get("p"), "value": w.get("value"), "tag": w.get("tag", ""),
                "adoptable": bool(w.get("adoptable")),
                "neural_lift": w.get("neural_lift"),
            })
    repl = _read_json(_path("_ar_replication.json", base), {})
    stamps = (repl or {}).get(sig) or []
    return {
        "sig": sig,
        "genome": gd,
        "levers": levers_of(gd) if gd else [],
        "findings": findings,
        "verdicts": verdicts(base).get(sig, []),
        "clears": len(stamps) if isinstance(stamps, list) else 0,
    }


def similar(sig, k=10, base=None):
    """Neighbours by shared levers, closest first.

    Overlap is Jaccard over the lever sets. Two genomes that share a dropped
    feature and a DSL operation are the same idea measured twice, and that is
    what the search kept rediscovering.
    """
    by_sig = _genomes_by_sig(base)
    mine = set(levers_of(by_sig.get(sig)))
    if not mine:
        return []
    out = []
    for other, gd in by_sig.items():
        if other == sig:
            continue
        theirs = set(levers_of(gd))
        shared = mine & theirs
        if not shared:
            continue
        union = mine | theirs
        out.append({"sig": other, "shared": sorted(shared),
                    "overlap": len(shared) / len(union)})
    out.sort(key=lambda r: (-r["overlap"], r["sig"]))
    return out[:k]


def unresolved(base=None):
    """Signatures the join could not place, by source.

    A clear or an A/B verdict whose genome never appears in the journal is a
    fact about the join, so it is named rather than dropped. Most of them are
    composed-with-reference variants, the case ab_build._gate_by_sig exists to
    unfold, and silently discarding them would hide exactly that.
    """
    known = set(journalled_sigs(base))
    repl = _read_json(_path("_ar_replication.json", base), {})
    return {
        "replication": sorted(s for s in (repl or {}) if s not in known),
        "verdicts": sorted(s for s in verdicts(base) if s not in known),
    }


_GENERATIONS_SQL = """
    SELECT model_version,
           COUNT(*)                       AS n,
           SUM(correct IS NOT NULL)       AS reconciled,
           AVG(CASE WHEN correct IS NOT NULL THEN correct END) AS accuracy,
           MIN(date)                      AS first,
           MAX(date)                      AS last
      FROM prediction_log
     WHERE model_version IS NOT NULL
  GROUP BY model_version
  ORDER BY MAX(date) DESC
"""


def generations(db_path=None):
    """One row per model generation, newest first.

    `model_version` is `core.features.feature_version()`, a hash of the active
    feature list, so an adopted genome that changes the feature space gets its
    own generation. That is the only join the project has between a research
    result and what production then did.

    Accuracy is over reconciled rows only. Averaging NULLs as zeros would read
    an unreconciled prediction as a wrong one.
    """
    try:
        con = sqlite3.connect("file:%s?mode=ro" % (db_path or DB_PATH), uri=True)
    except sqlite3.Error:
        return []
    try:
        rows = con.execute(_GENERATIONS_SQL).fetchall()
    except sqlite3.Error:
        return []
    finally:
        con.close()
    return [{"model_version": r[0], "n": r[1], "reconciled": r[2],
             "accuracy": r[3], "first": r[4], "last": r[5]} for r in rows]
