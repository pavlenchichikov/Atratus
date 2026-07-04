"""Compounding research wiki for the auto_research agent (Karpathy 'LLM Wiki' pattern).

The findings journal (ar_memory) is an append-only LOG; this distills it into a small set
of interlinked markdown topic pages the LLM proposer can read, so accumulated learning
compounds across runs instead of evaporating through a 5-record window. Env-gated
(GTRADE_AR_WIKI, default OFF). The LLM backend is imported lazily; this module imports
ar_memory but never auto_research (no cycle)."""

import json
import os

from core import ar_memory

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIKI_DIR = os.path.join(BASE, "_ar_wiki")
PAGES = ["labeling", "features", "regime", "neural", "calibration", "general", "changes"]


def wiki_on():
    """GTRADE_AR_WIKI truthy (default OFF)."""
    return (os.getenv("GTRADE_AR_WIKI") or "").strip() in ("1", "true", "True")


def _page_path(page):
    return os.path.join(WIKI_DIR, "%s.md" % page)


def _state_path():
    return os.path.join(WIKI_DIR, "_state.json")


def _read(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def _write(path, text):
    os.makedirs(WIKI_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def wiki_summary(max_chars=6000):
    """The distilled wiki text for the proposer prompt (concatenated pages, truncated).
    Reads the files fresh; '' when absent/empty. Never raises."""
    parts = []
    for page in PAGES:
        t = _read(_page_path(page)).strip()
        if t:
            parts.append("## %s\n%s" % (page, t))
    return "\n\n".join(parts)[:max_chars]


def note_replicated(sig, detail):
    """Append a high-confidence entry to general.md when a finding clears the replication
    gate (compounding of CONFIRMED knowledge). Never raises."""
    try:
        line = "- (high) REPLICATED %s: %s\n" % (str(sig)[:80], str(detail)[:200])
        _write(_page_path("general"), _read(_page_path("general")) + line)
    except Exception:
        pass


def _backend():
    """The LLM proposer backend (lazy import so there is no import cycle)."""
    from core.llm_proposer import _backend as b
    return b()


def _load_state():
    try:
        with open(_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"cursor": 0}


def _save_state(state):
    os.makedirs(WIKI_DIR, exist_ok=True)
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(state, f)


def _apply_sections(text):
    """Write each recognized '## <page>' section from the LLM reply to its page file."""
    import re
    for block in re.split(r"(?m)^##\s+", text):
        block = block.strip()
        if not block:
            continue
        head, _, body = block.partition("\n")
        page = head.strip().lower()
        if page in PAGES and body.strip():
            _write(_page_path(page), body.strip() + "\n")


def compile_wiki():
    """Fold findings NEW since the last compile into the wiki (LLM rewrites the affected
    '## <page>' sections with confidence tags, contradiction reconciliation, pruning).
    Returns the number of new findings folded in; 0 and the wiki unchanged when off, when
    there are no new findings, or on any error (never raises)."""
    if not wiki_on():
        return 0
    try:
        journal = ar_memory._load(ar_memory.FINDINGS_PATH, [])
        state = _load_state()
        cur = int(state.get("cursor", 0))
        new = journal[cur:]
        if not new:
            return 0
        prompt = (
            "You maintain a compounding research wiki for a trading-model search. Fold the "
            "NEW findings into the CURRENT wiki: update claims, tag each (stated|high|low), "
            "reconcile contradictions to current truth, prune stale/duplicate claims. Use "
            "ONLY these page names as '## <page>' sections: " + ", ".join(PAGES) +
            ".\nCURRENT WIKI:\n" + (wiki_summary(max_chars=8000) or "(empty)") +
            "\nNEW FINDINGS (JSON):\n" + json.dumps(new, ensure_ascii=True)[:6000] +
            "\nReturn the FULL updated wiki as '## <page>' sections, no prose.")
        out = (_backend()(prompt) or "").strip()
        if not out:
            return 0
        _apply_sections(out)
        state["cursor"] = len(journal)
        _save_state(state)
        return len(new)
    except Exception:
        return 0


def lint_wiki():
    """Maintenance pass: the LLM reconciles contradictions and prunes stale/duplicate
    claims across pages, no new experiments. Never raises."""
    if not wiki_on():
        return
    try:
        current = wiki_summary(max_chars=8000)
        if not current:
            return
        prompt = (
            "Reconcile contradictions and prune stale/duplicate claims in this research "
            "wiki; keep the (stated|high|low) confidence tags. Return the FULL wiki as "
            "'## <page>' sections using ONLY these page names: " + ", ".join(PAGES) +
            ".\n" + current)
        out = (_backend()(prompt) or "").strip()
        if out:
            _apply_sections(out)
    except Exception:
        pass
