"""A team of specialists behind the analyst's verdict.

Each specialist reads only its slice of the dossier, may ask only for its own
tools, and returns a LEAN, not a verdict: {"lean", "strength", "findings",
"evidence"}. The lead then judges with the whole dossier plus the reports,
through agent.judge, so parsing, tools, critic and storage are the solo path's.
Every lean is logged (analyst_team_log) so each role's hit rate is measured on
its own: a role that carries nothing shows up as a coin flip beside the others.
"""

import json

from core.analyst import tools
from core.analyst.agent import _first_json_object, plain

SPECIALISTS = ("macro", "fundamental", "technical", "news")
LEANS = ("up", "down", "flat")
COMMON = ("asset", "date", "close", "atr", "ret_1", "ret_5", "ret_20")
ROLE_FIELDS = {
    "macro": ("policy_rate", "policy_rate_bank", "policy_rate_direction",
              "policy_rate_prev", "policy_rate_days_since_change", "vix_level",
              "vix_chg_20", "benchmark", "benchmark_ret_1", "benchmark_ret_20",
              "breadth_above_sma50_pct", "breadth_positive_20d_pct",
              "cross_asset_corr", "cross_asset_corr_label", "macro_events",
              "beta", "beta_1y", "corr_to_benchmark_60", "regime_trend",
              "regime_vol", "regime_momentum"),
    "fundamental": ("pe", "pb", "ps", "ev_ebitda", "ebitda_margin", "debt_ebitda",
                    "roe", "roa", "nim", "div_yield", "div_yield_pref",
                    "ex_dividend_date", "market_cap", "float_shares", "short_ratio",
                    "next_earnings", "fundamentals_asof", "industry", "sector"),
    "technical": ("atr_pct", "atr_vs_90d", "atr_to_high_20", "atr_to_low_20",
                  "high_20", "low_20", "drawdown_60", "vol_20", "vol_20_vs_60",
                  "vol_1y", "streak_days", "rsi_14", "range_atr", "gap_open",
                  "volume_vs_20", "turnover", "off_52w_high", "max_dd_1y",
                  "ret_60", "ret_1y", "ret_ytd", "sector_momentum", "sector_trend",
                  "regime_trend", "regime_vol", "regime_momentum"),
    "news": ("headlines", "news_publishers", "next_earnings", "sector", "industry"),
}
ROLE_TOOLS = {
    "macro": ("macro_series", "compare"),
    "fundamental": ("company_financials", "insider_filings"),
    "technical": ("price_history", "sector_peers"),
    "news": ("news_search", "attention"),
}
ROLE_BRIEF = {
    "macro": "rates, volatility, the benchmark and the market around this asset",
    "fundamental": "the company's valuation, profitability, balance sheet and calendar",
    "technical": "price action, trend, volatility and volume of this asset itself",
    "news": "what is being reported about this asset, across publishers",
}


def slice_dossier(dossier, role):
    keep = COMMON + ROLE_FIELDS[role]
    return {k: dossier.get(k) for k in keep if k in dossier}


def _filled(v):
    return v not in (None, [], "", {}, 0)


def has_material(dossier, role, today=None):
    """False when the role has nothing to read and no tool that applies."""
    if any(_filled(dossier.get(k)) for k in ROLE_FIELDS[role]):
        return True
    return bool(tools.available(today, dossier.get("asset"), only=ROLE_TOOLS[role]))


def parse_report(text, allowed):
    """A validated report, or None. Unknown evidence names are dropped; a report
    left with no evidence is rejected, as a judgment is."""
    data = _first_json_object(text or "")
    if not isinstance(data, dict) or data.get("lean") not in LEANS:
        return None
    if data.get("strength") not in (1, 2, 3, 4, 5):
        return None
    evidence = [e for e in (data.get("evidence") or [])
                if isinstance(e, str) and e in allowed]
    if not evidence:
        return None
    findings = [plain(str(f))[:300] for f in (data.get("findings") or [])][:5]
    return {"lean": data["lean"], "strength": int(data["strength"]),
            "findings": findings, "evidence": evidence}


SPECIALIST_PROMPT = (
    "You are the %(role)s specialist on an analysis team, looking at %(brief)s. "
    "Other specialists cover everything else; stay in your lane. Below is what "
    "you have. Say which way it leans over %(span)s.\n\n%(facts)s\n\n"
    "Return STRICT JSON: {\"lean\": \"up|down|flat\", \"strength\": 1-5, "
    "\"findings\": [\"up to five short facts that drive the lean\"], "
    "\"evidence\": [\"names of the fields above you used\"]}\n"
    "No price targets, no percentages, nobody else's opinion.%(menu)s\n")

MAX_TRIES = 2


def consult(role, dossier, call, today=None, tool_calls=None, horizon=1, on_reject=None):
    """One specialist's report, or None after MAX_TRIES unusable answers.
    TerminalCallError / ProviderUnavailable propagate: the run must stop."""
    from core.analyst.agent import _span
    from core.llm_proposer import ProviderUnavailable, TerminalCallError

    facts = slice_dossier(dossier, role)
    own = ROLE_TOOLS[role]
    menu = (tools.spec_lines(today, dossier.get("asset"), only=own)
            if tool_calls is not None and tools.max_calls() else "")
    prompt = SPECIALIST_PROMPT % {
        "role": role, "brief": ROLE_BRIEF[role],
        "span": _span(horizon, dossier.get("asset")),
        "facts": json.dumps(facts, indent=2, ensure_ascii=True), "menu": menu}
    extra, rounds, tries = "", (tools.max_rounds() if menu else 0), 0
    allowed = set(facts)
    while tries < MAX_TRIES:
        try:
            answer = call(prompt + extra)
        except (TerminalCallError, ProviderUnavailable):
            raise
        except Exception as exc:
            tries += 1
            if on_reject is not None:
                on_reject("%s: call failed: %s" % (role, exc))
            continue
        if rounds > 0:
            reqs = [r for r in tools.parse_requests(_first_json_object(answer or ""))
                    if r["tool"] in own][:tools.max_calls()]
            if reqs:
                rounds -= 1
                for req in reqs:
                    entry = tools.call(req, asset=dossier.get("asset"), today=today)
                    entry["role"] = role
                    tool_calls.append(entry)
                    # A finding from a tool result has no dossier field name, so
                    # the tool's own name counts as evidence.
                    allowed.add(req["tool"])
                    extra += "\n\nYou asked for %s and received:\n%s\n" % (
                        req["tool"], json.dumps(entry.get("result", entry.get("error")),
                                                ensure_ascii=True)[:1800])
                extra += "Now return the report JSON.\n"
                continue
        report = parse_report(answer, allowed)
        if report is not None:
            return report
        tries += 1
        if on_reject is not None:
            on_reject("%s: unusable report" % role)
    return None


def _reports_text(reports):
    lines = ["", "", "Your specialists reported (a skipped role had nothing usable):"]
    for r in reports:
        if r.get("skipped"):
            lines.append("  %s: skipped (%s)" % (r["role"], r["skipped"]))
        else:
            lines.append("  %s: %s" % (r["role"], json.dumps(
                {k: r[k] for k in ("lean", "strength", "findings")}, ensure_ascii=True)))
    lines.append("Weigh them yourself; a majority is not a verdict.")
    return "\n".join(lines) + "\n"


def run_team(dossier, lead_call, call_for, horizon=1, depth="full", today=None,
             on_reject=None, tool_calls=None, notes=None):
    """(the lead's judgment or None, one report per specialist)."""
    from core.analyst import agent, brains

    reports = []
    for role in SPECIALISTS:
        base = {"role": role, "brain": brains.label(role)}
        if not has_material(dossier, role, today):
            reports.append({**base, "skipped": "nothing to read"})
            continue
        rep = consult(role, dossier, call_for(role), today=today,
                      tool_calls=tool_calls, horizon=horizon, on_reject=on_reject)
        reports.append({**base, **rep} if rep else {**base, "skipped": "no usable report"})
    j = agent.judge(dossier, call=lead_call, depth=depth, horizon=horizon,
                    on_reject=on_reject, today=today, tool_calls=tool_calls,
                    notes=notes, reports_text=_reports_text(reports))
    return j, reports
