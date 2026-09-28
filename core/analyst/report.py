"""The analyst's day in one markdown file: what the scout picked and why, and
each verdict with its thesis. Written by `analyst.py auto`."""

import json
import os
import sqlite3

from core.analyst import store


def _cell(text):
    return " ".join(str(text or "").replace("|", "/").split())


def write_daily(date, picks, mode_label=None, path_dir="reports", db_path=None):
    """`date` names the file (the calendar day of the run). Each pick is shown
    at its OWN newest judgment: BTC's bar can be Sunday while NVDA's is Friday."""
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, "analyst_%s.md" % date)
    con = sqlite3.connect(db_path or store.DB_PATH)
    con.row_factory = sqlite3.Row
    rows = {}
    try:
        con.execute(store.DDL)
        store._migrate(con)
        for p in picks:
            sql, args = "SELECT * FROM analyst_log WHERE asset=?", [p["asset"]]
            if mode_label:
                sql += " AND mode=?"
                args.append(mode_label)
            r = con.execute(sql + " ORDER BY date DESC, horizon LIMIT 1", args).fetchone()
            if r is not None:
                rows[p["asset"]] = dict(r)
    finally:
        con.close()
    lines = ["# Analyst, %s" % date, "", "| Asset | Picked because | Verdict |",
             "|---|---|---|"]
    for p in picks:
        r = rows.get(p["asset"])
        verdict = ("%s, conviction %s (bar %s)" % (r["direction"], r["conviction"], r["date"])
                   if r else "no verdict")
        lines.append("| %s | %s (%s) | %s |" % (p["asset"], _cell(p["reason"]), p["by"], verdict))
    for p in picks:
        r = rows.get(p["asset"])
        if not r:
            continue
        lines += ["", "## %s: %s" % (p["asset"], r["direction"]), "",
                  r.get("thesis") or "", "", "Key risk: %s" % (r.get("key_risk") or "-")]
        if r.get("pre_critic_json"):
            lines.append("Before the critic: %s"
                         % json.loads(r["pre_critic_json"]).get("direction"))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
