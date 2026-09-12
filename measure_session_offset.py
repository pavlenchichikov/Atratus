"""The registered session-offset test, with the composition printed first.

Spec: docs/superpowers/specs/2026-09-12-session-offset-accuracy-preregistration.md
Class: _session_offset_class.json, frozen by rule before any score was read.

Composition comes first on purpose. The frozen class is whatever the rule
admits, and a verdict about "sessions opening shortly after the US close" reads
very differently when the class is mostly one exchange's small caps. That has to
sit beside the number, not in a footnote nobody reaches.
"""
import json
import os
import statistics as st

import config

BASE = os.path.dirname(os.path.abspath(__file__))
CLASS_PATH = os.path.join(BASE, "_session_offset_class.json")
QUALITY_PATH = os.path.join(BASE, "models", "quality_report.json")

FLOOR = 0.020          # registered: class mean must beat the rest by this much
ALPHA = 0.01           # registered: stricter, the hypothesis came from a look
MIN_IN, MIN_OUT = 25, 100


def group_of(asset):
    for group, members in config.ASSET_TYPES.items():
        if group != "TOP SIGNALS" and asset in members:
            return group
    return "OTHER"


def main():
    with open(CLASS_PATH, encoding="utf-8") as fh:
        blob = json.load(fh)
    with open(QUALITY_PATH, encoding="utf-8") as fh:
        quality = json.load(fh)
    rows = [r for r in quality
            if isinstance(r, dict) and isinstance(r.get("Ens_AUC"), (int, float))]
    auc = {r["Asset"]: float(r["Ens_AUC"]) for r in rows}
    acc = {r["Asset"]: float(r["CB_Acc"]) for r in rows
           if isinstance(r.get("CB_Acc"), (int, float))}

    inside = [a for a in blob["in_class"] if a in auc]
    outside = [a for a in blob["out_of_class"] if a in auc]

    print("rule: %s" % blob["rule"])
    print("frozen %s\n" % blob["frozen"])
    print("composition of the class (what the verdict will describe):")
    counts = {}
    for a in inside:
        counts[group_of(a)] = counts.get(group_of(a), 0) + 1
    for group, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print("  %-20s %4d  (%.0f%% of the class)" % (group, n, 100.0 * n / len(inside)))
    print()

    if len(inside) < MIN_IN or len(outside) < MIN_OUT:
        print("not measurable here: %d in class, %d out, floors are %d and %d"
              % (len(inside), len(outside), MIN_IN, MIN_OUT))
        return 0

    a_in = [auc[a] for a in inside]
    a_out = [auc[a] for a in outside]
    delta = st.mean(a_in) - st.mean(a_out)
    from scipy.stats import mannwhitneyu
    p = float(mannwhitneyu(a_in, a_out, alternative="two-sided").pvalue)

    print("Ens_AUC  in class %.4f (n=%d)   out %.4f (n=%d)   delta %+.4f"
          % (st.mean(a_in), len(a_in), st.mean(a_out), len(a_out), delta))
    print("median   in class %.4f            out %.4f"
          % (st.median(a_in), st.median(a_out)))
    if acc:
        print("CB_Acc   in class %.4f            out %.4f"
              % (st.mean(acc[a] for a in inside if a in acc),
                 st.mean(acc[a] for a in outside if a in acc)))
    print("Mann-Whitney p = %.4g   floor %+.3f   alpha %.2f" % (p, FLOOR, ALPHA))
    verdict = "CLEARS" if (delta >= FLOOR and p < ALPHA) else "does NOT clear"
    print("\nverdict: %s the registered bar" % verdict)
    if delta > 0 and delta < FLOOR:
        print("  positive but under the floor: recorded, and the scope of the book"
              " is NOT changed on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
