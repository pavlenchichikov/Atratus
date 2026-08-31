"""Stage C tick: refit the timing Q on the newest data, with an anchor.

Run:  python train_timing_online.py [--assets A,B] [--iters 6]

There is no transition store to maintain. build_asset_series reconstructs every
asset's scorable history from market.db and the champions, so a tick that runs
it picks up the new bars by construction; a second copy of the transitions
would cost hundreds of megabytes and its own consistency rules against the
database it was copied from.

Nothing here is served. The tick maintains a SHADOW challenger; crossing into
production still means the Stage-B gate in train_timing.py.
"""
import argparse
import json
import random
from datetime import datetime

import train_timing as tt
from core import timing_fqi as fq
from core import timing_online as on
from core import timing_policy as tp


def _sides_of(policy, series):
    """The positions a policy holds over one series, for the trust region."""
    if hasattr(policy, "apply_series"):
        sides, _a, _r = policy.apply_series(series)
        return sides
    sides, _a, _r = policy.apply(
        series["probs"], series["buy_thr"], series["sell_thr"],
        series["atr"], series["taleb_hi"], series["risky"],
        next_ret=series["next_ret"])
    return sides


def _behaviour_for(assets, anchor, champion, self_share, rng):
    """Which policy generates each asset's transitions: {asset: policy}.

    The anchor and the policy that COLLECTS the data are two different jobs the
    anchor used to do at once. The trust region stays the rules permanently,
    which is the invariant; the data does not have to.

    self_share is the fraction of assets rolled out under the CURRENT champion
    Q instead. Mixed rather than swapped, and by ASSET rather than by bar, for
    the failure this guards against: a Q that is wrong collects data only where
    it is wrong and locks that in. Keeping most of the buffer on the anchor
    leaves the fit something to correct against. share=0 or no champion yet
    means every asset rolls out under the anchor, which is what happened before
    this existed.
    """
    names = sorted(assets)
    if champion is None or self_share <= 0.0:
        return {a: anchor for a in names}
    k = round(self_share * len(names))
    chosen = set(rng.sample(names, min(k, len(names))))
    return {a: (champion if a in chosen else anchor) for a in names}


def tick(by_asset, state, iters=6, gamma=0.97, epsilon=0.1, seed=0,
         anchor=None, ts=None, champion=None, self_share=0.0):
    """One scheduled update. Returns (new_state, report).

    A halted stack returns before the fit, not after it: the expensive part is
    the CatBoost ladder, and a halted schedule that still pays for it has not
    stopped anything.

    `champion` is the last ACCEPTED generation's Q and `self_share` how much of
    the transition buffer it collects. At the default 0.0 the tick is exactly
    what it was: every transition comes from the anchor plus epsilon noise, so
    the Q learns to correct the RULES' mistakes and the ten percent of flipped
    actions, and never sees the states its own policy would reach.
    """
    ts = ts or datetime.now().isoformat(timespec="seconds")
    if state.get("halted"):
        verdict, reason = on.decide(state, 0.0, 0.0)
        return (on.apply_decision(state, verdict, reason, 0.0, 0.0, ts),
                {"verdict": verdict, "reason": reason, "agreement": 0.0,
                 "score": 0.0, "selected_on_val": None, "assets": 0})

    anchor = anchor or tp.load_policy() or tp.RulesPolicy(dict(tp.DEFAULT_PARAMS))
    splits = {a: tt.split_series(s) for a, s in by_asset.items()}
    train = {a: v[0] for a, v in splits.items()}
    val = {a: v[1] for a, v in splits.items()}
    recent = {a: v[2] for a, v in splits.items()}

    rng = random.Random(seed)
    behaviour = _behaviour_for(train, anchor, champion, self_share,
                               random.Random(seed))
    batches = [fq.rollout(train[a], behaviour[a], rng, epsilon=epsilon,
                          costs=tt._costs(train[a]))
               for a in sorted(train)]
    n_self = sum(1 for a in train if behaviour[a] is not anchor)
    models = fq.fit_q(batches, iters=iters, gamma=gamma, seed=seed)
    candidates = [("q_iter_%d" % (k + 1), fq.FqiPolicy(m))
                  for k, m in enumerate(models)]

    val_scores = [(name, tt.fitness([tt.eval_policy(s, pol)["score"]
                                     for s in val.values()]))
                  for name, pol in candidates]
    best_name = max(val_scores, key=lambda kv: kv[1])[0]
    chosen = dict(candidates)[best_name]

    agree = on.agreement(recent, chosen, anchor, _sides_of)
    score = tt.fitness([tt.eval_policy(s, chosen)["score"]
                        for s in recent.values()])
    verdict, reason = on.decide(state, agree, score)
    new_state = on.apply_decision(state, verdict, reason, agree, score, ts)
    if verdict == "ACCEPT":
        # Only an ACCEPTED generation is kept. A rejected one collecting the
        # next tick's data would be the lock-in this guards against, handed the
        # keys.
        on.save_champion(chosen.model)
    return new_state, {"verdict": verdict, "reason": reason,
                       "agreement": agree, "score": score,
                       "selected_on_val": best_name, "assets": len(by_asset),
                       "self_rollouts": n_self}


def main():
    ap = argparse.ArgumentParser(description="Stage C: one online tick")
    ap.add_argument("--assets", default="")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-share", dest="self_share", type=float, default=0.0,
                    help="fraction of assets whose transitions are collected "
                         "under the CURRENT accepted Q instead of the anchor "
                         "rules (default 0.0, the behaviour before this "
                         "existed). The trust region stays the rules whatever "
                         "this is set to; only the data moves")
    args = ap.parse_args()

    import config
    assets = ([a.strip() for a in args.assets.split(",") if a.strip()]
              or [a for grp in config.ASSET_TYPES.values() for a in grp])
    series = {}
    for i, a in enumerate(assets, 1):
        s = tt.build_asset_series(a)
        if s is not None:
            series[a] = s
        print("[online] [%d/%d] %s: %s" % (
            i, len(assets), a, "%d bars" % len(s["probs"]) if s else "skipped"),
            flush=True)
    state = on.load_state()
    champion = on.load_champion() if args.self_share > 0 else None
    if args.self_share > 0 and champion is None:
        print("[online] --self-share %.2f but no accepted generation is stored "
              "yet; this tick collects entirely under the anchor."
              % args.self_share, flush=True)
    state, report = tick(series, state, iters=args.iters, gamma=args.gamma,
                         epsilon=args.epsilon, seed=args.seed,
                         champion=champion, self_share=args.self_share)
    on.save_state(state)
    print("[online] generation %d | agreement %.3f | score %.3f | on val %s "
          "| self-collected %d/%d assets"
          % (state["generation"], report["agreement"], report["score"],
             report["selected_on_val"], report.get("self_rollouts", 0),
             report["assets"]))
    print("[online] %s: %s" % (report["verdict"], report["reason"]))
    print("[online] " + json.dumps(
        {k: report[k] for k in ("verdict", "agreement", "score", "assets")}))


if __name__ == "__main__":
    main()
