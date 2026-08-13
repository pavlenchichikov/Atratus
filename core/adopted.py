"""The adopted genome: what production trains and serves with, and why.

Read by config.py, which every entry point imports, so one adoption reaches the
trainer, the predictor, the alert bot and the web UI alike.

Standard library only, on purpose: predict.py must never pull the research
machinery in just to learn its own configuration.

Nothing here raises. A corrupt file reads as no adoption, because a bad edit must
not take down live predictions.
"""

import json
import os

PATH = os.getenv("GTRADE_ADOPTED_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "adopted_genome.json")

# Defaults mirror the Genome dataclass in auto_research. A gene sitting on its
# default is omitted from the overrides, so an unadopted run sets nothing.
_DEFAULTS = {
    "drops": [], "extra": [], "label_mode": "direction", "label_window": 30,
    "cb_depth_delta": 0, "cb_lr_mult": 1.0, "cb_iter_mult": 1.0,
    "lookback_delta": 0, "net_seeds": 1, "net_uniqueness": 0,
    "cb_uniqueness": 0,
    "net_calibrate": 0, "thr_margin": 0.0, "band_delta": 0.0,
    "regime_mode": "both",
}


def load(path=None):
    """The adopted record, or None when there is nothing usable to adopt."""
    try:
        with open(path or PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("genome"), dict):
        return None
    return data


def specs(record):
    """The adopted DSL specs, or [] when there are none."""
    if not isinstance(record, dict):
        return []
    extra = (record.get("genome") or {}).get("extra")
    if not isinstance(extra, list):
        return []
    return [s for s in extra if isinstance(s, dict)]


def _g(genome, key):
    if not isinstance(genome, dict):
        return _DEFAULTS[key]
    return genome.get(key, _DEFAULTS[key])


def env_overrides(genome):
    """The genome as GTRADE_* training and serving overrides.

    The single definition of these rules: auto_research.genome_to_env delegates
    here, so what production applies cannot drift from what an A/B measured.

    GTRADE_DSL_SPECS is deliberately NOT emitted. Research points that at a
    per-candidate temp file; production reads the adopted specs directly, and a
    temp path would die on the next reboot.
    """
    env = {}
    extra_names = [s["name"] for s in (_g(genome, "extra") or [])
                   if isinstance(s, dict) and s.get("name")]
    if extra_names:
        env["GTRADE_EXTRA_FEATURES"] = ",".join(extra_names)
    drops = _g(genome, "drops")
    if drops:
        env["GTRADE_DROP_FEATURES"] = ",".join(drops)
    label_mode = _g(genome, "label_mode")
    if label_mode == "triple_barrier":
        env["GTRADE_LABEL_MODE"] = "triple_barrier"
        env["GTRADE_LABEL_HORIZON"] = str(_g(genome, "label_window"))
    elif label_mode != "direction":
        env["GTRADE_LABEL_MODE"] = label_mode
        env["GTRADE_LABEL_WINDOW"] = str(_g(genome, "label_window"))
    if _g(genome, "cb_depth_delta"):
        env["GTRADE_CB_DEPTH_DELTA"] = str(_g(genome, "cb_depth_delta"))
    if _g(genome, "cb_lr_mult") != 1.0:
        env["GTRADE_CB_LR_MULT"] = str(_g(genome, "cb_lr_mult"))
    if _g(genome, "cb_iter_mult") != 1.0:
        env["GTRADE_CB_ITER_MULT"] = str(_g(genome, "cb_iter_mult"))
    if _g(genome, "lookback_delta"):
        env["GTRADE_LOOKBACK_DELTA"] = str(_g(genome, "lookback_delta"))
    if _g(genome, "net_seeds") > 1:
        env["GTRADE_NET_SEEDS"] = str(_g(genome, "net_seeds"))
    if _g(genome, "net_uniqueness"):
        env["GTRADE_NET_UNIQUENESS"] = "1"
    if _g(genome, "cb_uniqueness"):
        env["GTRADE_CB_UNIQUENESS"] = "1"
    if _g(genome, "net_calibrate"):
        env["GTRADE_NET_CALIBRATE"] = "1"
    if _g(genome, "thr_margin"):
        env["GTRADE_THR_MARGIN"] = str(_g(genome, "thr_margin"))
    if _g(genome, "band_delta"):
        env["GTRADE_BAND_DELTA"] = str(_g(genome, "band_delta"))
    if _g(genome, "regime_mode") != "both":
        env["GTRADE_REGIME_MODE"] = _g(genome, "regime_mode")
    return env


def apply(genome, environ=None):
    """Set the overrides that are not already set. Returns the keys it set.

    An existing value always wins, so a one-off experiment exported in the shell
    beats the adopted default without needing to revert anything.
    """
    target = os.environ if environ is None else environ
    done = []
    for key, value in env_overrides(genome).items():
        if key not in target:
            target[key] = value
            done.append(key)
    return done
