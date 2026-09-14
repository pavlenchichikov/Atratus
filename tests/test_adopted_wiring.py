"""The adoption reaches the processes that need it."""

import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GENOME = {
    "label": "T",
    "genome": {
        "drops": ["rsi"],
        "extra": [{"name": "zscore_vol_z_20", "op": "zscore",
                   "inputs": ["vol_z"], "params": {"window": 20}}],
        "label_mode": "rel_median", "label_window": 30, "thr_margin": 0.02,
    },
}


def _env_without_genome_vars(adopted_path):
    env = dict(os.environ)
    env["GTRADE_ADOPTED_PATH"] = adopted_path
    for k in ("GTRADE_DROP_FEATURES", "GTRADE_LABEL_MODE", "GTRADE_LABEL_WINDOW",
              "GTRADE_THR_MARGIN", "GTRADE_EXTRA_FEATURES", "GTRADE_DSL_SPECS",
              # GTRADE_FEATURE_SET=base would give 24 features and fail the pin
              # for a reason that has nothing to do with adoption.
              "GTRADE_FEATURE_SET"):
        env.pop(k, None)
    return env


def _last_line(code, env):
    out = subprocess.run([sys.executable, "-c", code], cwd=BASE, env=env,
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-800:]
    return out.stdout.strip().splitlines()[-1]


def _run(code, adopted_path):
    return _last_line(code, _env_without_genome_vars(adopted_path))


def _write(tmp_path, payload=None):
    p = tmp_path / "a.json"
    open(str(p), "w", encoding="utf-8").write(json.dumps(payload or GENOME))
    return str(p)


def test_importing_config_applies_the_adoption(tmp_path):
    line = _run("import config, os;"
                "print(os.getenv('GTRADE_LABEL_MODE'), os.getenv('GTRADE_THR_MARGIN'))",
                _write(tmp_path))
    assert line == "rel_median 0.02"


def test_a_shell_value_still_wins(tmp_path):
    env = _env_without_genome_vars(_write(tmp_path))
    env["GTRADE_THR_MARGIN"] = "0.09"
    line = _last_line("import config, os; print(os.getenv('GTRADE_THR_MARGIN'))",
                      env)
    assert line == "0.09"


def test_no_adoption_changes_nothing(tmp_path):
    # The regression that matters most: an unadopted checkout must behave exactly
    # as it does today. Both values were captured from the tree before any of this
    # work landed; feature_version hashes the active feature list, so either
    # number moving means an unadopted run changed.
    line = _run("import config, os;"
                "from core.features import active_candidate_features, feature_version;"
                "print(len(active_candidate_features()), feature_version())",
                str(tmp_path / "absent.json"))
    n, ver = line.split()
    # Re-captured 2026-09-14, from 34 / c48ee7bf, when the non-price inputs were
    # added to CANDIDATE_FEATURES_EXT: three breadth columns and two COT ones.
    # The baseline moved because the input space did, which is the one reason
    # this pin may be rewritten - never to make a failure go away.
    assert int(n) == 39, f"unadopted feature count changed from 39 to {n}"
    assert ver == "db1ee1dd", f"unadopted feature_version changed to {ver}"


def test_an_adoption_changes_the_feature_version(tmp_path):
    # The other side of the same coin: adopting MUST move feature_version, so the
    # live track record cannot blend two model generations.
    #
    # Compared against the UNADOPTED version computed here, not against a frozen
    # literal. It used to assert `!= "c48ee7bf"`, and on 2026-09-14 the baseline
    # moved away from that string on its own: the test would then have passed
    # even if adoption changed nothing at all, which is the failure mode it
    # exists to catch.
    code = ("import config;"
            "from core.features import feature_version; print(feature_version())")
    unadopted = _run(code, str(tmp_path / "absent.json"))
    adopted = _run(code, _write(tmp_path))
    assert adopted != unadopted, "adoption left the feature version untouched"


def test_load_dsl_specs_falls_back_to_the_adopted_specs(tmp_path):
    line = _run("import config;"
                "from core.feature_dsl import load_dsl_specs;"
                "print([s['name'] for s in load_dsl_specs()])",
                _write(tmp_path))
    assert line == "['zscore_vol_z_20']"


def test_the_research_temp_file_still_wins_over_the_adopted_specs(tmp_path):
    spec_file = tmp_path / "specs.json"
    open(str(spec_file), "w", encoding="utf-8").write(json.dumps(
        [{"name": "research_only", "op": "lag", "inputs": ["ret_5"],
          "params": {"k": 2}}]))
    env = _env_without_genome_vars(_write(tmp_path))
    env["GTRADE_DSL_SPECS"] = str(spec_file)
    line = _last_line("import config;"
                      "from core.feature_dsl import load_dsl_specs;"
                      "print([s['name'] for s in load_dsl_specs()])", env)
    assert line == "['research_only']"


def test_a_corrupt_adoption_does_not_stop_a_run(tmp_path):
    # A bad hand-edit must not take down predict.py.
    bad = tmp_path / "bad.json"
    open(str(bad), "w", encoding="utf-8").write("{not json")
    line = _run("import config, os;"
                "print('survived', os.getenv('GTRADE_LABEL_MODE'))", str(bad))
    assert line == "survived None"
