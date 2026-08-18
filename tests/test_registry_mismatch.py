"""Finding assets whose champion files and registry entry disagree.

The two used to be written at different times: model files per asset as each was
promoted, the registry once at the end of the run. Any interruption in between
left an asset whose .cbm expects one feature count while the entry the serving
path builds its pool from names another, and CatBoost refuses with "Feature N is
present in model but not in pool" - the asset silently vanishes from the signals.
Six assets were in that state on 2026-08-18 and the only way to find them was to
notice the error scroll past.
"""

import json
import os
import time

import model_health


def _repo(tmp_path, registry, files):
    models = tmp_path / "models"
    models.mkdir()
    (models / "champion_registry.json").write_text(json.dumps(registry),
                                                   encoding="utf-8")
    for asset, when in files.items():
        p = models / ("%s_cb.cbm" % asset.lower())
        p.write_text("x", encoding="utf-8")
        os.utime(p, (when, when))
    return str(tmp_path)


def _t(iso):
    return time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))


def test_a_model_written_after_its_entry_is_damaged(tmp_path):
    base = _repo(tmp_path,
                 {"QCOM": {"updated_at": "2026-07-28T22:24:17"}},
                 {"QCOM": _t("2026-08-18T14:59:15")})
    rows = model_health.mismatched_registry(base)
    assert [r["asset"] for r in rows] == ["QCOM"]


def test_an_entry_written_with_its_files_is_healthy(tmp_path):
    base = _repo(tmp_path,
                 {"NVDA": {"updated_at": "2026-08-18T15:00:00"}},
                 {"NVDA": _t("2026-08-18T14:59:00")})
    assert model_health.mismatched_registry(base) == []


def test_the_newest_damage_is_listed_first(tmp_path):
    base = _repo(tmp_path,
                 {"OLD": {"updated_at": "2026-07-01T00:00:00"},
                  "NEW": {"updated_at": "2026-07-01T00:00:00"}},
                 {"OLD": _t("2026-08-01T00:00:00"),
                  "NEW": _t("2026-08-18T00:00:00")})
    assert [r["asset"] for r in model_health.mismatched_registry(base)] == ["NEW", "OLD"]


def test_an_asset_with_no_model_file_is_not_damage(tmp_path):
    """It was never trained here. Nothing disagrees with anything."""
    base = _repo(tmp_path, {"GONE": {"updated_at": "2026-07-01T00:00:00"}}, {})
    assert model_health.mismatched_registry(base) == []


def test_an_unreadable_registry_reports_nothing_rather_than_raising(tmp_path):
    """This runs from a menu screen; a corrupt file must not take it down."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "champion_registry.json").write_text("{not json", encoding="utf-8")
    assert model_health.mismatched_registry(str(tmp_path)) == []
