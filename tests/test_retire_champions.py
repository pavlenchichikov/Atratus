"""Quarantining champions that predate the feature chain that fed them."""
import os
import time

import retire_stale_champions as R


def _champion(d, table, age_days=0.0):
    for suffix in R.CHAMPION_FILES:
        p = d / (table + suffix)
        p.write_text("x", encoding="utf-8")
        if age_days:
            t = time.time() - age_days * 86400
            os.utime(p, (t, t))


def test_a_champion_older_than_the_chain_is_stale(tmp_path):
    _champion(tmp_path, "aapl", age_days=5)
    _champion(tmp_path, "msft", age_days=0)
    stale = R.stale_assets(str(tmp_path), since=time.time() - 86400,
                           universe=["AAPL", "MSFT"])
    assert [a for a, _ in stale] == ["AAPL"]


def test_the_cutoff_is_what_selects_it(tmp_path):
    """Positive control: move the cutoff back and the same champion is fine, so
    the test above is reading the mtime and not the filename."""
    _champion(tmp_path, "aapl", age_days=5)
    assert R.stale_assets(str(tmp_path), since=time.time() - 30 * 86400,
                          universe=["AAPL"]) == []


def test_an_asset_with_no_champion_is_not_reported(tmp_path):
    assert R.stale_assets(str(tmp_path), since=time.time(),
                          universe=["AAPL"]) == []


def test_retiring_moves_every_member_and_deletes_nothing(tmp_path):
    _champion(tmp_path, "aapl")
    dest = tmp_path / "_retired"
    moved, where = R.retire(["AAPL"], str(tmp_path), str(dest))
    assert moved == len(R.CHAMPION_FILES)
    assert not (tmp_path / "aapl_cb.cbm").exists()
    # Nothing is deleted: restoring one is a move back.
    assert (dest / "aapl_cb.cbm").exists()
    assert sorted(os.listdir(where)) == sorted("aapl" + s for s in R.CHAMPION_FILES)


def test_retiring_skips_members_that_are_absent(tmp_path):
    (tmp_path / "aapl_cb.cbm").write_text("x", encoding="utf-8")
    moved, _ = R.retire(["AAPL"], str(tmp_path), str(tmp_path / "_r"))
    assert moved == 1
