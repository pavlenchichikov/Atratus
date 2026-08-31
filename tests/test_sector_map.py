"""config.SECTOR_MAP and RADAR_GROUPS: what the maps are allowed to leave out.

Both are derived from ASSET_TYPES so that adding an asset flows everywhere, and
both had stopped naming the categories that were added after them. Nothing
noticed, because a missing sector is not an error anywhere: the rotation heatmap
just leaves the asset out, portfolio counts it under OTHER, and radar_category
falls back to "us".
"""

import correlation_alert as ca
from config import ASSET_TYPES, FULL_ASSET_MAP, RADAR_GROUPS, SECTOR_MAP, radar_category
from portfolio import SECTOR_LIMITS


def test_every_asset_in_the_map_belongs_to_a_sector():
    """523 of 847 belonged to none: invisible to the rotation heatmap, and
    pooled into portfolio's single OTHER ceiling for exposure."""
    covered = {a for members in SECTOR_MAP.values() for a in members}
    orphans = sorted(a for a in FULL_ASSET_MAP if a not in covered)
    assert orphans == [], "%d assets have no sector: %s" % (
        len(orphans), ", ".join(orphans[:15]))


def test_every_asset_type_category_reaches_a_sector():
    """The failure mode is adding a category to ASSET_TYPES and forgetting the
    map, which is exactly how twelve of them went missing. Checked against the
    categories rather than the assets so a new EMPTY category is caught too."""
    covered = {a for members in SECTOR_MAP.values() for a in members}
    for name, members in ASSET_TYPES.items():
        if name == "TOP SIGNALS":       # a display shortlist, not a category
            continue
        missing = [a for a in members if a not in covered]
        assert not missing, "%s: %d assets reach no sector (%s)" % (
            name, len(missing), ", ".join(missing[:6]))


def test_every_sector_has_an_exposure_ceiling_or_the_documented_default():
    """A sector with no entry silently takes check_sector_limit's 0.50. That is
    fine as a default and wrong as a surprise, so the ones without one are named
    here rather than discovered in a position."""
    without = sorted(k for k in SECTOR_MAP if k not in SECTOR_LIMITS)
    assert without == ["US Consumer", "US Finance", "US Health", "US Industrial"], (
        "a sector's ceiling changed: %s" % without)
    assert not [k for k in SECTOR_LIMITS if k != "OTHER" and k not in SECTOR_MAP], (
        "a limit names a sector that does not exist")


def test_second_tier_moscow_names_are_russian():
    """radar_category falls back to "us" for anything unlisted, so 130 MOEX
    names read as American: _is_moex went false, can_have_earnings went TRUE,
    and the earnings scan bought one Yahoo 404 per name asking about a bare
    Moscow ticker. AAPL is the control."""
    from core.events import can_have_earnings

    for asset in ("ABIO", "AFKS", "AKRN", "TRNFP", "SNGSP", "SELG", "SBER"):
        assert radar_category(asset) == "ru", asset
        assert not can_have_earnings(FULL_ASSET_MAP.get(asset), asset), asset
    assert radar_category("AAPL") == "us"
    assert can_have_earnings("AAPL", "AAPL")


def test_no_asset_is_claimed_by_two_sectors():
    """A double membership would double-count the asset in the rotation average
    and against two exposure ceilings at once."""
    seen = {}
    for sector, members in SECTOR_MAP.items():
        for a in members:
            assert a not in seen, "%s is in both %s and %s" % (a, seen[a], sector)
            seen[a] = sector


def test_the_correlation_matrix_is_small_and_real():
    """It is read as a picture, so it stays short; but every name has to exist,
    and a name that is not in the asset map silently drops a row."""
    assert 15 <= len(ca.KEY_ASSETS) <= 30, len(ca.KEY_ASSETS)
    assert len(set(ca.KEY_ASSETS)) == len(ca.KEY_ASSETS), "a duplicate row"
    missing = [a for a in ca.KEY_ASSETS if a not in FULL_ASSET_MAP]
    assert missing == [], missing
    # rates and credit were both absent, which are the two that move first
    assert {"TNX", "TLT", "HYG"} <= set(ca.KEY_ASSETS)


def test_the_radar_groups_and_the_sector_map_agree_about_russia():
    """Two lists of the same thing drift. The Russian sector and the MOEX radar
    group are the pair that already did."""
    assert set(SECTOR_MAP["Russia"]) == set(RADAR_GROUPS["MOEX"])
