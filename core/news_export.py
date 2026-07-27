"""Supabase row builders for the mobile news layer.

Pure functions over what news_analyzer already returns, so the export is
testable without a network. push_signals owns the HTTP; this module owns the
shapes and the bounds.
"""

import hashlib

GENERAL_LIMIT = 40   # general digest items exported per run
ASSET_CAP = 30       # assets that get per-asset news (one RSS request each)
PER_ASSET = 6        # articles per asset


def news_id(asset, link, title):
    """Deterministic row id, so a re-run upserts instead of duplicating.

    Scoped by asset: the same article can legitimately appear in the general
    feed and against an asset, and those are two rows.
    """
    key = "%s|%s" % (asset or "", link or title or "")
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _row(item, asset, date):
    """One news row, or None when the item carries no usable title.

    `description` is deliberately dropped: it is the largest field and the
    phone shows title, source and sentiment.
    """
    title = (item.get("title") or "").strip()
    if not title:
        return None
    link = item.get("link") or None
    return {
        "id": news_id(asset, link, title),
        "asset": asset,
        "date": date,
        "published": item.get("published") or None,
        "title": title,
        "link": link,
        "source": item.get("source") or None,
        "category": item.get("category") or None,
        "sentiment": item.get("weighted_score"),
        "label": item.get("sentiment_label") or None,
    }


def _rows(items, asset, date, limit):
    out, seen = [], set()
    for item in items:
        if len(out) >= limit:
            break
        row = _row(item, asset, date)
        if row is None or row["id"] in seen:
            continue
        seen.add(row["id"])
        out.append(row)
    return out


def general_rows(items, date, limit=GENERAL_LIMIT):
    """General-feed rows. `asset` is None so the client can split on it."""
    return _rows(items, None, date, limit)


def asset_rows(asset, items, date, limit=PER_ASSET):
    """Per-asset news rows."""
    return _rows(items, asset, date, limit)


def pick_assets(hist_rows, today, cap=ASSET_CAP):
    """Assets worth a per-asset news fetch: today's actionable calls, most
    confident first.

    Bounded on purpose - one RSS request per asset, and 208 of them would
    invite rate limiting and stretch the run.
    """
    ranked = []
    for row in hist_rows:
        if row.get("date") != today:
            continue
        if (row.get("signal") or "").upper() not in ("BUY", "SELL"):
            continue
        prob = row.get("prob")
        if prob is None:
            continue
        ranked.append((abs(prob - 0.5), row["asset"]))
    # Asset name breaks ties so the selection is deterministic run to run.
    ranked.sort(key=lambda pair: (-pair[0], pair[1]))
    out = []
    for _conf, asset in ranked:
        if asset not in out:
            out.append(asset)
        if len(out) >= cap:
            break
    return out
