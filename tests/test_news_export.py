"""Unit tests for core.news_export (pure; no network, no Supabase)."""

from core import news_export


def item(title, link="http://x/1", published="Fri, 24 Jul 2026 10:00:00 GMT",
         source="Reuters", category="markets", weighted=0.5, label="POSITIVE"):
    return {"title": title, "link": link, "published": published,
            "source": source, "description": "a long description body",
            "category": category, "credibility": 1.0, "sentiment_score": 0.4,
            "weighted_score": weighted, "sentiment_label": label}


def h(asset, date, signal, prob):
    return {"asset": asset, "date": date, "signal": signal, "prob": prob}


def test_id_is_stable_and_scoped_by_asset():
    a = news_export.news_id(None, "http://x/1", "T")
    b = news_export.news_id(None, "http://x/1", "T")
    c = news_export.news_id("SBER", "http://x/1", "T")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_id_falls_back_to_title_when_link_missing():
    assert news_export.news_id(None, None, "T") == news_export.news_id(None, "", "T")


def test_general_rows_shape_and_no_description():
    rows = news_export.general_rows([item("Oil up")], "2026-07-26")
    assert len(rows) == 1
    row = rows[0]
    assert row["asset"] is None
    assert row["date"] == "2026-07-26"
    assert row["title"] == "Oil up"
    assert row["sentiment"] == 0.5
    assert row["label"] == "POSITIVE"
    assert row["category"] == "markets"
    assert "description" not in row


def test_asset_rows_carry_the_asset_and_null_category():
    items = [item("SBER beats", category=None)]
    rows = news_export.asset_rows("SBER", items, "2026-07-26")
    assert rows[0]["asset"] == "SBER"
    assert rows[0]["category"] is None


def test_rows_are_capped():
    items = [item(f"t{i}", link=f"http://x/{i}") for i in range(50)]
    assert len(news_export.general_rows(items, "2026-07-26")) == 40
    assert len(news_export.asset_rows("BTC", items, "2026-07-26")) == 6


def test_duplicate_ids_are_dropped_within_a_batch():
    # A PK collision inside one POST chunk would fail the whole chunk.
    items = [item("same", link="http://x/1"), item("same", link="http://x/1")]
    assert len(news_export.general_rows(items, "2026-07-26")) == 1


def test_untitled_items_are_skipped():
    assert news_export.general_rows([item("   ")], "2026-07-26") == []


def test_pick_assets_ranks_by_confidence_and_filters_to_actionable():
    rows = [h("A", "2026-07-26", "BUY", 0.55), h("B", "2026-07-26", "SELL", 0.9),
            h("C", "2026-07-26", "WAIT", 0.99), h("D", "2026-07-25", "BUY", 0.95)]
    assert news_export.pick_assets(rows, "2026-07-26") == ["B", "A"]


def test_pick_assets_dedupes_and_respects_the_cap():
    rows = [h(f"A{i}", "2026-07-26", "BUY", 0.9) for i in range(40)]
    rows += [h("A0", "2026-07-26", "BUY", 0.9)]
    picked = news_export.pick_assets(rows, "2026-07-26")
    assert len(picked) == 30
    assert len(set(picked)) == 30


def test_pick_assets_skips_rows_without_a_probability():
    rows = [h("A", "2026-07-26", "BUY", None)]
    assert news_export.pick_assets(rows, "2026-07-26") == []
