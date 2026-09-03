"""Tests for guru_report.py's fundamentals-resolution helper."""

import guru_report as gr


def test_resolve_fundamentals_prefers_smartlab():
    smartlab = {"SBER": {"pe": 4.1, "roe": 0.23, "debt": 0.0, "div": 12.0}}
    fund = gr.resolve_fundamentals("SBER", "SBER", smartlab)
    assert fund["_source"] == "smartlab"
    assert fund["pe"] == 4.1
    assert fund["roe"] == 0.23


def test_resolve_fundamentals_yfinance_for_non_moex(monkeypatch):
    monkeypatch.setattr(gr, "fetch_yf_deep",
                         lambda symbol: {"_source": "yfinance_live", "pe": 20.0})
    fund = gr.resolve_fundamentals("TSLA", "TSLA", {})
    assert fund == {"_source": "yfinance_live", "pe": 20.0}


def test_resolve_fundamentals_backup_fallback_non_moex(monkeypatch):
    monkeypatch.setattr(gr, "fetch_yf_deep", lambda symbol: None)
    fund = gr.resolve_fundamentals("TSLA", "TSLA", {})
    assert fund is None  # TSLA has no GLOBAL_BACKUP entry


def test_resolve_fundamentals_moex_backup():
    fund = gr.resolve_fundamentals("SBER", "SBER", {})  # smartlab empty, MOEX backup branch
    assert fund["_source"] == "backup"
    assert fund["pe"] == 4.2  # GLOBAL_BACKUP['SBER']['pe']


def test_resolve_fundamentals_moex_no_backup_returns_none():
    fund = gr.resolve_fundamentals("IMOEX", "IMOEX", {})  # MOEX asset, no GLOBAL_BACKUP entry
    assert fund is None


def test_stock_assets_excludes_non_stock_and_dedups():
    assets = gr.stock_assets()
    s = set(assets)
    assert {"AAPL", "ASML", "SBER"} <= s          # US / EU / RU stocks present
    for a in ("BTC", "EURUSD", "GOLD", "SP500", "DAX"):
        assert a not in s                          # crypto/forex/commodity/index out
    assert assets.count("TSLA") == 1               # TOP SIGNALS + US TECH -> once
    assert len(assets) == len(set(assets))


def test_recalc_all_stocks_scrapes_once_skips_na_logs_real(monkeypatch):
    monkeypatch.setattr(gr, "stock_assets", lambda: ["AAPL", "SBER"])
    scrapes = {"n": 0}

    def _smart():
        scrapes["n"] += 1
        return {}

    monkeypatch.setattr(gr, "fetch_smartlab_data", _smart)
    monkeypatch.setattr(gr, "resolve_fundamentals",
                        lambda name, symbol, sl: {"_source": "x", "price": 10.0})
    monkeypatch.setattr(gr, "get_technical", lambda name: None)
    monkeypatch.setattr(gr, "technical_context", lambda t: None)

    seq = iter(["yfinance_live", "technical"])   # AAPL real, SBER N/A

    def _analysis(fund, tech):
        return {"data_source": next(seq),
                "council": {"pct": 70.0, "verdict": "BUY"},
                "lynch": {"_score": 1}, "buffett": {"_score": 1},
                "graham": {"_score": 1}, "munger": {"_score": 1}}

    monkeypatch.setattr(gr, "get_guru_analysis", _analysis)
    import guru_tracker
    logged = []
    monkeypatch.setattr(guru_tracker, "log_guru_verdict",
                        lambda name, *a, **k: logged.append(name))

    progress = []
    res = gr.recalc_all_stocks(progress=lambda d, t, a: progress.append((d, t, a)))

    assert scrapes["n"] == 1                        # Smart-Lab scraped exactly once
    assert logged == ["AAPL"]                       # real logged, N/A skipped
    assert res == {"total": 2, "updated": 1, "skipped": 1, "errors": 0}
    assert progress[0] == (0, 2, None)              # total reported before the scrape
    assert progress[-1][0] == 2


def test_resolve_fundamentals_finds_a_renamed_moscow_name():
    """HH.ru trades as HEAD and the guru report looked it up as HHRU, so it fell
    through to the yfinance branch, which cannot resolve a bare MOEX ticker
    either, and the council scored it on nothing.

    The second half is the control: a foreign name whose map value collides with
    a Russian ticker must NOT pick up the Russian row."""
    import guru_report as gr

    sl = {"HEAD": {"pe": 6.1, "roe": 0.0, "debt": -0.6, "div": 17.3},
          "ROST": {"pe": -6.4, "roe": 0.0, "debt": 2.4, "div": 0.0}}

    got = gr.resolve_fundamentals("HHRU", "HEAD", sl)
    assert got["_source"] == "smartlab"
    assert got["pe"] == 6.1 and got["dividend_yield"] == 17.3

    # ROSS is Ross Stores; ROST here is a Russian company at a negative P/E
    ross = gr.resolve_fundamentals("ROSS", "ROST", sl) or {}
    assert ross.get("_source") != "smartlab", ross
    assert ross.get("pe") != -6.4, "a US retailer got Russian fundamentals"


# --- the Russian list and the preferred shares -------------------------------

def test_the_russian_list_is_the_asset_map_not_a_second_copy():
    """It used to be 51 tickers written down here while the map carried 181, so
    130 Russian names were sent to yfinance, answered 404 three times each, and
    ended with no fundamentals at all."""
    from config import MOEX_ASSETS

    assert gr.MOEX_ASSETS == set(MOEX_ASSETS)
    assert len(gr.MOEX_ASSETS) > 100
    for late_addition in ("BANEP", "CNTLP", "NKNCP"):
        assert late_addition in gr.MOEX_ASSETS


def test_a_russian_preferred_takes_the_company_numbers_from_its_ordinary():
    """Smart-Lab publishes a COMPANY under its ordinary ticker, so NKNCP came
    back blank while NKNC was right there. Fourteen assets were in that state."""
    sl = {"NKNC": {"pe": 5.5, "debt": 1.2, "div": 9.9, "roe": 0}}
    d = gr.resolve_fundamentals("NKNCP", "NKNCP", sl)
    assert d["pe"] == 5.5 and d["debt_equity"] == 1.2
    assert "NKNC" in d["_source"], "the source has to say whose numbers these are"
    # The dividend yield is the ORDINARY share's ("ДД ао, %") and a preferred
    # pays differently, so it is dropped rather than borrowed.
    assert d["dividend_yield"] == 0
    assert gr.resolve_fundamentals("NKNC", "NKNC", sl)["dividend_yield"] == 9.9


def test_the_fallback_needs_the_ordinary_to_actually_exist():
    """Otherwise any Russian ticker ending in P would silently pick up whatever
    happened to be listed one letter shorter."""
    assert gr.resolve_fundamentals("NKNCP", "NKNCP", {}) is None
    # and a non-Russian name is never remapped at all
    sl = {"AD": {"pe": 1.0, "debt": 0, "div": 0, "roe": 0}}
    d = gr.resolve_fundamentals("ADP", "ADP", sl)
    assert d is None or "AD" not in str(d.get("_source", ""))
