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
