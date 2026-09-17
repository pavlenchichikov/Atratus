"""Unit tests for net.yf_session (no network: the proxy lookup is stubbed)."""

import net


def test_no_session_on_the_direct_path(monkeypatch):
    # Handing yfinance a custom session on the direct path breaks its
    # cookie/crumb handshake and yields an empty .info, so None is required.
    monkeypatch.setattr(net, "proxies_for", lambda route="auto": None)
    assert net.yf_session() is None


def test_no_session_for_an_empty_proxy_mapping(monkeypatch):
    monkeypatch.setattr(net, "proxies_for", lambda route="auto": {})
    assert net.yf_session() is None


def test_a_live_proxy_is_handed_to_yfinance_not_to_the_caller(monkeypatch):
    """yfinance 1.1.0 raises YFDataException on a requests.Session, and
    guru_report's retry loop swallowed it: every non-MOEX asset reported no
    fundamentals. The proxy goes into yfinance's own transport instead."""
    import yfinance as yf
    proxies = {"http": "socks5://127.0.0.1:1080",
               "https": "socks5://127.0.0.1:1080"}
    monkeypatch.setattr(net, "proxies_for", lambda route="auto": proxies)
    monkeypatch.setattr(yf.config.network, "proxy", None, raising=False)
    assert net.yf_session() is None
    assert yf.config.network.proxy == proxies
