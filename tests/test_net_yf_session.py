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


def test_a_live_proxy_yields_a_configured_session(monkeypatch):
    proxies = {"http": "socks5://127.0.0.1:1080",
               "https": "socks5://127.0.0.1:1080"}
    monkeypatch.setattr(net, "proxies_for", lambda route="auto": proxies)
    monkeypatch.setattr(net, "ssl_verify", lambda: False)
    session = net.yf_session()
    assert session is not None
    assert session.proxies["https"] == proxies["https"]
    assert session.verify is False
