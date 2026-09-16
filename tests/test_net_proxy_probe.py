"""is_proxy_alive must prove the endpoint speaks SOCKS, not merely that it listens.

Deliberately a separate module from test_net.py: that file's autouse fixture
replaces net.is_proxy_alive with a lambda, so tests written there would
exercise the stub and never the probe.

Real loopback servers rather than stubbed sockets. The bug being guarded is a
handshake that never completes, and a stub that returns bytes on demand cannot
reproduce a peer that stays silent - which is exactly what xray's HTTP port
does when a config points at it by mistake.
"""
import socket
import threading
import time

import pytest

import net


def _server(behaviour):
    """Listen on a free loopback port and run `behaviour(conn)` for one client.

    Returns the port. The thread is a daemon and every error is swallowed: the
    probe closes its socket the moment it has an answer, so the server side is
    expected to die mid-write on the happy paths.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        try:
            conn, _ = srv.accept()
            with conn:
                behaviour(conn)
        except Exception:
            pass
        finally:
            try:
                srv.close()
            except Exception:
                pass

    threading.Thread(target=run, daemon=True).start()
    return port


def _dead_port():
    """A port nothing listens on: bound, read, released."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def probe(monkeypatch):
    """Call the REAL probe against a given URL, with a short probe budget."""
    def _call(url, timeout=0.3):
        monkeypatch.setattr(net, "SOCKS5_PROXY", url)
        monkeypatch.setattr(net, "_PROXY_MODE", "auto")
        monkeypatch.setattr(net, "_PROBE_TIMEOUT", timeout)
        return net.is_proxy_alive(force=True)
    return _call


def test_a_real_socks5_greeting_is_accepted(probe):
    """VER=5 plus a chosen method is what a SOCKS5 server answers (RFC 1928)."""
    port = _server(lambda c: (c.recv(3), c.sendall(b"\x05\x00")))
    assert probe(f"socks5h://127.0.0.1:{port}") is True


def test_a_port_that_listens_but_never_answers_is_not_alive(probe):
    """The bug this exists for. xray puts its HTTP proxy beside its SOCKS5 one
    (10809 next to 10808), and a config on the wrong port connects fine, probed
    "alive", then burned the whole connect budget on a handshake the peer never
    answers: 5.0s per attempt, three attempts, per asset, on 2026-09-16."""
    port = _server(lambda c: time.sleep(1.5))
    assert probe(f"socks5h://127.0.0.1:{port}") is False


def test_an_http_proxy_answering_http_is_not_mistaken_for_socks(probe):
    port = _server(lambda c: (c.recv(3),
                              c.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")))
    assert probe(f"socks5h://127.0.0.1:{port}") is False


def test_a_socks5_server_refusing_every_method_is_not_usable(probe):
    """0xFF means it accepted none of the methods offered. It is a SOCKS5
    server, but not one this project can use unauthenticated, and calling it
    alive would hand http_get a route that fails on every request."""
    port = _server(lambda c: (c.recv(3), c.sendall(b"\x05\xff")))
    assert probe(f"socks5h://127.0.0.1:{port}") is False


def test_a_plain_http_proxy_url_is_still_judged_by_tcp_alone(probe):
    """A http:// proxy is a legitimate value here - requests takes one in the
    same dict - and it speaks no SOCKS greeting. Demanding one would disable a
    working configuration, so the handshake is asked of SOCKS schemes only."""
    port = _server(lambda c: time.sleep(1.5))
    assert probe(f"http://127.0.0.1:{port}") is True


def test_nothing_listening_is_not_alive(probe):
    assert probe(f"socks5h://127.0.0.1:{_dead_port()}") is False


def test_an_empty_address_short_circuits(probe):
    assert probe("") is False
