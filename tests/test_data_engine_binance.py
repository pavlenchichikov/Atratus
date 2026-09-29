"""Binance klines: only finished candles are stored, volume in dollars."""
import data_engine as de


def _k(open_ms, close_ms, close="0.00000499", quote_vol="1000.5"):
    return [open_ms, "0.00000477", "0.00000536", "0.00000476", close, "9e12",
            close_ms, quote_vol, 1, "0", "0", "0"]


def test_the_candle_still_running_is_dropped():
    day = 86_400_000
    klines = [_k(0, day - 1), _k(day, 2 * day - 1)]
    df = de._binance_frame(klines, now_ms=day + 5)
    assert len(df) == 1
    assert df["Close"].iloc[0] == 4.99e-06
    assert df["Volume"].iloc[0] == 1000.5, "quote (USDT) volume, not coin count"


def test_a_restricted_location_falls_back_to_the_market_data_mirror(monkeypatch):
    import data_engine

    day = 86400 * 1000
    kline = [0, "1", "2", "0.5", "1.5", "10", day - 1, "15"]
    seen = []

    class _R:
        def __init__(self, body):
            self._body = body

        def json(self):
            return self._body

    def fake_get(url, **kw):
        seen.append(url)
        if "api.binance.com" in url:
            return _R({"code": 0, "msg": "Service unavailable from a restricted location"})
        return _R([kline])

    monkeypatch.setattr(data_engine.net, "http_get", fake_get)
    df = data_engine.fetch_binance("PEPE", None)
    assert df is not None and len(df) == 1
    assert any("data-api.binance.vision" in u for u in seen)
