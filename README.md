# G-Trade

ML trading signals for ~150 assets: crypto, US and Russian stocks, forex, commodities. Each asset has an ensemble of 4 models (CatBoost, LSTM, Transformer, TCN). The best one is picked by walk-forward backtest with commissions. Position sizing is Kelly-based, with drawdown stops.

## How it works

1. `data_engine.py` downloads daily and weekly quotes from Yahoo Finance and MOEX into `market.db` (SQLite).
2. `train_hybrid.py` builds features: returns, volatility, tail risk (Taleb kurtosis, skew, VaR), RSI, MACD, SMA, ATR, weekly and cross-asset correlations. Trains the ensemble and saves the champion together with its scaler and probability calibrator.
3. `predict.py` prints BUY/SELL/WAIT with confidence for all assets.
4. `backtest.py` checks champions on held-out data: PnL, win rate, Sharpe, directional accuracy, Brier, alpha vs buy & hold.
5. `risk_manager.py` and `portfolio.py` do position sizing, loss limits and correlations. Tail risk is gated by the Taleb index: position size shrinks above the soft cap and new buys are blocked above the hard cap.

On top of the signals there are supporting layers: a Guru Council fundamental verdict (Lynch, Buffett, Graham, Munger) from `guru_report.py`, news sentiment from `news_analyzer.py`, and a market regime / fear-greed read. `db_check.py` is a read-only audit of `market.db` (freshness, OHLC sanity, gaps, coverage).

`app.py` is a Streamlit dashboard. The Telegram bot sends signals every hour.

## Web UI

```
uvicorn webapp:app --host 0.0.0.0 --port 8000
```

Lightweight web interface, no TensorFlow needed, reads predictions from the database, starts instantly. Pages:

- `/` signal radar: BUY/SELL/WAIT per asset with confidence, live accuracy, a Taleb tail-risk column, market regime and fear-greed gauges
- `/asset/BTC` per-asset detail: price and candle charts, signal history, model consensus, Taleb tail risk, and the Guru Council verdict with on-demand recalculate
- `/risk` interactive risk manager: open and close positions, edit and persist risk limits, manually halt or resume trading, plus a Taleb tail-risk watchlist of the highest-risk assets
- `/market`, `/sectors`, `/correlations`, `/performance`, `/news`, `/guru`, `/models` analytics pages

Same data as JSON under `/api/...`. Pages auto-refresh; works from a phone on the same network.

## Telegram bot

`python alert_bot.py` runs the hourly scan and also:

- commands /top, /signal BTC, /risk, /digest (owner only)
- morning digest, hour is set by GTRADE_DIGEST_HOUR, default 9
- degradation warnings: data older than 7 days or accuracy below 40% on the last 20 verified signals

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # telegram token, proxy if needed

python data_engine.py         # download market data
python train_hybrid.py        # train models
python predict.py             # console signals
streamlit run app.py          # dashboard
```

`python launcher.py` opens a text menu over all of the above (full cycle, dashboard, web UI, predict, DB audit, and more). `python db_check.py` runs a read-only audit of `market.db`; add `--fix` to repair duplicates and date formats.

`python scheduler.py` runs as a daemon: data every 6h, predictions every 4h, daily DB check.

## Network

If SOCKS5_PROXY is set in `.env`, outbound requests go through it. `net.py` checks if the proxy is alive and falls back to a direct connection.

- GTRADE_PROXY_MODE=auto|on|off, default auto
- GTRADE_SSL_VERIFY=0 disables TLS certificate checks. Verification is on by default, turn it off only if your proxy intercepts TLS

## GPU

TensorFlow on Windows is CPU-only since 2.11, so training runs on CPU. Good enough for daily data. For a GPU use WSL2 and `pip install tensorflow[and-cuda]`.

## Config

- `.env` - telegram credentials, proxy
- `config.py` - asset list and buy/sell thresholds
- `auto_trader_config.json` - paper trading settings

## Tests

```bash
pytest -q
ruff check .
```

## License

MIT
