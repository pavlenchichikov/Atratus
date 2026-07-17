"""Walk-forward splits, PnL simulation, fold scoring."""

import os

import numpy as np
import pandas as pd

# Trading cost defaults
COMMISSION = 0.001
SLIPPAGE = 0.0015
FOREX_COMMISSION = 0.0003
FOREX_SLIPPAGE = 0.0002
MAX_TRADE_RET = 0.04
INITIAL_CAPITAL = 1000.0
POSITION_FRACTION = 0.10


def adaptive_split_params(n_rows: int) -> dict | None:
    """Determine walk-forward split sizes based on dataset length."""
    if n_rows >= 3000:
        return {"min_train": 500, "val_size": 120, "test_size": 120, "step": 360}
    if n_rows >= 1500:
        return {"min_train": 500, "val_size": 120, "test_size": 120, "step": 240}
    if n_rows >= 900:
        return {"min_train": 500, "val_size": 120, "test_size": 120, "step": 120}
    if n_rows >= 600:
        return {"min_train": 320, "val_size": 90, "test_size": 90, "step": 90}
    if n_rows >= 360:
        return {"min_train": 220, "val_size": 60, "test_size": 60, "step": 60}
    if n_rows >= 220:
        return {"min_train": 140, "val_size": 40, "test_size": 40, "step": 40}
    return None


def make_walk_forward_splits(
    n: int,
    min_train: int = 500,
    val_size: int = 120,
    test_size: int = 120,
    step: int = 120,
    embargo: int = 0,
) -> list[tuple]:
    """Generate walk-forward cross-validation windows.

    Returns list of (train_slice, val_slice, test_slice) tuples.

    embargo inserts a gap of `embargo` bars between train/val and val/test. This
    prevents leakage from overlapping label and rolling-feature windows (e.g. a
    sequence model whose val sequences would otherwise reuse the tail of train).
    """
    splits = []
    start = min_train
    while start + embargo + val_size + embargo + test_size <= n:
        tr_end = start
        va_start = tr_end + embargo
        va_end = va_start + val_size
        te_start = va_end + embargo
        te_end = te_start + test_size
        splits.append((slice(0, tr_end), slice(va_start, va_end), slice(te_start, te_end)))
        start += step
    return splits


def pnl_from_signals(
    signals: np.ndarray,
    next_ret: np.ndarray,
    commission: float = COMMISSION,
    slippage: float = SLIPPAGE,
) -> tuple[float, int, float]:
    """Simulate PnL from trading signals.

    Args:
        signals: array of {-1, 0, +1} (sell, hold, buy)
        next_ret: next-bar returns
        commission: per-trade commission
        slippage: per-trade slippage

    Returns:
        (profit_pct, n_trades, win_rate_pct)
    """
    bal = INITIAL_CAPITAL
    trades = 0
    wins = 0
    exec_cost = commission + slippage
    for s, r in zip(signals, next_ret):
        if np.isnan(r) or s == 0:
            continue
        trades += 1
        raw_ret = r if s > 0 else -r
        raw_ret = float(np.clip(raw_ret, -MAX_TRADE_RET, MAX_TRADE_RET))
        trade_ret = raw_ret - exec_cost
        if trade_ret > 0:
            wins += 1
        bal *= (1 + trade_ret * POSITION_FRACTION)
    profit = (bal / INITIAL_CAPITAL - 1.0) * 100.0
    winrate = (wins / trades * 100.0) if trades else 0.0
    return profit, trades, winrate


def max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Compute max drawdown (%) from a sequence of per-period returns."""
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1 + np.array(returns, dtype=float))
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / (peak + 1e-9)
    return abs(dd.min()) * 100.0


def sharpe_from_returns(returns, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from a sequence of per-trade returns.

    Returns 0.0 for degenerate inputs (too few trades or zero variance).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def score_strategy(
    profit: float,
    max_dd: float,
    winrate: float,
    trades: int,
    sharpe: float | None = None,
    min_trades: int = 10,
) -> float:
    """Risk-adjusted strategy score.

    Base term penalizes drawdown and rewards win rate:
        profit - 0.5 * maxDD + 0.1 * winrate
    When `sharpe` is provided the score additionally rewards return-per-unit-risk
    (weight 2.0), so a strategy is preferred for *consistent* edge rather than a
    few lucky large trades. Omitting `sharpe` reproduces the original composite,
    keeping older callers unchanged.

    Returns -999 if fewer than `min_trades` trades (unreliable signal); the
    position-aware v2 objective passes 5 because positions run about half the
    per-bar count.
    """
    if trades < min_trades:
        return -999.0
    base = profit - 0.5 * max_dd + 0.1 * winrate
    if sharpe is not None:
        base += 2.0 * sharpe
    return base


def make_signals(
    prob: np.ndarray,
    buy_thr: float,
    sell_thr: float,
    no_trade_band: float,
) -> np.ndarray:
    """Convert probability array to trading signals {-1, 0, +1}.

    Args:
        prob: predicted probability of next bar going up
        buy_thr: probability threshold for BUY signal
        sell_thr: probability threshold for SELL signal
        no_trade_band: neutral zone width around thresholds
    """
    out = np.zeros_like(prob, dtype=int)
    for i, p in enumerate(prob):
        if (buy_thr - no_trade_band) <= p <= (sell_thr + no_trade_band):
            out[i] = 0
        elif p >= buy_thr:
            out[i] = 1
        elif p <= sell_thr:
            out[i] = -1
        else:
            out[i] = 0
    return out


def apply_regime_filter(
    signals: np.ndarray,
    close: np.ndarray,
    sma200: np.ndarray,
    taleb: np.ndarray,
    risk_cap: float,
    mode: str = "both",
) -> np.ndarray:
    """Suppress signals during high-risk regimes.

    - Blocks all signals when taleb_risk > risk_cap
    - Blocks BUY when close < SMA200 (downtrend)
    - Blocks SELL when close > SMA200 (uptrend)

    mode (auto-research regime axis; default = today's exact behavior):
    "both" applies the Taleb cap and the trend blocks; "off" returns an
    untouched copy; "sma_only" skips the Taleb cap; "taleb_only" skips the
    trend blocks. An unknown mode falls back to "both"."""
    if mode == "off":
        return signals.copy()
    if mode not in ("both", "sma_only", "taleb_only"):
        mode = "both"
    filt = signals.copy()
    for i in range(len(filt)):
        if mode != "sma_only" and taleb[i] > risk_cap:
            filt[i] = 0
            continue
        if mode != "taleb_only":
            if filt[i] > 0 and close[i] < sma200[i]:
                filt[i] = 0
            if filt[i] < 0 and close[i] > sma200[i]:
                filt[i] = 0
    return filt


# --- objective v2: position-aware simulation (env-gated) ----------------------
# The old pnl_from_signals charges a round-trip on EVERY signal bar; the serve
# side (core/positions.py) holds a position while the signal repeats and exits
# on WAIT or the opposite call. simulate_positions models exactly that display
# convention: costs only on side CHANGES (a flip = 2 legs), a daily equity
# series (zeros when flat) for honest Sharpe/drawdown, and a per-asset
# vol-scaled cap instead of the fixed +-4% clip. All of it activates only via
# GTRADE_OBJECTIVE_V2; the old functions stay untouched.


def objective_v2_on() -> bool:
    return (os.getenv("GTRADE_OBJECTIVE_V2") or "").strip() in ("1", "true", "True")


def vol_cap(next_ret, window: int = 20, k: float = 6.0,
            floor: float = MAX_TRADE_RET) -> np.ndarray:
    """Per-bar |return| cap = max(floor, k * causal rolling std).

    Keeps the dirty-data protection of the old fixed clip (a bad feed bar
    cannot flip a champion selection) without truncating honest volatile
    bars: the cap is never TIGHTER than the old 4%. Warm-up bars (fewer than
    min_periods=10 observations) fall back to the floor."""
    s = pd.Series(np.asarray(next_ret, dtype=float))
    sd = s.rolling(window, min_periods=10).std().shift(1)
    cap = (k * sd).clip(lower=floor)
    return cap.fillna(floor).to_numpy()


def simulate_positions(signals, next_ret, commission: float = COMMISSION,
                       slippage: float = SLIPPAGE, cap=None):
    """Position-aware PnL under the core/positions.py convention.

    Returns (profit_pct, n_trades, win_rate_pct, daily_returns):
    - side per bar = the signal itself (0 = flat); legs on a change =
      abs(new - old), each leg costing commission+slippage;
    - daily_returns[i] = side_i * capped_ret_i - legs_i * leg_cost, INCLUDING
      zeros on flat bars, UNSCALED by POSITION_FRACTION (Sharpe/drawdown
      input); profit compounds with POSITION_FRACTION like pnl_from_signals;
    - n_trades = opened positions; win rate = share of segments whose chained
      return net of their legs is positive (a still-open final segment is
      judged on its to-date return, matching the display's open trade);
    - a NaN return bar earns 0 and does not break the position."""
    sig = np.asarray(signals)
    ret = np.asarray(next_ret, dtype=float)
    leg_cost = commission + slippage
    n = len(sig)
    daily = np.zeros(n, dtype=float)
    bal = INITIAL_CAPITAL
    pos = 0
    trades = wins = 0
    seg_factor = 1.0
    seg_costs = 0.0
    for i in range(n):
        s = int(sig[i])
        legs = abs(s - pos)
        if legs:
            if pos != 0:  # closing (or flipping out of) the old segment
                if seg_factor - 1.0 - seg_costs - leg_cost > 0:
                    wins += 1
            if s != 0:    # opening a new segment (its entry leg)
                trades += 1
                seg_factor, seg_costs = 1.0, leg_cost
            pos = s
        r = 0.0 if np.isnan(ret[i]) else float(ret[i])
        if cap is not None and r != 0.0:
            c = float(cap[i])
            r = float(np.clip(r, -c, c))
        bar_ret = pos * r - legs * leg_cost
        daily[i] = bar_ret
        bal *= (1.0 + bar_ret * POSITION_FRACTION)
        if pos != 0:
            seg_factor *= (1.0 + pos * r)
    if pos != 0 and seg_factor - 1.0 - seg_costs > 0:  # open final segment
        wins += 1
    profit = (bal / INITIAL_CAPITAL - 1.0) * 100.0
    winrate = (wins / trades * 100.0) if trades else 0.0
    return profit, trades, winrate, daily


def evaluate_signals_v2(sig, ret, comm, slip):
    """(profit, trades, winrate, mdd, sharpe) under the v2 position-aware
    objective, regardless of the flag - used for the always-on dual-score
    emission (quality rows carry Score_v2 so A/B arms share one yardstick)."""
    ret_arr = np.asarray(ret, dtype=float)
    profit, trades, winrate, daily = simulate_positions(
        sig, ret_arr, commission=comm, slippage=slip, cap=vol_cap(ret_arr))
    return (profit, trades, winrate,
            max_drawdown_from_returns(daily), sharpe_from_returns(daily))


def evaluate_signals(sig, ret, comm, slip):
    """(profit, trades, winrate, mdd, sharpe) under the ACTIVE objective.

    GTRADE_OBJECTIVE_V2 on: the position-aware path (see simulate_positions).
    Off (default): the original per-bar path, byte-identical to the inline
    computation train_hybrid carried before this helper existed."""
    if objective_v2_on():
        return evaluate_signals_v2(sig, ret, comm, slip)
    ret_stream = [(float(np.clip((r if g > 0 else -r), -MAX_TRADE_RET, MAX_TRADE_RET)) - (comm + slip))
                  for g, r in zip(sig, ret) if g != 0 and not np.isnan(r)]
    p, t, w = pnl_from_signals(sig, ret, commission=comm, slippage=slip)
    return (p, t, w,
            max_drawdown_from_returns(ret_stream), sharpe_from_returns(ret_stream))
