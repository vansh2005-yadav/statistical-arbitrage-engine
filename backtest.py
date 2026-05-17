"""
backtest.py  —  Simple daily-bar backtest (no transaction costs in v1).

Position sizing (approximately dollar-neutral)
----------------------------------------------
  Long  spread: long  $1 of ticker1, short $hedge_ratio of ticker2
  Short spread: short $1 of ticker1, long  $hedge_ratio of ticker2

  Total capital deployed per unit  =  1 + |hedge_ratio|
  Daily P&L per unit               =  signal × (ret1 − β·ret2)
  Return on capital                =  P&L / (1 + |hedge_ratio|)

Signal is lagged by 1 day to avoid look-ahead bias
(we see today's z-score → trade at next open).

Performance metrics
-------------------
  Total return      : cumulative growth of the strategy
  Annualised Sharpe : daily Sharpe × √252
  Max drawdown      : worst peak-to-trough drawdown
  Trade signals     : number of position changes
"""

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    ticker1: str,
    ticker2: str,
    hedge_ratio: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate the pair-trade strategy on historical prices.

    Parameters
    ----------
    prices      : pd.DataFrame  columns [ticker1, ticker2]
    signals     : pd.Series     +1 / -1 / 0
    ticker1     : str
    ticker2     : str
    hedge_ratio : float         β from OLS

    Returns
    -------
    results_df  : pd.DataFrame  with all strategy columns
    metrics     : dict          performance summary
    """
    df = prices[[ticker1, ticker2]].copy()
    df["signal"] = signals

    # Daily percentage returns
    df["ret1"] = df[ticker1].pct_change()
    df["ret2"] = df[ticker2].pct_change()

    # Lag signal by 1 day  →  trade at next-day open (avoids look-ahead bias)
    df["signal_lag"] = df["signal"].shift(1).fillna(0)

    # Gross pair return for a long-spread position
    gross_return = df["ret1"] - hedge_ratio * df["ret2"]

    # Normalise by total capital per unit to get return on capital
    capital_per_unit  = 1.0 + abs(hedge_ratio)
    df["pair_return"] = df["signal_lag"] * gross_return / capital_per_unit

    # Cumulative growth and portfolio value
    df["cumulative_return"] = (1.0 + df["pair_return"]).cumprod()
    df["portfolio_value"]   = INITIAL_CAPITAL * df["cumulative_return"]

    # ── Performance Metrics ──────────────────────────────────────────
    valid_returns = df["pair_return"].dropna()

    total_return = float(df["cumulative_return"].iloc[-1] - 1.0)

    daily_mean = valid_returns.mean()
    daily_std  = valid_returns.std()
    sharpe     = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0

    rolling_max  = df["cumulative_return"].cummax()
    drawdown     = (df["cumulative_return"] - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Count position changes (entries + exits + reversals)
    num_trades = int((df["signal"].diff().abs() > 0).sum())

    metrics = {
        "total_return"      : total_return,
        "annualized_sharpe" : sharpe,
        "max_drawdown"      : max_drawdown,
        "num_trades"        : num_trades,
    }

    return df, metrics
