"""
backtest.py  —  Daily-bar backtest with transaction costs (v2).

New in v2
---------
Transaction costs
  Every time the position changes (entry, exit, or reversal), we subtract
  the cost of both legs:  2 × TRANSACTION_COST from that day's return.
  This prevents the strategy from overtrading and reflects real execution.

period_metrics()
  A helper that computes Sharpe, drawdown, and return for any sub-period
  of the results DataFrame. Used in main.py to report training-period
  and out-of-sample performance separately.

Position sizing (unchanged from v1, approximately dollar-neutral)
-----------------------------------------------------------------
  Long  spread: long  $1 of ticker1, short $hedge_ratio of ticker2
  Short spread: reverse of above
  Capital per unit = 1 + |hedge_ratio|
  Return on capital = signal × (ret1 − β·ret2) / (1 + |β|)

Signal is lagged by 1 day (no look-ahead bias).
"""

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, TRANSACTION_COST


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    ticker1: str,
    ticker2: str,
    hedge_ratio: float,
    transaction_cost: float = TRANSACTION_COST,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate the pair-trade strategy on historical prices.

    Parameters
    ----------
    prices           : pd.DataFrame  columns [ticker1, ticker2]
    signals          : pd.Series     +1 / -1 / 0
    ticker1          : str
    ticker2          : str
    hedge_ratio      : float    β from OLS (estimated on training data)
    transaction_cost : float    cost per leg per trade (default from config)

    Returns
    -------
    results_df : pd.DataFrame  with all strategy columns
    metrics    : dict          full-period performance summary
    """
    df = prices[[ticker1, ticker2]].copy()
    df["signal"] = signals

    # Daily percentage returns
    df["ret1"] = df[ticker1].pct_change()
    df["ret2"] = df[ticker2].pct_change()

    # Lag signal by 1 day → trade executes at next-day open
    df["signal_lag"] = df["signal"].shift(1).fillna(0)

    # Gross pair return (long ticker1, short hedge_ratio × ticker2)
    gross_return     = df["ret1"] - hedge_ratio * df["ret2"]
    capital_per_unit = 1.0 + abs(hedge_ratio)
    df["pair_return_gross"] = df["signal_lag"] * gross_return / capital_per_unit

    # Transaction cost: deducted whenever the lagged position changes
    # (2 legs × cost each time we enter, exit, or reverse)
    position_changed  = df["signal_lag"].diff().abs() > 0
    df["trade_cost"]  = position_changed.astype(float) * 2.0 * transaction_cost

    # Net return after costs
    df["pair_return"] = df["pair_return_gross"] - df["trade_cost"]

    # Cumulative growth and portfolio value
    df["cumulative_return"] = (1.0 + df["pair_return"]).cumprod()
    df["portfolio_value"]   = INITIAL_CAPITAL * df["cumulative_return"]

    # Full-period metrics
    metrics = period_metrics(df, df.index[0])

    return df, metrics


def period_metrics(results_df: pd.DataFrame, start_date, end_date=None) -> dict:
    """
    Compute performance metrics for any sub-period of results_df.

    Used in main.py to report training and out-of-sample periods separately.

    Parameters
    ----------
    results_df : output DataFrame from run_backtest()
    start_date : first date of the period (inclusive)
    end_date   : last date of the period (inclusive); None = until end

    Returns
    -------
    dict with total_return, annualized_sharpe, max_drawdown, num_trades
    """
    sub = results_df.loc[start_date:end_date] if end_date else results_df.loc[start_date:]

    pv      = sub["portfolio_value"].dropna()
    returns = sub["pair_return"].dropna()

    if len(pv) < 2 or len(returns) < 2:
        return {
            "total_return"      : 0.0,
            "annualized_sharpe" : 0.0,
            "max_drawdown"      : 0.0,
            "num_trades"        : 0,
        }

    # Return relative to the start of THIS period (not overall inception)
    total_return = float(pv.iloc[-1] / pv.iloc[0]) - 1.0

    daily_mean = returns.mean()
    daily_std  = returns.std()
    sharpe     = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0

    rolling_max  = pv.cummax()
    max_drawdown = float(((pv - rolling_max) / rolling_max).min())

    num_trades = int((sub["signal"].diff().abs() > 0).sum()) if "signal" in sub.columns else 0

    return {
        "total_return"      : total_return,
        "annualized_sharpe" : sharpe,
        "max_drawdown"      : max_drawdown,
        "num_trades"        : num_trades,
    }