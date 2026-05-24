"""
data.py  —  Fetch real adjusted closing prices + train/test split.

New in v2
---------
split_prices() divides the full price history into:
  - training period  : used ONLY for hedge ratio estimation
  - test period      : all signals and reported performance come from here

This fixes v1's core flaw where the hedge ratio was estimated on the same
data being traded, inflating apparent performance (in-sample bias).
"""

import pandas as pd
import yfinance as yf

from config import TICKER_1, TICKER_2, START_DATE, END_DATE, TRAIN_RATIO


def fetch_prices() -> pd.DataFrame:
    """
    Download adjusted closing prices for TICKER_1 and TICKER_2.

    Returns
    -------
    pd.DataFrame
        Columns: [TICKER_1, TICKER_2], tz-naive DatetimeIndex, no NaNs.
    """
    hist1 = yf.Ticker(TICKER_1).history(start=START_DATE, end=END_DATE)["Close"]
    hist2 = yf.Ticker(TICKER_2).history(start=START_DATE, end=END_DATE)["Close"]

    if hist1.empty or hist2.empty:
        raise ValueError(
            f"No data returned. Check tickers ({TICKER_1}, {TICKER_2}) "
            f"and date range ({START_DATE} → {END_DATE})."
        )

    if hist1.index.tz is not None:
        hist1 = hist1.tz_localize(None)
    if hist2.index.tz is not None:
        hist2 = hist2.tz_localize(None)

    prices = pd.DataFrame({TICKER_1: hist1, TICKER_2: hist2}).dropna()
    return prices


def split_prices(
    prices: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split price history into training and out-of-sample test periods.

    Parameters
    ----------
    prices      : full price DataFrame from fetch_prices()
    train_ratio : fraction of rows used for training (default from config)

    Returns
    -------
    train_prices : pd.DataFrame  — used for hedge ratio estimation only
    test_prices  : pd.DataFrame  — used for backtesting and all reported metrics
    """
    split_idx = int(len(prices) * train_ratio)
    return prices.iloc[:split_idx].copy(), prices.iloc[split_idx:].copy()