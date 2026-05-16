"""
data.py  —  Fetch real adjusted closing prices from Yahoo Finance.
Uses yfinance Ticker.history() which is robust across library versions.
"""

import pandas as pd
import yfinance as yf

from config import TICKER_1, TICKER_2, START_DATE, END_DATE


def fetch_prices() -> pd.DataFrame:
    """
    Download adjusted closing prices for TICKER_1 and TICKER_2.

    yfinance .history() returns auto-adjusted prices by default.
    The index is tz-aware (America/New_York) — we strip the timezone
    so pandas datetime operations stay simple throughout the project.

    Returns
    -------
    pd.DataFrame
        Columns: [TICKER_1, TICKER_2], DatetimeIndex (tz-naive), no NaNs.

    Raises
    ------
    ValueError
        If no data is returned for either ticker.
    """
    hist1 = yf.Ticker(TICKER_1).history(start=START_DATE, end=END_DATE)["Close"]
    hist2 = yf.Ticker(TICKER_2).history(start=START_DATE, end=END_DATE)["Close"]

    if hist1.empty or hist2.empty:
        raise ValueError(
            f"No data returned. Check tickers ({TICKER_1}, {TICKER_2}) "
            f"and date range ({START_DATE} → {END_DATE})."
        )

    # Strip timezone so DatetimeIndex is tz-naive
    if hist1.index.tz is not None:
        hist1 = hist1.tz_localize(None)
    if hist2.index.tz is not None:
        hist2 = hist2.tz_localize(None)

    prices = pd.DataFrame({TICKER_1: hist1, TICKER_2: hist2}).dropna()
    return prices