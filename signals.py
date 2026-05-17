"""
signals.py  —  Compute spread, rolling z-score, and entry/exit signals.

Logic
-----
Spread   =  ticker1  −  β·ticker2  −  α
           Under cointegration this is stationary (mean-reverting).

Z-score  =  (spread − rolling_mean) / rolling_std
           Standardises the spread so entry/exit thresholds are scale-free.

Signals
-------
  z > +ENTRY  →  spread overvalued  →  SHORT spread  (signal = -1)
  z < -ENTRY  →  spread undervalued →  LONG  spread  (signal = +1)
  Flat otherwise.

Exit
----
  Long  position: exit when z reverts back above  -EXIT_THRESHOLD
  Short position: exit when z reverts back below  +EXIT_THRESHOLD
"""

import numpy as np
import pandas as pd

from config import ZSCORE_WINDOW, ENTRY_THRESHOLD, EXIT_THRESHOLD


def compute_spread(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    hedge_ratio: float,
    intercept: float,
) -> pd.Series:
    """
    Compute the cointegration spread.

    spread = ticker1 − β·ticker2 − α
    """
    spread = prices[ticker1] - hedge_ratio * prices[ticker2] - intercept
    spread.name = "spread"
    return spread


def compute_zscore(
    spread: pd.Series,
    window: int = ZSCORE_WINDOW,
) -> pd.Series:
    """
    Rolling z-score of the spread.

    z = (spread − μ_rolling) / σ_rolling

    First `window` values will be NaN (insufficient history).
    """
    mu    = spread.rolling(window=window).mean()
    sigma = spread.rolling(window=window).std()
    zscore = (spread - mu) / sigma
    zscore.name = "zscore"
    return zscore


def generate_signals(zscore: pd.Series) -> pd.Series:
    """
    Convert z-score series into position signals (+1 / -1 / 0).

    Iterates day-by-day to apply stateful entry/exit rules.
    NaN z-scores (warm-up period) → signal = 0.

    Returns
    -------
    pd.Series of float  (+1, -1, or 0)
    """
    signals  = pd.Series(0.0, index=zscore.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z > ENTRY_THRESHOLD:
                position = -1          # spread too high → short it
            elif z < -ENTRY_THRESHOLD:
                position = 1           # spread too low  → long it

        elif position == 1:            # long spread, waiting for reversion upward
            if z >= -EXIT_THRESHOLD:   # z has crossed back above -EXIT band
                position = 0

        elif position == -1:           # short spread, waiting for reversion downward
            if z <= EXIT_THRESHOLD:    # z has crossed back below +EXIT band
                position = 0

        signals.iloc[i] = position

    return signals