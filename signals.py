"""
signals.py  —  Compute spread, rolling z-score, and entry/exit signals.

New in v2
---------
Stop-loss: if a position moves further against us beyond STOP_LOSS_Z
(instead of reverting), we cut immediately rather than waiting.

  Long  spread entered at z = -2.5 → stop out if z drops below -3.5
  Short spread entered at z = +2.5 → stop out if z rises above +3.5

This prevents runaway losses when the cointegration relationship
temporarily breaks down.

Signal logic summary
--------------------
  z > +ENTRY  →  short spread  (signal = -1)
  z < -ENTRY  →  long  spread  (signal = +1)
  flat otherwise

Exit (long spread):
  z >= -EXIT_THRESHOLD  →  normal exit (reversion)
  z <  -STOP_LOSS_Z     →  stop-loss exit

Exit (short spread):
  z <=  EXIT_THRESHOLD  →  normal exit (reversion)
  z >   STOP_LOSS_Z     →  stop-loss exit
"""

import numpy as np
import pandas as pd

from config import ZSCORE_WINDOW, ENTRY_THRESHOLD, EXIT_THRESHOLD, STOP_LOSS_Z


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

    Under cointegration this series is stationary (mean-reverting).
    The hedge ratio β and intercept α are estimated from the TRAINING
    period only (v2 improvement over v1).
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

    First `window` values are NaN (insufficient history for rolling stats).
    """
    mu     = spread.rolling(window=window).mean()
    sigma  = spread.rolling(window=window).std()
    zscore = (spread - mu) / sigma
    zscore.name = "zscore"
    return zscore


def generate_signals(
    zscore: pd.Series,
    stop_loss_z: float = STOP_LOSS_Z,
) -> pd.Series:
    """
    Convert z-score series into position signals (+1 / -1 / 0).

    Applies stateful entry, exit, and stop-loss rules day-by-day.
    NaN z-scores (warm-up period) → signal = 0.

    Parameters
    ----------
    zscore      : rolling z-score series
    stop_loss_z : cut position if |z| exceeds this against the trade

    Returns
    -------
    pd.Series of float  (+1 long spread, -1 short spread, 0 flat)
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
                position = -1              # spread overvalued → short it
            elif z < -ENTRY_THRESHOLD:
                position = 1               # spread undervalued → long it

        elif position == 1:                # long spread: entered when z was very negative
            if z >= -EXIT_THRESHOLD:
                position = 0              # normal exit — z reverted toward zero
            elif z < -stop_loss_z:
                position = 0              # stop-loss — spread kept diverging

        elif position == -1:              # short spread: entered when z was very positive
            if z <= EXIT_THRESHOLD:
                position = 0              # normal exit — z reverted toward zero
            elif z > stop_loss_z:
                position = 0              # stop-loss — spread kept diverging

        signals.iloc[i] = position

    return signals