"""
plot.py  —  Four-panel strategy visualisation.

Panel 1: Normalised price series (both tickers rebased to 100)
Panel 2: Cointegration spread with mean line
Panel 3: Z-score with entry/exit bands + position shading
Panel 4: Portfolio value over time (USD)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np


def plot_results(
    prices    : pd.DataFrame,
    spread    : pd.Series,
    zscore    : pd.Series,
    signals   : pd.Series,
    results_df: pd.DataFrame,
    ticker1   : str,
    ticker2   : str,
    save_path : str = "results.png",
) -> None:
    """
    Generate and save (+ display) the 4-panel chart.

    Parameters
    ----------
    prices     : raw price DataFrame
    spread     : spread series
    zscore     : z-score series
    signals    : +1 / -1 / 0 signal series
    results_df : backtest output DataFrame (must contain 'portfolio_value')
    ticker1    : str
    ticker2    : str
    save_path  : filename for saved PNG
    """
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"Statistical Arbitrage Engine  —  v1  |  {ticker1} / {ticker2}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6)

    # ── Panel 1: Normalised prices ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    norm1 = prices[ticker1] / prices[ticker1].iloc[0] * 100
    norm2 = prices[ticker2] / prices[ticker2].iloc[0] * 100
    ax1.plot(prices.index, norm1, label=ticker1, color="steelblue",  linewidth=0.9)
    ax1.plot(prices.index, norm2, label=ticker2, color="darkorange", linewidth=0.9, alpha=0.85)
    ax1.set_title("Normalised Price (base = 100)", fontsize=10)
    ax1.set_ylabel("Index")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    # ── Panel 2: Spread ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(spread.index, spread, color="purple", linewidth=0.8, label="Spread")
    ax2.axhline(
        float(spread.mean()), color="black", linestyle="--",
        linewidth=0.8, label=f"Mean ({spread.mean():.2f})"
    )
    ax2.set_title(f"Spread  ({ticker1} − β·{ticker2} − α)", fontsize=10)
    ax2.set_ylabel("Spread (USD)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Z-score + position shading ───────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(zscore.index, zscore, color="teal", linewidth=0.8, label="Z-score")

    # Threshold lines
    for level, color, ls, lw in [
        ( 2.0, "red",   "--", 0.9),
        (-2.0, "green", "--", 0.9),
        ( 0.5, "gray",  ":",  0.7),
        (-0.5, "gray",  ":",  0.7),
        ( 0.0, "black", "-",  0.5),
    ]:
        ax3.axhline(level, color=color, linestyle=ls, linewidth=lw)

    # Shade long / short periods
    z_arr    = zscore.values
    z_finite = z_arr[np.isfinite(z_arr)]
    z_min    = float(z_finite.min()) if len(z_finite) else -3
    z_max    = float(z_finite.max()) if len(z_finite) else  3

    long_mask  = (signals == 1).values
    short_mask = (signals == -1).values

    ax3.fill_between(
        zscore.index, z_min, z_max,
        where=long_mask, alpha=0.12, color="green", label="Long spread"
    )
    ax3.fill_between(
        zscore.index, z_min, z_max,
        where=short_mask, alpha=0.12, color="red", label="Short spread"
    )

    ax3.set_title("Z-Score  |  Entry ±2.0  |  Exit ±0.5", fontsize=10)
    ax3.set_ylabel("Z-score")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    # ── Panel 4: Portfolio value ───────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    pv = results_df["portfolio_value"].dropna()
    ax4.plot(pv.index, pv, color="darkgreen", linewidth=1.0, label="Strategy")
    ax4.axhline(
        float(pv.iloc[0]), color="gray", linestyle="--",
        linewidth=0.7, label="Initial capital"
    )
    ax4.set_title("Portfolio Value (USD)", fontsize=10)
    ax4.set_ylabel("USD")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved → {save_path}")