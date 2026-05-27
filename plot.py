"""
plot.py  —  Four-panel strategy visualisation (v2).

New in v2
---------
split_date parameter: draws a vertical dashed line across all four panels
marking where the training period ends and out-of-sample trading begins.
The out-of-sample region is lightly shaded on the portfolio value panel
to make the performance split immediately obvious.

Panels
------
1. Normalised price series (both tickers rebased to 100)
2. Cointegration spread with mean line
3. Z-score with entry/exit/stop-loss bands + position shading
4. Portfolio value — train region vs out-of-sample region clearly marked
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
    split_date=None,
    save_path : str = "results.png",
) -> None:
    """
    Generate and save the 4-panel strategy chart.

    Parameters
    ----------
    prices     : raw price DataFrame
    spread     : spread series
    zscore     : z-score series
    signals    : +1 / -1 / 0 signal series
    results_df : output DataFrame from run_backtest()
    ticker1    : str
    ticker2    : str
    split_date : train/test boundary date (adds vertical line if provided)
    save_path  : output PNG filename
    """
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"Statistical Arbitrage Engine  —  v2  |  {ticker1} / {ticker2}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(4, 1, figure=fig, hspace=0.6)

    # ── Panel 1: Normalised prices ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    norm1 = prices[ticker1] / prices[ticker1].iloc[0] * 100
    norm2 = prices[ticker2] / prices[ticker2].iloc[0] * 100
    ax1.plot(prices.index, norm1, label=ticker1, color="steelblue",  linewidth=0.9)
    ax1.plot(prices.index, norm2, label=ticker2, color="darkorange", linewidth=0.9, alpha=0.85)
    ax1.set_title("Normalised Price (base = 100)", fontsize=10)
    ax1.set_ylabel("Index")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    # ── Panel 2: Spread ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(spread.index, spread, color="purple", linewidth=0.8, label="Spread")
    ax2.axhline(
        float(spread.mean()), color="black", linestyle="--",
        linewidth=0.8, label=f"Mean ({spread.mean():.2f})"
    )
    ax2.set_title(f"Spread  ({ticker1} − β·{ticker2} − α)  |  β estimated on training data only", fontsize=10)
    ax2.set_ylabel("Spread (USD)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Z-score + position shading ────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(zscore.index, zscore, color="teal", linewidth=0.8, label="Z-score")

    # Entry, exit, and stop-loss threshold lines
    for level, color, ls, lw, label in [
        ( 3.5, "darkred",   "-.",  0.8, "Stop-loss"),
        (-3.5, "darkred",   "-.",  0.8, None),
        ( 2.0, "red",       "--",  0.9, "Entry ±2.0"),
        (-2.0, "green",     "--",  0.9, None),
        ( 0.5, "gray",      ":",   0.7, "Exit ±0.5"),
        (-0.5, "gray",      ":",   0.7, None),
        ( 0.0, "black",     "-",   0.5, None),
    ]:
        ax3.axhline(
            level, color=color, linestyle=ls, linewidth=lw,
            label=label if label else "_nolegend_"
        )

    z_arr    = zscore.values
    z_finite = z_arr[np.isfinite(z_arr)]
    z_min    = float(z_finite.min()) if len(z_finite) else -4
    z_max    = float(z_finite.max()) if len(z_finite) else  4

    ax3.fill_between(
        zscore.index, z_min, z_max,
        where=(signals == 1).values,  alpha=0.12, color="green", label="Long spread"
    )
    ax3.fill_between(
        zscore.index, z_min, z_max,
        where=(signals == -1).values, alpha=0.12, color="red",   label="Short spread"
    )

    ax3.set_title("Z-Score  |  Entry ±2.0  |  Exit ±0.5  |  Stop-loss ±3.5", fontsize=10)
    ax3.set_ylabel("Z-score")
    ax3.legend(fontsize=7, loc="upper right", ncol=2)
    ax3.grid(alpha=0.3)

    # ── Panel 4: Portfolio value with train/OOS shading ─────────────
    ax4 = fig.add_subplot(gs[3])
    pv = results_df["portfolio_value"].dropna()
    ax4.plot(pv.index, pv, color="darkgreen", linewidth=1.0, label="Strategy (net of costs)")
    ax4.axhline(
        float(pv.iloc[0]), color="gray", linestyle="--",
        linewidth=0.7, label="Initial capital"
    )

    if split_date is not None:
        # Shade the out-of-sample region
        ax4.axvspan(
            split_date, pv.index[-1],
            alpha=0.07, color="royalblue", label="Out-of-sample period"
        )

    ax4.set_title("Portfolio Value (USD)  —  net of transaction costs", fontsize=10)
    ax4.set_ylabel("USD")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # ── Train/test vertical line on ALL panels ──────────────────────
    if split_date is not None:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axvline(
                split_date, color="navy", linestyle="--",
                linewidth=1.0, alpha=0.6
            )

        # Label only on panel 1 so it is not repeated
        y_pos = ax1.get_ylim()[1]
        ax1.text(
            split_date, y_pos * 0.97,
            "  ← Train  |  Out-of-sample →",
            fontsize=7, color="navy", va="top"
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Chart saved → {save_path}")