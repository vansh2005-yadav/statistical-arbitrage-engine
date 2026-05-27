"""
main.py  —  Statistical Arbitrage Engine  v2

Run from the project folder:
    python main.py

What changed from v1
--------------------
  1. Walk-forward split   : hedge ratio estimated on training data only.
                            All reported performance is out-of-sample.
  2. Transaction costs    : 10 bps per leg deducted on every position change.
  3. Stop-loss            : position cut if z moves beyond ±3.5 (diverging).
  4. Dual reporting       : training-period vs out-of-sample metrics shown
                            separately so overfitting is immediately visible.

Pipeline
--------
1.  Fetch real price data  (Yahoo Finance)
2.  Split into train / test
3.  Engle-Granger cointegration test on TRAIN only
4.  Compute spread & z-score on ALL data using train parameters
5.  Generate signals on ALL data (with stop-loss)
6.  Backtest on ALL data with transaction costs
7.  Report training-period vs out-of-sample performance separately
8.  Plot with train/test boundary marked
"""

import numpy as np

from config import (
    TICKER_1, TICKER_2,
    START_DATE, END_DATE,
    TRAIN_RATIO,
    ZSCORE_WINDOW,
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    STOP_LOSS_Z,
)
from data import fetch_prices, split_prices
from cointegration import run_cointegration_test
from signals import compute_spread, compute_zscore, generate_signals
from backtest import run_backtest, period_metrics
from plot import plot_results


def print_metrics(label: str, m: dict, capital_start: float) -> None:
    """Pretty-print a metrics dict with a section label."""
    print(f"\n  {'─' * 44}")
    print(f"  {label}")
    print(f"  {'─' * 44}")
    final = capital_start * (1 + m["total_return"])
    pnl   = final - capital_start
    print(f"  Total Return      : {m['total_return'] * 100:>+10.2f}%")
    print(f"  Est. P&L          : ${pnl:>+12,.0f}")
    print(f"  Annualised Sharpe : {m['annualized_sharpe']:>10.3f}")
    print(f"  Max Drawdown      : {m['max_drawdown'] * 100:>10.2f}%")
    print(f"  Trade Signals     : {m['num_trades']:>10}")


def main() -> None:
    print("\n" + "═" * 56)
    print("  Statistical Arbitrage Engine  —  v2")
    print(f"  Pair   : {TICKER_1} / {TICKER_2}")
    print(f"  Period : {START_DATE}  →  {END_DATE}")
    print(f"  Split  : {int(TRAIN_RATIO*100)}% train | {int((1-TRAIN_RATIO)*100)}% out-of-sample")
    print(f"  Costs  : {TRANSACTION_COST*10000:.0f} bps/leg  |  Stop-loss z = ±{STOP_LOSS_Z}")
    print("═" * 56)

    # ── 1. Fetch data ────────────────────────────────────────────
    print("\n[1/6]  Fetching price data from Yahoo Finance...")
    prices = fetch_prices()
    print(f"       {len(prices)} trading days loaded.")

    # ── 2. Train / test split ────────────────────────────────────
    print("\n[2/6]  Splitting into training and out-of-sample periods...")
    train_prices, test_prices = split_prices(prices, TRAIN_RATIO)
    split_date = test_prices.index[0]
    print(f"       Training period   : {train_prices.index[0].date()}  →  {train_prices.index[-1].date()}  ({len(train_prices)} days)")
    print(f"       Out-of-sample     : {test_prices.index[0].date()}   →  {test_prices.index[-1].date()}  ({len(test_prices)} days)")

    # ── 3. Cointegration test — TRAINING DATA ONLY ───────────────
    print("\n[3/6]  Running Engle-Granger cointegration test on training data...")
    cr = run_cointegration_test(train_prices, TICKER_1, TICKER_2)

    print(f"       Test statistic  : {cr['score']:.4f}")
    print(f"       p-value         : {cr['pvalue']:.4f}")
    print(f"       Critical values : 1%={cr['critical_values'][0]:.2f}  "
          f"5%={cr['critical_values'][1]:.2f}  10%={cr['critical_values'][2]:.2f}")
    print(f"       Hedge ratio (β) : {cr['hedge_ratio']:.4f}   ← estimated on training data only")
    print(f"       Intercept  (α)  : {cr['intercept']:.4f}")
    flag = "YES  ✓" if cr["cointegrated"] else "NO   ✗"
    print(f"       Cointegrated    : {flag}")

    if not cr["cointegrated"]:
        print("\n  [WARNING] Pair not cointegrated at 5% level on training data.")
        print("  Proceeding — but treat results with caution.")

    # ── 4. Spread & z-score on ALL data ─────────────────────────
    print(f"\n[4/6]  Computing spread & z-score on full dataset (window={ZSCORE_WINDOW} days)...")
    print(f"       Using β={cr['hedge_ratio']:.4f} and α={cr['intercept']:.4f} from training period.")
    spread = compute_spread(prices, TICKER_1, TICKER_2, cr["hedge_ratio"], cr["intercept"])
    zscore = compute_zscore(spread)
    print(f"       Spread mean     : {spread.mean():.4f}")
    print(f"       Spread std      : {spread.std():.4f}")
    print(f"       Z-score range   : [{zscore.dropna().min():.2f},  {zscore.dropna().max():.2f}]")

    # ── 5. Signals on ALL data ───────────────────────────────────
    print(f"\n[5/6]  Generating signals (entry ±{2.0}, exit ±{0.5}, stop-loss ±{STOP_LOSS_Z})...")
    signals = generate_signals(zscore, stop_loss_z=STOP_LOSS_Z)
    print(f"       Long  days      : {int((signals ==  1).sum())}")
    print(f"       Short days      : {int((signals == -1).sum())}")
    print(f"       Flat  days      : {int((signals ==  0).sum())}")

    # ── 6. Backtest on ALL data (with costs) ─────────────────────
    print(f"\n[6/6]  Running backtest ({TRANSACTION_COST*10000:.0f} bps/leg transaction costs)...")
    results_df, _ = run_backtest(
        prices, signals, TICKER_1, TICKER_2,
        cr["hedge_ratio"], transaction_cost=TRANSACTION_COST,
    )

    # ── Performance: training period ────────────────────────────
    train_end   = train_prices.index[-1]
    m_train     = period_metrics(results_df, results_df.index[0], train_end)

    # ── Performance: out-of-sample period ───────────────────────
    oos_start_pv = float(results_df.loc[split_date, "portfolio_value"])
    m_oos        = period_metrics(results_df, split_date)

    print("\n" + "═" * 56)
    print("  PERFORMANCE SUMMARY")
    print_metrics("TRAINING PERIOD  (in-sample)", m_train, INITIAL_CAPITAL)
    print_metrics("OUT-OF-SAMPLE PERIOD  (true performance)", m_oos, oos_start_pv)

    print(f"\n  {'─' * 44}")
    print("  NOTE: Training period uses the same data the hedge ratio")
    print("  was estimated on — expect it to look better than OOS.")
    print(f"  {'─' * 44}")

    # ── Plot ─────────────────────────────────────────────────────
    print("\n  Generating plots  →  results.png")
    plot_results(
        prices, spread, zscore, signals,
        results_df, TICKER_1, TICKER_2,
        split_date=split_date,
    )

    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()