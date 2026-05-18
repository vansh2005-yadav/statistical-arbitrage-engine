"""
main.py  —  Statistical Arbitrage Engine  v1

Entry point. Run from the project folder:
    python main.py

Pipeline
--------
1. Fetch real price data (Yahoo Finance)
2. Engle-Granger cointegration test
3. Compute spread and rolling z-score
4. Generate entry/exit signals
5. Backtest and report performance
6. Plot results (saved to results.png)
"""

from config import (
    TICKER_1,
    TICKER_2,
    START_DATE,
    END_DATE,
    ZSCORE_WINDOW,
    INITIAL_CAPITAL,
)
from data import fetch_prices
from cointegration import run_cointegration_test
from signals import compute_spread, compute_zscore, generate_signals
from backtest import run_backtest
from plot import plot_results


def main() -> None:
    print("\n" + "═" * 56)
    print("  Statistical Arbitrage Engine  —  v1")
    print(f"  Pair   : {TICKER_1} / {TICKER_2}")
    print(f"  Period : {START_DATE}  →  {END_DATE}")
    print("═" * 56)

    # ── Step 1: Fetch data ───────────────────────────────────────
    print("\n[1/5]  Fetching price data from Yahoo Finance...")
    prices = fetch_prices()
    print(f"       {len(prices)} trading days loaded.")

    # ── Step 2: Cointegration test ───────────────────────────────
    print("\n[2/5]  Running Engle-Granger cointegration test...")
    cr = run_cointegration_test(prices, TICKER_1, TICKER_2)

    print(f"       Test statistic  : {cr['score']:.4f}")
    print(f"       p-value         : {cr['pvalue']:.4f}")
    print(f"       Critical values : 1%={cr['critical_values'][0]:.2f}  "
          f"5%={cr['critical_values'][1]:.2f}  10%={cr['critical_values'][2]:.2f}")
    print(f"       Hedge ratio (β) : {cr['hedge_ratio']:.4f}")
    print(f"       Intercept  (α)  : {cr['intercept']:.4f}")
    flag = "YES  ✓" if cr["cointegrated"] else "NO   ✗"
    print(f"       Cointegrated    : {flag}")

    if not cr["cointegrated"]:
        print("\n  [WARNING] Pair is not cointegrated at the 5% level.")
        print("  Signals will still be generated — interpret results with caution.")

    # ── Step 3: Spread & Z-score ─────────────────────────────────
    print(f"\n[3/5]  Computing spread & z-score  (window = {ZSCORE_WINDOW} days)...")
    spread = compute_spread(prices, TICKER_1, TICKER_2, cr["hedge_ratio"], cr["intercept"])
    zscore = compute_zscore(spread)

    print(f"       Spread mean     : {spread.mean():.4f}")
    print(f"       Spread std      : {spread.std():.4f}")
    print(f"       Z-score range   : [{zscore.dropna().min():.2f},  {zscore.dropna().max():.2f}]")

    # ── Step 4: Signals ──────────────────────────────────────────
    print("\n[4/5]  Generating trading signals...")
    signals = generate_signals(zscore)

    long_days  = int((signals ==  1).sum())
    short_days = int((signals == -1).sum())
    flat_days  = int((signals ==  0).sum())
    print(f"       Long  days      : {long_days}")
    print(f"       Short days      : {short_days}")
    print(f"       Flat  days      : {flat_days}")

    # ── Step 5: Backtest ─────────────────────────────────────────
    print("\n[5/5]  Running backtest...")
    results_df, metrics = run_backtest(
        prices, signals, TICKER_1, TICKER_2, cr["hedge_ratio"]
    )

    pv      = results_df["portfolio_value"].dropna()
    final_v = float(pv.iloc[-1])
    pnl     = final_v - INITIAL_CAPITAL

    print("\n" + "─" * 46)
    print("  PERFORMANCE SUMMARY")
    print("─" * 46)
    print(f"  Initial Capital   : ${INITIAL_CAPITAL:>12,.0f}")
    print(f"  Final Value       : ${final_v:>12,.0f}")
    print(f"  Total P&L         : ${pnl:>+12,.0f}")
    print(f"  Total Return      : {metrics['total_return'] * 100:>+10.2f}%")
    print(f"  Annualised Sharpe : {metrics['annualized_sharpe']:>10.3f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown'] * 100:>10.2f}%")
    print(f"  Trade Signals     : {metrics['num_trades']:>10}")
    print("─" * 46)

    # ── Plot ─────────────────────────────────────────────────────
    print("\n  Generating plots  →  results.png")
    plot_results(
        prices, spread, zscore, signals,
        results_df, TICKER_1, TICKER_2
    )

    print("\nDone. ✓\n")


if __name__ == "__main__":
    main()