# ─────────────────────────────────────────────────────────
#  Statistical Arbitrage Engine — v1
#  config.py  |  All parameters in one place
# ─────────────────────────────────────────────────────────

# Pair: GLD = SPDR Gold ETF  |  GDX = VanEck Gold Miners ETF
# Classic cointegrated pair — both driven by underlying gold price.
TICKER_1 = "GLD"
TICKER_2 = "GDX"

# Historical data window
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"

# Z-score parameters
ZSCORE_WINDOW    = 30     # rolling window in trading days
ENTRY_THRESHOLD  = 2.0    # open a trade when |z| > 2.0
EXIT_THRESHOLD   = 0.5    # close a trade when |z| < 0.5

# Capital
INITIAL_CAPITAL  = 100_000   # USD