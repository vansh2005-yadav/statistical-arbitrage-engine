# ─────────────────────────────────────────────────────────
#  Statistical Arbitrage Engine — v2
#  config.py  |  All parameters in one place
#
#  New in v2
#  ---------
#  TRAIN_RATIO      : walk-forward split — hedge ratio is estimated on
#                     training data only; all reported performance is
#                     genuinely out-of-sample.
#  STOP_LOSS_Z      : cut the position immediately if z moves this far
#                     against the trade (spread diverging, not reverting).
#  TRANSACTION_COST : realistic cost per leg per trade (one-way).
# ─────────────────────────────────────────────────────────

# Pair: GLD = SPDR Gold ETF  |  GDX = VanEck Gold Miners ETF
TICKER_1 = "GLD"
TICKER_2 = "GDX"

# Historical data window
START_DATE = "2018-01-01"
END_DATE   = "2023-12-31"

# ── Walk-forward split ──────────────────────────────────────────────
TRAIN_RATIO = 0.60        # 60 % in-sample (estimation) | 40 % out-of-sample

# ── Z-score parameters ──────────────────────────────────────────────
ZSCORE_WINDOW    = 30     # rolling window in trading days
ENTRY_THRESHOLD  = 2.0    # open trade when |z| > 2.0
EXIT_THRESHOLD   = 0.5    # close trade when |z| < 0.5
STOP_LOSS_Z      = 3.5    # NEW: stop out if z exceeds this (spread diverging)

# ── Transaction costs ────────────────────────────────────────────────
# 10 basis points (0.10 %) per leg, one-way.
# Two legs per trade → effective round-trip cost ≈ 0.40 % per complete trade.
TRANSACTION_COST = 0.001  # NEW: realistic for liquid ETFs

# ── Capital ─────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000