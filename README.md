# Statistical Arbitrage Engine — v1

A clean Python implementation of a pairs trading strategy using Engle-Granger cointegration. This is **v1 of a progressive project** — each version adds a real conceptual upgrade, not just more code.

---

## What This Does

Identifies a cointegrated pair of assets (GLD / GDX), models their long-run equilibrium relationship, and trades mean-reversion of the spread when it deviates significantly.

**Pipeline:**
```
Real price data (Yahoo Finance)
        ↓
Engle-Granger cointegration test
        ↓
OLS hedge ratio  →  spread  →  rolling z-score
        ↓
Entry/exit signals  (+1 / -1 / 0)
        ↓
Daily-bar backtest  →  Sharpe, drawdown, P&L
        ↓
4-panel results chart
```

---

## Setup

**Requirements:** Python 3.10+

```bash
# Clone and navigate into the folder
git clone <repo-url>
cd stat_arb_v1

# Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

**Output:**
- Live console report (cointegration stats, signal counts, performance summary)
- `results.png` — 4-panel chart: prices, spread, z-score, portfolio value

---

## Project Structure

```
stat_arb_v1/
├── config.py           # all parameters (pair, dates, thresholds, capital)
├── data.py             # Yahoo Finance data fetch via yfinance
├── cointegration.py    # Engle-Granger test + OLS hedge ratio
├── signals.py          # spread, rolling z-score, entry/exit signals
├── backtest.py         # daily-bar simulation, Sharpe, drawdown
├── plot.py             # 4-panel matplotlib chart
├── main.py             # entry point — runs the full pipeline
└── requirements.txt
```

---

## Key Concepts (v1)

| Concept | Implementation |
|---|---|
| Cointegration | Engle-Granger two-step test (`statsmodels.tsa.stattools.coint`) |
| Hedge ratio | OLS regression: `ticker1 = α + β·ticker2 + ε` |
| Spread | `ticker1 − β·ticker2 − α` |
| Z-score | Rolling 30-day standardisation of spread |
| Entry | `|z| > 2.0` |
| Exit | `|z| < 0.5` (from same side) |
| Sizing | Approximate dollar-neutral, no leverage |
| Look-ahead | Signal lagged 1 day before trade execution |

---

## Pair: GLD / GDX

- **GLD** — SPDR Gold ETF (tracks spot gold)
- **GDX** — VanEck Gold Miners ETF (tracks gold mining companies)

Both assets are driven by the underlying gold price, making them a canonical cointegrated pair used widely in academic and practitioner literature.

---

## Roadmap

| Version | Additions |
|---|---|
| **v1** (this) | Basic EG cointegration, z-score signals, simple backtest |
| **v2** | Proper backtesting framework, transaction costs, position sizing |
| **v3** | Kalman filter for dynamic hedge ratio, regime detection |
| **v4** | Multi-pair, risk management, production-level structure |

---

## Limitations (v1 — intentional)

- Static hedge ratio (estimated once on full sample — in-sample bias)
- No transaction costs or slippage
- Single pair only
- No regime awareness (trades through volatile periods)

These are addressed in later versions.