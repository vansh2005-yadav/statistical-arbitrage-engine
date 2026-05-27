# Statistical Arbitrage Engine — v2

Pairs trading strategy on GLD/GDX, built in Python. This is the second version of a progressive project — each version fixes something real that the previous one got wrong.

---

## What this does

Takes two ETFs that historically move together (GLD and GDX — both driven by gold prices), models the relationship between them, and trades when they drift too far apart on the assumption they'll revert back.

The core idea: if GLD gets expensive relative to GDX, short GLD and long GDX. Wait for the gap to close. Exit.

---

## What changed from v1

v1 had a fundamental problem — the hedge ratio (how many units of GDX offset one unit of GLD) was estimated on the full dataset, then the strategy was tested on that same dataset. That's not a real backtest. The model was essentially fitted to its own test.

v2 fixes this properly:

**Walk-forward validation** — the hedge ratio is now estimated on the first 60% of data only. All performance numbers come from the remaining 40% that the model never saw during calibration. This is the minimum bar for taking a backtest seriously.

**Transaction costs** — v1 assumed you could trade for free. v2 deducts 10 basis points per leg on every entry and exit, which is realistic for liquid ETFs. This hits overtrading strategies hard and forces the signals to actually earn their keep.

**Stop-loss** — cointegration is a long-run property and it can break temporarily. If the spread keeps moving against the position past z = ±3.5 instead of reverting, v2 cuts the loss immediately rather than holding and hoping.

---

## How to run

```bash
pip install -r requirements.txt
python main.py
```

You'll get a console summary showing training-period performance vs out-of-sample performance side by side, and a `results.png` with the train/test boundary clearly marked on all four panels.

---

## File structure

```
├── config.py           — all parameters in one place
├── data.py             — price fetch + train/test split
├── cointegration.py    — Engle-Granger test, OLS hedge ratio
├── signals.py          — spread, z-score, entry/exit/stop-loss signals
├── backtest.py         — simulation with transaction costs, period metrics
├── plot.py             — 4-panel chart with train/OOS split marked
└── main.py             — runs the full pipeline
```

---

## Parameters

Everything is in `config.py`. The ones that matter most:

| Parameter | Value | Why |
|---|---|---|
| `TRAIN_RATIO` | 0.60 | 60% for estimation, 40% for real testing |
| `ENTRY_THRESHOLD` | 2.0 | Enter when spread is 2 standard deviations off |
| `EXIT_THRESHOLD` | 0.5 | Exit when it's mostly reverted |
| `STOP_LOSS_Z` | 3.5 | Cut if it keeps going the wrong way |
| `TRANSACTION_COST` | 0.001 | 10 bps per leg — realistic for ETFs |

---

## Versions

| | |
|---|---|
| v1 | Basic cointegration, z-score signals, simple backtest |
| v2 (this) | Walk-forward validation, transaction costs, stop-loss |
| v3 | Kalman filter for dynamic hedge ratio, regime detection |
| v4 | Multi-pair, portfolio-level risk, production structure |