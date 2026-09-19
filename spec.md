# BTC Daily Factor Research — Finalized Spec (v1)

This is the locked-down version of `plan.md`, after grilling out every open decision. Anything not spelled out here inherits the original intent from `plan.md` (research goal, anti-overfitting rules, evaluation approach).

---

## 1. Objective (unchanged)

Build a data-driven BTC factor model to identify whether BTC is relatively cheap and whether current market conditions predict future BTC returns. Research and validation phase only — not live trading.

---

## 2. Environment

- **Language**: Python (pandas, numpy, scipy, statsmodels, scikit-learn)
- **Storage**: local CSV / Parquet files in this folder
- **Version control**: git (initialized)

---

## 3. Research Period

- Full window: ~3 years of daily data, ending today
- Test / holdout: most recent 3 months, completely locked (no peeking during training, factor design, or parameter selection)
- Training: everything before the holdout window

---

## 4. Data Sources (all free)

| Category | Data | Source |
|---|---|---|
| Price | OHLCV, EMA20, EMA50 | Binance API / Yahoo Finance |
| Price | Volatility | Computed: 30-day rolling std dev of daily returns |
| Price | Monthly price position | Computed: today's price position between this month's high and low, scaled 0–100 |
| Price | Drawdown from recent highs | Computed from OHLCV |
| Derivatives | Funding Rate | Binance Futures API |
| Derivatives | Open Interest | Coinalyze API (free tier, daily granularity not capped) |
| Derivatives | Long/Short Ratio | Coinalyze API |
| Macro | CPI, PCE, NFP, Fed Funds Rate, Treasury yields | FRED API — **actual reported values only**, not "surprise" vs. consensus (no clean free consensus-forecast source exists) |
| Macro | DXY (US Dollar Index) | Yahoo Finance, ticker `DX-Y.NYB` |
| Whale | BTC spot ETF flows | Farside Investors (free) — data only available from Jan 2024 onward; exact handling of the pre-2024 gap decided during factor-engineering, not now |
| News | Major dated events (ETF approval, hacks, FOMC surprises, etc.) | Manual list, built later — not part of the v1 quantitative model |

**Explicitly out of scope for v1**: futures basis/premium, liquidation volume, on-chain exchange netflows/reserves (all require paid data — revisit later if the simple model shows promise).

---

## 5. Feature Engineering

```text
Price Factor
├── Distance from EMA20
├── Distance from EMA50
├── 7D / 30D Return
├── 30D Drawdown
├── Volatility (30D rolling std dev of daily returns)
└── Monthly Price Position (0–100 within month's high/low range)

Derivatives Factor
├── Funding Rate
├── Open Interest (level + change)
└── Long/Short Ratio

Macro Factor
├── CPI (actual value / YoY change)
├── PCE, NFP (actual values)
├── Fed Funds Rate
├── Treasury Yield changes
└── DXY change

Whale Factor
└── ETF Flow (usable date range TBD during factor-engineering)
```

News/sentiment factors are deferred — not built in v1.

---

## 6. Target Variable

- **Primary and only modeled horizon: 90-day forward BTC return.**
- Other horizons (1D/3D/7D/14D/30D) from the original plan are **not separately modeled** — they may be computed as diagnostic sanity checks only (e.g. "does the signal also show up at 30D?"), never as separate targets.

---

## 7. Modeling Approach

Run three regression variants on the training data, predicting 90D forward return from all factors:

1. **OLS** — plain multivariate regression, baseline
2. **Ridge** — handles factors that move together (e.g. OI and funding rate) without breaking
3. **Lasso** — automatically zeroes out weak factors (matches the plan's "avoid excessive factors" rule)

**Selection method**: time-series cross-validation within the training period only (expanding-window train/test slices). Whichever of the three generalizes best on held-out training folds becomes the final model. The locked 3-month holdout is never touched during this comparison.

Regression coefficients double as factor weights — this satisfies the plan's requirement that weights be "determined using training data," and that factors be checked "after controlling for other factors."

---

## 8. Regime Stability Check

Split the training period into **bull vs. bear** using price vs. its 200-day moving average (above = bull, below = bear). Check whether factor relationships hold up in both regimes.

---

## 9. Statistical Significance Bar

A factor is only considered "meaningful" if **both** hold:

1. **p < 0.05** on its regression coefficient
2. **Effect size check**: the average 90D return gap between "factor high" days and "factor low" days must exceed one month of BTC's normal historical volatility

(p-value alone is not trusted here — daily data with overlapping 90-day forward windows is autocorrelated, which makes p-values look better than they are.)

---

## 10. Output: The Score

- **One unified 0–100 score per day**, built from the model's predicted 90D return, converted via **expanding-window percentile rank** (today's prediction ranked against every prior day's prediction — self-normalizing, can't go out of range even as new extremes appear).
- **Plus a factor-contribution breakdown**: each factor's contribution to the score (coefficient × today's value), grouped by category (Price / Derivatives / Macro / Whale), so it's clear *why* the score is what it is on any given day — not just the number.

This replaces the original plan's fixed-percentage category weighting (40/20/20/20) — that scheme assumed hand-set weights; the regression now determines weighting empirically.

---

## 11. Holdout Test & Evaluation (unchanged from plan.md)

- Finalize model on training data only, then run once against the locked 3-month holdout
- No changes to the model after seeing holdout results
- Evaluate: correlation between score and future return, average/median return by score bucket, directional accuracy, win rate

---

## 12. Anti-Overfitting Rules (unchanged from plan.md)

- No look-ahead bias — a factor on day T uses only information available at or before T
- No test-set leakage — the holdout period never influences feature selection, thresholds, weights, or parameters
- Avoid excessive factors — Lasso's automatic feature selection reinforces this

---

## 13. Deferred to Later Phases

- News/sentiment factor (manual event list)
- Whale/on-chain factors beyond ETF flows (CryptoQuant etc., paid)
- Futures basis/premium, liquidation volume
- CPI "surprise" vs. consensus (would require paid data)
- Walk-forward validation, multi-regime testing, paper trading, live accumulation (per plan.md Section 13)
