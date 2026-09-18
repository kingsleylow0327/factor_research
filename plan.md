# BTC Daily Factor Research Plan

## 1. Objective

Build a data-driven BTC factor model to identify whether BTC is relatively cheap and whether the identified market conditions can predict future BTC returns.

The initial goal is **research and validation**, not immediate live trading.

The model should determine whether a combination of market, positioning, macro, and crypto-related factors has a statistically meaningful relationship with future BTC price performance.

---

## 2. Research Period

Use approximately **3 years of historical data**.

```text
Historical Data
<-------------------------------------------------------------->

3 Years
|---------------------------------------------------------------|
                         |----------------------|
                         Training              Test
                         Period                Period
                                               Last 3 Months
```

### Training Period

Use all historical data except the most recent 3 months.

### Test / Holdout Period

Lock the **most recent 3 months** as an unseen dataset.

The model must not use any information from this period during training, parameter selection, or factor design.

---

## 3. Data Collection

Collect daily BTC data for the full 3-year period.

### Price & Market Data

* Open
* High
* Low
* Close
* Volume
* EMA20
* EMA50
* Monthly price position
* Drawdown from recent highs
* Volatility

### Derivatives Data

* Open Interest (OI)
* OI change
* Funding Rate
* Long/Short Ratio
* Long Liquidations
* Short Liquidations
* Futures basis / premium if available

### Whale / Large Player Data

Potential variables:

* Exchange netflows
* Large BTC transfers
* ETF flows
* Large transaction activity
* Exchange reserves

### Macro Data

Potential variables:

* CPI
* PCE
* NFP
* FOMC decisions
* Fed Funds Rate
* DXY
* US Treasury yields
* Other major macroeconomic events

### Crypto News

Collect major crypto-related events:

* ETF-related news
* Regulation
* Exchange events
* Hacks / security incidents
* Major protocol events
* Institutional adoption
* Major market-moving announcements

---

## 4. Feature Engineering

Convert raw data into measurable daily factors.

Examples:

```text
Price Factor
├── Distance from EMA20
├── Distance from EMA50
├── 7D Return
├── 30D Return
├── 30D Drawdown
└── Volatility

Derivatives Factor
├── OI
├── OI Change
├── Funding Rate
├── Liquidation Ratio
└── Long/Short Ratio

Whale Factor
├── Exchange Netflow
├── ETF Flow
└── Large Transaction Activity

Macro Factor
├── CPI Surprise
├── FOMC Event
├── DXY Change
└── Yield Change

News Factor
├── Event Category
├── Sentiment
└── Event Magnitude
```

Avoid relying on arbitrary thresholds initially.

---

## 5. Target Variables

The model should predict **future BTC returns**, rather than simply predicting whether BTC is currently cheap.

Calculate:

```text
Future Return 1D
Future Return 3D
Future Return 7D
Future Return 14D
Future Return 30D
Future Return 90D
```

Example:

```text
Date T
    ↓
Factors available at T
    ↓
Model Score
    ↓
BTC return from T → T+90 days
```

The primary prediction horizon is **3 months / 90 days**.

---

## 6. Training

Use the first approximately **33 months** of data for training.

```text
3 Years
├──────────────────────────────────────────┬───────────────┤
              Training Data                 Holdout Test
                                             Last 3 Months
```

The training process should investigate:

1. Which individual factors have predictive power?
2. Which factors are correlated with future BTC returns?
3. Which factors remain useful after controlling for other factors?
4. Which combinations of factors improve predictive performance?
5. Whether the relationship is stable across different market regimes.

---

## 7. Factor Analysis

Do not rely only on simple Pearson correlation.

For each factor, analyze:

### Correlation

```text
Factor → Future BTC Return
```

### Conditional Returns

For example:

```text
OI unusually high
        ↓
What is the average / median 30D return?

OI unusually high
+
Price below EMA50
        ↓
What is the average / median 30D return?
```

### Distribution

Analyze:

* Mean return
* Median return
* Standard deviation
* Win rate
* Maximum positive return
* Maximum negative return
* Number of observations

This helps determine whether a factor has useful predictive information rather than merely appearing correlated.

---

## 8. Relative Cheapness Score

Eventually combine multiple factors into a daily score.

Example:

```text
BTC Relative Cheapness Score
│
├── Price                  40%
├── Derivatives             20%
├── Whale / Institutional   20%
└── Macro / News            20%
```

The exact weights should **not be assumed to be correct**.

They should be determined or validated using the training data.

Example output:

```text
Date: 2026-06-15

Price Score:        72
Derivatives Score:  81
Whale Score:        68
Macro Score:        55

Total Score:        70
```

The score represents the model's estimate of whether BTC is relatively attractive compared with its historical conditions.

---

## 9. Holdout Test

The most recent 3 months must remain completely unseen during development.

After the model and parameters are finalized using the training period:

```text
TRAINING
~33 months
     ↓
Finalize Model
     ↓
LOCKED 3-MONTH DATA
     ↓
Generate Predictions
     ↓
Compare Predictions vs Actual BTC Returns
```

Do not modify the model based on the test results.

Otherwise, the test period is no longer a true out-of-sample test.

---

## 10. Evaluation

Evaluate whether the model successfully predicts the following 3-month period.

### Prediction Performance

Measure:

* Correlation between predicted score and future return
* Average future return by score bucket
* Median future return
* Directional accuracy
* Win rate
* Number of signals

Example:

```text
Score Range       90D Return
--------------------------------
0–20              ?
20–40             ?
40–60             ?
60–80             ?
80–100            ?
```

The key question is:

> Do higher relative-cheapness scores consistently correspond to better subsequent BTC returns?

---

## 11. Important Anti-Overfitting Rules

### No Look-Ahead Bias

A factor on day `T` can only use information that was available at or before day `T`.

For example:

```text
❌ Today's factor → uses tomorrow's price

✅ Today's factor → predicts tomorrow / future returns
```

### No Test Set Leakage

The latest 3 months must not influence:

* Feature selection
* Threshold selection
* Factor weights
* Model parameters
* Strategy rules

### Avoid Excessive Factors

Start simple.

Do not create hundreds of variables and select the ones that happen to work.

---

## 12. Final Research Question

The entire research process should answer one core question:

> **Can information available today identify BTC market conditions that have a statistically meaningful relationship with BTC returns over the following 3 months?**

If the answer is positive in the training data but fails on the locked 3-month test period, the model is likely overfitted or unstable.

If the relationship remains present in the unseen period, proceed to further validation using additional historical windows and eventually walk-forward testing.

---

## 13. Future Development

If the initial experiment shows promising results:

```text
Phase 1
3 Years Historical Data
        ↓
Train / Holdout
        ↓
3-Month Out-of-Sample Test

        ↓

Phase 2
Walk-Forward Validation

        ↓

Phase 3
Multiple Market Regimes

        ↓

Phase 4
Paper Trading

        ↓

Phase 5
Live BTC Spot Accumulation
```

The eventual strategy is intended to use the model to identify relatively attractive BTC accumulation periods rather than attempting to predict the exact BTC price.
