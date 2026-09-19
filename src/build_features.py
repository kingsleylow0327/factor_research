"""Build all factors per spec.md Section 5, from the aligned daily dataset (ticket 7).

Usage: python src/build_features.py
Input:  data/processed/aligned_daily.csv
Output: data/processed/features.csv (date, close, + every factor column below)

Every factor only ever looks backward (rolling/ewm/pct_change/diff over past days), so this
step introduces no new look-ahead risk -- that was already handled in align_data.py.

Factor definitions (documented since spec.md doesn't pin down every exact formula):

Price
  dist_ema20, dist_ema50      % distance of close from its 20/50-day EMA
  return_7d, return_30d       % close-to-close return over 7 / 30 days
  drawdown_30d                % below the trailing 30-day rolling high (<=0)
  volatility_30d              30-day rolling std dev of daily % returns
  monthly_price_position      0-100: close's position between this calendar month's
                               month-to-date high and low (resets each month, uses only
                               days up to and including today -- no look-ahead)

Derivatives
  funding_rate                daily mean funding rate, as pulled
  open_interest                level
  open_interest_change_7d     % change in OI over 7 days
  long_short_ratio             level

Macro
  cpi, pce, nfp, fed_funds_rate   actual reported levels (spec: "actual values only")
  cpi_yoy                      % change in CPI over ~365 days
  treasury_10y_change_30d      change in 10Y yield over 30 days, in percentage points
  dxy_change_30d               % change in DXY over 30 days
  (30-day window chosen for both "change" factors for consistency with Price's 30D window)

Whale
  etf_flow_total               daily net BTC spot ETF flow, US$m (NaN before 2024-01-11,
                                the Farside/spec-documented start of ETF data)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "aligned_daily.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.csv"


def add_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    df["dist_ema20"] = (df["close"] - ema20) / ema20 * 100
    df["dist_ema50"] = (df["close"] - ema50) / ema50 * 100

    df["return_7d"] = df["close"].pct_change(7) * 100
    df["return_30d"] = df["close"].pct_change(30) * 100

    rolling_high_30d = df["close"].rolling(30).max()
    df["drawdown_30d"] = (df["close"] - rolling_high_30d) / rolling_high_30d * 100

    daily_return = df["close"].pct_change()
    df["volatility_30d"] = daily_return.rolling(30).std() * 100

    month = df["date"].dt.to_period("M")
    month_high_mtd = df.groupby(month)["high"].cummax()
    month_low_mtd = df.groupby(month)["low"].cummin()
    denom = (month_high_mtd - month_low_mtd).replace(0, pd.NA)
    df["monthly_price_position"] = ((df["close"] - month_low_mtd) / denom * 100).fillna(50.0)

    return df


def add_derivatives_factors(df: pd.DataFrame) -> pd.DataFrame:
    df["open_interest_change_7d"] = df["open_interest"].pct_change(7) * 100
    # funding_rate, open_interest, long_short_ratio pass through as-is (already in aligned_daily)
    return df


def add_macro_factors(df: pd.DataFrame) -> pd.DataFrame:
    df["cpi_yoy"] = df["cpi"].pct_change(365) * 100
    df["treasury_10y_change_30d"] = df["treasury_10y"].diff(30)
    df["dxy_change_30d"] = df["dxy"].pct_change(30) * 100
    # cpi, pce, nfp, fed_funds_rate pass through as-is (already in aligned_daily)
    return df


def add_whale_factors(df: pd.DataFrame) -> pd.DataFrame:
    df["etf_flow_total"] = df["total"]
    return df


FACTOR_COLUMNS = [
    # price (risk regime, not trend -- trend factors dropped, see below)
    "volatility_30d",
    # derivatives (cost of staying positioned, not redundant with OI/long-short per ticket 11 v2 research)
    "funding_rate",
    # macro (YoY change only -- raw levels dropped, they were collinear with each other and
    # spuriously trend-matched to BTC price per ticket 11/12 findings)
    "cpi_yoy", "dxy_change_30d",
    # whale
    "etf_flow_total",
]

# Dropped for ticket 11 v2 (cut down from 19 -> 5 factors), 2026-09-19:
#   dist_ema20, dist_ema50, return_7d, return_30d, drawdown_30d, monthly_price_position
#     -- one redundant "price trend" cluster, corr 0.6-0.92 with each other, none passed
#        ticket 12's bull+bear regime check
#   cpi, pce, nfp, fed_funds_rate
#     -- raw levels, corr 0.94-0.99 with each other (Fed sets rates off CPI/PCE by definition),
#        looked significant only because they drift with BTC price over the same window
#   treasury_10y_change_30d
#     -- sign flips between bull and bear regimes (not a stable relationship)
#   open_interest, open_interest_change_7d, long_short_ratio
#     -- failed ticket 12's regime check; also correlate 0.6-0.7 with the price-trend cluster
# Kept 5 have <0.3 pairwise correlation with each other -- see conversation research, 2026-09-19.


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = add_price_factors(df)
    df = add_derivatives_factors(df)
    df = add_macro_factors(df)
    df = add_whale_factors(df)

    out = df[["date", "close"] + FACTOR_COLUMNS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(out)} rows, {len(FACTOR_COLUMNS)} factors -> {OUT_PATH}")
    print("\nNaN count per factor (expected: nonzero near the start, from warmup windows / pre-2024 ETF gap):")
    print(out[FACTOR_COLUMNS].isna().sum().to_string())


if __name__ == "__main__":
    main()
