"""Merge all raw data sources onto one daily date index (ticket 6).

Usage: python src/align_data.py
Input:  data/raw/price.csv, derivatives.csv, macro.csv, etf_flows.csv
Output: data/processed/aligned_daily.csv

Master index: every calendar day covered by price.csv (BTC trades 7 days/week, so this
is also the widest usable window -- everything else is merged onto it).

How each source is aligned (this is where "no look-ahead" actually gets enforced):

- price, derivatives (funding_rate/open_interest/long_short_ratio): already ~daily.
  Forward-filled for the rare missing day -- these are same-day market data, no lag.
- macro daily columns (treasury_10y, dxy): only exist on trading days (no weekend
  Treasury/FX quotes). Forward-filled -- same-day market data, no lag concern.
- macro monthly columns (cpi, pce, nfp, fed_funds_rate): reference a past period but
  aren't PUBLISHED until weeks later (e.g. September CPI is dated 2023-09-01 but wasn't
  released until 2023-10-12 -- see pull_macro.py). Using an as-of-date merge on each
  column's own *_release_date, so day D only ever sees values that were actually public
  knowledge by day D. This is the main look-ahead trap in this dataset -- naive
  forward-fill on the reference date would leak ~1-6 weeks of future information.
- etf flows: NaN before 2024-01-11 (ETFs didn't exist yet -- left as a flagged gap for
  ticket 7 to decide how to handle). From 2024-01-11 onward, missing days (weekends/
  holidays, ETFs don't trade) are filled with 0 flow -- a flow, not a level, so
  forward-filling the last traded day's value would be wrong (it would imply repeated
  phantom inflows); 0 correctly says "no trading, no flow."
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "aligned_daily.csv"

MONTHLY_MACRO_COLS = ["cpi", "pce", "nfp", "fed_funds_rate"]
DAILY_MACRO_COLS = ["treasury_10y", "dxy"]
ETF_START_DATE = pd.Timestamp("2024-01-11")


def load(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} -- run the corresponding pull_*.py script first")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def asof_merge_monthly_macro(master: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """As-of merge each monthly macro column on ITS OWN release date, not the reference date."""
    result = master.copy()
    for col in MONTHLY_MACRO_COLS:
        release_col = f"{col}_release_date"
        sub = macro[[release_col, col]].dropna(subset=[release_col, col]).copy()
        sub[release_col] = pd.to_datetime(sub[release_col])
        sub = sub.sort_values(release_col).rename(columns={release_col: "date"})

        merged = pd.merge_asof(
            master[["date"]].sort_values("date"),
            sub,
            on="date",
            direction="backward",  # only ever look at releases <= this date
        )
        result[col] = merged[col].values
    return result


def main() -> None:
    price = load("price.csv")
    derivatives = load("derivatives.csv")
    macro = load("macro.csv")
    etf = load("etf_flows.csv")

    start, end = price["date"].min(), price["date"].max()
    master = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    print(f"Master daily index: {start.date()} to {end.date()} ({len(master)} days)")

    # price + derivatives: same-day data, straight merge + forward-fill
    df = master.merge(price, on="date", how="left")
    df = df.merge(derivatives, on="date", how="left")
    price_cols = [c for c in price.columns if c != "date"]
    deriv_cols = [c for c in derivatives.columns if c != "date"]
    df[price_cols + deriv_cols] = df[price_cols + deriv_cols].ffill()

    # macro: daily columns straight-merge + ffill; monthly columns as-of merge on release date
    daily_macro = macro[["date"] + DAILY_MACRO_COLS]
    df = df.merge(daily_macro, on="date", how="left")
    df[DAILY_MACRO_COLS] = df[DAILY_MACRO_COLS].ffill()
    df = asof_merge_monthly_macro(df, macro)

    # etf flows: 0-fill non-trading days from launch onward, NaN before launch
    etf_cols = [c for c in etf.columns if c != "date"]
    df = df.merge(etf, on="date", how="left")
    post_launch = df["date"] >= ETF_START_DATE
    df.loc[post_launch, etf_cols] = df.loc[post_launch, etf_cols].fillna(0.0)

    # sanity check: every day a monthly macro value changes must be an actual release date
    for col in MONTHLY_MACRO_COLS:
        release_dates = set(pd.to_datetime(macro[f"{col}_release_date"]).dropna())
        changed = df["date"][df[col].ne(df[col].shift()) & df[col].notna()]
        bad = set(changed) - release_dates
        if bad:
            raise AssertionError(f"Look-ahead leak in '{col}': value changed on non-release date(s) {sorted(bad)[:5]}")
    print("No-look-ahead check passed: monthly macro values only ever change on their real release date")

    n_leading_nan = df[price_cols[0]].isna().sum() if price_cols else 0
    if n_leading_nan:
        print(f"Note: {n_leading_nan} leading day(s) still NaN for '{price_cols[0]}' (before first data point)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows, {len(df.columns)} columns -> {OUT_PATH}")
    print(f"ETF columns NaN before {ETF_START_DATE.date()}: {df.loc[~post_launch, 'total'].isna().all()}")


if __name__ == "__main__":
    main()
