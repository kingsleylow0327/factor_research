"""Pull macro data: CPI, PCE, NFP, Fed Funds Rate, 10Y Treasury yield (FRED) + DXY (Yahoo Finance).

Usage: python src/pull_macro.py
Output: data/raw/macro.csv
  columns: date, cpi, cpi_release_date, pce, pce_release_date, nfp, nfp_release_date,
           fed_funds_rate, fed_funds_rate_release_date, treasury_10y, dxy
  Raw, unaligned: each column only has values on its own release/trading days.
  Forward-filling to a daily index happens later, in the alignment step (ticket 6).

IMPORTANT (no-look-ahead): for the four monthly series (CPI, PCE, NFP, Fed Funds Rate),
`date` is the reference period (e.g. CPI for January is dated 2024-01-01), but that number
isn't actually published until roughly 2-6 weeks later. Using `date` as the availability
date would leak future information into the model. So each monthly series also carries a
`*_release_date` column (FRED's first-release vintage date via output_type=4) -- the
alignment step must forward-fill using *_release_date, not `date`.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "macro.csv"

YEARS_BACK = 3
BUFFER_DAYS = 10

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# monthly series, released weeks after the reference period -> pull first-release vintage
# (output_type=4) so we get both the value and its true publish date, with no later revisions
MONTHLY_FRED_SERIES = {
    "CPIAUCSL": "cpi",              # CPI, all urban consumers, seasonally adjusted, monthly
    "PCEPI": "pce",                 # PCE price index, monthly
    "PAYEMS": "nfp",                # total nonfarm payrolls, monthly
    "FEDFUNDS": "fed_funds_rate",   # effective federal funds rate, monthly
}

# daily series, published same/next business day -> no release-lag concern
DAILY_FRED_SERIES = {
    "DGS10": "treasury_10y",        # 10-year treasury constant maturity yield, daily
}

load_dotenv(ROOT / ".env")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# FRED convention for "all vintages ever published"
ALFRED_REALTIME_START = "1776-07-04"
ALFRED_REALTIME_END = "9999-12-31"


def fetch_fred_observations(series_id: str, start: datetime, end: datetime, output_type: int = 1) -> list:
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not set in .env")
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start.strftime("%Y-%m-%d"),
        "observation_end": end.strftime("%Y-%m-%d"),
        "output_type": output_type,
    }
    if output_type != 1:
        params["realtime_start"] = ALFRED_REALTIME_START
        params["realtime_end"] = ALFRED_REALTIME_END
    resp = requests.get(FRED_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"FRED {series_id} failed [{resp.status_code}]: {resp.text[:300]}")
    obs = resp.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"FRED {series_id} returned no observations")
    return obs


def fetch_fred_daily_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    obs = fetch_fred_observations(series_id, start, end, output_type=1)
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.dropna(subset=["value"])
    return df


def fetch_fred_monthly_series(series_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    """First-release value + true publish date for each reference period (no revisions, no lag leakage)."""
    obs = fetch_fred_observations(series_id, start, end, output_type=4)
    df = pd.DataFrame(obs)[["date", "realtime_start", "value"]]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["realtime_start"] = pd.to_datetime(df["realtime_start"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    return df


def fetch_dxy(start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download("DX-Y.NYB", start=start.date(), end=end.date(), interval="1d", progress=False)
    data = data.reset_index()
    data.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in data.columns]
    df = data[["date", "close"]].rename(columns={"close": "dxy"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS_BACK + BUFFER_DAYS)

    merged = None

    for series_id, col_name in MONTHLY_FRED_SERIES.items():
        release_col = f"{col_name}_release_date"
        try:
            df = fetch_fred_monthly_series(series_id, start, end)
            df = df.rename(columns={"value": col_name, "realtime_start": release_col})
            print(f"{series_id} -> {col_name}: {len(df)} observations (first-release vintage)")
        except Exception as exc:
            print(f"FRED {series_id} failed: {exc}", file=sys.stderr)
            df = pd.DataFrame(columns=["date", col_name, release_col])

        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    for series_id, col_name in DAILY_FRED_SERIES.items():
        try:
            df = fetch_fred_daily_series(series_id, start, end).rename(columns={"value": col_name})
            print(f"{series_id} -> {col_name}: {len(df)} observations")
        except Exception as exc:
            print(f"FRED {series_id} failed: {exc}", file=sys.stderr)
            df = pd.DataFrame(columns=["date", col_name])

        merged = df if merged is None else merged.merge(df, on="date", how="outer")

    try:
        dxy = fetch_dxy(start, end)
        print(f"DXY: {len(dxy)} rows")
    except Exception as exc:
        print(f"DXY fetch failed: {exc}", file=sys.stderr)
        dxy = pd.DataFrame(columns=["date", "dxy"])

    merged = merged.merge(dxy, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(merged)} rows ({merged['date'].min()} to {merged['date'].max()}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
