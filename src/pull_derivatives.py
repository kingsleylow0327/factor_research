"""Pull BTC derivatives data: funding rate (Binance Futures) + open interest & long/short ratio (Coinalyze).

Usage: python src/pull_derivatives.py
Output: data/raw/derivatives.csv
  columns: date, funding_rate, open_interest, long_short_ratio
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "derivatives.csv"

SYMBOL = "BTCUSDT"
YEARS_BACK = 3
BUFFER_DAYS = 10

BINANCE_FAPI_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
COINALYZE_BASE = "https://api.coinalyze.net/v1"

load_dotenv(ROOT / ".env")
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY", "")


# ---------- Funding rate (Binance, no key needed) ----------

def fetch_funding_rate(start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": SYMBOL, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        resp = requests.get(BINANCE_FAPI_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break

    if not rows:
        raise RuntimeError("Binance funding rate returned no data")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.date
    df["fundingRate"] = df["fundingRate"].astype(float)
    # multiple funding events per day (every 8h) -> daily mean
    daily = df.groupby("date", as_index=False)["fundingRate"].mean()
    daily = daily.rename(columns={"fundingRate": "funding_rate"})
    return daily


# ---------- Coinalyze (needs API key) ----------

def coinalyze_get(path: str, params: dict) -> dict:
    if not COINALYZE_API_KEY:
        raise RuntimeError("COINALYZE_API_KEY not set in .env")
    params = {**params, "api_key": COINALYZE_API_KEY}
    resp = requests.get(f"{COINALYZE_BASE}{path}", params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Coinalyze {path} failed [{resp.status_code}]: {resp.text[:500]}")
    return resp.json()


BINANCE_EXCHANGE_CODE = "A"  # from GET /exchanges: {'name': 'Binance', 'code': 'A'}


def find_binance_btc_perp_symbol() -> str:
    markets = coinalyze_get("/future-markets", {})
    candidates = [
        m for m in markets
        if str(m.get("base_asset", "")).upper() == "BTC"
        and str(m.get("exchange", "")) == BINANCE_EXCHANGE_CODE
        and str(m.get("quote_asset", "")).upper() == "USDT"
        and m.get("is_perpetual")
    ]
    if not candidates:
        raise RuntimeError(
            f"No Binance BTC USDT perpetual found in /future-markets. "
            f"Sample entries: {markets[:5]}"
        )
    return candidates[0]["symbol"]


def fetch_coinalyze_history(endpoint: str, symbol: str, start_s: int, end_s: int, value_keys) -> pd.DataFrame:
    data = coinalyze_get(
        endpoint,
        {"symbols": symbol, "interval": "daily", "from": start_s, "to": end_s},
    )
    if isinstance(data, list) and data and "history" in data[0]:
        history = data[0]["history"]
    elif isinstance(data, list):
        history = data
    else:
        raise RuntimeError(f"Unexpected Coinalyze response shape from {endpoint}: {str(data)[:500]}")

    rows = []
    for point in history:
        date = pd.to_datetime(point.get("t"), unit="s").date()
        value = None
        for key in value_keys:
            if key in point:
                value = point[key]
                break
        if value is None:
            raise RuntimeError(f"Could not find any of {value_keys} in point: {point}")
        rows.append({"date": date, "value": value})

    return pd.DataFrame(rows)


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS_BACK + BUFFER_DAYS)

    funding = fetch_funding_rate(int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    print(f"Funding rate: {len(funding)} daily rows")

    merged = funding

    if not COINALYZE_API_KEY:
        print("COINALYZE_API_KEY not set -> skipping open interest & long/short ratio", file=sys.stderr)
    else:
        try:
            symbol = find_binance_btc_perp_symbol()
            print(f"Coinalyze symbol resolved: {symbol}")

            oi = fetch_coinalyze_history(
                "/open-interest-history", symbol,
                int(start.timestamp()), int(end.timestamp()),
                value_keys=["c", "close", "value"],
            ).rename(columns={"value": "open_interest"})

            ls = fetch_coinalyze_history(
                "/long-short-ratio-history", symbol,
                int(start.timestamp()), int(end.timestamp()),
                value_keys=["r", "ratio", "c", "close", "value"],
            ).rename(columns={"value": "long_short_ratio"})

            print(f"Open interest: {len(oi)} rows, long/short ratio: {len(ls)} rows")

            merged = merged.merge(oi, on="date", how="outer")
            merged = merged.merge(ls, on="date", how="outer")
        except Exception as exc:
            print(f"Coinalyze fetch failed: {exc}", file=sys.stderr)
            print("Saving funding rate only.", file=sys.stderr)

    merged = merged.sort_values("date").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(merged)} rows ({merged['date'].min()} to {merged['date'].max()}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
