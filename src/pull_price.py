"""Pull BTC OHLCV (daily + monthly), full research window, from Binance (fallback: Yahoo Finance).

Usage: python src/pull_price.py
Output:
  data/raw/price.csv         (daily, columns: date, open, high, low, close, volume)
  data/raw/price_monthly.csv (monthly, same columns)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
YEARS_BACK = 3
BUFFER_DAYS = 10  # pad a bit past the exact window so downstream EMA/rolling calcs have warmup room

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# (binance interval, yfinance interval, output filename)
PULLS = [
    ("1d", "1d", "price.csv"),
    ("1M", "1mo", "price_monthly.csv"),
]


def fetch_binance(interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if not rows:
        raise RuntimeError("Binance returned no data")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
    df = df[["date", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df


def fetch_yfinance(interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download("BTC-USD", start=start.date(), end=end.date(), interval=interval, progress=False)
    data = data.reset_index()
    data.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in data.columns]
    date_col = "date" if "date" in data.columns else "datetime"
    df = data[[date_col, "open", "high", "low", "close", "volume"]].copy()
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def main() -> None:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * YEARS_BACK + BUFFER_DAYS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for binance_interval, yf_interval, filename in PULLS:
        try:
            df = fetch_binance(binance_interval, int(start.timestamp() * 1000), int(end.timestamp() * 1000))
            source = "binance"
        except Exception as exc:
            print(f"Binance fetch failed for {binance_interval} ({exc}), falling back to Yahoo Finance", file=sys.stderr)
            df = fetch_yfinance(yf_interval, start, end)
            source = "yfinance"

        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

        out_path = DATA_DIR / filename
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows ({df['date'].min()} to {df['date'].max()}) from {source} -> {out_path}")


if __name__ == "__main__":
    main()
