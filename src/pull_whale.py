"""Pull BTC spot ETF daily flow data from Farside Investors (US$ millions, per-fund + total).

Farside sits behind a Cloudflare bot check that plain `requests` can't pass, so this
uses `cloudscraper` instead. Data is only available from 11 Jan 2024 onward (spot ETF launch).

Usage: python src/pull_whale.py
Output: data/raw/etf_flows.csv
  columns: date, ibit, fbtc, bitb, arkb, btco, ezbc, brrr, hodl, btcw, msbt, gbtc, btc, total
  (all values in US$ millions; "-" on the site means zero flow that day)
"""

import io
import re
import sys
from pathlib import Path

import cloudscraper
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "raw" / "etf_flows.csv"

URL = "https://farside.co.uk/bitcoin-etf-flow-all-data/"


def parse_money(val) -> float:
    """Farside formats: "123.4" positive, "(123.4)" negative, "-" zero, "1,234" thousands comma."""
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    if s == "-" or s == "":
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "")
    try:
        num = float(s)
    except ValueError:
        return float("nan")
    return -num if negative else num


def fetch_page() -> str:
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def main() -> None:
    html = fetch_page()
    tables = pd.read_html(io.StringIO(html))
    df = tables[0]

    df = df.rename(columns={"Date": "date"})
    df.columns = [str(c).lower() for c in df.columns]

    # drop the leading blank row and the trailing "Total" summary row
    df = df[df["date"].notna()]
    df = df[df["date"].astype(str).str.match(r"^\d{1,2} \w{3} \d{4}$")]

    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y").dt.date

    value_cols = [c for c in df.columns if c != "date"]
    for col in value_cols:
        df[col] = df[col].apply(parse_money)

    df = df.sort_values("date").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows ({df['date'].min()} to {df['date'].max()}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
