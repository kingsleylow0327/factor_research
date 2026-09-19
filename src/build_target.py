"""Compute the 90-day forward BTC return target for every row (ticket 8).

Usage: python src/build_target.py
Input:  data/processed/features.csv
Output: data/processed/features_with_target.csv (adds target_90d_return, usable_for_training)

Two separate reasons a row's target can be unusable:
1. target_90d_return is NaN -- there's no price 90 days later yet, so no target can be
   computed at all.
2. usable_for_training is False -- a target CAN be computed, but the 90-day-forward
   window reaches into the locked holdout period, so computing it required holdout-period
   prices. Using this row to fit the model would leak holdout information into training
   even though the row's own date is before the holdout starts. These rows keep their
   real target_90d_return value (needed later for holdout evaluation in ticket 14) but
   must never be used to fit or select the model (ticket 9+).

Holdout window placement: "most recent 3 months" is defined relative to the most recent
date with a FULLY RESOLVED target, not relative to today. A 90-day-forward target for a
date in the literal last 3 months would need price data up to 3 months in the future,
which doesn't exist yet since this is live, ongoing data collection. So the holdout is
the last 3-month window that's actually evaluable today: it ends 90 days before the last
date in the dataset, not on the last date itself. Rows newer than that (the true last ~90
days) fall in neither training nor holdout -- their targets simply aren't resolved yet.

The date index is a continuous daily calendar (confirmed no gaps), so a 90-row shift
is exactly a 90-calendar-day shift.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "features.csv"
OUT_PATH = ROOT / "data" / "processed" / "features_with_target.csv"

FORWARD_DAYS = 90
HOLDOUT_MONTHS = 3


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    gaps = df["date"].diff().dropna().unique()
    if len(gaps) != 1 or pd.Timedelta(gaps[0]) != pd.Timedelta(days=1):
        raise AssertionError(f"Expected a gapless daily index, found gaps: {gaps}")

    future_close = df["close"].shift(-FORWARD_DAYS)
    df["target_90d_return"] = (future_close / df["close"] - 1) * 100

    last_date = df["date"].max()
    last_resolved_date = last_date - pd.Timedelta(days=FORWARD_DAYS)
    holdout_start = last_resolved_date - pd.DateOffset(months=HOLDOUT_MONTHS)
    df["holdout_start_date"] = holdout_start
    df["holdout_end_date"] = last_resolved_date

    window_end = df["date"] + pd.Timedelta(days=FORWARD_DAYS)
    leaks_holdout = window_end >= holdout_start
    df["usable_for_training"] = df["target_90d_return"].notna() & ~leaks_holdout
    df["in_holdout"] = df["target_90d_return"].notna() & df["date"].between(holdout_start, last_resolved_date)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    n_no_target = df["target_90d_return"].isna().sum()
    n_usable = df["usable_for_training"].sum()
    n_holdout = df["in_holdout"].sum()

    print(f"Holdout window (targets fully resolved): {holdout_start.date()} to {last_resolved_date.date()}")
    print(f"Saved {len(df)} rows -> {OUT_PATH}")
    print(f"  no target yet (too close to today, {last_resolved_date.date()} to {last_date.date()}): {n_no_target}")
    print(f"  usable for training: {n_usable}")
    print(f"  in locked holdout: {n_holdout}")


if __name__ == "__main__":
    main()
