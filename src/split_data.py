"""Split into training and locked holdout sets (ticket 9).

Usage: python src/split_data.py
Input:  data/processed/features_with_target.csv
Output: data/processed/train.csv    -- only usable_for_training rows, use freely
        data/processed/holdout_LOCKED.csv  -- the locked 3-month test set

ENFORCEMENT: everything from here through ticket 12 (factor analysis, model training/
selection, regime stability check) must only ever read train.csv. holdout_LOCKED.csv is
named the way it is on purpose -- it is not to be opened, inspected, or referenced again
until ticket 13/14 (holdout evaluation). If a script needs "all the data" for some other
reason, that's a sign it's about to touch the holdout and should stop and ask first.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "features_with_target.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
HOLDOUT_PATH = ROOT / "data" / "processed" / "holdout_LOCKED.csv"


def main() -> None:
    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    train = df[df["usable_for_training"]].reset_index(drop=True)
    holdout = df[df["in_holdout"]].reset_index(drop=True)

    train.to_csv(TRAIN_PATH, index=False)
    holdout.to_csv(HOLDOUT_PATH, index=False)

    print(f"Train:   {len(train)} rows, {train['date'].min().date()} to {train['date'].max().date()} -> {TRAIN_PATH}")
    print(f"Holdout: {len(holdout)} rows, {holdout['date'].min().date()} to {holdout['date'].max().date()} -> {HOLDOUT_PATH}")
    print("\nHoldout is locked. Tickets 10-12 must only read train.csv.")


if __name__ == "__main__":
    main()
