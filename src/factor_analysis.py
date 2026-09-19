"""Factor analysis on training data only (ticket 10).

Usage: python src/factor_analysis.py
Input:  data/processed/train.csv  (holdout_LOCKED.csv is never read here -- ticket 9 rule)
Output: data/processed/factor_analysis.csv

For each factor:
- Pearson correlation to target_90d_return, with its p-value (univariate screen -- the
  "after controlling for other factors" check happens later, in ticket 11's regression).
- Conditional return analysis: split rows into the top quartile ("high") and bottom
  quartile ("low") of the factor's value, then report mean/median/std/win rate/N of
  target_90d_return within each group. Quartiles (not median-split) chosen to give
  cleaner separation between the two conditions.
- effect_size = high_mean - low_mean, compared against one month of BTC's normal
  historical volatility (daily training-period return std, scaled to a 30-day horizon
  via sqrt(30), expressed as a % return) -- this is the spec.md Section 9 effect-size bar.
- passes_significance = (pearson_p < 0.05) AND (abs(effect_size) > monthly_vol_threshold),
  per spec.md Section 9's two-part significance bar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
OUT_PATH = ROOT / "data" / "processed" / "factor_analysis.csv"

TARGET_COL = "target_90d_return"
NON_FACTOR_COLS = {
    "date", "close", TARGET_COL,
    "holdout_start_date", "holdout_end_date", "usable_for_training", "in_holdout",
}
QUANTILE = 0.25  # top/bottom quartile


def monthly_vol_threshold(df: pd.DataFrame) -> float:
    daily_return = df["close"].pct_change()
    daily_std = daily_return.std()
    return daily_std * np.sqrt(30) * 100  # % return, scaled to a ~30-day horizon


def group_stats(returns: pd.Series) -> dict:
    return {
        "mean": returns.mean(),
        "median": returns.median(),
        "std": returns.std(),
        "win_rate": (returns > 0).mean() * 100,
        "n": len(returns),
    }


def analyze_factor(df: pd.DataFrame, factor: str, vol_threshold: float) -> dict:
    sub = df[[factor, TARGET_COL]].dropna()
    if len(sub) < 20:
        return {"factor": factor, "n": len(sub), "note": "too few observations, skipped"}

    r, p = stats.pearsonr(sub[factor], sub[TARGET_COL])

    lo_cut = sub[factor].quantile(QUANTILE)
    hi_cut = sub[factor].quantile(1 - QUANTILE)
    low = sub[sub[factor] <= lo_cut][TARGET_COL]
    high = sub[sub[factor] >= hi_cut][TARGET_COL]

    high_stats = group_stats(high)
    low_stats = group_stats(low)
    effect_size = high_stats["mean"] - low_stats["mean"]

    passes = bool(p < 0.05 and abs(effect_size) > vol_threshold)

    return {
        "factor": factor,
        "n": len(sub),
        "pearson_r": r,
        "pearson_p": p,
        "high_mean": high_stats["mean"], "high_median": high_stats["median"],
        "high_std": high_stats["std"], "high_win_rate": high_stats["win_rate"], "high_n": high_stats["n"],
        "low_mean": low_stats["mean"], "low_median": low_stats["median"],
        "low_std": low_stats["std"], "low_win_rate": low_stats["win_rate"], "low_n": low_stats["n"],
        "effect_size": effect_size,
        "monthly_vol_threshold": vol_threshold,
        "passes_significance": passes,
    }


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    factor_cols = [c for c in df.columns if c not in NON_FACTOR_COLS]
    vol_threshold = monthly_vol_threshold(df)

    results = [analyze_factor(df, f, vol_threshold) for f in factor_cols]
    out = pd.DataFrame(results)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Training rows: {len(df)}, monthly vol threshold: {vol_threshold:.2f}%")
    print(f"Saved {len(out)} factor rows -> {OUT_PATH}\n")

    passed = out[out.get("passes_significance", False) == True]
    print(f"Factors passing significance bar ({len(passed)}/{len(out)}):")
    if len(passed):
        print(passed[["factor", "pearson_r", "pearson_p", "effect_size"]].to_string(index=False))
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
