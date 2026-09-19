"""Regime stability check: do factor relationships hold in both bull and bear markets? (ticket 12)

Usage: python src/regime_check.py
Input:  data/processed/train.csv (holdout_LOCKED.csv is never read here -- ticket 9 rule)
Output: data/processed/regime_check.csv

Regime definition: bull = close above its 200-day moving average, bear = below (spec.md
Section 8). The first 199 rows have no 200-day MA yet and are excluded from this check.

Reuses ticket 10's exact per-factor method (Pearson r/p, top/bottom-quartile effect size)
but applied separately within each regime's rows. The monthly-volatility effect-size
threshold is kept as ONE fixed number (computed on the full training set, same as ticket
10) rather than recomputed per regime, so bull and bear are compared on the same bar
rather than bear's naturally higher volatility making its own bar easier to clear.

A factor is "stable" if it passes the ticket 10 significance bar in BOTH regimes AND its
correlation has the same sign in both -- i.e. the relationship isn't just an artifact of
one regime (or, worse, flips direction between them).
"""

from pathlib import Path

import pandas as pd

from factor_analysis import NON_FACTOR_COLS, TARGET_COL, analyze_factor, monthly_vol_threshold

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
OUT_PATH = ROOT / "data" / "processed" / "regime_check.csv"

MA_WINDOW = 200


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["ma_200"] = df["close"].rolling(MA_WINDOW).mean()
    df = df.dropna(subset=["ma_200"]).reset_index(drop=True)
    df["regime"] = pd.Series(["bull" if c > m else "bear" for c, m in zip(df["close"], df["ma_200"])])

    print(f"Rows with a 200-day MA: {len(df)}")
    print(df["regime"].value_counts().to_string())

    factor_cols = [c for c in df.columns if c not in NON_FACTOR_COLS and c not in ("ma_200", "regime")]
    # keep the same effect-size bar as ticket 10 (full training set), for a fair bull-vs-bear comparison
    vol_threshold = monthly_vol_threshold(df)

    bull = df[df["regime"] == "bull"]
    bear = df[df["regime"] == "bear"]

    rows = []
    for factor in factor_cols:
        bull_res = analyze_factor(bull, factor, vol_threshold)
        bear_res = analyze_factor(bear, factor, vol_threshold)

        r_bull = bull_res.get("pearson_r")
        r_bear = bear_res.get("pearson_r")
        same_sign = (r_bull is not None and r_bear is not None
                     and pd.notna(r_bull) and pd.notna(r_bear)
                     and (r_bull > 0) == (r_bear > 0))
        stable = bool(bull_res.get("passes_significance") and bear_res.get("passes_significance") and same_sign)

        rows.append({
            "factor": factor,
            "bull_n": bull_res.get("n"), "bull_r": r_bull, "bull_p": bull_res.get("pearson_p"),
            "bull_effect_size": bull_res.get("effect_size"), "bull_passes": bull_res.get("passes_significance"),
            "bear_n": bear_res.get("n"), "bear_r": r_bear, "bear_p": bear_res.get("pearson_p"),
            "bear_effect_size": bear_res.get("effect_size"), "bear_passes": bear_res.get("passes_significance"),
            "same_sign": same_sign,
            "stable_in_both_regimes": stable,
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(out)} factor rows -> {OUT_PATH}\n")
    stable = out[out["stable_in_both_regimes"]]
    print(f"Factors stable in both regimes ({len(stable)}/{len(out)}):")
    if len(stable):
        print(stable[["factor", "bull_r", "bear_r"]].to_string(index=False))
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
