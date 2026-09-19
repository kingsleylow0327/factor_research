"""Model training & selection: OLS vs Ridge vs Lasso, via expanding-window time-series CV (ticket 11).

Usage: python src/train_model.py
Input:  data/processed/train.csv (holdout_LOCKED.csv is never read here -- ticket 9 rule)
Output: data/processed/model_comparison.csv   -- per-model, per-fold CV scores
        data/processed/model_weights.csv      -- final chosen model's factor weights

Rows with a NaN in ANY factor are dropped (complete-case analysis) -- a joint regression
needs every predictor present. This mostly costs the cpi_yoy 365-day warmup and the
pre-2024 ETF gap, shrinking 832 training rows down to a smaller complete-case set.

Ridge/Lasso are scale-sensitive (the regularization penalty treats all coefficients the
same regardless of the factor's natural units), so every fold standardizes X using a
scaler fit on that fold's training slice only. For the final chosen model, coefficients
are reported in standardized-feature space; model_weights.csv also saves each factor's
mean/std from the scaler, so ticket 13's scoring pipeline can standardize new values the
same way before multiplying by these coefficients.

Model comparison: expanding-window TimeSeriesSplit (5 folds) over the chronologically
sorted training set. Ridge/Lasso each pick their own regularization strength via their
built-in CV (RidgeCV/LassoCV) using only that fold's training slice -- no peeking at the
fold's test slice, and never at the locked holdout. Average out-of-fold R^2 across the 5
folds decides the winner.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
COMPARISON_PATH = ROOT / "data" / "processed" / "model_comparison.csv"
WEIGHTS_PATH = ROOT / "data" / "processed" / "model_weights.csv"

TARGET_COL = "target_90d_return"
NON_FACTOR_COLS = {
    "date", "close", TARGET_COL,
    "holdout_start_date", "holdout_end_date", "usable_for_training", "in_holdout",
}
N_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
LASSO_ALPHAS = np.logspace(-3, 2, 25)


def fit_predict(model_name: str, X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_name == "OLS":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = RidgeCV(alphas=RIDGE_ALPHAS)
    elif model_name == "Lasso":
        model = LassoCV(alphas=LASSO_ALPHAS, max_iter=20000)
    else:
        raise ValueError(model_name)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    return model, scaler, preds


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    factor_cols = [c for c in df.columns if c not in NON_FACTOR_COLS]
    complete = df.dropna(subset=factor_cols + [TARGET_COL]).reset_index(drop=True)
    print(f"Complete-case training rows: {len(complete)} (of {len(df)}), "
          f"{complete['date'].min().date()} to {complete['date'].max().date()}")

    X = complete[factor_cols].values
    y = complete[TARGET_COL].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows = []
    for model_name in ["OLS", "Ridge", "Lasso"]:
        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            _, _, preds = fit_predict(model_name, X[train_idx], y[train_idx], X[test_idx])
            r2 = r2_score(y[test_idx], preds)
            rmse = mean_squared_error(y[test_idx], preds) ** 0.5
            rows.append({
                "model": model_name, "fold": fold_i,
                "train_n": len(train_idx), "test_n": len(test_idx),
                "r2": r2, "rmse": rmse,
            })

    comparison = pd.DataFrame(rows)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    summary = comparison.groupby("model")[["r2", "rmse"]].mean().sort_values("r2", ascending=False)
    print(f"\nSaved per-fold CV results -> {COMPARISON_PATH}")
    print("\nMean out-of-fold performance across folds:")
    print(summary.to_string())

    best_model_name = summary.index[0]
    print(f"\nSelected model: {best_model_name} (highest mean out-of-fold R^2)")

    # refit the selected model on ALL complete-case training data
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    if best_model_name == "OLS":
        final_model = LinearRegression()
    elif best_model_name == "Ridge":
        final_model = RidgeCV(alphas=RIDGE_ALPHAS)
    else:
        final_model = LassoCV(alphas=LASSO_ALPHAS, max_iter=20000)
    final_model.fit(X_s, y)

    weights = pd.DataFrame({
        "factor": factor_cols,
        "coefficient": final_model.coef_,
        "scaler_mean": scaler.mean_,
        "scaler_std": scaler.scale_,
    })
    weights["intercept"] = final_model.intercept_
    weights["selected_model"] = best_model_name
    weights = weights.sort_values("coefficient", key=abs, ascending=False)

    weights.to_csv(WEIGHTS_PATH, index=False)
    print(f"Saved final factor weights -> {WEIGHTS_PATH}\n")
    print(weights[["factor", "coefficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
