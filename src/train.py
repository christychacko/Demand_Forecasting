"""
End-to-end demand forecasting pipeline.

Data:   Kaggle "Store Item Demand Forecasting Challenge"
        5 years (2013-2017) of daily sales, 50 items x 10 stores.
Task:   Predict daily sales per (store, item) for a held-out final
        3-month test window, using only past data (no look-ahead).

Run:    python src/train.py
Output: outputs/metrics.json, outputs/forecast_plot.png,
        outputs/feature_importance.png
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from features import build_feature_frame, FEATURE_COLUMNS, TARGET_COLUMN
from metrics import evaluate_all

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_DAYS = 90  # last 3 months held out, mirroring the original Kaggle task


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def time_based_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test


def naive_baseline(test: pd.DataFrame, full: pd.DataFrame, fallback_mean: float) -> np.ndarray:
    """Predict today's sales = same store/item's sales from 7 days ago (vectorized merge)."""
    lookup = full[["store", "item", "date", "sales"]].copy()
    lookup["date"] = lookup["date"] + pd.Timedelta(days=7)  # shift so it aligns to future date
    lookup = lookup.rename(columns={"sales": "naive_pred"})
    merged = test.merge(lookup, on=["store", "item", "date"], how="left")
    return merged["naive_pred"].fillna(fallback_mean).values


def moving_average_baseline(test_features: pd.DataFrame) -> np.ndarray:
    """Predict = trailing 30-day rolling mean feature already computed."""
    return test_features["roll_mean_30"].values


def main():
    t0 = time.time()
    print("Loading data...")
    df = load_data()

    print("Building features (calendar + lag + rolling, no look-ahead)...")
    feat_df = build_feature_frame(df)

    train_feat, test_feat = time_based_split(feat_df, TEST_DAYS)
    _, test_raw = time_based_split(df, TEST_DAYS)  # for naive baseline lookup
    print(f"Train rows: {len(train_feat):,}  Test rows: {len(test_feat):,}")

    X_train, y_train = train_feat[FEATURE_COLUMNS], train_feat[TARGET_COLUMN]
    X_test, y_test = test_feat[FEATURE_COLUMNS], test_feat[TARGET_COLUMN]

    results = {}

    # --- Baselines ---
    print("Evaluating naive (7-day-ago) baseline...")
    naive_preds = naive_baseline(test_feat[["store", "item", "date"]], df, y_train.mean())
    results["Naive_7day"] = evaluate_all(y_test, naive_preds)

    print("Evaluating moving-average (30-day) baseline...")
    ma_preds = moving_average_baseline(test_feat)
    results["MovingAverage_30day"] = evaluate_all(y_test, ma_preds)

    # --- ML models ---
    print("Training Random Forest (lightweight - small n_estimators/depth)...")
    rf = RandomForestRegressor(
        n_estimators=40, max_depth=10, min_samples_leaf=10,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["RandomForest"] = evaluate_all(y_test, rf_preds)

    print("Training HistGradientBoosting (histogram-based GBM, LightGBM-style)...")
    gb = HistGradientBoostingRegressor(
        max_iter=250, max_depth=6, learning_rate=0.08, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    results["HistGradientBoosting"] = evaluate_all(y_test, gb_preds)

    # --- Business framing: translate WMAPE improvement into safety-stock impact ---
    best_model = min(
        ["RandomForest", "HistGradientBoosting"],
        key=lambda m: results[m]["WMAPE_%"]
    )
    baseline_wmape = results["Naive_7day"]["WMAPE_%"]
    best_wmape = results[best_model]["WMAPE_%"]
    improvement_pct = (baseline_wmape - best_wmape) / baseline_wmape * 100
    results["_summary"] = {
        "best_model": best_model,
        "wmape_improvement_over_naive_%": round(improvement_pct, 2),
        "note": (
            "Safety stock is typically sized proportional to forecast error "
            "(std dev of errors over the lead time). A WMAPE reduction of "
            f"{improvement_pct:.1f}% vs. the naive baseline implies a roughly "
            "proportional reduction in required safety stock for the same "
            "service level, holding lead-time variability constant."
        ),
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Feature importance plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=True).tail(15)
    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="#1f4e79")
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # --- Actual vs predicted plot for one representative store/item ---
    sample_store, sample_item = 1, 1
    mask = (test_feat["store"] == sample_store) & (test_feat["item"] == sample_item)
    plot_dates = test_feat.loc[mask, "date"]
    plt.figure(figsize=(11, 5))
    plt.plot(plot_dates, y_test[mask], label="Actual", color="black", linewidth=1.5)
    plt.plot(plot_dates, naive_preds[mask.values], label="Naive (7-day-ago)",
              linestyle="--", alpha=0.6)
    plt.plot(plot_dates, gb_preds[mask.values], label="HistGradientBoosting",
              color="#c00000", linewidth=1.5)
    plt.title(f"Store {sample_store}, Item {sample_item} - Actual vs. Forecast (test period)")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forecast_plot.png", dpi=150)
    plt.close()

    print(f"\nDone in {time.time()-t0:.1f}s. Results:")
    for model, m in results.items():
        if model == "_summary":
            continue
        print(f"  {model:20s} WMAPE={m['WMAPE_%']:.2f}%  MAPE={m['MAPE_%']:.2f}%  "
              f"Bias={m['Bias_%']:+.2f}%  RMSE={m['RMSE']:.2f}")
    print(f"\nBest model: {results['_summary']['best_model']}")
    print(results["_summary"]["note"])


if __name__ == "__main__":
    main()
