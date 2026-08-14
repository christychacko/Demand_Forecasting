"""
M5 Walmart Demand Forecasting Pipeline
========================================
Place calendar.csv, sales_train_validation.csv, sell_prices.csv in DATA_DIR.
(sales_train_evaluation.csv / sample_submission.csv not needed for this pipeline.)

pip install pandas numpy lightgbm scikit-learn matplotlib
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older sklearn
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5
import matplotlib.pyplot as plt

DATA_DIR = "src\M5_forecasting"          # <-- point this at your folder

N_ITEMS  = 50                # subset of SKUs for fast iteration; set None for all ~30k
FORECAST_HORIZON = 28        # days to hold out for validation

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data():
    cal = pd.read_csv(f"{DATA_DIR}/calendar.csv", parse_dates=["date"])
    sales = pd.read_csv(f"{DATA_DIR}/sales_train_validation.csv")
    prices = pd.read_csv(f"{DATA_DIR}/sell_prices.csv")
    return cal, sales, prices

# ---------------------------------------------------------------------------
# 2. RESHAPE wide -> long, merge calendar + prices
# ---------------------------------------------------------------------------
def to_long(cal, sales, prices, n_items=N_ITEMS):
    if n_items:
        sales = sales.sample(n=n_items, random_state=42).reset_index(drop=True)

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    long_df = sales.melt(id_vars=id_cols, value_vars=day_cols,
                          var_name="d", value_name="sales")

    long_df = long_df.merge(cal[["d", "date", "wm_yr_wk", "wday", "month", "year",
                                  "event_name_1", "event_type_1",
                                  "snap_CA", "snap_TX", "snap_WI"]],
                             on="d", how="left")

    long_df = long_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_df = long_df.sort_values(["id", "date"]).reset_index(drop=True)
    return long_df

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def add_features(df):
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["has_event"] = df["event_name_1"].notna().astype(int)

    snap_map = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    df["snap"] = df.apply(lambda r: r[snap_map[r["state_id"]]], axis=1)

    g = df.groupby("id")["sales"]
    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = g.shift(lag)
    for win in [7, 30]:
        df[f"roll_mean_{win}"] = g.shift(1).rolling(win).mean().reset_index(0, drop=True)
        df[f"roll_std_{win}"] = g.shift(1).rolling(win).std().reset_index(0, drop=True)

    df["sell_price"] = df.groupby("id")["sell_price"].ffill().bfill()

    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[c] = df[c].astype("category").cat.codes

    df = df.dropna(subset=["lag_28"]).reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# 4. BASELINE: naive "same day last week"
# ---------------------------------------------------------------------------
def naive_baseline(df):
    df["naive_pred"] = df["lag_7"]
    return df

# ---------------------------------------------------------------------------
# 5. TRAIN/TEST SPLIT (time-based, last FORECAST_HORIZON days = test)
# ---------------------------------------------------------------------------
def split(df):
    cutoff = df["date"].max() - pd.Timedelta(days=FORECAST_HORIZON)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test

FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id",
            "dow", "is_weekend", "has_event", "snap", "sell_price",
            "lag_7", "lag_14", "lag_28",
            "roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30",
            "month", "year"]

# ---------------------------------------------------------------------------
# 6. TRAIN LightGBM
# ---------------------------------------------------------------------------
def train_lgbm(train, test):
    train_set = lgb.Dataset(train[FEATURES], label=train["sales"])
    val_set = lgb.Dataset(test[FEATURES], label=test["sales"], reference=train_set)

    params = {
        "objective": "poisson",       # sales counts -> poisson/tweedie fit well
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "verbose": -1,
    }
    model = lgb.train(params, train_set, num_boost_round=500,
                       valid_sets=[val_set],
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    return model

# ---------------------------------------------------------------------------
# 7. METRICS (business-relevant)
# ---------------------------------------------------------------------------
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def bias(y_true, y_pred):
    return np.sum(y_pred - y_true) / np.sum(y_true) * 100

def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print(f"RMSE : {root_mean_squared_error(y_true, y_pred):.3f}")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"WMAPE: {wmape(y_true, y_pred):.2f}%")
    print(f"Bias : {bias(y_true, y_pred):+.2f}%  (positive = over-forecasting)")

# ---------------------------------------------------------------------------
# 8. PLOT: actual vs predicted for one SKU
# ---------------------------------------------------------------------------
def plot_example(test, sample_id):
    sub = test[test["id"] == sample_id].sort_values("date")
    plt.figure(figsize=(12, 4))
    plt.plot(sub["date"], sub["sales"], label="Actual", marker="o")
    plt.plot(sub["date"], sub["naive_pred"], label="Naive baseline", linestyle="--")
    plt.plot(sub["date"], sub["lgbm_pred"], label="LightGBM", linestyle="--")
    plt.title(f"Demand Forecast: {sample_id}")
    plt.xlabel("Date"); plt.ylabel("Units Sold"); plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_example.png", dpi=150)
    print("Saved plot -> forecast_example.png")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cal, sales, prices = load_data()
    long_df = to_long(cal, sales, prices)
    long_df = add_features(long_df)
    long_df = naive_baseline(long_df)

    train, test = split(long_df)
    model = train_lgbm(train, test)
    test["lgbm_pred"] = model.predict(test[FEATURES])
    test["lgbm_pred"] = test["lgbm_pred"].clip(lower=0)

    evaluate("Naive baseline (same day last week)", test["sales"], test["naive_pred"])
    evaluate("LightGBM", test["sales"], test["lgbm_pred"])

    # rough business translation
    base_wmape = wmape(test["sales"], test["naive_pred"])
    model_wmape = wmape(test["sales"], test["lgbm_pred"])
    improvement = base_wmape - model_wmape
    print(f"\nWMAPE improved by {improvement:.2f} points vs naive baseline "
          f"-> proportionally less safety stock needed to hedge forecast error.")

    plot_example(test, test["id"].iloc[0])

    lgb.plot_importance(model, max_num_features=15, figsize=(8, 6))
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved plot -> feature_importance.png")