"""
Feature engineering for the demand forecasting pipeline.

Given a long-format dataframe with columns [date, store, item, sales],
build the calendar, lag, and rolling-window features used by the models.
"""
import numpy as np
import pandas as pd


LAGS = [7, 14, 28, 90, 365]
ROLL_WINDOWS = [7, 30, 90]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    # cyclical encodings so e.g. December and January are "close"
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT: lags/rolling stats are computed per (store, item) series,
    and rolling windows are shifted by 1 day so no feature ever leaks
    the value we are trying to predict (no look-ahead bias).
    """
    df = df.sort_values(["store", "item", "date"]).copy()
    group_key = ["store", "item"]

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(group_key)["sales"].shift(lag)

    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}"] = df.groupby(group_key)["sales"].transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
        df[f"roll_std_{window}"] = df.groupby(group_key)["sales"].transform(
            lambda x: x.shift(1).rolling(window).std()
        )
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    # first 365 days per series will have NaNs from the lag_365 feature -> drop
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    ["store", "item", "day_of_week", "day_of_month", "month", "year",
     "week_of_year", "is_weekend", "is_month_start", "is_month_end",
     "month_sin", "month_cos", "dow_sin", "dow_cos"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
)
TARGET_COLUMN = "sales"
