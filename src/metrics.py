"""
Business-relevant forecast evaluation metrics.

Plain RMSE doesn't tell a supply-chain stakeholder much on its own.
MAPE/WMAPE and bias are the numbers planners actually use to size
safety stock and judge whether a model over- or under-forecasts.
"""
import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred, epsilon: float = 1.0):
    """Mean Absolute Percentage Error. epsilon avoids div-by-zero on days with 0 sales."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(y_true == 0, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def wmape(y_true, y_pred):
    """Weighted MAPE - weights errors by volume, standard in demand planning
    because it isn't distorted by low-volume SKUs the way plain MAPE is."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def bias(y_true, y_pred):
    """Mean signed error as % of actual. Positive = systematically over-forecasting
    (risk of overstock); negative = under-forecasting (risk of stockouts)."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sum(y_pred - y_true) / np.sum(y_true) * 100)


def evaluate_all(y_true, y_pred) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "WMAPE_%": wmape(y_true, y_pred),
        "Bias_%": bias(y_true, y_pred),
    }
