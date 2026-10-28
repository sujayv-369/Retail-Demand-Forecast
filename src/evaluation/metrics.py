"""
Evaluation metrics — pure functions for MAE, WAPE with proper edge-case handling.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error = sum(|actual - predicted|) / n."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)))


def wape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    """
    Weighted Absolute Percentage Error.
    WAPE = 100 × sum(|actual - predicted|) / sum(|actual|)

    Returns None if the denominator (total actual demand) is zero.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return None
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return None  # Undefined — not fabricated zero
    return float(100.0 * np.sum(np.abs(actual - predicted)) / denom)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error = sqrt(mean((actual - predicted)**2))."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(actual) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return a dict with mae, wape, rmse, and n_observations."""
    return {
        "mae": mae(actual, predicted),
        "wape": wape(actual, predicted),
        "rmse": rmse(actual, predicted),
        "n_observations": int(len(actual)),
    }



def compute_grouped_metrics(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    group_col: str,
) -> list[dict]:
    """Compute MAE/WAPE per group (e.g., per product, per category)."""
    results = []
    for group_val, grp in df.groupby(group_col):
        m = compute_metrics(grp[actual_col].values, grp[pred_col].values)
        m["scope_value"] = str(group_val)
        results.append(m)
    return results


def compute_horizon_metrics(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    horizon_col: str = "horizon_step",
) -> list[dict]:
    """Compute MAE/WAPE per forecast horizon step (1–4)."""
    results = []
    for h, grp in df.groupby(horizon_col):
        m = compute_metrics(grp[actual_col].values, grp[pred_col].values)
        m["horizon_week"] = int(h)
        results.append(m)
    return results
