"""
Baseline forecasting models: Naive (last-value) and 4-week Moving Average.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.config import SERIES_KEY


def naive_forecast(
    history: pd.DataFrame,
    forecast_origins: list[pd.Timestamp],
    horizon: int = 4,
) -> pd.DataFrame:
    """
    Naive baseline: each forecast equals the last observed demand.

    For each origin, for each series, the forecast for all horizon steps
    equals the most recent observed units_sold before the origin.
    """
    rows = []
    history = history.sort_values(SERIES_KEY + ["week_start_date"])

    for origin in forecast_origins:
        hist_before = history[history["week_start_date"] <= origin]
        if hist_before.empty:
            continue

        # Last value per series
        last_vals = (
            hist_before
            .groupby(SERIES_KEY, sort=False)
            .last()
            .reset_index()
        )

        for step in range(1, horizon + 1):
            forecast_date = origin + pd.Timedelta(weeks=step)
            for _, row in last_vals.iterrows():
                rows.append({
                    **{k: row[k] for k in SERIES_KEY},
                    "forecast_origin": origin,
                    "forecast_week_date": forecast_date,
                    "horizon_step": step,
                    "predicted_units": float(row["units_sold"]),
                    "model_type": "naive",
                })

    return pd.DataFrame(rows)


def moving_average_forecast(
    history: pd.DataFrame,
    forecast_origins: list[pd.Timestamp],
    window: int = 4,
    horizon: int = 4,
) -> pd.DataFrame:
    """
    Moving-average baseline: each forecast equals the mean of the
    latest `window` completed weeks of demand.
    """
    rows = []
    history = history.sort_values(SERIES_KEY + ["week_start_date"])

    for origin in forecast_origins:
        hist_before = history[history["week_start_date"] <= origin]
        if hist_before.empty:
            continue

        # Last `window` values per series
        tail_vals = (
            hist_before
            .groupby(SERIES_KEY, sort=False)
            .apply(lambda g: g.tail(window)["units_sold"].mean(), include_groups=False)
            .reset_index(name="ma_pred")
        )

        for step in range(1, horizon + 1):
            forecast_date = origin + pd.Timedelta(weeks=step)
            for _, row in tail_vals.iterrows():
                rows.append({
                    **{k: row[k] for k in SERIES_KEY},
                    "forecast_origin": origin,
                    "forecast_week_date": forecast_date,
                    "horizon_step": step,
                    "predicted_units": float(row["ma_pred"]),
                    "model_type": "ma4",
                })

    return pd.DataFrame(rows)
