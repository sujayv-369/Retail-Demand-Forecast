"""
Recursive 4-week demand forecasting using CatBoost.

At each forecast origin, predicts week 1, then uses that prediction
to update lag features for week 2, and so on through week 4.

Lag feature update rules (matching train_cb.py):
    lag_1_sales[t]       = units_sold[t-1]      (or predicted[t-1] if t-1 is forecast)
    lag_4_sales[t]       = units_sold[t-4]      (or predicted[t-4] if t-4 is forecast)
    rolling_4_week_avg[t] = mean(units_sold[t-1], units_sold[t-2], units_sold[t-3], units_sold[t-4])
                            (using predictions for any forecasted period)
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import (
    SERIES_KEY, TARGET, FEATURES, CAT_FEATURES,
    MODEL_PATH, FORECAST_HORIZON,
)

logger = logging.getLogger(__name__)


def load_model(model_path: str | Path | None = None) -> CatBoostRegressor:
    """Load a saved CatBoost model."""
    path = str(model_path or MODEL_PATH)
    model = CatBoostRegressor()
    model.load_model(path)
    logger.info("Loaded CatBoost model from %s", path)
    return model


def recursive_forecast(
    model: CatBoostRegressor,
    history_df: pd.DataFrame,
    forecast_origin: pd.Timestamp,
    horizon: int = FORECAST_HORIZON,
    future_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate recursive multi-step forecasts for ALL series from a
    single forecast origin.

    Parameters
    ----------
    model : Trained CatBoost model.
    history_df : Historical data with all columns, sorted by series+date.
    forecast_origin : The last observed week; forecasts start from origin + 1 week.
    horizon : Number of weeks to forecast (default 4).
    future_features_df : Optional DataFrame with known future calendar/promo features.
                         If not provided, calendar features are imputed from the date.

    Returns
    -------
    DataFrame with columns: SERIES_KEY + forecast_week_date, horizon_step, predicted_units, ...
    """
    hist = history_df[history_df["week_start_date"] <= forecast_origin].copy()
    hist = hist.sort_values(SERIES_KEY + ["week_start_date"])

    # Build recent-history buffer per series (need last 4 weeks for lags)
    series_groups = hist.groupby(SERIES_KEY)
    series_buffers = {}
    series_static = {}

    for key, grp in series_groups:
        tail = grp.tail(8).copy()  # Keep extra for rolling calculations
        series_buffers[key] = list(tail[TARGET].values)
        # Store static features from the latest row
        last_row = grp.iloc[-1]
        series_static[key] = {
            "category": last_row.get("category", "UNKNOWN"),
            "subcategory": last_row.get("subcategory", "UNKNOWN"),
            "brand": last_row.get("brand", "UNKNOWN"),
            "region": last_row.get("region", "UNKNOWN"),
            "season": last_row.get("season", "UNKNOWN"),
            "unit_price": last_row.get("unit_price", 0),
            "discount_pct": last_row.get("discount_pct", 0),
            "promotion_flag": last_row.get("promotion_flag", 0),
            "promotion_type": str(last_row.get("promotion_type", "UNKNOWN")),
            "opening_stock": last_row.get("opening_stock", 0),
            "stock_received": last_row.get("stock_received", 0),
            "supplier_lead_time_days": last_row.get("supplier_lead_time_days", 0),
            "product_name": last_row.get("product_name", ""),
        }

    results = []

    for step in range(1, horizon + 1):
        forecast_date = forecast_origin + pd.Timedelta(weeks=step)
        step_rows = []

        for key, buffer in series_buffers.items():
            # Compute lag features from buffer
            lag_1 = buffer[-1] if len(buffer) >= 1 else 0.0
            lag_4 = buffer[-4] if len(buffer) >= 4 else 0.0
            rolling_4 = float(np.mean(buffer[-4:])) if len(buffer) >= 4 else float(np.mean(buffer))

            # Calendar features from forecast date
            month = forecast_date.month
            week_number = forecast_date.isocalendar()[1]
            holiday_flag = 0  # Conservative default

            # Use future features if provided
            if future_features_df is not None:
                future_row = future_features_df[
                    (future_features_df["week_start_date"] == forecast_date)
                ]
                if len(key) == 3:
                    future_row = future_row[
                        (future_row["product_id"] == key[0]) &
                        (future_row["store_id"] == key[1]) &
                        (future_row["sales_channel"] == key[2])
                    ]
                if not future_row.empty:
                    fr = future_row.iloc[0]
                    holiday_flag = fr.get("holiday_flag", 0)
                    for col in ["season", "festival_name", "promotion_type",
                                "unit_price", "discount_pct", "promotion_flag",
                                "opening_stock", "stock_received"]:
                        if col in fr and pd.notna(fr[col]):
                            series_static[key][col] = fr[col]

            static = series_static[key]
            festival = "UNKNOWN"
            if future_features_df is not None:
                pass  # Already handled above
            # Determine season from month
            season = static.get("season", "UNKNOWN")

            row_dict = {
                "product_id": key[0] if len(key) == 3 else key,
                "store_id": key[1] if len(key) >= 2 else "",
                "sales_channel": key[2] if len(key) >= 3 else "",
                "category": static["category"],
                "subcategory": static["subcategory"],
                "brand": static["brand"],
                "region": static["region"],
                "season": season,
                "festival_name": str(static.get("festival_name", "UNKNOWN")),
                "promotion_type": str(static.get("promotion_type", "UNKNOWN")),
                "month": month,
                "week_number": week_number,
                "holiday_flag": holiday_flag,
                "lag_1_sales": lag_1,
                "lag_4_sales": lag_4,
                "rolling_4_week_avg": rolling_4,
                "unit_price": static["unit_price"],
                "discount_pct": static["discount_pct"],
                "promotion_flag": static["promotion_flag"],
                "opening_stock": static["opening_stock"],
                "stock_received": static["stock_received"],
                "supplier_lead_time_days": static["supplier_lead_time_days"],
            }
            step_rows.append(row_dict)

        if not step_rows:
            continue

        step_df = pd.DataFrame(step_rows)

        # Prepare features in the correct order
        X = step_df[FEATURES].copy()
        for col in CAT_FEATURES:
            X[col] = X[col].fillna("UNKNOWN").astype(str)

        predictions = model.predict(X)
        predictions = np.clip(predictions, 0, None)  # Demand >= 0

        # Record results and update buffers
        for i, (key, pred) in enumerate(zip(series_buffers.keys(), predictions)):
            series_buffers[key].append(float(pred))
            static = series_static[key]
            results.append({
                "product_id": key[0] if len(key) == 3 else key,
                "store_id": key[1] if len(key) >= 2 else "",
                "sales_channel": key[2] if len(key) >= 3 else "",
                "forecast_origin": forecast_origin,
                "forecast_week_date": forecast_date,
                "horizon_step": step,
                "predicted_units": float(pred),
                "product_name": static.get("product_name", ""),
                "category": static.get("category", ""),
                "model_type": "catboost",
            })

    return pd.DataFrame(results)


def generate_forecast_run(
    model: CatBoostRegressor,
    history_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Generate a forecast from the latest available data point.
    Returns (forecast_df, forecast_origin).
    """
    forecast_origin = history_df["week_start_date"].max()
    logger.info("Forecast origin: %s, horizon: %d weeks", forecast_origin, horizon)

    forecast_df = recursive_forecast(model, history_df, forecast_origin, horizon)
    logger.info("Generated %d forecast rows", len(forecast_df))
    return forecast_df, forecast_origin
