"""
ARIMA forecasting implementation using StatsForecast AutoARIMA.
Compatible with rolling-origin held-out backtesting.
"""

from __future__ import annotations
import logging
import time
import pandas as pd
import numpy as np

from src.config import SERIES_KEY, TARGET

logger = logging.getLogger(__name__)


def arima_forecast(
    history: pd.DataFrame,
    forecast_origins: list[pd.Timestamp],
    horizon: int = 4,
    series_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Generate multi-step ARIMA forecasts for series up to each forecast origin.
    Returns (forecasts_df, runtime_seconds).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    t0 = time.time()
    rows = []
    history = history.sort_values(SERIES_KEY + ["week_start_date"])

    # Format unique_id for statsforecast
    history = history.copy()
    history["unique_id"] = (
        history["product_id"].astype(str) + "___" +
        history["store_id"].astype(str) + "___" +
        history["sales_channel"].astype(str)
    )

    if series_subset is not None:
        if any("___" in str(x) for x in series_subset):
            history = history[history["unique_id"].isin(series_subset)].copy()
        else:
            history = history[history["product_id"].isin(series_subset)].copy()

    for origin in forecast_origins:
        hist_before = history[history["week_start_date"] <= origin].copy()
        if hist_before.empty:
            continue

        # Prepare panel for StatsForecast
        sf_df = hist_before[["unique_id", "week_start_date", TARGET]].rename(
            columns={"week_start_date": "ds", TARGET: "y"}
        )

        # Filter out series with fewer than 10 observations for ARIMA
        counts = sf_df.groupby("unique_id").size()
        valid_uids = counts[counts >= 10].index
        sf_df = sf_df[sf_df["unique_id"].isin(valid_uids)]

        if sf_df.empty:
            continue

        try:
            sf = StatsForecast(
                models=[AutoARIMA(seasonal=False, alias="ARIMA", max_p=2, max_q=2, stepwise=True)],
                freq="W-MON",
                n_jobs=1,
            )
            sf.fit(sf_df)
            preds = sf.predict(h=horizon)

            for uid, group in preds.groupby("unique_id"):
                parts = uid.split("___")
                if len(parts) != 3:
                    continue
                pid, sid, ch = parts

                for step_idx, (_, r) in enumerate(group.sort_values("ds").iterrows(), start=1):
                    pred_val = float(r["ARIMA"]) if pd.notna(r["ARIMA"]) else 0.0
                    pred_val = max(0.0, pred_val)  # demand cannot be negative

                    rows.append({
                        "product_id": pid,
                        "store_id": sid,
                        "sales_channel": ch,
                        "forecast_origin": origin,
                        "forecast_week_date": pd.to_datetime(r["ds"]),
                        "horizon_step": step_idx,
                        "predicted_units": pred_val,
                        "model_type": "arima",
                    })

        except Exception as e:
            logger.warning("StatsForecast ARIMA batch failed at origin %s: %s", origin, e)

    runtime = time.time() - t0
    return pd.DataFrame(rows), runtime
