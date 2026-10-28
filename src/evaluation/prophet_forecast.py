"""
Prophet forecasting implementation using Facebook Prophet.
Compatible with rolling-origin held-out backtesting.
"""

from __future__ import annotations
import logging
import time
import pandas as pd
import numpy as np

from src.config import SERIES_KEY, TARGET

logger = logging.getLogger(__name__)


def prophet_forecast(
    history: pd.DataFrame,
    forecast_origins: list[pd.Timestamp],
    horizon: int = 4,
    series_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, float]:
    """
    Generate multi-step Prophet forecasts for series up to each forecast origin.
    Returns (forecasts_df, runtime_seconds).
    """
    from prophet import Prophet

    # Suppress verbose cmdstanpy and prophet logs during batch evaluation
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    t0 = time.time()
    rows = []
    history = history.sort_values(SERIES_KEY + ["week_start_date"])

    # Series identifier
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

        for uid, grp in hist_before.groupby("unique_id"):
            if len(grp) < 10:
                continue

            parts = uid.split("___")
            if len(parts) != 3:
                continue
            pid, sid, ch = parts

            p_df = grp[["week_start_date", TARGET]].rename(
                columns={"week_start_date": "ds", TARGET: "y"}
            ).dropna()

            try:
                # Fast weekly Prophet setup
                m = Prophet(
                    growth="linear",
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    n_changepoints=10,
                )
                m.fit(p_df)

                future = m.make_future_dataframe(periods=horizon, freq="W-MON", include_history=False)
                forecast = m.predict(future)

                for step_idx, (_, r) in enumerate(forecast.iterrows(), start=1):
                    pred_val = float(r["yhat"]) if pd.notna(r["yhat"]) else 0.0
                    pred_val = max(0.0, pred_val)

                    rows.append({
                        "product_id": pid,
                        "store_id": sid,
                        "sales_channel": ch,
                        "forecast_origin": origin,
                        "forecast_week_date": pd.to_datetime(r["ds"]),
                        "horizon_step": step_idx,
                        "predicted_units": pred_val,
                        "model_type": "prophet",
                    })

            except Exception as e:
                logger.debug("Prophet series fit failed for %s at %s: %s", uid, origin, e)

    runtime = time.time() - t0
    return pd.DataFrame(rows), runtime
