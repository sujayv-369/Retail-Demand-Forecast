import os

# Configure thread limits BEFORE numerical libraries are imported to prevent memory exhaustion on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

warnings.filterwarnings("default")
load_dotenv()

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing from .env")

SERIES_KEY = ["product_id", "store_id", "sales_channel"]
TARGET = "units_sold"

HORIZON = 4
N_WINDOWS = 3
STEP_SIZE = 4

MIN_ARIMA_TRAIN_WEEKS = 30
MIN_SEASONAL_TRAIN_WEEKS = 104

# Default to 2 workers on Windows to prevent virtual memory paging exhaustion (WinError 1455)
DEFAULT_N_JOBS = "2" if os.name == "nt" else "-1"
N_JOBS = int(os.getenv("ARIMA_N_JOBS", DEFAULT_N_JOBS))

# Enable ONLY when a missing weekly row is confirmed to mean
# zero sales, rather than unknown or unrecorded sales.
FILL_MISSING_WEEKS_WITH_ZERO = False

OUTPUT_DIR = Path("artifacts/arima")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MODEL SEARCH CONFIGURATIONS
# ---------------------------------------------------------

def make_arima():
    return AutoARIMA(
        seasonal=False,
        alias="ARIMA",
        max_p=3,
        max_q=3,
        stepwise=True,
        nmodels=30,
    )


def make_sarima():
    return AutoARIMA(
        seasonal=True,
        season_length=52,
        alias="SARIMA_52",
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
        stepwise=True,
        nmodels=30,
    )


# ---------------------------------------------------------
# BASELINES AND METRICS
# ---------------------------------------------------------

def add_baselines(cv, history):
    cv = cv.reset_index(drop=True).copy()

    cv["cutoff"] = pd.to_datetime(cv["cutoff"])
    cv["ds"] = pd.to_datetime(cv["ds"])

    history_arrays = {}

    for uid, group in history.groupby("unique_id", sort=False):
        group = group.sort_values("ds")
        history_arrays[uid] = (
            group["ds"].to_numpy(dtype="datetime64[ns]"),
            group["y"].to_numpy(dtype=float),
        )

    cv["Naive"] = np.nan
    cv["MovingAverage4"] = np.nan

    for (uid, cutoff), idx in cv.groupby(
        ["unique_id", "cutoff"], sort=False
    ).groups.items():

        dates, values = history_arrays[uid]
        cutoff64 = pd.Timestamp(cutoff).to_datetime64()

        # Only observations up through the forecast origin
        # may be used for a baseline prediction.
        pos = np.searchsorted(
            dates, cutoff64, side="right"
        )

        if pos == 0:
            continue

        last_value = values[pos - 1]
        recent_values = values[max(0, pos - 4):pos]

        cv.loc[idx, "Naive"] = last_value
        cv.loc[idx, "MovingAverage4"] = recent_values.mean()

    return cv


def summarize_metrics(scored, group_cols):
    grouped = scored.groupby(
        ["model"] + group_cols,
        dropna=False,
    )

    result = grouped.agg(
        n_observations=("y", "size"),
        MAE=("abs_error", "mean"),
        total_absolute_error=("abs_error", "sum"),
        total_actual=("y", "sum"),
    ).reset_index()

    result["WAPE_pct"] = np.where(
        result["total_actual"] > 0,
        100 * result["total_absolute_error"]
        / result["total_actual"],
        np.nan,
    )

    return result


def save_evaluation(cv, lookup, model_columns, prefix):
    cv = cv.merge(
        lookup,
        on="unique_id",
        how="left",
        validate="many_to_one",
    )

    cv["horizon"] = (
        (cv["ds"] - cv["cutoff"]).dt.days / 7
    ).round().astype(int)

    cv.to_csv(
        OUTPUT_DIR / f"{prefix}_predictions.csv",
        index=False,
    )

    id_columns = [
        "unique_id",
        "ds",
        "cutoff",
        "y",
        "horizon",
        "product_id",
        "store_id",
        "sales_channel",
        "category",
    ]

    scored = cv.melt(
        id_vars=id_columns,
        value_vars=model_columns,
        var_name="model",
        value_name="prediction",
    )

    scored = scored.dropna(subset=["prediction"]).copy()

    scored["abs_error"] = (
        scored["y"] - scored["prediction"]
    ).abs()

    scored["scope"] = prefix

    overall = summarize_metrics(scored, [])
    by_horizon = summarize_metrics(scored, ["horizon"])
    by_category = summarize_metrics(scored, ["category"])

    by_series = summarize_metrics(
        scored,
        [
            "unique_id",
            "product_id",
            "store_id",
            "sales_channel",
            "category",
        ],
    )

    overall.to_csv(
        OUTPUT_DIR / f"{prefix}_metrics_overall.csv",
        index=False,
    )

    by_horizon.to_csv(
        OUTPUT_DIR / f"{prefix}_metrics_by_horizon.csv",
        index=False,
    )

    by_category.to_csv(
        OUTPUT_DIR / f"{prefix}_metrics_by_category.csv",
        index=False,
    )

    by_series.to_csv(
        OUTPUT_DIR / f"{prefix}_metrics_by_series.csv",
        index=False,
    )

    print(f"\nEvaluation scope: {prefix}")
    print(
        overall[
            ["model", "n_observations", "MAE", "WAPE_pct"]
        ].to_string(index=False)
    )

    return overall


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )

    # 1. LOAD ONLY THE REQUIRED DATA
    query = text("""
        SELECT
            product_id,
            store_id,
            sales_channel,
            category,
            week_start_date,
            units_sold
        FROM retail
        ORDER BY week_start_date
    """)

    raw = pd.read_sql(query, engine)

    if raw.empty:
        raise ValueError("The retail table contains no records.")

    raw["week_start_date"] = pd.to_datetime(
        raw["week_start_date"], errors="coerce"
    )
    raw["units_sold"] = pd.to_numeric(
        raw["units_sold"], errors="coerce"
    )

    # Deduplicate identical records (e.g. from repeated table ingestion)
    n_raw = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    if len(raw) < n_raw:
        print(f"Removed {n_raw - len(raw)} exact duplicate records from retail table.")

    required = SERIES_KEY + ["week_start_date", "units_sold"]
    invalid = raw[required].isna().any(axis=1)

    if invalid.any():
        raw.loc[invalid].to_csv(
            OUTPUT_DIR / "invalid_input_records.csv", index=False
        )
        raise ValueError(
            f"{invalid.sum()} invalid records. "
            "See invalid_input_records.csv."
        )

    if (raw["units_sold"] < 0).any():
        raw.loc[raw["units_sold"] < 0].to_csv(
            OUTPUT_DIR / "negative_sales_records.csv", index=False
        )
        raise ValueError(
            "Negative units_sold values need investigation. "
            "Define whether they represent returns or net sales "
            "before fitting the demand model."
        )

    raw[SERIES_KEY] = raw[SERIES_KEY].astype(str)
    raw["category"] = raw["category"].fillna("UNKNOWN").astype(str)

    # Derive the actual weekly frequency instead of assuming
    # that week_start_date is always a Monday.
    weekdays = raw["week_start_date"].dt.dayofweek.unique()

    if len(weekdays) != 1:
        raise ValueError(
            "week_start_date contains inconsistent weekdays. "
            "Standardize weekly dates before forecasting."
        )

    weekday_name = (
        raw["week_start_date"].dt.day_name().iloc[0][:3].upper()
    )
    freq = f"W-{weekday_name}"

    # 2. ASSIGN ONE ID TO EACH PRODUCT/STORE/CHANNEL SERIES
    series_keys = (
        raw[SERIES_KEY]
        .drop_duplicates()
        .sort_values(SERIES_KEY)
        .reset_index(drop=True)
    )

    series_keys["unique_id"] = series_keys.index.astype(str)

    df = raw.merge(
        series_keys,
        on=SERIES_KEY,
        how="left",
        validate="many_to_one",
    )

    df = df.rename(
        columns={
            "week_start_date": "ds",
            "units_sold": "y",
        }
    )

    # A weekly series must have at most one observation per unique_id and date.
    duplicate_mask = df.duplicated(
        ["unique_id", "ds"], keep=False
    )

    if duplicate_mask.any():
        df.loc[duplicate_mask].to_csv(
            OUTPUT_DIR / "duplicate_series_weeks.csv", index=False
        )
        print(
            f"Notice: {duplicate_mask.sum()} duplicate series/week observations found. "
            "Deduplicating to retain unique weekly observations..."
        )
        df = df.drop_duplicates(subset=["unique_id", "ds"], keep="first").reset_index(drop=True)

    series_lookup = (
        df.groupby("unique_id", as_index=False)
        .agg(
            product_id=("product_id", "first"),
            store_id=("store_id", "first"),
            sales_channel=("sales_channel", "first"),
            category=("category", "first"),
        )
    )

    global_last_week = df["ds"].max()

    # Use series with observations through the common final week.
    series_end = df.groupby("unique_id")["ds"].max()
    active_ids = series_end[
        series_end == global_last_week
    ].index

    inactive_count = series_end.shape[0] - len(active_ids)

    df = df[df["unique_id"].isin(active_ids)].copy()
    series_lookup = series_lookup[
        series_lookup["unique_id"].isin(active_ids)
    ].copy()

    if df.empty:
        raise ValueError("No series end at the latest observed week.")

    # 3. CHECK WEEKLY CALENDAR COMPLETENESS
    gap_rows = []
    regular_parts = []

    for uid, group in df.groupby("unique_id", sort=False):
        group = group.sort_values("ds")

        expected_dates = pd.date_range(
            start=group["ds"].min(),
            end=group["ds"].max(),
            freq=freq,
        )

        actual_dates = pd.DatetimeIndex(group["ds"])
        missing_dates = expected_dates.difference(actual_dates)

        for missing_date in missing_dates:
            gap_rows.append({
                "unique_id": uid,
                "missing_week": missing_date,
            })

        if FILL_MISSING_WEEKS_WITH_ZERO:
            indexed = group.set_index("ds")[["y"]]
            indexed = indexed.reindex(expected_dates)
            indexed.index.name = "ds"

            indexed["unique_id"] = uid
            indexed["y"] = indexed["y"].fillna(0.0)

            regular_parts.append(
                indexed.reset_index()[["unique_id", "ds", "y"]]
            )

    if gap_rows:
        pd.DataFrame(gap_rows).to_csv(
            OUTPUT_DIR / "missing_week_report.csv", index=False
        )

        if not FILL_MISSING_WEEKS_WITH_ZERO:
            raise ValueError(
                "Missing weeks detected. Inspect missing_week_report.csv. "
                "Only set FILL_MISSING_WEEKS_WITH_ZERO=True after "
                "confirming that omitted weeks represent zero sales."
            )

    if FILL_MISSING_WEEKS_WITH_ZERO:
        panel = pd.concat(regular_parts, ignore_index=True)
    else:
        panel = df[["unique_id", "ds", "y"]].copy()

    panel = panel.sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)

    # 4. SELECT SERIES WITH ENOUGH TRAINING HISTORY
    counts = panel.groupby("unique_id").size()

    required_arima_history = (
        MIN_ARIMA_TRAIN_WEEKS + HORIZON * N_WINDOWS
    )

    arima_ids = counts[
        counts >= required_arima_history
    ].index

    if len(arima_ids) == 0:
        raise ValueError(
            "No series have enough historical data for ARIMA "
            "evaluation. Review the data coverage and thresholds."
        )

    arima_data = panel[
        panel["unique_id"].isin(arima_ids)
    ].copy()

    seasonal_history_required = (
        MIN_SEASONAL_TRAIN_WEEKS + HORIZON * N_WINDOWS
    )

    seasonal_ids = counts[
        counts >= seasonal_history_required
    ].index

    seasonal_data = panel[
        panel["unique_id"].isin(seasonal_ids)
    ].copy()

    print(f"Weekly frequency: {freq}")
    print(f"Latest week: {global_last_week.date()}")
    print(f"All source series: {series_end.shape[0]}")
    print(f"Series ending at latest week: {len(active_ids)}")
    print(f"Series excluded as inactive/stale: {inactive_count}")
    print(f"ARIMA evaluation series: {len(arima_ids)}")
    print(f"Seasonal evaluation series: {len(seasonal_ids)}")
    print(f"Using n_jobs: {N_JOBS}")

    # 5. EVALUATE NON-SEASONAL AUTOARIMA
    print("\nRunning ARIMA cross-validation...")
    sf_arima = StatsForecast(
        models=[make_arima()],
        freq=freq,
        n_jobs=N_JOBS,
    )

    cv_arima = sf_arima.cross_validation(
        df=arima_data,
        h=HORIZON,
        step_size=STEP_SIZE,
        n_windows=N_WINDOWS,
    )

    cv_arima = add_baselines(cv_arima, arima_data)

    save_evaluation(
        cv=cv_arima,
        lookup=series_lookup[
            series_lookup["unique_id"].isin(arima_ids)
        ],
        model_columns=[
            "Naive",
            "MovingAverage4",
            "ARIMA",
        ],
        prefix="arima_all_eligible",
    )

    # 6. EVALUATE SARIMA ON THE SAME ELIGIBLE SERIES
    sf_seasonal = None

    if len(seasonal_ids) > 0:
        print("\nRunning SARIMA cross-validation...")
        sf_seasonal = StatsForecast(
            models=[
                make_arima(),
                make_sarima(),
            ],
            freq=freq,
            n_jobs=N_JOBS,
        )

        cv_seasonal = sf_seasonal.cross_validation(
            df=seasonal_data,
            h=HORIZON,
            step_size=STEP_SIZE,
            n_windows=N_WINDOWS,
        )

        cv_seasonal = add_baselines(
            cv_seasonal,
            seasonal_data,
        )

        save_evaluation(
            cv=cv_seasonal,
            lookup=series_lookup[
                series_lookup["unique_id"].isin(seasonal_ids)
            ],
            model_columns=[
                "Naive",
                "MovingAverage4",
                "ARIMA",
                "SARIMA_52",
            ],
            prefix="arima_vs_sarima_common_series",
        )
    else:
        print(
            "\nSARIMA evaluation skipped: no series have "
            f"{seasonal_history_required} or more weeks "
            "required for the configured evaluation."
        )

    # 7. GENERATE FOUR-WEEK FUTURE FORECASTS
    print("\nFitting final models and generating future forecasts...")
    future_parts = []

    future_arima = sf_arima.forecast(
        df=arima_data,
        h=HORIZON,
    )

    future_arima = future_arima.rename(
        columns={"ARIMA": "predicted_units"}
    )
    future_arima["model"] = "ARIMA"
    future_parts.append(future_arima)

    if sf_seasonal is not None:
        future_seasonal = sf_seasonal.forecast(
            df=seasonal_data,
            h=HORIZON,
        )

        seasonal_arima = future_seasonal[
            ["unique_id", "ds", "ARIMA"]
        ].rename(
            columns={"ARIMA": "predicted_units"}
        )
        seasonal_arima["model"] = "ARIMA_common_subset"
        future_parts.append(seasonal_arima)

        sarima_future = future_seasonal[
            ["unique_id", "ds", "SARIMA_52"]
        ].rename(
            columns={"SARIMA_52": "predicted_units"}
        )
        sarima_future["model"] = "SARIMA_52"
        future_parts.append(sarima_future)

    future = pd.concat(future_parts, ignore_index=True)

    future = future.merge(
        series_lookup,
        on="unique_id",
        how="left",
        validate="many_to_one",
    )

    future["forecast_origin_week"] = global_last_week

    future["horizon"] = (
        future.sort_values(["unique_id", "ds"])
        .groupby(["unique_id", "model"])
        .cumcount() + 1
    )

    future["predicted_units"] = (
        future["predicted_units"].clip(lower=0)
    )

    future.to_csv(
        OUTPUT_DIR / "four_week_forecasts.csv",
        index=False,
    )

    # 8. SAVE RUN METADATA
    metadata = {
        "data_table": "retail",
        "target": TARGET,
        "series_key": SERIES_KEY,
        "frequency": freq,
        "last_observed_week": str(global_last_week.date()),
        "forecast_horizon_weeks": HORIZON,
        "evaluation_windows": N_WINDOWS,
        "evaluation_window_size": STEP_SIZE,
        "arima_min_training_weeks": MIN_ARIMA_TRAIN_WEEKS,
        "seasonal_period_weeks": 52,
        "seasonal_min_training_weeks": MIN_SEASONAL_TRAIN_WEEKS,
        "arima_evaluation_series": int(len(arima_ids)),
        "seasonal_evaluation_series": int(len(seasonal_ids)),
        "fill_missing_weeks_with_zero": FILL_MISSING_WEEKS_WITH_ZERO,
        "n_jobs": N_JOBS,
    }

    with open(
        OUTPUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nCompleted ARIMA evaluation and forecast generation.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
