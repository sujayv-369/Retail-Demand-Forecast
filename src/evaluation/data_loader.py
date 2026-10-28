"""
Data loading and feature preparation for evaluation.
Loads from CSV (authoritative source) and recomputes lag features
correctly within each series to prevent cross-series contamination.
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

from src.config import (
    RAW_CSV_PATH, SERIES_KEY, TARGET, FEATURES, CAT_FEATURES,
)

logger = logging.getLogger(__name__)


def load_raw_data(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw dataset and parse dates."""
    path = csv_path or str(RAW_CSV_PATH)
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="raise")
    df = df.sort_values(SERIES_KEY + ["week_start_date"]).reset_index(drop=True)
    logger.info("Loaded %d rows, %d weeks", len(df), df["week_start_date"].nunique())
    return df


def recompute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute lag_1_sales, lag_4_sales, rolling_4_week_avg
    strictly within each series (product_id × store_id × sales_channel).
    This prevents cross-series contamination.
    """
    df = df.sort_values(SERIES_KEY + ["week_start_date"]).copy()
    group = df.groupby(SERIES_KEY, sort=False)[TARGET]

    df["lag_1_sales"] = group.shift(1)
    df["lag_4_sales"] = group.shift(4)
    df["rolling_4_week_avg"] = (
        df.groupby(SERIES_KEY, sort=False)[TARGET]
        .transform(lambda s: s.shift(1).rolling(4).mean())
    )
    return df


def prepare_features(
    df: pd.DataFrame,
    recompute_lags: bool = True,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Full feature preparation pipeline:
    1. Recompute lags within series
    2. Fill categorical NaNs
    3. Drop rows lacking required features
    """
    if recompute_lags:
        df = recompute_lag_features(df)

    # Fill categorical NaNs consistently with training code
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    if drop_na:
        before = len(df)
        df = df.dropna(subset=FEATURES + [TARGET]).copy()
        logger.info("Dropped %d rows with NaN features (%d remaining)", before - len(df), len(df))

    return df


def get_unique_weeks(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Return sorted unique week_start_dates."""
    return sorted(df["week_start_date"].unique())


def get_train_test_split(
    df: pd.DataFrame,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no random shuffling."""
    train = df[df["week_start_date"] < test_start].copy()
    test = df[df["week_start_date"] >= test_start].copy()
    return train, test


def get_series_history_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-series summary: total weeks, first/last date, total demand,
    number of zero-demand weeks, missing-week gaps.
    """
    all_weeks = set(df["week_start_date"].unique())

    def _summarize(g):
        weeks_present = set(g["week_start_date"].unique())
        date_range_weeks = (
            (g["week_start_date"].max() - g["week_start_date"].min()).days // 7
        ) + 1
        return pd.Series({
            "n_weeks": len(weeks_present),
            "date_range_weeks": date_range_weeks,
            "first_date": g["week_start_date"].min(),
            "last_date": g["week_start_date"].max(),
            "total_demand": g[TARGET].sum(),
            "mean_demand": g[TARGET].mean(),
            "max_demand": g[TARGET].max(),
            "zero_weeks": (g[TARGET] == 0).sum(),
            "missing_weeks": date_range_weeks - len(weeks_present),
        })

    return df.groupby(SERIES_KEY).apply(_summarize, include_groups=False).reset_index()
