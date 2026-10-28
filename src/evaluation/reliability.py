"""
Reliability flags — documented rules for identifying series
with insufficient history, high error, sparsity, demand spikes,
or stockout distortion.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from src.config import SERIES_KEY, TARGET


# ── Flag rule thresholds (documented, not claimed as calibrated) ───
MIN_HISTORY_WEEKS = 12
SPARSE_ZERO_FRACTION = 0.5     # >50% zero-demand weeks
SPIKE_ZSCORE_THRESHOLD = 3.0   # demand > 3 std devs above mean
HIGH_ERROR_WAPE = 80.0         # WAPE > 80% flagged as unreliable
STOCKOUT_FRACTION = 0.2        # >20% stockout weeks


def flag_insufficient_history(
    series_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Flag series with fewer than MIN_HISTORY_WEEKS of data."""
    flagged = series_summary[series_summary["n_weeks"] < MIN_HISTORY_WEEKS].copy()
    flagged["flag_type"] = "insufficient_history"
    flagged["flag_reason"] = flagged["n_weeks"].apply(
        lambda n: f"Only {n} weeks of history (minimum {MIN_HISTORY_WEEKS} required)"
    )
    flagged["severity"] = "high"
    return flagged[SERIES_KEY + ["flag_type", "flag_reason", "severity"]]


def flag_missing_periods(
    series_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Flag series with significant gaps in their weekly data."""
    flagged = series_summary[series_summary["missing_weeks"] > 4].copy()
    flagged["flag_type"] = "missing_periods"
    flagged["flag_reason"] = flagged["missing_weeks"].apply(
        lambda m: f"{m} missing weeks detected in date range"
    )
    flagged["severity"] = "medium"
    return flagged[SERIES_KEY + ["flag_type", "flag_reason", "severity"]]


def flag_sparse_demand(
    series_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Flag series where most weeks have zero demand (intermittent)."""
    series_summary = series_summary.copy()
    series_summary["zero_frac"] = series_summary["zero_weeks"] / series_summary["n_weeks"].clip(lower=1)
    flagged = series_summary[series_summary["zero_frac"] > SPARSE_ZERO_FRACTION].copy()
    flagged["flag_type"] = "sparse_demand"
    flagged["flag_reason"] = flagged.apply(
        lambda r: f"{r['zero_weeks']}/{r['n_weeks']} weeks ({r['zero_frac']*100:.0f}%) have zero demand",
        axis=1,
    )
    flagged["severity"] = "medium"
    return flagged[SERIES_KEY + ["flag_type", "flag_reason", "severity"]]


def flag_demand_spikes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag series that have unusually large demand spikes."""
    stats = df.groupby(SERIES_KEY)[TARGET].agg(["mean", "std", "max"]).reset_index()
    stats["std"] = stats["std"].fillna(0)
    stats["z_max"] = np.where(
        stats["std"] > 0,
        (stats["max"] - stats["mean"]) / stats["std"],
        0,
    )
    flagged = stats[stats["z_max"] > SPIKE_ZSCORE_THRESHOLD].copy()
    flagged["flag_type"] = "demand_spike"
    flagged["flag_reason"] = flagged.apply(
        lambda r: f"Max demand {r['max']:.0f} is {r['z_max']:.1f} std devs above mean ({r['mean']:.1f})",
        axis=1,
    )
    flagged["severity"] = "medium"
    return flagged[SERIES_KEY + ["flag_type", "flag_reason", "severity"]]


def flag_stockout_distortion(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag series with high stockout frequency, which suppresses true demand."""
    if "stockout_flag" not in df.columns:
        return pd.DataFrame(columns=SERIES_KEY + ["flag_type", "flag_reason", "severity"])

    stats = df.groupby(SERIES_KEY).agg(
        stockout_weeks=("stockout_flag", "sum"),
        total_weeks=("stockout_flag", "count"),
    ).reset_index()
    stats["stockout_frac"] = stats["stockout_weeks"] / stats["total_weeks"].clip(lower=1)

    flagged = stats[stats["stockout_frac"] > STOCKOUT_FRACTION].copy()
    flagged["flag_type"] = "stockout_distortion"
    flagged["flag_reason"] = flagged.apply(
        lambda r: (
            f"Stockouts in {r['stockout_weeks']}/{r['total_weeks']} weeks "
            f"({r['stockout_frac']*100:.0f}%) — true demand may be higher than recorded"
        ),
        axis=1,
    )
    flagged["severity"] = "high"
    return flagged[SERIES_KEY + ["flag_type", "flag_reason", "severity"]]


def flag_high_error(
    eval_metrics: pd.DataFrame,
    threshold: float = HIGH_ERROR_WAPE,
) -> pd.DataFrame:
    """Flag products with WAPE above threshold in evaluation."""
    flagged = eval_metrics[
        (eval_metrics["wape"].notna()) & (eval_metrics["wape"] > threshold)
    ].copy()
    flagged["flag_type"] = "high_forecast_error"
    flagged["flag_reason"] = flagged["wape"].apply(
        lambda w: f"Out-of-sample WAPE = {w:.1f}% (threshold {threshold}%)"
    )
    flagged["severity"] = "high"
    cols = [c for c in SERIES_KEY if c in flagged.columns]
    if "scope_value" in flagged.columns and "product_id" not in flagged.columns:
        flagged["product_id"] = flagged["scope_value"]
        cols = ["product_id"]
    return flagged[cols + ["flag_type", "flag_reason", "severity"]]


def compute_all_flags(
    df: pd.DataFrame,
    series_summary: pd.DataFrame,
    eval_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run all reliability flag rules and return combined results."""
    flags = [
        flag_insufficient_history(series_summary),
        flag_missing_periods(series_summary),
        flag_sparse_demand(series_summary),
        flag_demand_spikes(df),
        flag_stockout_distortion(df),
    ]
    if eval_metrics is not None and not eval_metrics.empty:
        flags.append(flag_high_error(eval_metrics))

    combined = pd.concat(flags, ignore_index=True)
    combined = combined.fillna("")
    return combined
