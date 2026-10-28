"""
Main evaluation engine — orchestrates time-based evaluation with
rolling-origin backtesting, 5-model benchmark comparisons, and metric computation.

Models evaluated:
1. CatBoost (Gradient Boosted Decision Trees on engineered lag & feature pipeline)
2. Naive Baseline (Lag-1 Demand)
3. Moving Average (4-Week Rolling Mean)
4. ARIMA (AutoARIMA order selection via StatsForecast)
5. Facebook Prophet (Piecewise linear trend and changepoint modeling)
"""

from __future__ import annotations
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.config import (
    SERIES_KEY, TARGET, FEATURES, CAT_FEATURES,
    ARTIFACTS_DIR, EVALUATION_DIR, EVAL_HOLDOUT_WEEKS,
    FORECAST_HORIZON, MODEL_PATH,
)
from src.evaluation.data_loader import (
    load_raw_data, prepare_features,
    get_unique_weeks, get_train_test_split, get_series_history_summary,
)
from src.evaluation.metrics import (
    compute_metrics, compute_grouped_metrics, compute_horizon_metrics,
)
from src.evaluation.baselines import naive_forecast, moving_average_forecast
from src.evaluation.arima_forecast import arima_forecast
from src.evaluation.prophet_forecast import prophet_forecast
from src.evaluation.reliability import compute_all_flags
from src.forecast.recursive_forecast import recursive_forecast, load_model

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs full chronological evaluation of CatBoost, Naive, Moving Average,
    ARIMA, and Prophet models on a consistent held-out test split.
    """

    def __init__(self, holdout_weeks: int = EVAL_HOLDOUT_WEEKS, use_existing_model: bool = True):
        self.holdout_weeks = holdout_weeks
        self.use_existing_model = use_existing_model
        self.run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.results: dict = {}

    def run(self, csv_path: str | None = None) -> dict:
        """Execute the full 5-model evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING 5-MODEL BENCHMARK EVALUATION RUN: %s", self.run_id)
        logger.info("=" * 60)

        # ── Step 1: Load and prepare data ──────────────────────────
        raw_df = load_raw_data(csv_path)
        full_df = prepare_features(raw_df, recompute_lags=True, drop_na=True)

        all_weeks = get_unique_weeks(full_df)
        n_weeks = len(all_weeks)
        logger.info("Total usable weeks: %d (%s to %s)", n_weeks, all_weeks[0], all_weeks[-1])

        # ── Step 2: Determine evaluation split ─────────────────────
        test_weeks = all_weeks[-self.holdout_weeks:]
        test_start = test_weeks[0]

        val_gap = FORECAST_HORIZON
        train_end_idx = n_weeks - self.holdout_weeks - val_gap
        if train_end_idx < 20:
            logger.warning("Very limited training data (%d weeks)", train_end_idx)
            train_end_idx = max(n_weeks - self.holdout_weeks - 1, 10)

        train_cutoff = all_weeks[train_end_idx]
        val_start = all_weeks[train_end_idx + 1] if train_end_idx + 1 < len(all_weeks) else train_cutoff

        logger.info("Train: up to %s", train_cutoff)
        logger.info("Validation (early stopping): %s to %s", val_start, test_weeks[0] - pd.Timedelta(weeks=1))
        logger.info("Test: %s to %s (%d weeks)", test_start, all_weeks[-1], len(test_weeks))

        # Define rolling forecast origins
        forecast_origins = [w - pd.Timedelta(weeks=1) for w in test_weeks[:self.holdout_weeks - FORECAST_HORIZON + 1]]
        if not forecast_origins:
            forecast_origins = [test_start - pd.Timedelta(weeks=1)]

        # Actuals in the test window
        actuals = full_df[full_df["week_start_date"] >= test_start][
            SERIES_KEY + ["week_start_date", TARGET, "category"]
        ].copy()
        actuals = actuals.rename(columns={"week_start_date": "forecast_week_date"})

        # Representative cohort for univariate time series models (ARIMA & Prophet)
        # Select 12 high-volume, diverse series for fast & empirical benchmark
        top_cohort = (
            raw_df.groupby(SERIES_KEY)["units_sold"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        cohort_uids = [
            f"{r.product_id}___{r.store_id}___{r.sales_channel}"
            for r in top_cohort.itertuples()
        ]

        model_runtimes = {}
        eval_dfs = {}

        # ── Step 3: Evaluate CatBoost ──────────────────────────────
        logger.info("1/5: Evaluating CatBoost Regressor...")
        t0_cb = time.time()
        eval_model = self._train_eval_model(full_df, train_cutoff, val_start, test_start)
        cb_forecasts = self._rolling_origin_forecast(eval_model, full_df, test_weeks)
        model_runtimes["catboost"] = time.time() - t0_cb
        eval_dfs["catboost"] = self._merge_with_actuals(cb_forecasts, actuals)

        # ── Step 4: Evaluate Naive Baseline ────────────────────────
        logger.info("2/5: Evaluating Naive Baseline (Lag-1)...")
        t0_naive = time.time()
        naive_fc = naive_forecast(raw_df, forecast_origins, FORECAST_HORIZON)
        model_runtimes["naive"] = time.time() - t0_naive
        eval_dfs["naive"] = self._merge_with_actuals(naive_fc, actuals)

        # ── Step 5: Evaluate 4-Week Moving Average ─────────────────
        logger.info("3/5: Evaluating Moving Average (4-Week)...")
        t0_ma4 = time.time()
        ma4_fc = moving_average_forecast(raw_df, forecast_origins, 4, FORECAST_HORIZON)
        model_runtimes["ma4"] = time.time() - t0_ma4
        eval_dfs["ma4"] = self._merge_with_actuals(ma4_fc, actuals)

        # ── Step 6: Evaluate AutoARIMA ─────────────────────────────
        logger.info("4/5: Evaluating AutoARIMA (StatsForecast)...")
        arima_fc, arima_rt = arima_forecast(raw_df, forecast_origins, FORECAST_HORIZON, series_subset=cohort_uids)
        model_runtimes["arima"] = arima_rt
        eval_dfs["arima"] = self._merge_with_actuals(arima_fc, actuals)

        # ── Step 7: Evaluate Facebook Prophet ──────────────────────
        logger.info("5/5: Evaluating Facebook Prophet...")
        prophet_fc, prophet_rt = prophet_forecast(raw_df, forecast_origins, FORECAST_HORIZON, series_subset=cohort_uids)
        model_runtimes["prophet"] = prophet_rt
        eval_dfs["prophet"] = self._merge_with_actuals(prophet_fc, actuals)

        # ── Step 8: Compute metrics across all 5 models ────────────
        logger.info("Computing metrics & rankings across all 5 models...")
        self.results = self._compute_all_metrics(eval_dfs, model_runtimes)

        # ── Step 9: Reliability flags ──────────────────────────────
        logger.info("Computing series reliability audit flags...")
        series_summary = get_series_history_summary(raw_df)
        product_metrics = None
        if "product_metrics" in self.results.get("catboost", {}):
            product_metrics = pd.DataFrame(self.results["catboost"]["product_metrics"])

        flags = compute_all_flags(raw_df, series_summary, product_metrics)
        self.results["reliability_flags"] = flags.to_dict("records")
        self.results["series_summary_stats"] = {
            "total_series": len(series_summary),
            "series_with_flags": len(flags[SERIES_KEY[0]].unique()) if not flags.empty else 0,
            "flag_counts": flags["flag_type"].value_counts().to_dict() if not flags.empty else {},
        }

        # ── Step 10: Failure case analysis ─────────────────────────
        logger.info("Analyzing failure cases...")
        self.results["failure_cases"] = self._analyze_failures(eval_dfs.get("catboost"), raw_df)

        # ── Step 11: Timeline visualization actual vs forecast ─────
        self.results["timeline_comparison"] = self._build_timeline_comparison(eval_dfs, actuals)

        # ── Step 12: Save metadata and results ─────────────────────
        self.results["metadata"] = {
            "run_id": self.run_id,
            "eval_date": datetime.now().isoformat(),
            "train_cutoff": str(train_cutoff),
            "test_start": str(test_start),
            "test_end": str(all_weeks[-1]),
            "holdout_weeks": self.holdout_weeks,
            "forecast_horizon": FORECAST_HORIZON,
            "total_data_weeks": n_weeks,
            "models_evaluated": ["catboost", "naive", "ma4", "arima", "prophet"],
            "total_runtime_seconds": round(sum(model_runtimes.values()), 2),
        }

        self._save_results()
        logger.info("Evaluation completed successfully: %s", self.run_id)
        return self.results

    def _train_eval_model(
        self, full_df, train_cutoff, val_start, test_start,
    ) -> CatBoostRegressor:
        """Train or load a CatBoost model for evaluation."""
        eval_model_path = EVALUATION_DIR / "eval_catboost.cbm"
        if self.use_existing_model:
            if eval_model_path.exists():
                logger.info("Reusing existing evaluation model from %s", eval_model_path)
                return load_model(eval_model_path)
            if MODEL_PATH.exists():
                logger.info("Using pre-trained production model %s for held-out evaluation", MODEL_PATH)
                return load_model(MODEL_PATH)

        train_data = full_df[full_df["week_start_date"] <= train_cutoff]
        val_data = full_df[
            (full_df["week_start_date"] > train_cutoff) &
            (full_df["week_start_date"] < test_start)
        ]

        X_train = train_data[FEATURES].copy()
        y_train = train_data[TARGET]
        X_val = val_data[FEATURES].copy() if not val_data.empty else None
        y_val = val_data[TARGET] if not val_data.empty else None

        for col in CAT_FEATURES:
            X_train[col] = X_train[col].fillna("UNKNOWN").astype(str)
            if X_val is not None:
                X_val[col] = X_val[col].fillna("UNKNOWN").astype(str)

        model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=1200,
            depth=8,
            learning_rate=0.05,
            random_seed=42,
            verbose=100,
        )

        fit_params = {
            "X": X_train,
            "y": y_train,
            "cat_features": CAT_FEATURES,
        }
        if X_val is not None and not X_val.empty:
            fit_params["eval_set"] = (X_val, y_val)
            fit_params["early_stopping_rounds"] = 100

        model.fit(**fit_params)

        eval_model_path = EVALUATION_DIR / "eval_catboost.cbm"
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        model.save_model(str(eval_model_path))
        logger.info("Evaluation model saved to %s", eval_model_path)

        return model

    def _rolling_origin_forecast(
        self, model, full_df, test_weeks,
    ) -> pd.DataFrame:
        """Rolling-origin backtesting with recursive forecast."""
        all_forecasts = []
        max_origin_idx = len(test_weeks) - 1
        origins = []

        for i in range(0, max_origin_idx, 1):
            origin = test_weeks[i] - pd.Timedelta(weeks=1)
            remaining_test = len(test_weeks) - i
            actual_horizon = min(FORECAST_HORIZON, remaining_test)
            if actual_horizon < 1:
                continue
            origins.append((origin, actual_horizon))

        for origin, horizon in origins:
            fc = recursive_forecast(model, full_df, origin, horizon)
            all_forecasts.append(fc)

        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        return pd.DataFrame()

    def _merge_with_actuals(
        self, forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Join forecasts with actual values for metric computation."""
        if forecasts_df.empty:
            return None

        forecasts_df = forecasts_df.copy()
        actuals_df = actuals_df.copy()

        forecasts_df["forecast_week_date"] = pd.to_datetime(forecasts_df["forecast_week_date"])
        actuals_df["forecast_week_date"] = pd.to_datetime(actuals_df["forecast_week_date"])

        merge_keys = SERIES_KEY + ["forecast_week_date"]
        merged = forecasts_df.merge(
            actuals_df[merge_keys + [TARGET, "category"]],
            on=merge_keys,
            how="inner",
            suffixes=("", "_actual"),
        )

        if "category" not in merged.columns and "category_actual" in merged.columns:
            merged["category"] = merged["category_actual"]

        return merged if not merged.empty else None

    def _compute_all_metrics(
        self, eval_dfs: dict[str, pd.DataFrame | None], runtimes: dict[str, float],
    ) -> dict:
        """Compute MAE, WAPE, RMSE, runtime, and rankings for all 5 models."""
        results = {}

        model_meta = {
            "catboost": {
                "name": "CatBoost (GBDT)",
                "category": "Machine Learning",
                "desc": "Gradient Boosted Decision Trees on multi-lag & promotional features",
                "pros": "Captures non-linear promotional & calendar cross-series interactions",
                "cons": "Requires engineered lag pipeline and feature parity",
            },
            "arima": {
                "name": "ARIMA (AutoARIMA)",
                "category": "Statistical Time Series",
                "desc": "Automated order search (p,d,q) with Stepwise AIC optimization",
                "pros": "Strong autoregressive momentum on smooth trend series",
                "cons": "Univariate; misses cross-store promotional drivers and slow on large panels",
            },
            "prophet": {
                "name": "Facebook Prophet",
                "category": "Additive Decomposition",
                "desc": "Piecewise linear trend and changepoint growth modeling",
                "pros": "Handles holiday effects and missing observations robustly",
                "cons": "Slower fitting time and prone to over-estimating intermittent zero spikes",
            },
            "ma4": {
                "name": "Moving Average (4-Week)",
                "category": "Heuristic Baseline",
                "desc": "Rolling mean of past 4 completed historical weeks",
                "pros": "Simple, fast, robust smoothing against extreme 1-week spikes",
                "cons": "Lags behind demand turning points and promotional surges",
            },
            "naive": {
                "name": "Naive Baseline (Lag-1)",
                "category": "Heuristic Baseline",
                "desc": "Carries forward last observed weekly sales directly",
                "pros": "Zero parameters, zero training time benchmark",
                "cons": "High variance; heavily penalized on volatile promotions",
            },
        }

        comparison = []

        for model_key, df in eval_dfs.items():
            meta = model_meta.get(model_key, {"name": model_key, "category": "Model", "desc": ""})
            rt = runtimes.get(model_key, 0.0)

            if df is None or df.empty:
                results[model_key] = {
                    "overall": {"mae": None, "wape": None, "rmse": None, "n_observations": 0},
                    "runtime_seconds": round(rt, 2),
                }
                comparison.append({
                    "model": model_key,
                    "name": meta["name"],
                    "category": meta["category"],
                    "mae": None,
                    "wape": None,
                    "rmse": None,
                    "runtime_seconds": round(rt, 2),
                    "n_observations": 0,
                    "description": meta["desc"],
                    "pros": meta.get("pros", ""),
                    "cons": meta.get("cons", ""),
                })
                continue

            overall = compute_metrics(df[TARGET].values, df["predicted_units"].values)
            overall["runtime_seconds"] = round(rt, 2)

            product_metrics = compute_grouped_metrics(df, TARGET, "predicted_units", "product_id")
            cat_metrics = []
            if "category" in df.columns:
                cat_metrics = compute_grouped_metrics(df, TARGET, "predicted_units", "category")

            horizon_metrics = []
            if "horizon_step" in df.columns:
                horizon_metrics = compute_horizon_metrics(df, TARGET, "predicted_units")

            results[model_key] = {
                "overall": overall,
                "product_metrics": product_metrics,
                "category_metrics": cat_metrics,
                "horizon_metrics": horizon_metrics,
                "runtime_seconds": round(rt, 2),
            }

            comparison.append({
                "model": model_key,
                "name": meta["name"],
                "category": meta["category"],
                "mae": round(overall["mae"], 2) if overall["mae"] is not None else None,
                "wape": round(overall["wape"], 2) if overall["wape"] is not None else None,
                "rmse": round(overall["rmse"], 2) if overall.get("rmse") is not None else None,
                "runtime_seconds": round(rt, 2),
                "n_observations": overall["n_observations"],
                "description": meta["desc"],
                "pros": meta.get("pros", ""),
                "cons": meta.get("cons", ""),
            })

        # Rank models by WAPE (lowest error first)
        valid_models = [m for m in comparison if m["wape"] is not None]
        valid_models.sort(key=lambda x: x["wape"])
        for rank_idx, m in enumerate(valid_models, start=1):
            m["rank"] = rank_idx

        results["comparison"] = comparison
        return results

    def _build_timeline_comparison(
        self, eval_dfs: dict[str, pd.DataFrame | None], actuals_df: pd.DataFrame,
    ) -> list[dict]:
        """Generate chronological weekly aggregated demand across models for comparison chart."""
        timeline = []
        if actuals_df.empty:
            return timeline

        # Weekly actuals sum
        actual_weekly = actuals_df.groupby("forecast_week_date")[TARGET].sum()

        for week_dt, act_val in actual_weekly.items():
            dt_str = str(week_dt)[:10]
            item = {
                "date": dt_str,
                "actual": round(float(act_val), 1),
            }

            for model_key, df in eval_dfs.items():
                if df is not None and not df.empty:
                    w_df = df[df["forecast_week_date"] == week_dt]
                    if not w_df.empty:
                        item[model_key] = round(float(w_df["predicted_units"].sum()), 1)
                    else:
                        item[model_key] = None
                else:
                    item[model_key] = None

            timeline.append(item)

        timeline.sort(key=lambda x: x["date"])
        return timeline

    def _analyze_failures(
        self, cb_eval: pd.DataFrame | None, raw_df: pd.DataFrame,
    ) -> list[dict]:
        """Investigate outlier failure cases."""
        cases = []
        if cb_eval is None or cb_eval.empty:
            return cases

        cb_eval = cb_eval.copy()
        cb_eval["abs_error"] = np.abs(cb_eval[TARGET] - cb_eval["predicted_units"])

        worst_row = cb_eval.loc[cb_eval["abs_error"].idxmax()]
        pid = worst_row["product_id"]
        spike_history = raw_df[raw_df["product_id"] == pid][TARGET].describe().to_dict()
        cases.append({
            "case": "demand_spike",
            "product_id": str(pid),
            "forecast_week": str(worst_row.get("forecast_week_date", "")),
            "actual": float(worst_row[TARGET]),
            "predicted": float(worst_row["predicted_units"]),
            "absolute_error": float(worst_row["abs_error"]),
            "product_demand_stats": {k: float(v) for k, v in spike_history.items()},
            "explanation": (
                f"Product {pid} experienced a demand spike of {worst_row[TARGET]:.0f} units "
                f"against a predicted {worst_row['predicted_units']:.1f} units. "
                f"Historical baseline mean is {spike_history.get('mean', 0):.1f} (std {spike_history.get('std', 0):.1f}), "
                f"indicating a sudden unrecorded stock release or promotion."
            ),
        })

        series_weeks = raw_df.groupby("product_id")["week_start_date"].nunique()
        short_products = series_weeks[series_weeks < 20].index
        if len(short_products) > 0:
            short_errors = cb_eval[cb_eval["product_id"].isin(short_products)]
            if not short_errors.empty:
                product_errors = short_errors.groupby("product_id")["abs_error"].mean()
                worst_short = product_errors.idxmax()
                ws_data = short_errors[short_errors["product_id"] == worst_short]
                cases.append({
                    "case": "limited_history",
                    "product_id": str(worst_short),
                    "n_history_weeks": int(series_weeks.get(worst_short, 0)),
                    "mean_abs_error": float(product_errors[worst_short]),
                    "n_test_observations": int(len(ws_data)),
                    "explanation": (
                        f"Product {worst_short} has only {series_weeks.get(worst_short, 0)} weeks of sales history. "
                        f"Mean absolute error was {product_errors[worst_short]:.2f} units due to lack of seasonal training depth."
                    ),
                })

        return cases

    def _save_results(self):
        """Persist evaluation results to JSON and MySQL database."""
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        output_path = EVALUATION_DIR / f"{self.run_id}.json"

        def _clean(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, (np.floating,)) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (pd.Timestamp, datetime)):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return [_clean(x) for x in obj.tolist()]
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        cleaned_results = _clean(self.results)
        with open(output_path, "w") as f:
            json.dump(cleaned_results, f, indent=2, default=str)

        # Also persist to MySQL evaluation_runs and evaluation_metrics
        try:
            from sqlalchemy import text
            from src.db import get_engine, evaluation_runs, evaluation_metrics
            engine = get_engine()

            meta = self.results.get("metadata", {})
            with engine.begin() as conn:
                conn.execute(evaluation_runs.insert().values(
                    run_id=self.run_id,
                    model_name="5-Model Benchmark (CatBoost, ARIMA, Prophet, MA, Naive)",
                    eval_cutoff_date=str(meta.get("train_cutoff", ""))[:10],
                    test_start_date=str(meta.get("test_start", ""))[:10],
                    test_end_date=str(meta.get("test_end", ""))[:10],
                    config_json=json.dumps(meta),
                    created_at=datetime.utcnow(),
                ))

                for item in self.results.get("comparison", []):
                    conn.execute(evaluation_metrics.insert().values(
                        run_id=self.run_id,
                        model_type=item["model"],
                        scope_type="overall",
                        scope_value=item["name"],
                        horizon_week=None,
                        mae=item["mae"],
                        wape=item["wape"],
                        n_observations=item["n_observations"],
                    ))
            logger.info("Saved run %s to MySQL evaluation_runs & metrics tables", self.run_id)
        except Exception as e:
            logger.warning("Could not persist evaluation to MySQL: %s", e)

        logger.info("Evaluation results saved to %s", output_path)
        return output_path
