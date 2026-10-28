"""
CLI entry point for generating recursive forecasts.

Usage:
    python -m src.forecast.run_forecast [--horizon 4]
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from datetime import datetime

from src.config import EVALUATION_DIR, FORECAST_HORIZON
from src.evaluation.data_loader import load_raw_data, prepare_features
from src.forecast.recursive_forecast import load_model, generate_forecast_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate recursive demand forecasts")
    parser.add_argument(
        "--horizon", type=int, default=FORECAST_HORIZON,
        help="Number of forecast weeks ahead (default: 4)",
    )
    args = parser.parse_args()

    raw_df = load_raw_data()
    full_df = prepare_features(raw_df, recompute_lags=True, drop_na=True)
    model = load_model()

    forecast_df, origin = generate_forecast_run(model, full_df, horizon=args.horizon)

    run_id = f"forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    output_path = EVALUATION_DIR / f"{run_id}.json"

    forecast_data = {
        "run_id": run_id,
        "forecast_origin": str(origin),
        "created_at": datetime.now().isoformat(),
        "n_forecasts": len(forecast_df),
        "horizon_weeks": args.horizon,
        "forecasts": forecast_df.to_dict("records"),
    }

    with open(output_path, "w") as f:
        json.dump(forecast_data, f, indent=2, default=str)

    logger.info("Saved %d forecast rows to %s", len(forecast_df), output_path)

    # Clear analytics tools cache
    try:
        from src.analytics.tools import clear_cache
        clear_cache()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("FORECAST RUN COMPLETE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Forecast Origin: {origin}")
    print(f"Horizon: {args.horizon} weeks")
    print(f"Total Forecast Predictions: {len(forecast_df):,}")
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
