"""
CLI entry point for running the model evaluation.

Usage:
    python -m src.evaluation.run_evaluation [--holdout-weeks 8] [--csv path/to/csv]
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run CatBoost model evaluation")
    parser.add_argument(
        "--holdout-weeks", type=int, default=8,
        help="Number of weeks for held-out test set (default: 8)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to raw CSV dataset (uses default if not specified)",
    )
    parser.add_argument(
        "--retrain", action="store_true", default=False,
        help="Force retraining of evaluation model rather than using pre-trained model",
    )
    args = parser.parse_args()

    from src.evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        holdout_weeks=args.holdout_weeks,
        use_existing_model=not args.retrain,
    )
    results = evaluator.run(csv_path=args.csv)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if "comparison" in results:
        print(f"\n{'Model':<12} {'MAE':>8} {'WAPE':>8} {'N obs':>8}")
        print("-" * 40)
        for row in results["comparison"]:
            mae_str = f"{row['mae']:.3f}" if row['mae'] is not None else "N/A"
            wape_str = f"{row['wape']:.1f}%" if row['wape'] is not None else "N/A"
            print(f"{row['model']:<12} {mae_str:>8} {wape_str:>8} {row['n_observations']:>8}")

    if "catboost" in results and "horizon_metrics" in results["catboost"]:
        print("\nCatBoost by forecast horizon:")
        for hm in results["catboost"]["horizon_metrics"]:
            mae_str = f"{hm['mae']:.3f}" if hm.get('mae') is not None else "N/A"
            wape_str = f"{hm['wape']:.1f}%" if hm.get('wape') is not None else "N/A"
            print(f"  Week {hm['horizon_week']}: MAE={mae_str}, WAPE={wape_str}, N={hm['n_observations']}")

    meta = results.get("metadata", {})
    print(f"\nRun ID: {meta.get('run_id', 'N/A')}")
    print(f"Test period: {meta.get('test_start', '?')} to {meta.get('test_end', '?')}")
    print(f"Results saved to: artifacts/evaluation/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
