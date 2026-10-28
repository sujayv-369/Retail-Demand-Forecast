"""
Unit tests for recursive demand forecasting and data loading.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.data_loader import recompute_lag_features, prepare_features
from src.evaluation.metrics import mae, wape, compute_metrics


def test_recompute_lag_features_no_cross_contamination():
    """Verify that lags do not leak across different product/store series."""
    df = pd.DataFrame({
        "product_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
        "store_id": ["S1", "S1", "S1", "S1", "S1", "S1"],
        "sales_channel": ["offline", "offline", "offline", "offline", "offline", "offline"],
        "week_start_date": pd.to_datetime([
            "2025-01-01", "2025-01-08", "2025-01-15",
            "2025-01-01", "2025-01-08", "2025-01-15",
        ]),
        "units_sold": [10, 20, 30, 100, 200, 300],
    })

    result = recompute_lag_features(df)

    # First row of P2 should NOT take P1's last value as lag_1
    p2_first_lag = result.loc[(result["product_id"] == "P2") & (result["week_start_date"] == "2025-01-01"), "lag_1_sales"].values[0]
    assert np.isnan(p2_first_lag), f"Expected NaN for first row of P2, got {p2_first_lag}"

    # P1 second row should have lag_1 = 10
    p1_second_lag = result.loc[(result["product_id"] == "P1") & (result["week_start_date"] == "2025-01-08"), "lag_1_sales"].values[0]
    assert p1_second_lag == 10.0


def test_metrics_computation():
    """Verify MAE and WAPE match standard formulas."""
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    predicted = np.array([12.0, 18.0, 33.0, 38.0])

    m = compute_metrics(actual, predicted)
    expected_mae = (2 + 2 + 3 + 2) / 4.0
    expected_wape = 100.0 * (2 + 2 + 3 + 2) / 100.0

    assert m["mae"] == pytest.approx(expected_mae, rel=1e-3)
    assert m["wape"] == pytest.approx(expected_wape, rel=1e-3)
    assert m["n_observations"] == 4


def test_wape_zero_actuals():
    """WAPE should return None instead of dividing by zero when demand is 0."""
    actual = np.array([0.0, 0.0])
    predicted = np.array([5.0, 5.0])
    assert wape(actual, predicted) is None
