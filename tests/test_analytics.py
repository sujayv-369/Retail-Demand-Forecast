"""
Unit tests for analytics tools and allowlisted dispatcher.
"""

import pytest
from src.analytics.dispatcher import ToolDispatcher, ALLOWED_TOOLS
from src.analytics.tools import get_data_quality_report, compare_forecast_models


def test_dispatcher_rejects_unauthorized_functions():
    """Ensure dispatcher rejects functions not in the allowlist."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("drop_table", {"table_name": "orders"})

    assert "error" in result
    assert "not authorized" in result["error"].lower()


def test_dispatcher_validates_arguments():
    """Ensure dispatcher sanitizes and enforces parameter types."""
    dispatcher = ToolDispatcher()
    # Invalid integer for weeks parameter
    result = dispatcher.dispatch("get_product_forecast", {"product_id": "P0001", "weeks": "invalid_number"})

    assert "error" in result
    assert "invalid value" in result["error"].lower()


def test_compare_forecast_models_schema():
    """Verify compare_forecast_models returns the expected schema."""
    result = compare_forecast_models()
    assert "overall_comparison" in result
    assert "calculation_method" in result

    comparison = result["overall_comparison"]
    assert len(comparison) >= 3
    models = [c["model"] for c in comparison]
    assert "catboost" in models
    assert "naive" in models
    assert "ma4" in models


def test_revenue_and_pricing_overview_tool():
    """Verify get_revenue_and_pricing_overview returns valid revenue and price metrics."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("get_revenue_and_pricing_overview", {"group_by": "category", "period": "last_4_weeks"})

    assert "error" not in result
    assert "total_revenue" in result
    assert result["total_revenue"] > 0
    assert "currency" in result
    assert "unit_price_statistics" in result
    assert result["unit_price_statistics"]["average_unit_price"] > 0
    assert "breakdown" in result
    assert len(result["breakdown"]) > 0
    assert "revenue" in result["breakdown"][0]
    assert "revenue_share_pct" in result["breakdown"][0]


def test_demand_aggregates_includes_revenue():
    """Verify get_demand_aggregates includes both units and monetary revenue."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("get_demand_aggregates", {"group_by": "category", "period": "last_4_weeks"})

    assert "error" not in result
    assert "aggregates" in result
    assert len(result["aggregates"]) > 0
    first_item = result["aggregates"][0]
    assert "current_revenue" in first_item
    assert "previous_revenue" in first_item
    assert "average_unit_price" in first_item
    assert "current_units_sold" in first_item


def test_search_products_tool():
    """Verify search_products finds products and returns full names."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("search_products", {"query": "Vivo", "top_n": 5})

    assert "error" not in result
    assert "products" in result
    assert len(result["products"]) > 0
    first = result["products"][0]
    assert "product_id" in first
    assert "product_name" in first
    assert "brand" in first
    assert "Vivo" in first["brand"] or "Vivo" in first["product_name"]


def test_top_demand_changes_includes_product_names():
    """Verify get_top_demand_changes returns human-readable product names."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("get_top_demand_changes", {"direction": "highest", "top_n": 5})

    assert "error" not in result
    assert "products" in result
    assert len(result["products"]) > 0
    first = result["products"][0]
    assert "product_id" in first
    assert "product_name" in first
    assert len(first["product_name"]) > 0
    assert "brand" in first
    assert "category" in first
    assert isinstance(first["forecast_units"], int)
    assert isinstance(first["historical_units"], int)
    assert isinstance(first["units_change"], int)


def test_product_forecast_by_name_resolution():
    """Verify get_product_forecast accepts product names and returns product_name."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch("get_product_forecast", {"product_id": "Vivo V9", "weeks": 4})

    assert "error" not in result
    assert result["product_id"] == "P0001"
    assert "product_name" in result
    assert "Vivo V9" in result["product_name"]
    assert "brand" in result
    assert result["brand"] == "Vivo"


def test_geographic_demand_aggregates_state_and_brand():
    """Verify get_demand_aggregates can rank states for Samsung smartphones."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch(
        "get_demand_aggregates",
        {"group_by": "state", "brand": "Samsung", "category": "Smartphones", "period": "last_4_weeks"}
    )

    assert "error" not in result
    assert "aggregates" in result
    assert len(result["aggregates"]) > 0
    first = result["aggregates"][0]
    assert "state" in first
    assert "region" in first
    assert first["current_units_sold"] > 0
    assert "current_revenue" in first


def test_geographic_demand_aggregates_city():
    """Verify get_demand_aggregates attaches state and region when grouping by city."""
    dispatcher = ToolDispatcher()
    result = dispatcher.dispatch(
        "get_demand_aggregates",
        {"group_by": "city", "period": "last_4_weeks", "top_n": 5}
    )

    assert "error" not in result
    assert "aggregates" in result
    assert len(result["aggregates"]) > 0
    first = result["aggregates"][0]
    assert "city" in first
    assert "state" in first
    assert "region" in first
    assert first["state"] != "Unknown"



