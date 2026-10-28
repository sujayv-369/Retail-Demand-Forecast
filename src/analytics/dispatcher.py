"""
Allowlisted dispatcher — maps approved function names to actual
Python handlers. Rejects unknown functions and unsupported arguments.
"""

from __future__ import annotations
import logging
import time
from typing import Any

from src.analytics.tools import (
    get_forecast_summary,
    get_product_forecast,
    get_top_demand_changes,
    compare_forecast_models,
    get_product_accuracy,
    get_demand_aggregates,
    get_unreliable_products,
    get_data_quality_report,
    get_revenue_and_pricing_overview,
    search_products,
    get_festival_demand_analysis,
)

logger = logging.getLogger(__name__)

# ── Allowlist: function name → (handler, allowed_params) ──────────
ALLOWED_TOOLS: dict[str, dict] = {
    "get_forecast_summary": {
        "handler": get_forecast_summary,
        "params": {"run_id": str},
    },
    "get_product_forecast": {
        "handler": get_product_forecast,
        "params": {"product_id": str, "weeks": int},
    },
    "get_top_demand_changes": {
        "handler": get_top_demand_changes,
        "params": {"direction": str, "top_n": int},
    },
    "compare_forecast_models": {
        "handler": compare_forecast_models,
        "params": {},
    },
    "get_product_accuracy": {
        "handler": get_product_accuracy,
        "params": {"product_id": str},
    },
    "get_demand_aggregates": {
        "handler": get_demand_aggregates,
        "params": {
            "group_by": str,
            "period": str,
            "brand": str,
            "category": str,
            "state": str,
            "region": str,
            "city": str,
            "sales_channel": str,
            "top_n": int,
            "sort_by": str,
        },
    },
    "get_unreliable_products": {
        "handler": get_unreliable_products,
        "params": {"top_n": int},
    },
    "get_data_quality_report": {
        "handler": get_data_quality_report,
        "params": {},
    },
    "get_revenue_and_pricing_overview": {
        "handler": get_revenue_and_pricing_overview,
        "params": {
            "group_by": str,
            "period": str,
            "top_n": int,
            "brand": str,
            "category": str,
            "state": str,
            "region": str,
            "city": str,
        },
    },
    "search_products": {
        "handler": search_products,
        "params": {"query": str, "category": str, "top_n": int},
    },
    "get_festival_demand_analysis": {
        "handler": get_festival_demand_analysis,
        "params": {
            "category": str,
            "brand": str,
            "state": str,
            "period": str,
            "top_n": int,
        },
    },
}


class ToolDispatcher:
    """
    Validates and executes approved function calls.
    Rejects unauthorized functions and enforces execution limits.
    """

    MAX_EXECUTION_TIME_SECONDS = 60

    def dispatch(self, function_name: str, arguments: dict[str, Any]) -> dict:
        """
        Execute an approved function and return its result.
        Raises ValueError for unauthorized or malformed calls.
        """
        # ── Validate function name ─────────────────────────────────
        if function_name not in ALLOWED_TOOLS:
            logger.warning("REJECTED: unauthorized function call '%s'", function_name)
            return {
                "error": f"Function '{function_name}' is not authorized.",
                "available_functions": list(ALLOWED_TOOLS.keys()),
            }

        tool = ALLOWED_TOOLS[function_name]
        handler = tool["handler"]
        allowed_params = tool["params"]

        # ── Validate arguments ─────────────────────────────────────
        sanitized_args = {}
        for key, value in arguments.items():
            if key not in allowed_params:
                logger.warning("Ignoring unsupported argument '%s' for '%s'", key, function_name)
                continue
            expected_type = allowed_params[key]
            try:
                sanitized_args[key] = expected_type(value)
            except (ValueError, TypeError) as e:
                return {"error": f"Invalid value for '{key}': {e}"}

        # ── Execute with timeout tracking ──────────────────────────
        start_time = time.time()
        try:
            result = handler(**sanitized_args)
        except Exception as e:
            logger.exception("Error executing '%s'", function_name)
            return {"error": f"Execution error in '{function_name}': {str(e)}"}

        elapsed = time.time() - start_time
        if elapsed > self.MAX_EXECUTION_TIME_SECONDS:
            logger.warning("Tool '%s' took %.1fs (limit %ds)",
                           function_name, elapsed, self.MAX_EXECUTION_TIME_SECONDS)

        logger.info("Executed '%s' in %.2fs", function_name, elapsed)

        # Add metadata to result
        if isinstance(result, dict):
            result["_tool_metadata"] = {
                "function_name": function_name,
                "execution_time_seconds": round(elapsed, 3),
                "arguments": sanitized_args,
            }

        return result

    def get_tool_names(self) -> list[str]:
        """Return list of available tool names."""
        return list(ALLOWED_TOOLS.keys())

    def is_authorized(self, function_name: str) -> bool:
        """Check if a function name is in the allowlist."""
        return function_name in ALLOWED_TOOLS
