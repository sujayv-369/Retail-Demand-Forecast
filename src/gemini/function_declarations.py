"""
Function declarations for the Gemini function-calling protocol.
These declare what tools are available to the LLM.
"""

from __future__ import annotations
from google.genai import types


def get_tool_declarations() -> list[types.Tool]:
    """Return the list of tool declarations for Gemini."""

    declarations = [
        types.FunctionDeclaration(
            name="get_forecast_summary",
            description=(
                "Get a summary of the latest demand forecast evaluation, including "
                "model metrics, comparison results, reliability information, and test period details."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "run_id": types.Schema(
                        type=types.Type.STRING,
                        description="Optional evaluation run ID. Omit to get the latest.",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="search_products",
            description=(
                "Search the product catalog by name, brand, subcategory, or product ID. "
                "Use this to find product names and their canonical IDs (e.g., 'Vivo V9', 'iPhone', 'boAt'). "
                "Returns product_id, product_name, brand, category, subcategory, and unit_price."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Product name, brand, model, or search keyword.",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Optional category filter (e.g., 'Smartphones', 'Chargers and cables').",
                    ),
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of matches to return (default 10).",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="get_product_forecast",
            description=(
                "Get the demand forecast for a specific product by product ID or product name (e.g., 'P0001' or 'Vivo V9'). "
                "Returns the full human-readable product name, brand, category, predicted units, unit price, "
                "and projected monetary revenue for each forecast week."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "product_id": types.Schema(
                        type=types.Type.STRING,
                        description="The product ID (e.g., 'P0001') or product name/keyword (e.g., 'Vivo V9', 'iPhone 13').",
                    ),
                    "weeks": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of forecast weeks to return (1–8, default 4).",
                    ),
                },
                required=["product_id"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_top_demand_changes",
            description=(
                "Rank products by forecast demand. Direction can be: "
                "'highest' (or 'most_demanding' for top products by total predicted units sold), "
                "'increase' (for largest projected demand gain vs historical), "
                "'decrease' (for largest projected drop), or 'highest_revenue' (for top projected revenue). "
                "Returns full product names, brands, categories, predicted units, unit prices, and revenue."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "direction": types.Schema(
                        type=types.Type.STRING,
                        description="'highest' (for most demanding products), 'increase' (largest gain), 'decrease' (largest drop), or 'highest_revenue'.",
                    ),
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of products to return (default 10, max 50).",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="compare_forecast_models",
            description=(
                "Compare CatBoost, naive baseline, and 4-week moving average models. "
                "Returns MAE, WAPE, and observation counts for each model."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),

        types.FunctionDeclaration(
            name="get_product_accuracy",
            description=(
                "Get a product's out-of-sample forecast accuracy metrics, including "
                "MAE, WAPE, sample counts, and any reliability flags."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "product_id": types.Schema(
                        type=types.Type.STRING,
                        description="The product ID to check accuracy for.",
                    ),
                },
                required=["product_id"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_demand_aggregates",
            description=(
                "Calculate total historical unit demand and monetary revenue broken down by "
                "state, city, region, category, brand, sales_channel, or product_id. "
                "Supports filters by brand (e.g. 'Samsung'), category (e.g. 'Smartphones' or 'mobiles'), "
                "state, region, or city. Always use this tool when asked which state, region, or city "
                "has the highest demand for a brand or product category."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "group_by": types.Schema(
                        type=types.Type.STRING,
                        description="Grouping dimension: 'state', 'city', 'region', 'category', 'brand', 'sales_channel', or 'product_id'. Default 'state' or 'category'.",
                    ),
                    "brand": types.Schema(
                        type=types.Type.STRING,
                        description="Optional brand filter (e.g., 'Samsung', 'Apple', 'Vivo', 'Xiaomi').",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Optional category filter (e.g., 'Smartphones', 'Laptops', 'Televisions', 'Earbuds').",
                    ),
                    "state": types.Schema(
                        type=types.Type.STRING,
                        description="Optional state filter (e.g., 'Maharashtra', 'Karnataka', 'Delhi', 'Haryana').",
                    ),
                    "region": types.Schema(
                        type=types.Type.STRING,
                        description="Optional region filter (e.g., 'South', 'North', 'West', 'East').",
                    ),
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="Optional city filter (e.g., 'Mumbai', 'Bengaluru', 'Delhi', 'Pune').",
                    ),
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="Comparison period: 'last_4_weeks', 'last_8_weeks', 'last_12_weeks', or 'all_time'. Default 'last_4_weeks'.",
                    ),
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of items to return (default 30, max 50).",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="get_revenue_and_pricing_overview",
            description=(
                "Comprehensive revenue, price, discount, and projected monetary revenue overview across "
                "the entire retail dataset. Can be grouped by 'state', 'city', 'region', 'category', 'brand', "
                "'sales_channel', or 'product_id', and filtered by brand, category, state, region, or city."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "group_by": types.Schema(
                        type=types.Type.STRING,
                        description="Grouping dimension: 'state', 'city', 'region', 'category', 'brand', 'sales_channel', or 'product_id' (default 'category').",
                    ),
                    "brand": types.Schema(
                        type=types.Type.STRING,
                        description="Optional brand filter (e.g., 'Samsung', 'Apple').",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Optional category filter (e.g., 'Smartphones', 'Laptops').",
                    ),
                    "state": types.Schema(
                        type=types.Type.STRING,
                        description="Optional state filter (e.g., 'Maharashtra').",
                    ),
                    "region": types.Schema(
                        type=types.Type.STRING,
                        description="Optional region filter (e.g., 'South', 'West').",
                    ),
                    "city": types.Schema(
                        type=types.Type.STRING,
                        description="Optional city filter.",
                    ),
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="Historical time window: 'last_4_weeks', 'last_8_weeks', 'last_12_weeks', or 'all_time' (default 'last_4_weeks').",
                    ),
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of top groups to return (default 15).",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="get_unreliable_products",
            description=(
                "Get products flagged as unreliable due to insufficient history, "
                "sparse demand, demand spikes, stockout distortion, or high forecast error."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of flagged items to return (default 20).",
                    ),
                },
            ),
        ),

        types.FunctionDeclaration(
            name="get_data_quality_report",
            description=(
                "Get a data quality report including row counts, missing values, "
                "target distribution, and coverage statistics."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),

        types.FunctionDeclaration(
            name="get_festival_demand_analysis",
            description=(
                "Analyze and rank retail demand across Indian festivals (Navratri, Diwali, Dhanteras, Eid, "
                "Christmas, Republic Day, Independence Day, etc.) and compare festival uplifts against "
                "regular non-festival baseline periods. ALWAYS invoke this tool when asked which festival "
                "had more demand, festival sales rankings, or the impact of festivals on product sales."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Optional product category filter (e.g. 'Smartphones', 'Televisions').",
                    ),
                    "brand": types.Schema(
                        type=types.Type.STRING,
                        description="Optional brand filter (e.g. 'Samsung', 'Apple').",
                    ),
                    "state": types.Schema(
                        type=types.Type.STRING,
                        description="Optional state filter (e.g. 'Maharashtra', 'Delhi').",
                    ),
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="Analysis time period: 'all_time' (default) or 'last_52_weeks'.",
                    ),
                    "top_n": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of festivals to return (default 20).",
                    ),
                },
            ),
        ),
    ]

    return [types.Tool(function_declarations=declarations)]
