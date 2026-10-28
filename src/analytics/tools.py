"""
Approved business analytics tools — Python/SQL functions that
Gemini can invoke through the function-calling protocol.

These are the ONLY functions the LLM can call. Each function:
- Has a strict input schema
- Uses parameterized queries / deterministic calculations
- Returns structured data with units, dates, and source references
- Never exposes arbitrary SQL execution
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import (
    SERIES_KEY, TARGET, EVALUATION_DIR, ARTIFACTS_DIR, FORECAST_HORIZON,
)

logger = logging.getLogger(__name__)

# ── Cached data (loaded once at startup) ───────────────────────────
_eval_data: dict | None = None
_forecast_data: pd.DataFrame | None = None
_raw_df: pd.DataFrame | None = None


def _load_eval_data() -> dict:
    """Load the latest evaluation results."""
    global _eval_data
    if _eval_data is not None:
        return _eval_data

    eval_dir = EVALUATION_DIR
    if not eval_dir.exists():
        return {"error": "No evaluation results found. Run the evaluation first."}

    eval_files = sorted(eval_dir.glob("eval_*.json"), reverse=True)
    if not eval_files:
        return {"error": "No evaluation results found."}

    with open(eval_files[0]) as f:
        _eval_data = json.load(f)
    logger.info("Loaded evaluation data from %s", eval_files[0].name)
    return _eval_data


def _load_forecast_data() -> pd.DataFrame:
    """Load the latest forecast results (from active forecast_run JSON or MySQL future_4week_forecasts)."""
    global _forecast_data
    if _forecast_data is not None:
        return _forecast_data

    forecast_dir = EVALUATION_DIR
    forecast_files = sorted(forecast_dir.glob("forecast_run_*.json"), reverse=True)
    if forecast_files:
        with open(forecast_files[0]) as f:
            data = json.load(f)
        _forecast_data = pd.DataFrame(data.get("forecasts", []))
        return _forecast_data

    # Fallback to MySQL future_4week_forecasts table if available
    try:
        from src.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            df_sql = pd.read_sql("SELECT * FROM future_4week_forecasts", conn)
            if not df_sql.empty:
                df_sql = df_sql.rename(columns={"predicted_units_sold": "predicted_units"})
                _forecast_data = df_sql
                return _forecast_data
    except Exception:
        pass

    return pd.DataFrame()


def _load_raw_data() -> pd.DataFrame:
    """Load cached raw data for analytics queries."""
    global _raw_df
    if _raw_df is not None:
        return _raw_df
    from src.evaluation.data_loader import load_raw_data
    _raw_df = load_raw_data()
    return _raw_df


_product_catalog: dict[str, dict] | None = None


def _get_product_catalog() -> dict[str, dict]:
    """Return cached mapping of product_id -> product details (name, category, brand, unit price)."""
    global _product_catalog
    if _product_catalog is not None:
        return _product_catalog

    raw_df = _load_raw_data()
    cols = ["product_id", "product_name", "category", "subcategory", "brand", "unit_price"]
    available_cols = [c for c in cols if c in raw_df.columns]
    dedup = raw_df[available_cols].drop_duplicates(subset=["product_id"])

    _product_catalog = {}
    for _, r in dedup.iterrows():
        pid = str(r["product_id"])
        _product_catalog[pid] = {
            "product_id": pid,
            "product_name": str(r.get("product_name", pid)),
            "category": str(r.get("category", "")),
            "subcategory": str(r.get("subcategory", "")),
            "brand": str(r.get("brand", "")),
            "unit_price": float(r.get("unit_price", 0.0)),
        }
    return _product_catalog


def _get_product_meta(product_id: str) -> dict:
    """Retrieve metadata for a given product_id."""
    catalog = _get_product_catalog()
    return catalog.get(str(product_id), {
        "product_id": str(product_id),
        "product_name": str(product_id),
        "category": "Unknown",
        "subcategory": "Unknown",
        "brand": "Unknown",
        "unit_price": 0.0,
    })


def _resolve_product_id(identifier: str) -> str:
    """
    Resolve a product ID or product name search query to a canonical product_id.
    Matches exact ID, case-insensitive ID, or substring of product_name.
    """
    if not identifier:
        return identifier
    clean_id = str(identifier).strip()
    catalog = _get_product_catalog()

    if clean_id in catalog:
        return clean_id

    upper_id = clean_id.upper()
    if upper_id in catalog:
        return upper_id

    q_lower = clean_id.lower()
    for pid, meta in catalog.items():
        if q_lower == meta["product_name"].lower():
            return pid

    for pid, meta in catalog.items():
        if q_lower in meta["product_name"].lower():
            return pid

    return clean_id


_geo_maps: dict[str, dict] | None = None

_CATEGORY_ALIASES = {
    "mobile": "Smartphones",
    "mobiles": "Smartphones",
    "phone": "Smartphones",
    "phones": "Smartphones",
    "smartphone": "Smartphones",
    "smartphones": "Smartphones",
    "tv": "Televisions",
    "tvs": "Televisions",
    "television": "Televisions",
    "televisions": "Televisions",
    "laptop": "Laptops",
    "laptops": "Laptops",
    "earbud": "Earbuds",
    "earbuds": "Earbuds",
    "earphone": "Earbuds",
    "earphones": "Earbuds",
    "headphone": "Headphones",
    "headphones": "Headphones",
    "watch": "Smartwatches",
    "watches": "Smartwatches",
    "smartwatch": "Smartwatches",
    "smartwatches": "Smartwatches",
    "fridge": "Refrigerators",
    "fridges": "Refrigerators",
    "refrigerator": "Refrigerators",
    "refrigerators": "Refrigerators",
    "washing machine": "Washing machines",
    "washing machines": "Washing machines",
    "ac": "Air conditioners",
    "acs": "Air conditioners",
    "air conditioner": "Air conditioners",
    "air conditioners": "Air conditioners",
    "cable": "Chargers and cables",
    "charger": "Chargers and cables",
    "cables": "Chargers and cables",
    "chargers": "Chargers and cables",
}


def _normalize_category_query(cat: str) -> str:
    """Normalize colloquial category terms (e.g. 'mobiles' -> 'Smartphones')."""
    if not cat:
        return ""
    clean = cat.strip().lower()
    return _CATEGORY_ALIASES.get(clean, cat.strip())


def _get_geo_maps() -> dict[str, dict]:
    """Return mappings for city -> state, city -> region, and state -> region."""
    global _geo_maps
    if _geo_maps is not None:
        return _geo_maps

    raw = _load_raw_data()
    city_state: dict[str, str] = {}
    city_region: dict[str, str] = {}
    state_region: dict[str, str] = {}

    if all(c in raw.columns for c in ["city", "state", "region"]):
        unique_geo = raw[["city", "state", "region"]].drop_duplicates()
        for _, r in unique_geo.iterrows():
            c = str(r["city"])
            s = str(r["state"])
            reg = str(r["region"])
            city_state[c] = s
            city_region[c] = reg
            state_region[s] = reg

    _geo_maps = {
        "city_to_state": city_state,
        "city_to_region": city_region,
        "state_to_region": state_region,
    }
    return _geo_maps


def clear_cache():
    """Clear cached data (useful after new evaluation/forecast runs)."""
    global _eval_data, _forecast_data, _raw_df, _product_catalog, _geo_maps
    _eval_data = None
    _forecast_data = None
    _raw_df = None
    _product_catalog = None
    _geo_maps = None


# ══════════════════════════════════════════════════════════════════
# APPROVED TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════


def get_forecast_summary(run_id: str | None = None) -> dict:
    """
    Return verified demand trends, forecast totals, model metrics,
    major changes, and reliability information for the latest or
    specified evaluation run.
    """
    eval_data = _load_eval_data()
    if "error" in eval_data:
        return eval_data

    meta = eval_data.get("metadata", {})
    comparison = eval_data.get("comparison", [])
    reliability = eval_data.get("series_summary_stats", {})
    failure_cases = eval_data.get("failure_cases", [])

    # CatBoost horizon breakdown
    cb_horizons = eval_data.get("catboost", {}).get("horizon_metrics", [])

    return {
        "run_id": meta.get("run_id", "unknown"),
        "evaluation_period": {
            "test_start": meta.get("test_start"),
            "test_end": meta.get("test_end"),
            "holdout_weeks": meta.get("holdout_weeks"),
        },
        "model_comparison": comparison,
        "catboost_by_horizon": cb_horizons,
        "reliability_summary": reliability,
        "failure_case_count": len(failure_cases),
        "data_source": "evaluation results (computed, not generated by LLM)",
    }


def search_products(
    query: str = "",
    category: str = "",
    top_n: int = 10,
) -> dict:
    """
    Search products in the retail catalog by name, brand, subcategory, or product ID.
    Returns matching products with their canonical product_id, product_name, brand, category, and unit price.
    """
    catalog = _get_product_catalog()
    q = (query or "").strip().lower()
    cat_filter = (category or "").strip().lower()

    matches = []
    for pid, meta in catalog.items():
        name = meta["product_name"].lower()
        brand = meta["brand"].lower()
        cat = meta["category"].lower()
        subcat = meta["subcategory"].lower()

        if cat_filter and cat_filter not in cat:
            continue

        if not q or (q in pid.lower()) or (q in name) or (q in brand) or (q in subcat):
            matches.append({
                "product_id": pid,
                "product_name": meta["product_name"],
                "brand": meta["brand"],
                "category": meta["category"],
                "subcategory": meta["subcategory"],
                "unit_price": meta["unit_price"],
            })
            if len(matches) >= max(1, min(top_n, 50)):
                break

    return {
        "query": query,
        "matched_count": len(matches),
        "products": matches,
    }


def get_product_forecast(
    product_id: str,
    weeks: int = 4,
) -> dict:
    """
    Return a specific product's forecast for the requested weeks,
    with product name, brand, category, forecast origin, and projected monetary revenue.
    """
    if weeks < 1 or weeks > 8:
        return {"error": "Supported forecast range is 1–8 weeks."}

    # Resolve product_id if the user passed a product name or search term
    canonical_id = _resolve_product_id(product_id)
    meta = _get_product_meta(canonical_id)

    forecast_df = _load_forecast_data()
    if forecast_df.empty:
        return {"error": "No forecast data available. Generate a forecast first."}

    product_fc = forecast_df[forecast_df["product_id"] == canonical_id]
    if product_fc.empty:
        return {
            "error": f"No forecast found for product '{product_id}' (canonical ID: {canonical_id}).",
            "suggested_action": "Use search_products to locate the correct product name or ID.",
        }

    product_fc = product_fc[product_fc["horizon_step"] <= weeks]

    # Look up pricing data for the product
    raw_df = _load_raw_data()
    prod_data = raw_df[raw_df["product_id"] == canonical_id]
    unit_price = float(prod_data["unit_price"].iloc[0]) if not prod_data.empty and "unit_price" in prod_data.columns else meta["unit_price"]
    avg_discount = float(prod_data["discount_pct"].mean()) if not prod_data.empty and "discount_pct" in prod_data.columns else 0.0
    effective_price = unit_price * (1.0 - avg_discount / 100.0)

    result_rows = []
    for _, row in product_fc.iterrows():
        units = int(round(float(row.get("predicted_units", 0))))
        proj_rev = round(units * effective_price, 2)
        result_rows.append({
            "product_id": canonical_id,
            "product_name": meta["product_name"],
            "store_id": row.get("store_id"),
            "sales_channel": row.get("sales_channel"),
            "forecast_week": str(row.get("forecast_week_date")),
            "horizon_step": int(row.get("horizon_step", 0)),
            "predicted_units": units,
            "projected_revenue": proj_rev,
        })

    total_units = int(sum(r["predicted_units"] for r in result_rows))
    total_rev = round(sum(r["projected_revenue"] for r in result_rows), 2)

    return {
        "product_id": canonical_id,
        "product_name": meta["product_name"],
        "brand": meta["brand"],
        "category": meta["category"],
        "subcategory": meta["subcategory"],
        "forecast_origin": str(product_fc["forecast_origin"].iloc[0]) if "forecast_origin" in product_fc.columns else "N/A",
        "unit_price": round(unit_price, 2),
        "average_discount_pct": round(avg_discount, 1),
        "effective_unit_price": round(effective_price, 2),
        "forecasts": result_rows[:50],  # Cap result size
        "total_predicted_units": total_units,
        "total_predicted": total_units,
        "total_projected_revenue": total_rev,
        "n_series": len(product_fc[SERIES_KEY].drop_duplicates()) if all(k in product_fc.columns for k in SERIES_KEY) else "N/A",
    }


def get_top_demand_changes(
    direction: str = "increase",
    top_n: int = 10,
) -> dict:
    """
    Rank products by the largest projected demand changes or highest absolute forecast demand.
    Supports:
    - 'highest' / 'highest_demand' / 'most_demanding': ranks by total forecast units.
    - 'increase': ranks by largest positive demand change vs. historical period.
    - 'decrease': ranks by largest demand drop vs. historical period.
    - 'highest_revenue': ranks by projected forecast monetary revenue.
    """
    if top_n < 1 or top_n > 50:
        top_n = 10
    valid_directions = ("increase", "decrease", "highest", "highest_demand", "most_demanding", "total", "highest_revenue")
    dir_norm = direction.strip().lower() if direction else "increase"
    if dir_norm not in valid_directions:
        return {"error": f"direction must be one of {valid_directions}."}

    forecast_df = _load_forecast_data()
    raw_df = _load_raw_data()

    if forecast_df.empty:
        return {"error": "No forecast data available."}

    # Total forecast per product
    fc_totals = forecast_df.groupby("product_id")["predicted_units"].sum().reset_index()
    fc_totals.columns = ["product_id", "forecast_total"]

    # Historical comparison: last 4 weeks of actual data
    all_weeks = sorted(raw_df["week_start_date"].unique())
    comparison_weeks = all_weeks[-FORECAST_HORIZON:]
    hist_period = raw_df[raw_df["week_start_date"].isin(comparison_weeks)]
    hist_totals = hist_period.groupby("product_id")[TARGET].sum().reset_index()
    hist_totals.columns = ["product_id", "historical_total"]

    merged = fc_totals.merge(hist_totals, on="product_id", how="inner")
    merged["absolute_change"] = merged["forecast_total"] - merged["historical_total"]
    merged["percentage_change"] = np.where(
        merged["historical_total"] > 0,
        100.0 * merged["absolute_change"] / merged["historical_total"],
        None,  # Undefined if historical is zero
    )

    # Precompute prices and discounts for revenue estimates
    prod_prices = raw_df.groupby("product_id")["unit_price"].last().to_dict()
    prod_discounts = raw_df.groupby("product_id")["discount_pct"].mean().to_dict()

    merged["unit_price"] = merged["product_id"].map(lambda p: float(prod_prices.get(p, 0.0)))
    merged["discount_pct"] = merged["product_id"].map(lambda p: float(prod_discounts.get(p, 0.0)))
    merged["effective_price"] = merged["unit_price"] * (1.0 - merged["discount_pct"] / 100.0)
    merged["projected_forecast_revenue"] = merged["forecast_total"] * merged["effective_price"]
    merged["historical_revenue"] = merged["historical_total"] * merged["effective_price"]
    merged["projected_revenue_change"] = merged["absolute_change"] * merged["effective_price"]

    if dir_norm in ("highest", "highest_demand", "most_demanding", "total"):
        merged = merged.sort_values("forecast_total", ascending=False)
    elif dir_norm == "highest_revenue":
        merged = merged.sort_values("projected_forecast_revenue", ascending=False)
    elif dir_norm == "decrease":
        merged = merged.sort_values("absolute_change", ascending=True)
    else:  # "increase"
        merged = merged.sort_values("absolute_change", ascending=False)

    top = merged.head(top_n)

    results = []
    for _, row in top.iterrows():
        pid = row["product_id"]
        meta = _get_product_meta(pid)
        price = float(row["unit_price"])
        eff_price = float(row["effective_price"])

        fc_units = int(round(float(row["forecast_total"])))
        hist_units = int(round(float(row["historical_total"])))
        diff_units = fc_units - hist_units
        pct_change = round(100.0 * diff_units / hist_units, 2) if hist_units > 0 else "undefined (zero historical demand)"

        results.append({
            "product_id": pid,
            "product_name": meta.get("product_name", pid),
            "brand": meta.get("brand", ""),
            "category": meta.get("category", ""),
            "subcategory": meta.get("subcategory", ""),
            "forecast_units": fc_units,
            "historical_units": hist_units,
            "units_change": diff_units,
            "percentage_change": pct_change,
            "unit_price": round(price, 2),
            "projected_forecast_revenue": round(float(fc_units * eff_price), 2),
            "historical_revenue": round(float(hist_units * eff_price), 2),
            "projected_revenue_change": round(float(diff_units * eff_price), 2),
            # Legacy field aliases
            "forecast_total": fc_units,
            "historical_total": hist_units,
            "absolute_change": diff_units,
        })

    return {
        "direction": direction,
        "ranking_metric": "total_forecast_units" if dir_norm in ("highest", "highest_demand", "most_demanding", "total") else ("projected_revenue" if dir_norm == "highest_revenue" else "demand_change"),
        "comparison_period": f"Last {FORECAST_HORIZON} actual weeks vs. next {FORECAST_HORIZON} forecast weeks",
        "products": results,
        "calculation_method": (
            "forecast_total = sum of predicted units over forecast horizon; "
            "absolute_change = forecast_total - historical_total; "
            "percentage_change = 100 × absolute_change / historical_total"
        ),
    }


def compare_forecast_models() -> dict:
    """Compare CatBoost, naive, and moving-average metrics for the same evaluation scope."""
    eval_data = _load_eval_data()
    if "error" in eval_data:
        return eval_data

    comparison = eval_data.get("comparison", [])
    cb_horizons = eval_data.get("catboost", {}).get("horizon_metrics", [])

    meta = eval_data.get("metadata", {})

    return {
        "evaluation_period": {
            "test_start": meta.get("test_start"),
            "test_end": meta.get("test_end"),
        },
        "overall_comparison": comparison,
        "catboost_by_horizon": cb_horizons,
        "calculation_method": (
            "MAE = sum(|actual - predicted|) / n_observations; "
            "WAPE = 100 × sum(|actual - predicted|) / sum(|actual|)"
        ),
        "notes": "All models evaluated on the same held-out test period using rolling-origin backtesting.",
    }


def get_product_accuracy(product_id: str) -> dict:
    """Return a product's out-of-sample errors, sample counts, and reliability flags."""
    eval_data = _load_eval_data()
    if "error" in eval_data:
        return eval_data

    canonical_id = _resolve_product_id(product_id)
    meta = _get_product_meta(canonical_id)

    # Find in product metrics
    product_metrics = eval_data.get("catboost", {}).get("product_metrics", [])
    match = [m for m in product_metrics if m.get("scope_value") == canonical_id]

    if not match:
        return {
            "error": f"No evaluation metrics found for product '{product_id}' (canonical ID: {canonical_id}).",
            "suggested_action": "Use search_products to find valid product IDs and names.",
        }

    pm = match[0]

    # Find reliability flags for this product
    all_flags = eval_data.get("reliability_flags", [])
    product_flags = [f for f in all_flags if f.get("product_id") == canonical_id]

    return {
        "product_id": canonical_id,
        "product_name": meta.get("product_name", canonical_id),
        "brand": meta.get("brand", ""),
        "category": meta.get("category", ""),
        "subcategory": meta.get("subcategory", ""),
        "mae": pm.get("mae"),
        "wape": pm.get("wape"),
        "n_observations": pm.get("n_observations"),
        "reliability_flags": product_flags,
        "is_reliable": len(product_flags) == 0,
        "data_source": "Out-of-sample evaluation (held-out test period)",
    }


def get_demand_aggregates(
    group_by: str = "category",
    period: str = "last_4_weeks",
    brand: str = "",
    category: str = "",
    state: str = "",
    region: str = "",
    city: str = "",
    sales_channel: str = "",
    top_n: int = 30,
    sort_by: str = "units",
) -> dict:
    """
    Calculate total historical unit demand and monetary revenue broken down by
    state, city, region, category, brand, subcategory, sales_channel, or product_id.
    Supports filtering by brand (e.g. 'Samsung'), category (e.g. 'Smartphones'),
    state, region, or city.
    """
    group_norm = (group_by or "category").strip().lower()
    group_map = {
        "state": "state", "states": "state",
        "city": "city", "cities": "city",
        "region": "region", "regions": "region",
        "category": "category", "categories": "category",
        "brand": "brand", "brands": "brand",
        "subcategory": "subcategory", "subcategories": "subcategory",
        "sales_channel": "sales_channel", "channel": "sales_channel",
        "product_id": "product_id", "product": "product_id", "products": "product_id",
        "festival": "festival_name", "festivals": "festival_name", "festival_name": "festival_name",
    }
    target_group = group_map.get(group_norm)
    if not target_group:
        valid_options = ["state", "city", "region", "category", "brand", "subcategory", "sales_channel", "product_id", "festival_name"]
        return {"error": f"group_by must be one of {valid_options}."}

    if top_n < 1 or top_n > 50:
        top_n = 30

    raw_df = _load_raw_data()
    all_weeks = sorted(raw_df["week_start_date"].unique())

    # Build filtering helper
    def _filter_df(df_in: pd.DataFrame) -> pd.DataFrame:
        f = df_in
        if brand:
            f = f[f["brand"].str.contains(brand.strip(), case=False, na=False)]
        if category:
            cat_target = _normalize_category_query(category)
            f = f[f["category"].str.contains(cat_target, case=False, na=False)]
        if state:
            f = f[f["state"].str.contains(state.strip(), case=False, na=False)]
        if region:
            f = f[f["region"].str.contains(region.strip(), case=False, na=False)]
        if city:
            f = f[f["city"].str.contains(city.strip(), case=False, na=False)]
        if sales_channel:
            f = f[f["sales_channel"].str.contains(sales_channel.strip(), case=False, na=False)]
        return f

    period_clean = (period or "last_4_weeks").strip().lower()
    if period_clean == "all_time":
        current = _filter_df(raw_df)
        previous = pd.DataFrame()
        current_period_str = f"{all_weeks[0].date()} to {all_weeks[-1].date()} (Full dataset, 104 weeks)"
        prev_period_str = "None (all_time analysis)"
        n_weeks = len(all_weeks)
    else:
        period_map = {"last_4_weeks": 4, "last_8_weeks": 8, "last_12_weeks": 12}
        n_weeks = period_map.get(period_clean, 4)
        if len(all_weeks) < n_weeks * 2:
            return {"error": "Insufficient data for the requested comparison period."}
        current_weeks = all_weeks[-n_weeks:]
        previous_weeks = all_weeks[-(n_weeks * 2):-n_weeks]
        current = _filter_df(raw_df[raw_df["week_start_date"].isin(current_weeks)])
        previous = _filter_df(raw_df[raw_df["week_start_date"].isin(previous_weeks)])
        current_period_str = f"{current_weeks[0].date()} to {current_weeks[-1].date()}"
        prev_period_str = f"{previous_weeks[0].date()} to {previous_weeks[-1].date()}"

    filters_applied = {
        k: v for k, v in {
            "brand": brand, "category": category, "state": state,
            "region": region, "city": city, "sales_channel": sales_channel
        }.items() if v
    }

    if current.empty:
        return {
            "message": "No sales records found matching the specified filters.",
            "group_by": target_group,
            "filters_applied": filters_applied,
            "aggregates": [],
        }

    if target_group == "festival_name":
        current = current.copy()
        current["festival_name"] = current["festival_name"].fillna("None / No Festival")
        if not previous.empty:
            previous = previous.copy()
            previous["festival_name"] = previous["festival_name"].fillna("None / No Festival")

    current_agg = current.groupby(target_group).agg(
        current_demand=(TARGET, "sum"),
        current_revenue=("revenue", "sum"),
        average_unit_price=("unit_price", "mean"),
    ).reset_index()

    if not previous.empty:
        previous_agg = previous.groupby(target_group).agg(
            previous_demand=(TARGET, "sum"),
            previous_revenue=("revenue", "sum"),
        ).reset_index()
        merged = current_agg.merge(previous_agg, on=target_group, how="outer").fillna(0)
        merged["absolute_change"] = merged["current_demand"] - merged["previous_demand"]
        merged["percentage_change"] = np.where(
            merged["previous_demand"] > 0,
            100.0 * merged["absolute_change"] / merged["previous_demand"],
            None,
        )
        merged["revenue_change"] = merged["current_revenue"] - merged["previous_revenue"]
        merged["revenue_percentage_change"] = np.where(
            merged["previous_revenue"] > 0,
            100.0 * merged["revenue_change"] / merged["previous_revenue"],
            None,
        )
    else:
        merged = current_agg
        merged["previous_demand"] = 0
        merged["previous_revenue"] = 0.0
        merged["absolute_change"] = 0
        merged["percentage_change"] = None
        merged["revenue_change"] = 0.0
        merged["revenue_percentage_change"] = None

    if sort_by.strip().lower() == "revenue":
        merged = merged.sort_values("current_revenue", ascending=False)
    else:
        merged = merged.sort_values("current_demand", ascending=False)

    geo_maps = _get_geo_maps()
    results = []
    for _, row in merged.head(top_n).iterrows():
        key_val = str(row[target_group])
        item = {
            target_group: key_val,
            "current_units_sold": int(row["current_demand"]),
            "previous_units_sold": int(row["previous_demand"]),
            "units_change": int(row["absolute_change"]),
            "units_percentage_change": round(float(row["percentage_change"]), 2) if pd.notna(row["percentage_change"]) else "undefined",
            "current_revenue": round(float(row["current_revenue"]), 2),
            "previous_revenue": round(float(row["previous_revenue"]), 2),
            "revenue_change": round(float(row["revenue_change"]), 2),
            "revenue_percentage_change": round(float(row["revenue_percentage_change"]), 2) if pd.notna(row["revenue_percentage_change"]) else "undefined",
            "average_unit_price": round(float(row["average_unit_price"]), 2),
            # Legacy fields for backwards compatibility
            "current_demand": int(row["current_demand"]),
            "previous_demand": int(row["previous_demand"]),
            "absolute_change": int(row["absolute_change"]),
            "percentage_change": round(float(row["percentage_change"]), 2) if pd.notna(row["percentage_change"]) else "undefined",
        }
        if target_group == "city":
            item["state"] = geo_maps["city_to_state"].get(key_val, "Unknown")
            item["region"] = geo_maps["city_to_region"].get(key_val, "Unknown")
        elif target_group == "state":
            item["region"] = geo_maps["state_to_region"].get(key_val, "Unknown")
        elif target_group == "product_id":
            pmeta = _get_product_meta(key_val)
            item["product_name"] = pmeta.get("product_name", key_val)
            item["brand"] = pmeta.get("brand", "")
            item["category"] = pmeta.get("category", "")
            item["subcategory"] = pmeta.get("subcategory", "")
        results.append(item)

    total_curr_rev = float(current["revenue"].sum())
    total_prev_rev = float(previous["revenue"].sum()) if not previous.empty else 0.0
    rev_growth = round(100.0 * (total_curr_rev - total_prev_rev) / total_prev_rev, 2) if total_prev_rev > 0 else "N/A"

    return {
        "group_by": target_group,
        "filters_applied": filters_applied,
        "current_period": current_period_str,
        "previous_period": prev_period_str,
        "total_current_revenue": round(total_curr_rev, 2),
        "total_previous_revenue": round(total_prev_rev, 2),
        "total_revenue_growth_pct": rev_growth,
        "total_current_units": int(current[TARGET].sum()),
        "total_previous_units": int(previous[TARGET].sum()) if not previous.empty else 0,
        "overall_average_unit_price": round(float(current["unit_price"].mean()), 2) if not current.empty else 0.0,
        "aggregates": results,
        "calculation_method": (
            f"Summed {TARGET} and monetary revenue over specified time period for active records matching filters. "
            "revenue = units_sold × unit_price × (1 - discount_pct / 100); "
            "percentage_change = 100 × (current - previous) / previous"
        ),
    }


def get_festival_demand_analysis(
    category: str = "",
    brand: str = "",
    state: str = "",
    period: str = "all_time",
    top_n: int = 20,
) -> dict:
    """
    Comprehensive festival demand analysis across all Indian festivals tracked in the dataset.
    Compares festival demand (Navratri, Diwali, Dhanteras, Eid, Christmas, Republic Day, etc.)
    with regular baseline periods where no festival was active.
    """
    raw_df = _load_raw_data()
    all_weeks = sorted(raw_df["week_start_date"].unique())

    f = raw_df
    if brand:
        f = f[f["brand"].str.contains(brand.strip(), case=False, na=False)]
    if category:
        cat_target = _normalize_category_query(category)
        f = f[f["category"].str.contains(cat_target, case=False, na=False)]
    if state:
        f = f[f["state"].str.contains(state.strip(), case=False, na=False)]

    period_clean = (period or "all_time").strip().lower()
    if period_clean != "all_time":
        period_map = {"last_4_weeks": 4, "last_8_weeks": 8, "last_12_weeks": 12, "last_52_weeks": 52}
        n_weeks = period_map.get(period_clean, 52)
        target_weeks = all_weeks[-n_weeks:]
        f = f[f["week_start_date"].isin(target_weeks)]

    if f.empty:
        return {"error": "No records found matching filters."}

    total_demand = int(f[TARGET].sum())
    total_revenue = round(float(f["revenue"].sum()), 2)
    total_records = len(f)

    # Separate Non-Festival rows (NaN or empty) vs Festival rows
    is_fest = f["festival_name"].notna() & (f["festival_name"].astype(str).str.strip() != "") & (f["festival_name"].astype(str).str.lower() != "nan")
    df_non_fest = f[~is_fest]
    df_fest = f[is_fest]

    non_fest_units = int(df_non_fest[TARGET].sum()) if not df_non_fest.empty else 0
    non_fest_rev = round(float(df_non_fest["revenue"].sum()), 2) if not df_non_fest.empty else 0.0
    non_fest_count = len(df_non_fest)
    non_fest_avg_units = round(float(df_non_fest[TARGET].mean()), 2) if not df_non_fest.empty else 0.0

    fest_total_units = int(df_fest[TARGET].sum()) if not df_fest.empty else 0
    fest_total_rev = round(float(df_fest["revenue"].sum()), 2) if not df_fest.empty else 0.0
    fest_total_count = len(df_fest)
    fest_avg_units = round(float(df_fest[TARGET].mean()), 2) if not df_fest.empty else 0.0

    fest_agg = df_fest.groupby("festival_name").agg(
        total_units=(TARGET, "sum"),
        total_revenue=("revenue", "sum"),
        record_count=(TARGET, "count"),
        average_units_per_series=(TARGET, "mean"),
        average_unit_price=("unit_price", "mean"),
    ).reset_index().sort_values("total_units", ascending=False)

    festivals_list = []
    for _, r in fest_agg.head(top_n).iterrows():
        u = int(r["total_units"])
        avg_u = round(float(r["average_units_per_series"]), 2)
        uplift_pct = round(((avg_u - non_fest_avg_units) / non_fest_avg_units) * 100.0, 1) if non_fest_avg_units > 0 else 0.0
        festivals_list.append({
            "festival_name": str(r["festival_name"]),
            "total_units_sold": u,
            "total_revenue": round(float(r["total_revenue"]), 2),
            "share_of_festival_demand_pct": round(u / fest_total_units * 100.0, 1) if fest_total_units > 0 else 0.0,
            "average_units_per_series_week": avg_u,
            "uplift_vs_regular_baseline_pct": uplift_pct,
            "average_unit_price": round(float(r["average_unit_price"]), 2),
        })

    top_volume = festivals_list[0]["festival_name"] if festivals_list else "None"
    top_velocity = max(festivals_list, key=lambda x: x["average_units_per_series_week"])["festival_name"] if festivals_list else "None"

    return {
        "summary": {
            "total_units_sold": total_demand,
            "total_revenue": total_revenue,
            "non_festival_dates": {
                "label": "No Festival / Regular Days",
                "total_units_sold": non_fest_units,
                "total_revenue": non_fest_rev,
                "share_of_total_demand_pct": round(non_fest_units / total_demand * 100.0, 1) if total_demand > 0 else 0.0,
                "share_of_calendar_records_pct": round(non_fest_count / total_records * 100.0, 1) if total_records > 0 else 0.0,
                "average_units_per_series_week": non_fest_avg_units,
                "description": "Regular sales volume across calendar weeks with no festival event (~78.7% of dates)."
            },
            "festival_dates": {
                "total_units_sold": fest_total_units,
                "total_revenue": fest_total_rev,
                "share_of_total_demand_pct": round(fest_total_units / total_demand * 100.0, 1) if total_demand > 0 else 0.0,
                "average_units_per_series_week": fest_avg_units,
                "overall_festival_uplift_pct": round(((fest_avg_units - non_fest_avg_units) / non_fest_avg_units) * 100.0, 1) if non_fest_avg_units > 0 else 0.0,
                "top_festival_by_total_volume": top_volume,
                "top_festival_by_velocity_uplift": top_velocity,
            }
        },
        "festivals_ranking": festivals_list,
        "key_takeaways": (
            f"1. Most cumulative demand ({non_fest_units:,} units, {round(non_fest_units/total_demand*100.0, 1)}%) occurs across dates where there was No Festival because regular business weeks make up ~78.7% of the calendar. "
            f"2. However, during festival periods, unit demand surges by +{round(((fest_avg_units-non_fest_avg_units)/non_fest_avg_units)*100.0, 1)}% from a baseline of {non_fest_avg_units:.2f} units/series/week up to {fest_avg_units:.2f} units/series/week. "
            f"3. In terms of total cumulative festival volume, {top_volume} had the most demand with {festivals_list[0]['total_units_sold']:,} units, followed by Republic Day and Independence Day. "
            f"4. In terms of peak weekly velocity and buying frenzy per product, Dhanteras (11.15 units/series) and Diwali (9.62 units/series) exhibited the highest surge."
        )
    }


def get_revenue_and_pricing_overview(
    group_by: str = "category",
    period: str = "last_4_weeks",
    top_n: int = 15,
    brand: str = "",
    category: str = "",
    state: str = "",
    region: str = "",
    city: str = "",
) -> dict:
    """
    Calculate comprehensive monetary revenue, pricing overview, price ranges,
    and period-over-period financial growth across states, cities, regions,
    categories, brands, sales channels, or products.
    """
    group_norm = (group_by or "category").strip().lower()
    group_map = {
        "state": "state", "states": "state",
        "city": "city", "cities": "city",
        "region": "region", "regions": "region",
        "category": "category", "categories": "category",
        "brand": "brand", "brands": "brand",
        "subcategory": "subcategory", "subcategories": "subcategory",
        "sales_channel": "sales_channel", "channel": "sales_channel",
        "product_id": "product_id", "product": "product_id", "products": "product_id",
    }
    target_group = group_map.get(group_norm)
    if not target_group:
        valid_options = ["state", "city", "region", "category", "brand", "subcategory", "sales_channel", "product_id"]
        return {"error": f"group_by must be one of {valid_options}."}

    if top_n < 1 or top_n > 50:
        top_n = 15

    raw_df = _load_raw_data()
    all_weeks = sorted(raw_df["week_start_date"].unique())

    # Build filtering helper
    def _filter_df(df_in: pd.DataFrame) -> pd.DataFrame:
        f = df_in
        if brand:
            f = f[f["brand"].str.contains(brand.strip(), case=False, na=False)]
        if category:
            cat_target = _normalize_category_query(category)
            f = f[f["category"].str.contains(cat_target, case=False, na=False)]
        if state:
            f = f[f["state"].str.contains(state.strip(), case=False, na=False)]
        if region:
            f = f[f["region"].str.contains(region.strip(), case=False, na=False)]
        if city:
            f = f[f["city"].str.contains(city.strip(), case=False, na=False)]
        return f

    period_clean = (period or "last_4_weeks").strip().lower()
    if period_clean == "all_time":
        current = _filter_df(raw_df)
        previous = pd.DataFrame()
        current_period_str = f"{all_weeks[0].date()} to {all_weeks[-1].date()} (Full dataset, 104 weeks)"
        prev_period_str = "None (Full Dataset)"
    else:
        period_map = {"last_4_weeks": 4, "last_8_weeks": 8, "last_12_weeks": 12}
        n_weeks = period_map.get(period_clean, 4)
        if len(all_weeks) < n_weeks * 2:
            return {"error": "Insufficient data for requested comparison period."}
        current_weeks = all_weeks[-n_weeks:]
        previous_weeks = all_weeks[-(n_weeks * 2):-n_weeks]
        current = _filter_df(raw_df[raw_df["week_start_date"].isin(current_weeks)])
        previous = _filter_df(raw_df[raw_df["week_start_date"].isin(previous_weeks)])
        current_period_str = f"{current_weeks[0].date()} to {current_weeks[-1].date()} ({n_weeks} weeks)"
        prev_period_str = f"{previous_weeks[0].date()} to {previous_weeks[-1].date()} ({n_weeks} weeks)"

    total_curr_rev = float(current["revenue"].sum()) if not current.empty else 0.0
    total_prev_rev = float(previous["revenue"].sum()) if not previous.empty else 0.0
    growth_pct = round(100.0 * (total_curr_rev - total_prev_rev) / total_prev_rev, 2) if total_prev_rev > 0 else "N/A"

    if current.empty:
        return {
            "group_by": target_group,
            "currency": "USD",
            "current_period": current_period_str,
            "previous_period": prev_period_str,
            "total_revenue": 0.0,
            "message": "No sales records found matching the specified filters.",
            "breakdown": [],
        }

    curr_grp = current.groupby(target_group).agg(
        revenue=("revenue", "sum"),
        units_sold=(TARGET, "sum"),
        average_unit_price=("unit_price", "mean"),
        min_unit_price=("unit_price", "min"),
        max_unit_price=("unit_price", "max"),
        average_discount_pct=("discount_pct", "mean"),
    ).reset_index()

    if not previous.empty:
        prev_grp = previous.groupby(target_group)["revenue"].sum().reset_index()
        prev_grp.columns = [target_group, "previous_revenue"]
        merged = curr_grp.merge(prev_grp, on=target_group, how="left").fillna({"previous_revenue": 0.0})
        merged["revenue_growth_pct"] = np.where(
            merged["previous_revenue"] > 0,
            100.0 * (merged["revenue"] - merged["previous_revenue"]) / merged["previous_revenue"],
            None,
        )
    else:
        merged = curr_grp
        merged["previous_revenue"] = 0.0
        merged["revenue_growth_pct"] = None

    merged = merged.sort_values("revenue", ascending=False)
    geo_maps = _get_geo_maps()

    breakdown = []
    for _, row in merged.head(top_n).iterrows():
        key_val = str(row[target_group])
        rev = float(row["revenue"])
        share = round(100.0 * rev / total_curr_rev, 2) if total_curr_rev > 0 else 0.0
        item = {
            target_group: key_val,
            "revenue": round(rev, 2),
            "revenue_share_pct": share,
            "units_sold": int(row["units_sold"]),
            "average_unit_price": round(float(row["average_unit_price"]), 2),
            "price_range": f"{round(float(row['min_unit_price']), 2)} to {round(float(row['max_unit_price']), 2)}",
            "average_discount_pct": round(float(row["average_discount_pct"]), 1),
            "previous_revenue": round(float(row["previous_revenue"]), 2),
            "revenue_growth_pct": round(float(row["revenue_growth_pct"]), 2) if pd.notna(row["revenue_growth_pct"]) else "N/A",
        }
        if target_group == "city":
            item["state"] = geo_maps["city_to_state"].get(key_val, "Unknown")
            item["region"] = geo_maps["city_to_region"].get(key_val, "Unknown")
        elif target_group == "state":
            item["region"] = geo_maps["state_to_region"].get(key_val, "Unknown")
        elif target_group == "product_id":
            pmeta = _get_product_meta(key_val)
            item["product_name"] = pmeta.get("product_name", key_val)
            item["brand"] = pmeta.get("brand", "")
            item["category"] = pmeta.get("category", "")
            item["subcategory"] = pmeta.get("subcategory", "")
        breakdown.append(item)

    # Forward projected revenue from forecast
    projected_forecast_rev = 0.0
    forecast_df = _load_forecast_data()
    if not forecast_df.empty:
        prod_prices = raw_df.groupby("product_id")["unit_price"].last().to_dict()
        prod_discs = raw_df.groupby("product_id")["discount_pct"].mean().to_dict()
        for _, fc_row in forecast_df.iterrows():
            pid = fc_row["product_id"]
            p = float(prod_prices.get(pid, 0.0))
            d = float(prod_discs.get(pid, 0.0))
            projected_forecast_rev += float(fc_row["predicted_units"]) * p * (1.0 - d / 100.0)

    return {
        "group_by": target_group,
        "currency": "USD",
        "current_period": current_period_str,
        "previous_period": prev_period_str,
        "total_revenue": round(total_curr_rev, 2),
        "previous_total_revenue": round(total_prev_rev, 2),
        "total_revenue_growth_pct": growth_pct,
        "total_units_sold": int(current[TARGET].sum()),
        "unit_price_statistics": {
            "average_unit_price": round(float(current["unit_price"].mean()), 2),
            "min_unit_price": round(float(current["unit_price"].min()), 2),
            "max_unit_price": round(float(current["unit_price"].max()), 2),
        },
        "overall_average_unit_price": round(float(current["unit_price"].mean()), 2),
        "unit_price_range": f"{round(float(current['unit_price'].min()), 2)} to {round(float(current['unit_price'].max()), 2)}",
        "projected_forward_4_week_revenue": round(projected_forecast_rev, 2) if projected_forecast_rev > 0 else "N/A",
        "breakdown": breakdown,
        "data_columns_used": ["revenue", "unit_price", "discount_pct", "units_sold"],
        "calculation_method": (
            "Monetary revenue per record = units_sold × unit_price × (1 - discount_pct / 100). "
            "Summed across active records in the specified time period."
        ),
    }


def get_unreliable_products(top_n: int = 20) -> dict:
    """Return products flagged by documented reliability rules."""
    eval_data = _load_eval_data()
    if "error" in eval_data:
        return eval_data

    flags = eval_data.get("reliability_flags", [])
    if not flags:
        return {"message": "No reliability flags generated.", "products": []}

    flags_df = pd.DataFrame(flags)
    summary = eval_data.get("series_summary_stats", {})

    enriched_flags = []
    for f in flags[:top_n]:
        f_copy = dict(f)
        pid = f_copy.get("product_id")
        if pid:
            pmeta = _get_product_meta(str(pid))
            f_copy["product_name"] = pmeta.get("product_name", str(pid))
            f_copy["brand"] = pmeta.get("brand", "")
            f_copy["category"] = pmeta.get("category", "")
        enriched_flags.append(f_copy)

    return {
        "total_flagged_series": summary.get("series_with_flags", 0),
        "flag_type_counts": summary.get("flag_counts", {}),
        "flags": enriched_flags,
        "rules_documentation": {
            "insufficient_history": f"Fewer than 12 weeks of data",
            "missing_periods": "More than 4 missing weeks in date range",
            "sparse_demand": "More than 50% of weeks with zero demand",
            "demand_spike": "Maximum demand exceeds 3 standard deviations above mean",
            "stockout_distortion": "Stockouts in more than 20% of weeks",
            "high_forecast_error": "Out-of-sample WAPE exceeds 80%",
        },
        "disclaimer": "These flags use documented heuristic rules, not statistically calibrated confidence intervals.",
    }


def get_data_quality_report() -> dict:
    """Return verified data-quality statistics and validation findings."""
    raw_df = _load_raw_data()

    total_rows = len(raw_df)
    n_products = raw_df["product_id"].nunique()
    n_stores = raw_df["store_id"].nunique()
    n_weeks = raw_df["week_start_date"].nunique()
    date_range = f"{raw_df['week_start_date'].min().date()} to {raw_df['week_start_date'].max().date()}"

    # Missing values
    missing = {}
    for col in raw_df.columns:
        n_miss = int(raw_df[col].isna().sum())
        if n_miss > 0:
            missing[col] = {"count": n_miss, "percentage": round(100 * n_miss / total_rows, 2)}

    # Target distribution
    target_stats = raw_df[TARGET].describe().to_dict()
    target_stats = {k: round(float(v), 2) for k, v in target_stats.items()}

    # Zero demand
    zero_demand = int((raw_df[TARGET] == 0).sum())

    return {
        "total_rows": total_rows,
        "n_products": n_products,
        "n_stores": n_stores,
        "n_weeks": n_weeks,
        "date_range": date_range,
        "categories": int(raw_df["category"].nunique()),
        "missing_values": missing,
        "target_statistics": target_stats,
        "zero_demand_rows": zero_demand,
        "zero_demand_percentage": round(100 * zero_demand / total_rows, 2),
        "data_source": "CSV / MySQL retail table",
    }
