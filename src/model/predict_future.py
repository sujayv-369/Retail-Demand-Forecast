import os
import json
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")

# 1. Load Environment Variables & DB Connection
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 2. Load Metadata and Model Artifacts
METADATA_PATH = "artifacts/model_metadata.json"
MODEL_PATH = "artifacts/catboost_demand.cbm"

if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("Model or metadata artifact missing. Please run training first.")

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

SERIES_KEY = metadata["series_key"]
CAT_FEATURES = metadata["categorical_features"]
FEATURES = metadata["features"]

from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.load_model(MODEL_PATH)
print("CatBoost model successfully loaded from artifacts.")

# 3. Load Latest Historical Data with product_name
print("Fetching historical sales data from MySQL...")
query = """
    SELECT
        product_id, product_name, store_id, sales_channel, category, subcategory, brand, region,
        season, festival_name, promotion_type, week_start_date, month, week_number,
        holiday_flag, unit_price, discount_pct, promotion_flag, opening_stock,
        stock_received, supplier_lead_time_days, units_sold
    FROM retail
    ORDER BY week_start_date ASC
"""

df = pd.read_sql(query, engine)
df["week_start_date"] = pd.to_datetime(df["week_start_date"])
df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)

# Sort strictly by series key and date
df = df.sort_values(SERIES_KEY + ["week_start_date"]).reset_index(drop=True)

# Map product_id to product_name for final join
product_name_map = df[['product_id', 'product_name']].drop_duplicates().set_index('product_id')['product_name'].to_dict()

# 4. Compute Lag Features on Historical DataFrame
print("Computing historical lag and rolling features...")
grouped = df.groupby(SERIES_KEY, group_keys=False)

df["lag_1_sales"] = grouped["units_sold"].shift(1)
df["lag_4_sales"] = grouped["units_sold"].shift(4)
df["rolling_4_week_avg"] = grouped["units_sold"].shift(1).rolling(window=4, min_periods=1).mean()

# Fill initial NaN lag values with 0
df["lag_1_sales"] = df["lag_1_sales"].fillna(0)
df["lag_4_sales"] = df["lag_4_sales"].fillna(0)
df["rolling_4_week_avg"] = df["rolling_4_week_avg"].fillna(0)

# 5. Recursive 4-Week Out-of-Sample Forecasting
print("\nGenerating 4-week future demand forecasts...")

future_predictions = []
last_date = df["week_start_date"].max()

# Get the most recent valid row per series as initial state
latest_states = df.groupby(SERIES_KEY).last().reset_index()

for step in range(1, 5):
    forecast_date = last_date + pd.Timedelta(weeks=step)
    
    # Create feature matrix for current step
    step_df = latest_states.copy()
    step_df["week_start_date"] = forecast_date
    step_df["month"] = forecast_date.month
    step_df["week_number"] = forecast_date.isocalendar().week

    # Format categorical features
    for col in CAT_FEATURES:
        step_df[col] = step_df[col].fillna("UNKNOWN").astype(str)

    # Format numeric features
    for col in FEATURES:
        if col not in CAT_FEATURES:
            step_df[col] = pd.to_numeric(step_df[col], errors="coerce").fillna(0)

    # Predict current week
    X_step = step_df[FEATURES]
    preds = np.maximum(0, model.predict(X_step))
    step_df["predicted_units_sold"] = np.round(preds, 2)
    step_df["forecast_step"] = f"Week +{step}"

    # Record predictions
    future_predictions.append(step_df[SERIES_KEY + ["week_start_date", "forecast_step", "predicted_units_sold"]])

    # Update recursive lag features for next iteration
    latest_states["lag_4_sales"] = latest_states["lag_1_sales"]
    latest_states["lag_1_sales"] = preds
    latest_states["rolling_4_week_avg"] = (
        0.75 * latest_states["rolling_4_week_avg"] + 0.25 * preds
    )

# 6. Aggregate & Map Product Names
df_future = pd.concat(future_predictions, ignore_index=True)

# Product-level aggregation across stores & channels
df_product_summary = df_future.groupby(["product_id", "week_start_date", "forecast_step"])["predicted_units_sold"].sum().reset_index()
df_product_summary["product_name"] = df_product_summary["product_id"].map(product_name_map)

# Reorder columns
df_product_summary = df_product_summary[["product_id", "product_name", "week_start_date", "forecast_step", "predicted_units_sold"]]

# Calculate Top 15 Demanded Products over 4 Weeks
df_top15 = df_product_summary.groupby(["product_id", "product_name"])["predicted_units_sold"].sum().reset_index()
df_top15.rename(columns={"predicted_units_sold": "total_4week_forecast_units"}, inplace=True)
df_top15 = df_top15.sort_values(by="total_4week_forecast_units", ascending=False).head(15).reset_index(drop=True)

print("\n===============================================================================")
print("                   TOP 15 DEMANDED PRODUCTS (NEXT 4 WEEKS)                    ")
print("===============================================================================")
print(df_top15.to_string(index=False))
print("===============================================================================\n")

# 7. Save Future Predictions to MySQL
print("Saving future predictions to MySQL (`future_4week_forecasts` table)...")
df_product_summary.to_sql(
    name="future_4week_forecasts",
    con=engine,
    if_exists="replace",
    index=False
)

print("Saved successfully! Database table `future_4week_forecasts` is updated with product names.")