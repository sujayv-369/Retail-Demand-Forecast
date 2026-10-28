import os
import json
import warnings
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# 1. Load Environment Variables and Database Connection
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 2. Define Feature Definitions and Keys
SERIES_KEY = ["product_id", "store_id", "sales_channel"]
TARGET = "units_sold"

CAT_FEATURES = [
    "product_id",
    "store_id",
    "sales_channel",
    "category",
    "subcategory",
    "brand",
    "region",
    "season",
    "festival_name",
    "promotion_type",
]

NUMERIC_FEATURES = [
    "month",
    "week_number",
    "holiday_flag",
    "lag_1_sales",
    "lag_4_sales",
    "rolling_4_week_avg",
    "unit_price",
    "discount_pct",
    "promotion_flag",
    "opening_stock",
    "stock_received",
    "supplier_lead_time_days",
]

FEATURES = CAT_FEATURES + NUMERIC_FEATURES

# 3. Load Dataset from MySQL
print("Fetching dataset from MySQL...")
query = """
    SELECT
        product_id, store_id, sales_channel, category, subcategory, brand, region,
        season, festival_name, promotion_type, week_start_date, month, week_number,
        holiday_flag, unit_price, discount_pct, promotion_flag, opening_stock,
        stock_received, supplier_lead_time_days, units_sold
    FROM retail
    ORDER BY week_start_date ASC
"""

df = pd.read_sql(query, engine)
print(f"Loaded {len(df):,} records.")

# 4. Data Type Formatting & Preprocessing
df["week_start_date"] = pd.to_datetime(df["week_start_date"])
df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)

# Sort strictly by series key and date before engineering lag features
df = df.sort_values(SERIES_KEY + ["week_start_date"]).reset_index(drop=True)

# 5. Robust Feature Engineering (No Data Leakage)
print("Engineering lag and rolling features...")
grouped = df.groupby(SERIES_KEY, group_keys=False)

df["lag_1_sales"] = grouped["units_sold"].shift(1)
df["lag_4_sales"] = grouped["units_sold"].shift(4)

# Shift by 1 first to prevent target leakage in 4-week rolling mean
df["rolling_4_week_avg"] = grouped["units_sold"].shift(1).rolling(window=4, min_periods=1).mean()

# Drop rows where lag features cannot be computed
df_model = df.dropna(subset=FEATURES + [TARGET]).copy()

# Fill missing categorical values explicitly
for col in CAT_FEATURES:
    df_model[col] = df_model[col].fillna("UNKNOWN").astype(str)

# Fill missing numeric features
for col in NUMERIC_FEATURES:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce").fillna(0)

# 6. Dynamic Chronological Train/Validation Split (Hold out final 4 weeks)
max_date = df_model["week_start_date"].max()
cutoff_date = max_date - pd.Timedelta(weeks=4)

train = df_model[df_model["week_start_date"] < cutoff_date]
valid = df_model[df_model["week_start_date"] >= cutoff_date]

print(f"Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
print(f"Training Rows: {len(train):,} | Validation Rows: {len(valid):,}")

if train.empty or valid.empty:
    raise ValueError("Training or validation set is empty. Check time cutoff logic.")

X_train, y_train = train[FEATURES], train[TARGET]
X_valid, y_valid = valid[FEATURES], valid[TARGET]

# 7. Train CatBoost Regressor
print("Training CatBoost Regressor...")
model = CatBoostRegressor(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=200
)

model.fit(
    X_train,
    y_train,
    cat_features=CAT_FEATURES,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100
)

# 8. Generate Predictions & Calculate Validation Scores
valid = valid.copy()
valid["forecast_units"] = np.maximum(0, model.predict(X_valid)) # Clip negative forecasts

# Overall Metrics
mae = mean_absolute_error(valid[TARGET], valid["forecast_units"])
rmse = np.sqrt(mean_squared_error(valid[TARGET], valid["forecast_units"]))
actual_sum = valid[TARGET].sum()
wape = (np.sum(np.abs(valid[TARGET] - valid["forecast_units"])) / actual_sum) if actual_sum != 0 else np.nan

print("\n--- CATBOOST MODEL EVALUATION METRICS ---")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"WAPE : {wape:.2%}")

# 9. Aggregate Results to Product Level & Save to MySQL
valid_product_level = valid.groupby(["product_id", "week_start_date"]).agg(
    actual_units=(TARGET, "sum"),
    forecast_units=("forecast_units", "sum")
).reset_index()

# Calculate Product-Level Metrics for database logging
product_metrics = []
for pid, group in valid_product_level.groupby("product_id"):
    p_actual = group["actual_units"].values
    p_pred = group["forecast_units"].values
    p_act_sum = np.sum(p_actual)
    
    p_mae = mean_absolute_error(p_actual, p_pred)
    p_rmse = np.sqrt(mean_squared_error(p_actual, p_pred))
    p_wape = (np.sum(np.abs(p_actual - p_pred)) / p_act_sum) if p_act_sum != 0 else np.nan
    
    product_metrics.append({
        "product_id": pid,
        "model_name": "CatBoost Regressor",
        "mae": float(p_mae),
        "rmse": float(p_rmse),
        "wape": float(p_wape) if not np.isnan(p_wape) else None
    })

df_catboost_metrics = pd.DataFrame(product_metrics)

print("\nAppending CatBoost metrics to MySQL (`model_comparison_metrics` table)...")
df_catboost_metrics.to_sql(
    name="model_comparison_metrics",
    con=engine,
    if_exists="append",
    index=False
)

# 10. Save Model Artifacts
os.makedirs("artifacts", exist_ok=True)
model.save_model("artifacts/catboost_demand.cbm")

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(
        {
            "model_name": "catboost_demand",
            "target": TARGET,
            "overall_mae": float(mae),
            "overall_rmse": float(rmse),
            "overall_wape": float(wape),
            "features": FEATURES,
            "categorical_features": CAT_FEATURES,
            "cutoff_date": str(cutoff_date.date()),
            "series_key": SERIES_KEY,
        },
        f,
        indent=2,
    )

print("\nProcessing completed successfully! Model and metadata saved in `artifacts/`.")