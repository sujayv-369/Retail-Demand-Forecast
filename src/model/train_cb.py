import os
import json

import pandas as pd
from sqlalchemy import create_engine, text
from catboost import CatBoostRegressor
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    os.environ["DATABASE_URL"],
    pool_pre_ping=True,
)


SERIES_KEY = [
    "product_id",
    "store_id",
    "sales_channel",
]

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


FEATURES = [
    *CAT_FEATURES,
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



query = text("""
    SELECT
        record_id,
        product_id,
        product_name,
        category,
        subcategory,
        brand,
        model_number,
        launch_date,
        store_id,
        city,
        state,
        region,
        sales_channel,
        week_start_date,
        week_number,
        month,
        year,
        season,
        holiday_flag,
        festival_name,
        units_sold,
        unit_price,
        discount_pct,
        promotion_flag,
        promotion_type,
        revenue,
        units_returned,
        opening_stock,
        stock_received,
        closing_stock,
        reorder_point,
        supplier_lead_time_days,
        stockout_flag,
        lag_1_sales,
        lag_4_sales,
        rolling_4_week_avg
    FROM retail
    ORDER BY week_start_date
""")

df = pd.read_sql(query, engine)

df["week_start_date"] = pd.to_datetime(
    df["week_start_date"]
)

df["units_sold"] = pd.to_numeric(
    df["units_sold"], errors="raise"
)

df = df.sort_values(
    SERIES_KEY + ["week_start_date"]
).copy()

# Recompute lag features within each individual series.
group = df.groupby(SERIES_KEY, sort=False)["units_sold"]

df["lag_1_sales"] = group.shift(1)
df["lag_4_sales"] = group.shift(4)

df["rolling_4_week_avg"] = (
    df.groupby(SERIES_KEY, sort=False)["units_sold"]
      .transform(
          lambda s: s.shift(1).rolling(4).mean()
      )
)

# Rows without sufficient history cannot supply all features.
df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

# Example only: replace this cutoff with one derived
# from your dataset's actual chronological coverage.
validation_start = pd.Timestamp("2025-10-01")

train = df[
    df["week_start_date"] < validation_start
]

valid = df[
    df["week_start_date"] >= validation_start
]

if train.empty or valid.empty:
    raise ValueError("Training or validation data is empty.")

X_train = train[FEATURES].copy()
y_train = train[TARGET]

X_valid = valid[FEATURES].copy()
y_valid = valid[TARGET]

# CatBoost categorical inputs should have consistent
# string representations in training and inference.

for col in CAT_FEATURES:
    X_train[col] = (
        X_train[col]
        .fillna("UNKNOWN")
        .astype(str)
    )

    X_valid[col] = (
        X_valid[col]
        .fillna("UNKNOWN")
        .astype(str)
    )


model = CatBoostRegressor(
    loss_function="MAE",
    eval_metric="MAE",
    iterations=1200,
    depth=8,
    learning_rate=0.05,
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    cat_features=CAT_FEATURES,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100,
)

os.makedirs("artifacts", exist_ok=True)

model.save_model("artifacts/catboost_demand.cbm")

with open("artifacts/model_metadata.json", "w") as f:
    json.dump(
        {
            "model_name": "catboost_demand",
            "target": TARGET,
            "features": FEATURES,
            "categorical_features": CAT_FEATURES,
            "validation_start": str(validation_start.date()),
            "series_key": SERIES_KEY,
        },
        f,
        indent=2,
    )