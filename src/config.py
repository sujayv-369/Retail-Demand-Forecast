"""
Central configuration — loads all settings from environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV_PATH = DATA_DIR / "raw" / "final_raw_datset.csv"
EVALUATION_DIR = ARTIFACTS_DIR / "evaluation"

load_dotenv(PROJECT_ROOT / ".env")

# ── database ───────────────────────────────────────────────────────
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major",
)

# ── model ──────────────────────────────────────────────────────────
MODEL_PATH = ARTIFACTS_DIR / "catboost_demand.cbm"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

SERIES_KEY = ["product_id", "store_id", "sales_channel"]
TARGET = "units_sold"

CAT_FEATURES = [
    "product_id", "store_id", "sales_channel",
    "category", "subcategory", "brand",
    "region", "season", "festival_name", "promotion_type",
]

FEATURES = [
    *CAT_FEATURES,
    "month", "week_number", "holiday_flag",
    "lag_1_sales", "lag_4_sales", "rolling_4_week_avg",
    "unit_price", "discount_pct", "promotion_flag",
    "opening_stock", "stock_received", "supplier_lead_time_days",
]

# ── Gemini ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MAX_TOOL_CALLS = int(os.environ.get("GEMINI_MAX_TOOL_CALLS", "8"))
GEMINI_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))

# ── evaluation ─────────────────────────────────────────────────────
EVAL_HOLDOUT_WEEKS = int(os.environ.get("EVAL_HOLDOUT_WEEKS", "8"))
FORECAST_HORIZON = int(os.environ.get("FORECAST_HORIZON", "4"))
