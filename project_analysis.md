# Project Analysis — Retail Demand Forecasting

## Current Project Structure

```
retail__data/
├── .env                          # DB connection: mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major
├── requirements.txt              # numpy, pandas, statsforecast, catboost, pymysql
├── artifacts/
│   ├── catboost_demand.cbm       # Trained model (~44MB)
│   └── model_metadata.json       # Model config & feature list
├── catboost_info/                # CatBoost training logs (560 iterations completed)
│   ├── learn_error.tsv           # Train MAE: 4.44 → 2.16
│   └── test_error.tsv            # Valid MAE: 4.56 → 2.44 (best ~2.443 at iter 391)
├── data/
│   ├── raw/final_raw_datset.csv  # ~265MB, 1,024,772 rows × 36 columns
│   └── processed/                # Empty
└── src/
    └── model/
        └── train_cb.py           # Single training script
```

## Key Findings

### Data Characteristics
| Attribute | Value |
|---|---|
| **Rows** | 1,024,772 |
| **Columns** | 36 |
| **Date range** | 2024-10-21 → 2026-10-12 |
| **Unique weeks** | 104 (exactly 2 years) |
| **Products** | 719 |
| **Stores** | 14 |
| **Sales channels** | 3 (offline, omnichannel, online) |
| **Categories** | 30 |
| **Series grain** | product_id × store_id × sales_channel |
| **Max possible series** | 719 × 14 × 3 = 30,198 |
| **Target** | units_sold (mean=5.43, max=468, heavily right-skewed) |

### Model Configuration
| Setting | Value |
|---|---|
| **Model** | CatBoostRegressor |
| **Loss** | MAE |
| **Target** | units_sold |
| **Iterations** | 1,200 (early stopped at ~560) |
| **Depth** | 8 |
| **Learning rate** | 0.05 |
| **Validation split** | 2025-10-01 (chronological) |
| **Best validation MAE** | ~2.443 |

### Training Script Analysis ([train_cb.py](file:///d:/AI_PROJECTS/retail__data/src/model/train_cb.py))

**What it does correctly:**
- Chronological train/validation split (before/after 2025-10-01)
- Recomputes lag features per series using `groupby(SERIES_KEY)` — prevents cross-series contamination
- `lag_1_sales = shift(1)`, `lag_4_sales = shift(4)`, `rolling_4_week_avg = shift(1).rolling(4).mean()`
- Drops rows with NaN features (first 4 weeks per series)
- Saves model artifact and metadata

**Issues identified:**
1. **Potential leakage**: `opening_stock`, `stock_received` may contain current-period info (depends on data semantics)
2. **No recursive forecasting**: Model trained as single-step predictor — no multi-step forecast pipeline exists
3. **No inference code**: No prediction/forecast generation script
4. **No evaluation code**: No test metrics, baselines, or comparison
5. **No frontend/backend**: No web application exists — must be built from scratch
6. **No Gemini integration**: No AI assistant code
7. **promotion_type column**: Appears as `float64` in CSV (likely encoded or has NaN), but treated as categorical

### What Does NOT Exist Yet
- ❌ No evaluation pipeline
- ❌ No baseline comparisons
- ❌ No recursive multi-step forecasting
- ❌ No web frontend or backend
- ❌ No Gemini integration
- ❌ No analytics tools/functions
- ❌ No tests
- ❌ No README

## Implementation Plan

### Phase 1: Model Evaluation (~core ML)
1. Create `src/evaluation/` — time-based eval, baselines, metrics, reliability flags
2. Create `src/forecast/` — recursive 4-week forecasting engine
3. Generate actual evaluation report with measured metrics

### Phase 2-4: Gemini Assistant (~backend)
1. Install `google-genai` SDK
2. Create `src/analytics/` — approved tool functions (SQL/Python calculations)
3. Create `src/gemini/` — function-calling orchestration, grounding validation
4. Create `src/api/` — Flask/FastAPI backend serving the chat + forecast data

### Phase 5: UI (~frontend)
1. Build web interface with forecast dashboard + chat
2. Grounded evidence display for analytical answers

### Phase 6-7: Tests + Documentation
1. Create `tests/` — forecasting, analytics, grounding, prompt injection
2. Write README, .env.example, evaluation report
