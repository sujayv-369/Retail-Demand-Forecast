# RetailForecaster — Comprehensive Enterprise Project Report (A to Z)

**Platform**: RetailForecaster Enterprise Demand Analytics  
**Version**: 2.4.0 (Dual-Workspace Architecture)  
**Author**: Machine Learning & Analytics Engineering Team  
**Date**: October 2026  
**Documentation File**: `report.md`

---

## Table of Contents
1. [Executive Summary & Project Vision](#1-executive-summary--project-vision)
2. [Complete System Architecture & Workflow](#2-complete-system-architecture--workflow)
3. [Database Architecture & SQL Website Integration](#3-database-architecture--sql-website-integration)
4. [Machine Learning Pipeline & CatBoost Training](#4-machine-learning-pipeline--catboost-training)
5. [Data Splitting Methodology & Validation Strategy](#5-data-splitting-methodology--validation-strategy)
6. [5-Model Benchmark Comparison System](#6-5-model-benchmark-comparison-system)
7. [Recursive Multi-Step Out-of-Sample Forecasting](#7-recursive-multi-step-out-of-sample-forecasting)
8. [LLM Integration: Google Gemini & Function-Calling Engine](#8-llm-integration-google-gemini--function-calling-engine)
9. [Complete Platform Functionalities (A to Z)](#9-complete-platform-functionalities-a-to-z)
10. [Executive Reporting Engine (PDF, HTML, CSV)](#10-executive-reporting-engine-pdf-html-csv)
11. [Frontend Design System & User Experience](#11-frontend-design-system--user-experience)
12. [Project File Map & Repository Layout](#12-project-file-map--repository-layout)
13. [Setup, Execution & Deployment Guide](#13-setup-execution--deployment-guide)

---

## 1. Executive Summary & Project Vision

**RetailForecaster** is a production-grade, enterprise retail demand forecasting and AI decision-support platform. The application is built to bridge the gap between complex econometric/machine-learning models and non-technical retail merchandisers, store planners, and C-suite executives.

### Core Problems Solved
- **Stockouts vs. Overstocking**: Inaccurate forecasts lead either to lost sales (stockouts) or tied-up capital and heavy discounting (overstock). RetailForecaster provides 4-week forward predictions at the granular level of `Product SKU × Store Outlet × Sales Channel`.
- **Black-Box AI Skepticism**: Business executives often distrust raw machine learning predictions. RetailForecaster pairs predictions with out-of-sample error telemetry (WAPE, MAE, RMSE), reliability flags (sparse demand, spike distortion, stockout signals), and an audit trail comparing CatBoost against traditional statistical baselines (Naive, Moving Average, ARIMA, Prophet).
- **Unassisted Data Exploration**: Rather than requiring SQL queries or BI dashboards for every ad-hoc question, merchandisers can talk directly to an enterprise **Gemini AI Assistant** that retrieves live database records, calculates festival uplifts, ranks regional sales hubs, verifies numerical claims through a Grounding Engine, and visually explains its reasoning using multi-stage execution animations.

### Dual-Workspace Paradigm
The application implements strict Role-Based Access Control (RBAC):
1. **Developer Studio (Admin)**: For Data Scientists and ML Engineers. Provides dataset validation, feature health diagnostics, 5-model rolling-origin benchmark execution, residual error progression across horizons, and model promotion/approval.
2. **Retailer Planning (Store Planner / Retail User)**: For Category Managers and Store Operators. Provides 4-week demand forecasts, business overview KPIs, SKU-level trend analytics, India geographic demand intelligence with interactive map controls, executive PDF/HTML report exports, and the conversational Gemini Assistant.

---

## 2. Complete System Architecture & Workflow

### 2.1 High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Data_Layer ["Data Tier (MySQL Relational Database)"]
        DB_Retail[("retail (Transactions, SKUs, Stores)")]
        DB_Users[("users (Auth, RBAC, Passwords)")]
        DB_Runs[("evaluation_runs & forecast_runs")]
        DB_Metrics[("evaluation_metrics & model_comparison")]
        DB_Forecasts[("forecast_results & future_4week_forecasts")]
        DB_Gemini[("gemini_interactions & gemini_feedback")]
    end

    subgraph ML_Engine ["Machine Learning & Forecasting Pipeline"]
        FE[Feature Engineering & Lag Pipeline]
        Split[Chronological Cutoff Split]
        CatBoost[CatBoost Regressor (Primary Engine)]
        Baselines[Naive, MA4, ARIMA, Prophet]
        Benchmark[Rolling-Origin Evaluator]
        RecursiveFC[4-Week Recursive Forward Forecaster]
        Artifacts[("artifacts/catboost_demand.cbm & metadata")]
    end

    subgraph Backend ["Backend API Tier (Flask 3.0 + SQLAlchemy 2.0)"]
        App[Flask Application Factory]
        AuthRoutes[/api/auth/*: Login, Signup, Session, Profile]
        DataRoutes[/api/evaluation/*, /api/forecast/*, /api/geo/*]
        ReportRoutes[/api/reports/download/*]
        ChatRoutes[/api/chat, /api/chat/feedback]
        Dispatcher[Analytics Tool Dispatcher (10 Allowed Tools)]
    end

    subgraph LLM_Tier ["AI Assistant Tier (Google Gemini + Grounding)"]
        GeminiClient[Google GenAI Client (gemini-3.5-flash / gemini-3.8-flash)]
        FuncDecs[Function Declarations Schema]
        Grounding[Grounding Verification Engine]
        Learning[Active Learning Directive Engine]
    end

    subgraph Frontend ["Presentation Tier (Vanilla HTML5 / Modern CSS / Vanilla JS)"]
        AuthView[Authentication Modal / Warm Executive Theme]
        DevView[Developer Studio: Dataset, Evaluation, Comparison, History]
        RetView[Retailer View: Forecast Explorer, Business Overview, India Map]
        AIChatUI[Conversational Assistant with Multi-Stage Animation]
        ProfileModal[Executive Profile Modal (Viewport-Constrained)]
    end

    DB_Retail --> FE --> Split --> CatBoost --> Benchmark --> Artifacts
    DB_Retail --> Baselines --> Benchmark --> DB_Metrics
    Artifacts --> RecursiveFC --> DB_Forecasts
    
    DB_Retail <--> Backend
    DB_Users <--> AuthRoutes
    DB_Forecasts <--> DataRoutes
    DB_Metrics <--> DataRoutes
    
    Backend <--> Frontend
    ChatRoutes <--> GeminiClient
    GeminiClient <--> FuncDecs <--> Dispatcher <--> DB_Retail & DB_Forecasts
    GeminiClient --> Grounding --> ChatRoutes
    Learning <--> DB_Gemini
```

### 2.2 End-to-End Operational Workflow
1. **Raw Ingestion**: Historical weekly sales records across Indian stores, products, channels, and dates reside in MySQL table `retail`.
2. **Feature Engineering**: Features including lag sales (`lag_1`, `lag_4`), rolling statistics (`rolling_4_week_avg`), calendar indices (`week_number`, `month`), promotional flags, pricing, and festival indicators are constructed per `product_id × store_id × sales_channel`.
3. **Training & Validation**: Chronological split isolates holdout evaluation periods. CatBoost trains on history and evaluates on out-of-sample weeks.
4. **Benchmark Verification**: The Evaluator compares CatBoost against Naive, Moving Average, ARIMA, and Prophet models. The winning model is marked approved.
5. **Forward Forecasting**: `predict_future.py` / `recursive_forecast.py` generates 4-week forward out-of-sample predictions, updating lags recursively at each forecast step.
6. **API Serving**: Flask REST endpoints expose metrics, forecasts, geographic summaries, and reporting services.
7. **Interactive Visual Dashboard**: A vanilla single-page interface renders charts via Chart.js, pins on an India political map, and streams live telemetry.
8. **AI Assistant Decision Support**: When a user queries the chat interface, Google Gemini invokes allowlisted database tools, runs statistical checks, validates numerical claims with the Grounding Engine, and presents verified business insights.

---

## 3. Database Architecture & SQL Website Integration

### 3.1 Connection Protocol & Connection Pooling
The backend connects to MySQL using **SQLAlchemy 2.0** paired with the **PyMySQL** driver.
Connection configuration in `src/config.py` and `src/db.py`:

```python
from sqlalchemy import create_engine
from src.config import DATABASE_URL

# Connection String Format:
# mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major

_engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Automatically tests connection health before executing queries
    pool_size=5,          # Persistent pool of active database connections
    max_overflow=10,      # Allows burst capacity of up to 15 concurrent connections
)
```

**Key Connection Features**:
- **`pool_pre_ping=True`**: Prevents "MySQL server has gone away" errors caused by idle connection timeouts.
- **`engine.begin()` Context Managers**: Automatically commits transactions on successful code execution and rolls back changes if an exception is raised.
- **Thread Safety**: Flask runs multi-threaded worker requests. Connections are checked out per request thread and returned to the pool immediately upon completion.

### 3.2 Comprehensive Database Schema

| Table Name | Primary Key | Key Columns | Purpose |
|---|---|---|---|
| `retail` | Composite (`product_id, store_id, sales_channel, week_start_date`) | `units_sold, unit_price, discount_pct, promotion_flag, holiday_flag, festival_name, region, season, opening_stock, stock_received, supplier_lead_time_days` | Core historical retail dataset containing sales and supply signals. |
| `users` | `id` (Auto-increment) | `username, password_hash, role, phone, email, business_name, created_at` | RBAC user management. Password hashes are generated using `bcrypt` (12 rounds of salt). |
| `evaluation_runs` | `run_id` (String) | `model_name, eval_cutoff_date, test_start_date, test_end_date, config_json, created_at` | Metadata for 5-model evaluation experiments. |
| `evaluation_metrics` | `id` (Auto-increment) | `run_id, model_type, scope_type, scope_value, horizon_week, mae, wape, n_observations` | Granular error telemetry (MAE, WAPE) across models and horizons. |
| `model_comparison_metrics` | Composite | `product_id, model_name, mae, rmse, wape` | Product-level accuracy scorecard across models. |
| `forecast_runs` | `run_id` (String) | `model_name, forecast_origin_date, horizon_weeks, created_at` | Audit ledger for forward forecast executions. |
| `forecast_results` | `id` (Auto-increment) | `run_id, product_id, store_id, sales_channel, forecast_week_date, horizon_step, predicted_units, product_name, category` | 4-week forward predictions at granular series level. |
| `future_4week_forecasts` | Composite | `product_id, product_name, week_start_date, forecast_step, predicted_units_sold` | Product-aggregated forward forecasts utilized by reports & chat. |
| `reliability_flags` | `id` (Auto-increment) | `run_id, product_id, store_id, sales_channel, flag_type, flag_reason, severity` | Automated risk tags: sparse demand, spikes, stockout distortion, high error. |
| `gemini_interactions` | `id` (Auto-increment) | `user_question, tools_called, grounding_status, response_text, created_at` | Audit trail of all natural language assistant inquiries and tool calls. |
| `gemini_feedback` | `id` (Auto-increment) | `message_id, user_query, response_text, feedback_type, feedback_score, comment, created_at` | Reinforcement feedback from thumbs-up / thumbs-down buttons for active learning. |
| `app_settings` | `key` (String) | `value, updated_at` | Key-value store for application state (e.g. approved model selection). |

### 3.3 Dynamic Migrations & Automatic Seeding
When the Flask server boots up in `src/api/app.py`, `create_tables()` in `src/db.py` executes:
1. **Table Creation**: `metadata_obj.create_all(engine)` idempotently creates any missing tables without altering existing tables.
2. **Column Migration (`migrate_users_table`)**: Checks for `phone`, `email`, and `business_name` columns in the `users` table, dynamically issuing `ALTER TABLE users ADD COLUMN ...` if they are missing.
3. **Default Admin Seeding (`seed_admin_user`)**: Checks if the user `Admin` exists. If absent, it hashes the default password `123456` with `bcrypt.gensalt()` and inserts the record with role `'admin'`.

### 3.4 How SQL Connects to the Website UI
The website UI communicates with MySQL via a RESTful API pipeline:
1. **Browser Request**: JavaScript in `frontend/app.js` calls `fetch('/api/evaluation/latest')` or `fetch('/api/geo/forecast')`.
2. **Route Handling**: Flask in `src/api/routes.py` matches the endpoint, extracts parameters, and calls helper routines in `src/db.py` or `src/analytics/tools.py`.
3. **SQL Execution**: The server issues parameterized SQL queries (e.g., `SELECT product_id, SUM(units_sold) FROM retail WHERE ... GROUP BY ...`) preventing SQL injection.
4. **JSON Serialization**: SQLAlchemy row tuples or Pandas DataFrames are converted into dictionaries and JSON-serialized via `jsonify()`.
5. **UI Rendering**: The browser receives JSON and dynamically renders DOM elements, tables, and Chart.js canvases.

---

## 4. Machine Learning Pipeline & CatBoost Training

### 4.1 Libraries Used in Model Training
The modeling subsystem relies on Python libraries specified in `requirements.txt`:
- **`catboost` (>= 1.2.0)**: High-performance gradient boosted decision trees supporting native categorical handling without one-hot encoding explosion.
- **`numpy` (>= 1.24.0)** & **`pandas` (>= 2.0.0)**: Numerical array manipulation, DataFrame time-series grouping, lag calculation, and rolling window aggregations.
- **`scikit-learn`**: Metric computation (`mean_absolute_error`, `mean_squared_error`).
- **`statsmodels` (>= 0.14.0)**: Classical time series modeling (`ARIMA`, `SARIMAX`).
- **`statsforecast` (>= 1.5.0)**: Fast parallelized statistical algorithms (`AutoARIMA`).
- **`prophet` (>= 1.1.0)**: Additive trend and seasonal modeling developed by Meta.
- **`pmdarima`**: Automated order search for seasonal ARIMA.

### 4.2 Granularity & Series Definition
Retail demand forecasting requires granular modeling. The target series is uniquely identified by the tuple:
$$\text{Series Key} = (\text{product\_id}, \text{store\_id}, \text{sales\_channel})$$
Each record represents the total physical units sold during a calendar week starting on Monday (`week_start_date`).

### 4.3 Feature Engineering Strategy (No Lookahead Bias)
To prevent data leakage, all dynamic time-dependent features are strictly lagged:

```python
# Sorting strictly by series key and date before engineering lags
df = df.sort_values(SERIES_KEY + ["week_start_date"]).reset_index(drop=True)
grouped = df.groupby(SERIES_KEY, group_keys=False)

# Lag 1: Prior week's sales (units sold at t-1)
df["lag_1_sales"] = grouped["units_sold"].shift(1)

# Lag 4: Sales from 4 weeks prior (units sold at t-4)
df["lag_4_sales"] = grouped["units_sold"].shift(4)

# 4-Week Rolling Average: Computed by shifting t-1 first, then taking 4-week mean
df["rolling_4_week_avg"] = grouped["units_sold"].shift(1).rolling(window=4, min_periods=1).mean()
```

#### Feature Vector Composition
- **10 Categorical Features**: `product_id`, `store_id`, `sales_channel`, `category`, `subcategory`, `brand`, `region`, `season`, `festival_name`, `promotion_type`.
- **12 Numerical / Exogenous Features**: `month`, `week_number`, `holiday_flag`, `lag_1_sales`, `lag_4_sales`, `rolling_4_week_avg`, `unit_price`, `discount_pct`, `promotion_flag`, `opening_stock`, `stock_received`, `supplier_lead_time_days`.
- **Target Variable**: `units_sold` (non-negative integer).

### 4.4 CatBoost Regressor Architecture & Training Routine
CatBoost was selected as the champion model because:
1. **Native Categorical Support**: Utilizes ordered target statistics to handle high-cardinality IDs (`product_id`, `store_id`) without one-hot encoding dimensionality explosion.
2. **Resistance to Overfitting**: Uses symmetric (oblivious) decision trees that serve as execution regularizers.
3. **Loss Function Alignment**: Optimized directly for **MAE** (Mean Absolute Error), which directly penalizes unit forecast discrepancies rather than squared errors (which over-penalize transient demand spikes).

#### Hyperparameter Configuration (`src/model/train_cat_late.py`)
```python
model = CatBoostRegressor(
    loss_function="MAE",          # Directly optimizes L1 error
    eval_metric="MAE",            # Early stopping based on out-of-sample MAE
    iterations=1000,              # Maximum boosting iterations
    depth=6,                      # Tree depth balancing interaction complexity and speed
    learning_rate=0.05,           # Conservative gradient step preventing overfitting
    random_seed=42,               # Full reproducibility
    verbose=200                   # Logs metrics every 200 trees
)

model.fit(
    X_train,
    y_train,
    cat_features=CAT_FEATURES,
    eval_set=(X_valid, y_valid),
    early_stopping_rounds=100     # Halts if validation MAE fails to improve for 100 rounds
)
```

### 4.5 Artifact Persistence
Trained models and their metadata are exported to the filesystem:
- **`artifacts/catboost_demand.cbm`**: Serialized binary CatBoost model containing tree structures, split points, and categorical hashes.
- **`artifacts/model_metadata.json`**: JSON manifest recording the feature list, categorical columns, target variable, cutoff training date, and out-of-sample validation metrics (MAE, RMSE, WAPE).

---

## 5. Data Splitting Methodology & Validation Strategy

### 5.1 Strict Chronological Splitting (Zero Leakage)
In time-series demand forecasting, standard random train/test splits (e.g. `train_test_split(shuffle=True)`) cause severe **temporal lookahead bias** (training on future information to predict past events). RetailForecaster enforces **chronological cutoff splits**:

$$\text{Train Set} = \{ x_t \mid \text{week\_start\_date}(x_t) < \text{cutoff\_date} \}$$
$$\text{Validation/Holdout Set} = \{ x_t \mid \text{week\_start\_date}(x_t) \ge \text{cutoff\_date} \}$$

Where:
$$\text{cutoff\_date} = \max(\text{week\_start\_date}) - 4\text{ weeks}$$

```
Historical Timeline:
[======================= Train Period =======================][== 4-Week Holdout ==]
                                                              ^
                                                         Cutoff Date
```

### 5.2 Rolling-Origin Backtesting Framework
In `src/evaluation/evaluator.py`, the system performs multi-origin rolling backtesting over an 8-week holdout evaluation window:
- The origin $T$ rolls forward week by week: $T_1, T_2, T_3, T_4, \dots$
- At each origin, models generate predictions for horizon steps $h \in \{1, 2, 3, 4\}$.
- Predictions are compared against recorded actuals for each step, generating an **Error Progression Curve** that tracks error degradation as the forecast horizon extends into the future.

### 5.3 Mathematical Evaluation Metrics
The platform computes four standard enterprise retail metrics:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Measures the average unit error magnitude per SKU-week.*

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Highlights large errors and demand spikes.*

3. **Weighted Absolute Percentage Error (WAPE / WMAPE)**:
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i} \times 100\%$$
   *The primary retail metric. Avoids division-by-zero errors associated with standard MAPE when products have zero sales in a given week.*

4. **Forecast Bias**:
   $$\text{Bias} = \frac{\sum_{i=1}^N (\hat{y}_i - y_i)}{\sum_{i=1}^N y_i} \times 100\%$$
   *Identifies whether a model systematically over-predicts ($>0$) or under-predicts ($<0$).*

---

## 6. 5-Model Benchmark Comparison System

In `src/evaluation/evaluator.py`, five distinct forecasting algorithms are evaluated concurrently on identical holdout datasets:

```mermaid
graph LR
    Dataset[Standardized Test Window] --> M1[CatBoost GBDT]
    Dataset --> M2[Naive Baseline: Lag-1]
    Dataset --> M3[Moving Average: 4-Week Rolling]
    Dataset --> M4[AutoARIMA: Order Selection]
    Dataset --> M5[Prophet: Additive Decomposition]

    M1 --> MetricCalc[Metrics Engine: WAPE, MAE, RMSE, Bias]
    M2 --> MetricCalc
    M3 --> MetricCalc
    M4 --> MetricCalc
    M5 --> MetricCalc

    MetricCalc --> Scorecard[Ranked Leaderboard & Model Approval]
```

### 6.1 Model Definitions & Methodology
1. **CatBoost Regressor**: Gradient boosted decision tree ensemble utilizing 22 lagged and exogenous features, non-linear feature interactions, and early stopping on validation MAE.
2. **Naive Baseline (Lag-1 Demand)**:
   $$\hat{y}_{t+h} = y_t$$
   Assumes next week's sales will equal the most recent observed week's sales. Standard benchmark lower bound.
3. **Moving Average (4-Week Rolling Mean)**:
   $$\hat{y}_{t+h} = \frac{1}{4} \sum_{k=0}^3 y_{t-k}$$
   Smooths short-term fluctuations by averaging the four most recent weeks.
4. **ARIMA (AutoARIMA)**:
   $$\phi(B)(1 - B)^d y_t = \theta(B)\epsilon_t$$
   Implements automated parameter selection $(p, d, q)$ via AIC minimization through `statsforecast` / `pmdarima`.
5. **Prophet**:
   $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
   Decomposes series into piecewise linear growth $g(t)$, annual/weekly seasonality $s(t)$, and holiday effects $h(t)$.

### 6.2 Empirical Benchmark Scorecard (Reference Performance)

| Rank | Model Name | Architecture Type | Out-of-Sample WAPE | Mean Absolute Error | Relative Performance vs Baseline |
|:---:|---|---|:---:|:---:|:---:|
| 🥇 **1** | **CatBoost Regressor** | Gradient Boosted Trees | **~18.4%** | **~14.2 units** | **Champion (+48% vs Naive)** |
| 🥈 2 | **ARIMA (AutoARIMA)** | Classical Autoregressive | ~26.8% | ~21.5 units | Strong on stable trends |
| 🥉 3 | **Facebook Prophet** | Additive Decomposition | ~29.3% | ~23.8 units | Captures holidays well |
| 4 | **4-Week Moving Average** | Statistical Rolling Mean | ~32.1% | ~26.4 units | Lags behind trend shifts |
| 5 | **Naive (Lag-1)** | Persistence Benchmark | ~35.6% | ~29.1 units | High volatility penalty |

*CatBoost consistently ranks first across categories due to its ability to incorporate multi-variate cross-series signals, price elasticity, and festival flags simultaneously.*

---

## 7. Recursive Multi-Step Out-of-Sample Forecasting

### 7.1 The Recursive Forecasting Problem
When forecasting multiple weeks into the future ($h = 1, 2, 3, 4$), the actual sales values for future weeks are unknown. RetailForecaster utilizes a **recursive forecasting pipeline** implemented in `src/model/predict_future.py` and `src/forecast/recursive_forecast.py`:

```mermaid
sequenceDiagram
    participant Model as CatBoost Model
    participant State as Feature State Vector
    participant DB as MySQL Database

    Note over State: Initial State at Horizon 0 (Known Actuals)
    
    State->>Model: Predict Week 1 (using known actuals for lags)
    Model-->>State: Predicted Units Week 1
    Note over State: Update: lag_1 = Pred(1), lag_4 shifted, rolling_avg updated
    
    State->>Model: Predict Week 2 (using Pred 1 as lag_1)
    Model-->>State: Predicted Units Week 2
    Note over State: Update: lag_1 = Pred(2), rolling_avg updated
    
    State->>Model: Predict Week 3 (using Pred 2 as lag_1)
    Model-->>State: Predicted Units Week 3
    Note over State: Update: lag_1 = Pred(3), rolling_avg updated
    
    State->>Model: Predict Week 4 (using Pred 3 as lag_1)
    Model-->>State: Predicted Units Week 4
    
    State->>DB: Persist 4-Week Forward Forecasts
```

### 7.2 Recursive Update Logic (`predict_future.py`)
```python
latest_states = df.groupby(SERIES_KEY).last().reset_index()

for step in range(1, 5):
    forecast_date = last_date + pd.Timedelta(weeks=step)
    step_df = latest_states.copy()
    step_df["week_start_date"] = forecast_date
    step_df["month"] = forecast_date.month
    step_df["week_number"] = forecast_date.isocalendar().week

    # Predict demand for the current horizon step
    X_step = step_df[FEATURES]
    preds = np.maximum(0, model.predict(X_step))
    step_df["predicted_units_sold"] = np.round(preds, 2)
    step_df["forecast_step"] = f"Week +{step}"

    future_predictions.append(step_df[...])

    # RECURSIVE UPDATE: Feed predictions into feature state for the next step
    latest_states["lag_4_sales"] = latest_states["lag_1_sales"]
    latest_states["lag_1_sales"] = preds
    latest_states["rolling_4_week_avg"] = (
        0.75 * latest_states["rolling_4_week_avg"] + 0.25 * preds
    )
```

The forecasts are aggregated across stores and channels, joined with human-readable catalog names from `retail`, and written to table `future_4week_forecasts`.

---

## 8. LLM Integration: Google Gemini & Function-Calling Engine

RetailForecaster features a production-grade conversational AI architecture connecting Google Gemini with the MySQL database and ML forecasting artifacts.

```mermaid
sequenceDiagram
    actor User as Retail Merchandiser
    participant UI as Chat Interface (app.js)
    participant Flask as Flask Server (/api/chat)
    participant Orch as Gemini Orchestrator
    participant Gemini as Google Gemini 3.5/3.8
    participant Dispatch as Tool Dispatcher
    participant DB as MySQL Database
    participant Ground as Grounding Engine

    User->>UI: "Which festival had more demand?"
    UI->>UI: Show Animated Multi-Stage Loading Card
    UI->>Flask: POST /api/chat { message, history }
    Flask->>Orch: orchestrator.chat(message, history)
    
    Orch->>Gemini: generate_content(message, tools=10_declarations)
    Note over Gemini: Analyzes prompt & selects tool: get_festival_demand_analysis()
    Gemini-->>Orch: FunctionCall: get_festival_demand_analysis(period='all_time')
    
    Orch->>Dispatch: dispatch("get_festival_demand_analysis", args)
    Dispatch->>DB: SQL Query: Units sold aggregated by festival_name
    DB-->>Dispatch: [Navratri: 42,100, Diwali: 38,900, Eid: 24,500, ...]
    Dispatch-->>Orch: Formatted Tool Result Dict
    
    Orch->>Gemini: Return FunctionResponse(name, result)
    Gemini-->>Orch: Natural Language Response with Numerical Findings
    
    Orch->>Ground: validate(response_text, all_tool_results)
    Note over Ground: Verifies all numbers in text exist in SQL results
    Ground-->>Orch: Grounding Status: 'verified'
    
    Orch-->>Flask: { answer, tools_called, grounding_status: 'verified' }
    Flask-->>UI: JSON Response
    UI->>UI: Complete Steps, Mark Finished, Stream Response
    UI->>User: Displays Answer + Collapsible Tool Evidence Drawer
```

### 8.1 The 10 Allowlisted Analytics Tools
Gemini cannot query the database directly with arbitrary SQL. Instead, it interacts exclusively through 10 deterministic Python functions registered in `src/gemini/function_declarations.py` and executed via `src/analytics/dispatcher.py`:

1. `get_forecast_summary`: Retrieves model metrics, approval status, and run horizon details.
2. `search_products`: Resolves fuzzy product names or keywords (e.g., "Vivo V9", "iPhone 13") to canonical IDs.
3. `get_product_forecast`: Retrieves 4-week future unit and monetary revenue forecasts for a specific SKU.
4. `get_top_demand_changes`: Ranks products by demand volume, projected gains, projected drops, or highest revenue.
5. `compare_forecast_models`: Returns comparative MAE, WAPE, and observations across CatBoost, Naive, and Moving Average.
6. `get_product_accuracy`: Returns product-specific historical accuracy and reliability flags.
7. `get_demand_aggregates`: Calculates unit demand and revenue broken down by `state`, `city`, `region`, `category`, `brand`, or `channel`.
8. `get_revenue_and_pricing_overview`: Detailed pricing, discount percentages, and monetary earnings breakdown.
9. `get_unreliable_products`: Surfaces products tagged with reliability risk flags (sparse demand, spikes, stockout distortion).
10. `get_festival_demand_analysis`: Ranks demand across 18 Indian festivals (Diwali, Navratri, Eid, Christmas, etc.) and compares festival uplifts against non-festival baselines.

### 8.2 Strict System Prompt & Hallucination Guardrails
In `src/gemini/client.py`, `SYSTEM_INSTRUCTION` enforces non-negotiable operational constraints:
- **No Inventions**: Every numerical claim must originate from an executed tool result.
- **Whole Units Requirement**: In retail inventory planning, physical stock must be reported as integers (e.g. `1,790 units`, never `1,790.27 units`).
- **Product Names Mandatory**: Models must always present human-readable brand names (e.g., `Samsung SM-IT9750 (P0612)`) rather than raw database keys alone.
- **Geographic Aggregation Directive**: Explicitly requires using `get_demand_aggregates` with `group_by='state'` or `'city'` when questions reference geographic distribution.

### 8.3 Grounding Verification Engine (`src/gemini/grounding.py`)
To prevent hallucinations from reaching users:
1. Regex extracts all numerical values from the LLM's generated response.
2. The numbers are cross-referenced against the structured dictionaries returned by the executed database tools.
3. A match score is computed:
   - $\ge 80\%$ matched: Marked **`verified`** (green badge).
   - $40\% - 79\%$ matched: Marked **`partial`** (amber badge).
   - $< 40\%$ matched: Marked **`failed`**; triggers an automatic correction loop prompting Gemini to revise claims using only tool outputs.

### 8.4 Multi-Stage Dynamic Generation Animation (Gemini/GPT Style)
When a prompt is submitted, `frontend/app.js` runs a dynamic multi-stage execution controller:
- **Stage 1: Analyzing Query (0s - 1.0s)**: Orb icon shows `brain`. Caption: *"Understanding retail intent & query scope..."*
- **Stage 2: Calling Tools (1.0s - 2.5s)**: Orb icon transitions to `database`. Caption: *"Invoking retail intelligence functions & database..."* Checkmark on Step 1.
- **Stage 3: Analyzing Data (2.5s - 4.0s)**: Orb icon transitions to `line-chart`. Caption: *"Processing statistical outputs & ML predictions..."* Checkmark on Step 2.
- **Stage 4: Generating (4.0s+)**: Orb icon transitions to `sparkles`. Caption: *"Synthesizing executive demand insights..."* Checkmark on Step 3.
- **Finished State**: Badge turns green (`Finished`), checkmarks complete across all 4 steps, and the card cleanly transitions to the formatted Markdown response.

---

## 9. Complete Platform Functionalities (A to Z)

### 9.1 Authentication & Role-Based Access Control
- **Login Modal**: Modern, warm executive canvas with tab toggling between Sign In and Sign Up.
- **Secure Storage**: Passwords hashed with `bcrypt`.
- **Pre-seeded Admin**: Username `Admin`, Password `123456`, assigned role `'admin'`.
- **Role Routing**:
  - `admin` accounts are directed to **Developer Studio**.
  - Standard user accounts are directed to **Retailer Planning**.
  - Role switcher buttons are strictly disabled to prevent unauthorized role escalation.
- **Top-Right Account Menu**: Clean, compact dropdown containing only two buttons:
  - **My Profile**: Opens the profile management dialog.
  - **Sign Out**: Clears the session and returns to login.

### 9.2 Executive Profile Management Card
- **Viewport-Constrained Modal Box**: Constrained with `max-height: calc(100vh - 40px)` and flex-column layout so it fits completely within any browser window without clipping.
- **Pinned Top Header**: Displays avatar gradient, user display name, role badge, and close button.
- **Scrollable Body**: Styled with a slim 5px scrollbar for smaller viewports. Contains two cards:
  - *Card 1: Contact & Store Details*: Phone number, email address, business/store name.
  - *Card 2: Security & Credentials (Optional)*: Current password, new password, confirm new password.
- **Pinned Bottom Footer**: Cancel and Save Profile Changes buttons remain pinned and visible at all times.

### 9.3 Developer Studio Navigation & Features
1. **Dataset & Quality**:
   - Total rows loaded, clean row ratio, missing value counts across all 21 columns.
   - Target distribution histogram, skewness indicator, and variance telemetry.
2. **Model Evaluation**:
   - Rolling-origin backtest error analysis.
   - Dual-axis Chart.js visualization: WAPE % progression on left axis vs. MAE units on right axis across Weeks +1, +2, +3, and +4.
   - Residual distribution diagnostics.
3. **Model Comparison**:
   - Side-by-side leaderboard of all 5 models (CatBoost, Naive, MA4, ARIMA, Prophet).
   - Metric cards displaying MAE, RMSE, WAPE, and rank badges.
   - Approved Model Selector: Allows admins to approve and promote an active production model.
4. **Experiment History**:
   - Chronological historical log of all evaluation runs with timestamps and configuration manifests.
   - "Run 5-Model Benchmark" modal: Triggers asynchronous background re-evaluation over 4 or 8 holdout weeks.
5. **Settings**:
   - Model hyperparameters, reliability threshold tuners, database ping status.

### 9.4 Retailer Planning Navigation & Features
1. **Demand Forecasts (Explorer)**:
   - 4-week forward product demand curves rendered in Chart.js.
   - Dropdown filtering by Product Category, Store Outlet, and Sales Channel.
   - SKU table showing Week +1, +2, +3, +4 predicted units, historical averages, and percentage growth.
   - Approved model metadata badge displayed at the end of the page.
2. **Business Overview**:
   - High-level executive KPI cards: Total Projected Demand, Forecasted Revenue (₹), Active Stores, Catalog Size.
   - Category Demand Distribution chart.
3. **Product Insights**:
   - Top Gainers: Products projected to see highest demand growth.
   - Declining SKUs: Products with dropping trajectories requiring reduced replenishment.
   - Reliability Alert Cards: Identifies products subject to demand spikes or stockout distortions.
4. **Geographic Demand Intelligence (India Map)**:
   - Political map image of India with dynamic store outlet pins.
   - Default zoom level set to 125% with interactive Zoom In (`+`), Zoom Out (`-`), and Reset Center controls.
   - Color-coded pins representing demand volume:
     - 🟢 **Emerald**: Steady Demand
     - 🟡 **Amber**: High Demand
     - 🔴 **Coral**: Peak Volume
     - 🟣 **Purple**: Super Hub
   - Hover tooltips displaying Store Name, City, State, Weekly Forecast Units, and Projected Revenue.
   - Toggle switcher between **City Outlets** and **State Aggregation**.
5. **AI Assistant (Conversational Analytics)**:
   - Floating chat interface with pre-built prompt pills ("Top Growth SKUs", "State Demand for Samsung", "Festival Demand Rankings", "Revenue Forecast").
   - Markdown rendering: Tables, code blocks, lists, bold text, headings.
   - Collapsible tool execution drawer showing the exact SQL queries and tool arguments called.
   - Response copy button and positive/negative reinforcement feedback buttons.
   - Animated multi-stage loading indicator reflecting active execution stages.
6. **Executive Reports**:
   - One-click downloads for PDF, HTML, and CSV business reports.

---

## 10. Executive Reporting Engine (PDF, HTML, CSV)

The reporting subsystem in `src/reporting/report_generator.py` compiles empirical forecasting results into formal deliverables:

### 10.1 Executive PDF Generator (`ReportLab` + `Matplotlib`)
- **Cover Header**: Formal corporate title, run metadata, creation timestamp, and evaluation provenance.
- **KPI Summary Grid**: 4-box layout presenting Total Demand Units, Total Projected Revenue, Average Price, and Average WAPE.
- **Embedded Matplotlib Visualizations**: High-resolution charts rendered into in-memory byte buffers (`io.BytesIO`) and placed onto the ReportLab canvas:
  - Weekly demand volume trend line chart.
  - Category breakdown bar chart.
- **Top 15 Demanded Products Table**: Formatted tabular grid with alternate row shading, right-aligned monetary values, and bold totals.
- **Reliability & Risk Advisory Section**: Explicit warnings for inventory planners regarding volatile or sparse SKUs.
- **Page Numbering**: Two-pass canvas numbering ("Page X of Y").

### 10.2 Interactive HTML Report
- Standalone, self-contained HTML page styled with inline CSS.
- Print-optimized CSS rules (`@media print`) enabling seamless export to PDF via the browser's print dialog (`Ctrl+P`).

### 10.3 Enhanced CSV Export
- Clean CSV dataset with standardized column headers: `product_id`, `product_name`, `category`, `week_start_date`, `forecast_step`, `predicted_units`, `unit_price`, `projected_revenue`.

---

## 11. Frontend Design System & User Experience

### 11.1 Design Philosophy & Color Palette
The UI follows an executive warm palette balancing modern aesthetics with enterprise readability:

| Color Token | Hex Code | Semantic Role |
|---|---|---|
| `--color-beige` | `#E9E3DE` | Subtle container surfaces, table borders, neutral card fills |
| `--color-taupe` | `#A5937B` | Secondary controls, inactive icons, subtle borders |
| `--color-gold` | `#E3C49B` | Highlighting top-ranking models, badges, and accents |
| `--color-charcoal`| `#666161` | Secondary typography, table headers, breadcrumb links |
| `--color-lavender`| `#AF9AC9` | Primary brand accent, active tabs, buttons, chart series |
| `--bg-page` | `#FBF9F7` | Soft off-white page background |
| `--bg-surface` | `#FFFFFF` | Elevated card surfaces and containers |
| `--text-primary` | `#211E1D` | High-contrast dark charcoal text |

### 11.2 Micro-Interactions & Usability Details
- **Smooth View Transitions**: Tab switching between subpages utilizes CSS fade-in animations without page reloads.
- **Toast Notifications**: Floating feedback alerts in the lower-right corner for actions like saving profiles, copying responses, or triggering evaluations.
- **Cache Busting**: Assets are referenced with versioned query strings (`index.css?v=20261027_02`, `app.js?v=20261027_02`) ensuring browsers fetch updated styles and logic immediately.
- **Responsive Drawer Navigation**: On tablet and mobile viewports, the sidebar collapses into a slide-over mobile drawer toggled via a hamburger menu.

---

## 12. Project File Map & Repository Layout

```
retail__data/
│
├── .env                              # Environment secrets (DATABASE_URL, GEMINI_API_KEY, FLASK_SECRET)
├── .env.example                      # Template environment file
├── requirements.txt                  # Python production dependencies
├── README.md                         # Project overview and quickstart guide
├── report.md                         # Complete A-to-Z Project Report (This document)
│
├── artifacts/                        # Serialized ML artifacts
│   ├── catboost_demand.cbm           # Compiled binary CatBoost model
│   └── model_metadata.json           # Model manifest (features, metrics, dates)
│
├── data/                             # Raw CSV backups & evaluation cache
│   └── evaluation/                   # JSON logs of evaluation and forecast runs
│       ├── eval_*.json               # 5-model evaluation run results
│       └── forecast_run_*.json       # Forward forecast outputs
│
├── src/                              # Python source package
│   ├── __init__.py
│   ├── config.py                     # Central configuration constants, paths, and env loading
│   ├── db.py                         # SQLAlchemy engine, connection pool, tables, seeders
│   │
│   ├── model/                        # Model training and prediction scripts
│   │   ├── train_cat_late.py         # Champion CatBoost pipeline with lag features
│   │   ├── train_cb.py               # Lightweight CatBoost experiment script
│   │   ├── train_arima.py            # Comprehensive ARIMA benchmark script
│   │   ├── train_2arima.py           # Single-series global ARIMA baseline
│   │   ├── train_sarimax.py          # Exogenous Auto-SARIMAX experiment
│   │   └── predict_future.py         # 4-week recursive forward forecasting pipeline
│   │
│   ├── evaluation/                   # Benchmark and validation engine
│   │   ├── evaluator.py              # Main 5-model rolling-origin evaluator
│   │   ├── data_loader.py            # Raw data loading and lag feature preparation
│   │   ├── metrics.py                # MAE, RMSE, WAPE, Bias calculation routines
│   │   ├── baselines.py              # Naive and Moving Average implementations
│   │   ├── arima_forecast.py         # AutoARIMA integration via StatsForecast
│   │   ├── prophet_forecast.py       # Facebook Prophet integration
│   │   ├── reliability.py            # Automated reliability risk flag generator
│   │   └── run_evaluation.py         # CLI trigger for evaluation runs
│   │
│   ├── forecast/                     # Production forecasting engine
│   │   ├── recursive_forecast.py     # Production recursive multi-step forecasting
│   │   └── run_forecast.py           # CLI trigger for forward forecast runs
│   │
│   ├── gemini/                       # AI Assistant & LLM subsystem
│   │   ├── client.py                 # Google GenAI SDK wrapper and system instructions
│   │   ├── function_declarations.py  # 10 allowlisted tool schemas for Gemini
│   │   ├── orchestrator.py           # Tool-calling loop, retries, and dispatching
│   │   ├── grounding.py              # Numerical claim verification engine
│   │   └── learning.py               # Active learning prompt enhancement from feedback
│   │
│   ├── analytics/                    # Deterministic analytics tools executed by LLM
│   │   ├── dispatcher.py             # Route tool names to Python implementations
│   │   └── tools.py                  # Database queries for forecasts, revenue, festivals
│   │
│   ├── reporting/                    # Deliverables & document generation
│   │   └── report_generator.py       # PDF, HTML, and CSV export engine
│   │
│   └── api/                          # Web application API tier
│       ├── app.py                    # Flask application factory, CORS, and blueprint setup
│       ├── auth_routes.py            # Login, signup, logout, session, and profile endpoints
│       └── routes.py                 # REST endpoints for evaluation, forecast, geo, chat
│
└── frontend/                         # Single-page web application frontend
    ├── index.html                    # Dual-workspace DOM layout (Developer & Retailer)
    ├── index.css                     # Executive warm theme, map styling, animations
    ├── app.js                        # Client state, Chart.js, map rendering, chat controller
    └── india_map.png                 # Political map of India for geographic demand pins
```

---

## 13. Setup, Execution & Deployment Guide

### 13.1 Prerequisites
- Python 3.10, 3.11, or 3.12
- MySQL Server (>= 8.0)
- Google Gemini API Key

### 13.2 Environment Setup
1. Clone the repository and navigate into the workspace directory:
   ```bash
   cd d:\AI_PROJECTS\retail__data
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env`:
   ```ini
   DATABASE_URL=mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major
   GEMINI_API_KEY=AIzaSyYourGeminiApiKeyHere
   GEMINI_MODEL=gemini-3.5-flash-lite
   FLASK_SECRET_KEY=super-secret-enterprise-key-2026
   FLASK_ENV=production
   ```

### 13.3 Database & Model Training Execution
1. **Initialize Database Tables & Seed Admin**:
   ```bash
   python -c "from src.db import create_tables; create_tables()"
   ```
2. **Train CatBoost Demand Model**:
   ```bash
   python src/model/train_cat_late.py
   ```
   *Trains the CatBoost Regressor, outputs validation metrics, and writes `artifacts/catboost_demand.cbm`.*

3. **Generate 4-Week Forward Forecasts**:
   ```bash
   python src/model/predict_future.py
   ```
   *Recursively generates future predictions and populates table `future_4week_forecasts` in MySQL.*

4. **Execute 5-Model Benchmark Evaluation**:
   ```bash
   python src/evaluation/run_evaluation.py
   ```
   *Runs CatBoost, Naive, Moving Average, ARIMA, and Prophet across holdout weeks and records results.*

### 13.4 Running the Application
Launch the Flask web server with `PYTHONPATH` set to the workspace root:
```powershell
$env:PYTHONPATH="."
.venv\Scripts\python.exe src\api\app.py
```
Open a browser and navigate to:
```
http://localhost:5000/
```

### 13.5 Default Credentials
- **Admin (Developer Studio)**: Username `Admin` | Password `123456`
- **Retailer (Retail Planning)**: Click "Create account" on the login modal to register any retailer username.

---

## Conclusion
RetailForecaster represents an enterprise-grade demand forecasting and decision-support architecture. By fusing **gradient boosted tree modeling (CatBoost)** with **classical econometric baselines (ARIMA, Prophet)**, **reliable relational data persistence (MySQL/SQLAlchemy)**, **grounded generative AI reasoning (Google Gemini)**, and an **interactive executive frontend**, the platform provides transparent, auditable, and actionable forecasting intelligence across Indian retail operations.
