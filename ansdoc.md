# RetailForecaster — Comprehensive System Architecture & Technical Overview

This document provides an end-to-end technical explanation of the **RetailForecaster** Enterprise Demand Analytics platform, covering data pipelines, forecasting models, model training and evaluation, the Google Gemini AI Assistant, MySQL database integration, and tool/function-calling mechanisms.

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [End-to-End System Workflow](#2-end-to-end-system-workflow)
3. [The Machine Learning & Time-Series Models](#3-the-machine-learning--time-series-models)
   - [CatBoost Regressor (Production Model)](#31-catboost-regressor-production-ml-model)
   - [AutoARIMA / ARIMA(1,1,1)](#32-autoarima--arima111)
   - [Facebook Prophet](#33-facebook-prophet)
   - [Moving Average Baseline (MA4)](#34-4-week-moving-average-ma4-baseline)
   - [Naive Baseline (Lag-1)](#35-naive-baseline-lag-1)
4. [Dataset Features & Columns Used](#4-dataset-features--columns-used)
5. [Model Training, Convergence & Verification](#5-model-training-convergence--verification)
6. [Model Evaluation Methodology & Error Metrics](#6-model-evaluation-methodology--error-metrics)
   - [Evaluation Protocol (Rolling-Origin Backtesting)](#61-evaluation-protocol-rolling-origin-backtesting)
   - [Mathematical Error Metrics (MAE, RMSE, WAPE)](#62-mathematical-error-metrics)
   - [Series Reliability Auditing & Volatility Flags](#63-series-reliability-auditing--volatility-flags)
7. [The AI Business Assistant (Google Gemini 2.5/1.5 Flash)](#7-the-ai-business-assistant-google-gemini)
   - [Which Model is Attached to the Assistant](#71-which-model-is-attached-to-the-assistant)
   - [How the Assistant Generates Answers](#72-how-the-assistant-generates-answers)
   - [Reinforcement Learning & Feedback Loop](#73-reinforcement-learning--feedback-loop)
8. [Tool & Function Calling Protocol (Zero-Hallucination)](#8-tool--function-calling-protocol-zero-hallucination)
   - [What is Tool Calling?](#81-what-is-tool-calling)
   - [Catalog of Declared Tools](#82-catalog-of-declared-tools)
   - [Grounding Validation Layer](#83-grounding-validation-layer)
9. [MySQL Database Connection & Data Architecture](#9-mysql-database-connection--data-architecture)
10. [Model Status & Production Summary Matrix](#10-model-status--production-summary-matrix)

---

## 1. Executive Summary & High-Level Architecture

**RetailForecaster** is an enterprise-grade retail demand forecasting and intelligence system built for multi-store, multi-channel retail operations. It combines:
- **Recursive Machine Learning Demand Forecasting**: Multi-step forward predictions using Gradient Boosted Decision Trees (CatBoost).
- **Time-Series Benchmarking**: Chronological backtesting against ARIMA, Facebook Prophet, Moving Averages, and Naive baselines.
- **Enterprise Web Studio**: A dual-view interface (Retailer Studio for business inventory decisions and Developer Studio for model evaluation, metrics, and data quality).
- **Grounded AI Business Assistant**: Powered by Google Gemini via deterministic tool-calling, connected directly to MySQL and CatBoost forecast artifacts with zero hallucination.

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|   +------------------------------------+   +----------------------------------+   |
|   |         RETAILER STUDIO            |   |         DEVELOPER STUDIO         |   |
|   | - 4-Week Demand Projections        |   | - 5-Model Benchmark Rankings     |   |
|   | - Store/Category Volume Ranking    |   | - Error Progression by Horizon   |   |
|   | - AI Demand Intelligence Assistant |   | - Data Ingestion & Quality Audit |   |
|   +------------------------------------+   +----------------------------------+   |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST API
+------------------------------------------v----------------------------------------+
|                            FLASK API BACKEND (src/api/)                           |
|   - Routes: /api/forecast, /api/evaluation/latest, /api/chat, /api/chat/feedback |
+----------------------+--------------------+---------------------------------------+
                       |                    |
        +--------------v-------+    +-------v----------------+
        |   GEMINI ASSISTANT   |    |  FORECAST & EVALUATION |
        |  (Google GenAI SDK)  |    |  - CatBoost Regressor  |
        |  - Tool Declarations |    |  - AutoARIMA & Prophet |
        |  - Grounding Guard   |    |  - Moving Avg & Naive  |
        |  - Dynamic Learning  |    +-------+----------------+
        +--------------+-------+            |
                       | Calls Tools        | Reads/Writes Artifacts
        +--------------v--------------------v----------------+
        |        BUSINESS ANALYTICS TOOLS & ARTIFACTS        |
        |  - artifacts/catboost_demand.cbm (Trained Model)   |
        |  - artifacts/evaluation/forecast_run_*.json        |
        |  - artifacts/assistant_learning.json               |
        +-----------------------------+----------------------+
                                      | Queries
        +-----------------------------v----------------------+
        |               MySQL DATABASE (major)               |
        |  - retail (1,024,772 rows raw sales transactions)  |
        |  - forecast_results (Model predictions & metrics)  |
        |  - evaluation_runs & evaluation_metrics            |
        |  - reliability_flags & gemini_interactions         |
        +----------------------------------------------------+
```

---

## 2. End-to-End System Workflow

1. **Data Ingestion**: Weekly retail sales data across 719 products, 14 stores, and 3 sales channels (offline, online, omnichannel) over 104 weeks (October 2024 to October 2026) is stored in the MySQL table `retail` and cached in `data/raw/final_raw_datset.csv`.
2. **Feature Engineering**: Autoregressive lag features (`lag_1_sales`, `lag_4_sales`, `rolling_4_week_avg`) are recomputed strictly within each series grain (`product_id × store_id × sales_channel`) to eliminate cross-series data leakage.
3. **Model Training**: The production CatBoost regressor is trained on historical data up to a validation cutoff date, optimizing for Mean Absolute Error (MAE) and saved as a binary artifact.
4. **Recursive Multi-Step Forecasting**: The system generates a 4-week forward demand projection for every individual series. When forecasting week $t+1$, predicted values feed recursively into lag features for week $t+2$, $t+3$, and $t+4$.
5. **Rigorous Backtesting**: The Evaluator runs chronological rolling-origin backtesting against 4 baseline/benchmark models (Naive Lag-1, 4-Week Moving Average, AutoARIMA, and Facebook Prophet).
6. **Data Presentation**: Flask API endpoints serve metrics, comparisons, and charts to the glassmorphic frontend.
7. **Conversational Intelligence**: When users ask natural language questions in the AI Assistant, Google Gemini detects analytical intents, triggers parameterized Python functions against MySQL/forecast artifacts, verifies the response against tool output, and returns grounded answers.

---

## 3. The Machine Learning & Time-Series Models

The project implements **5 distinct models**, each serving a specific role in evaluation and production:

### 3.1. CatBoost Regressor (Production ML Model)
- **Role**: **Primary Production Forecasting Engine**.
- **Files**: [`src/model/train_cat_late.py`](file:///d:/AI_PROJECTS/retail__data/src/model/train_cat_late.py), [`src/model/predict_future.py`](file:///d:/AI_PROJECTS/retail__data/src/model/predict_future.py), [`src/forecast/recursive_forecast.py`](file:///d:/AI_PROJECTS/retail__data/src/forecast/recursive_forecast.py)
- **Algorithm**: Symmetric (oblivious) Gradient Boosted Decision Trees (GBDT). CatBoost handles categorical features natively via target statistics without requiring one-hot encoding blowup.
- **Production Hyperparameters (`train_cat_late.py`)**:
  - `loss_function="MAE"`
  - `eval_metric="MAE"`
  - `iterations=1000`
  - `depth=6`
  - `learning_rate=0.05`
  - `early_stopping_rounds=100`
- **Validation Accuracy**:
  - **Overall MAE**: 2.75 units
  - **Overall RMSE**: 7.17 units
  - **Overall WAPE**: **40.75%** (a ~20% improvement over initial baseline iterations)
- **Why MAE Loss?**: Retail demand is heavily right-skewed with intermittent zeros (22.7% zero-demand weeks) and sporadic promotional spikes. Squared error ($L_2$) over-penalizes rare spikes, causing the model to over-forecast everyday demand. MAE ($L_1$) forecasts the conditional median, providing stable inventory stock levels.
- **Future Predictions**: Saved to MySQL table `future_4week_forecasts` and JSON artifact `artifacts/evaluation/forecast_run_*.json`.

### 3.2. AutoARIMA / ARIMA(1,1,1)
- **Role**: Classical statistical benchmark & standalone time series pipeline.
- **Files**: [`src/evaluation/arima_forecast.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/arima_forecast.py), [`src/model/train_2arima.py`](file:///d:/AI_PROJECTS/retail__data/src/model/train_2arima.py)
- **Algorithm**: Autoregressive Integrated Moving Average.
  - Model formulation: $ARIMA(p, d, q)$ where $p$ is autoregressive order, $d$ is differencing degree, and $q$ is moving average order.
  - In `train_2arima.py`: Fits an $ARIMA(1, 1, 1)$ model per product using a multi-process pool (`ProcessPoolExecutor`) reading from MySQL and writing forecast results back to MySQL table `forecast_results`.
  - In `evaluator.py`: Uses `statsforecast.models.AutoARIMA` to automatically search for optimal $(p, d, q)$ parameters based on Akaike Information Criterion (AIC).

### 3.3. Facebook Prophet
- **Role**: Structural time series benchmark with trend changepoints.
- **File**: [`src/evaluation/prophet_forecast.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/prophet_forecast.py)
- **Algorithm**: Decomposable Generalized Additive Model:
  $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
  Where $g(t)$ is piecewise linear growth, $s(t)$ represents yearly/weekly seasonality, $h(t)$ accounts for holiday effects, and $\epsilon_t$ is normally distributed error.
- **Evaluated Cohort**: Evaluated across high-volume representative series to provide an empirical Bayesian structural benchmark.

### 3.4. 4-Week Moving Average (MA4) Baseline
- **Role**: Statistical smoothing baseline.
- **File**: [`src/evaluation/baselines.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/baselines.py)
- **Algorithm**: Takes the unweighted arithmetic mean of the preceding 4 observed weeks for each series:
  $$\hat{y}_{t+h} = \frac{1}{4} \sum_{i=0}^{3} y_{t-i}$$
- **Key Empirical Finding**: In backtests on steady, non-seasonal products, MA4 often matches or beats complex models because it has zero variance and cannot overfit noisy weekly fluctuations.

### 3.5. Naive Baseline (Lag-1)
- **Role**: Minimal persistence baseline.
- **File**: [`src/evaluation/baselines.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/baselines.py)
- **Algorithm**: Simply repeats the most recent observed demand forward across all forecast horizons:
  $$\hat{y}_{t+h} = y_t$$
- **Purpose**: Any viable ML model must decisively outperform Naive Lag-1 to justify operational deployment.

---

## 4. Dataset Features & Columns Used

The underlying dataset contains **1,024,772 rows × 36 columns** covering 104 weeks. Here is how each model utilizes columns:

| Column Name | Data Type | CatBoost (ML) | ARIMA / Prophet | Moving Avg / Naive | Role / Description |
|---|---|:---:|:---:|:---:|---|
| `units_sold` | Integer/Float | **TARGET** ($y$) | **TARGET** ($y$) | **TARGET** ($y$) | Weekly sales volume. |
| `week_start_date` | Date / String | Time index | Time index (`ds`) | Time index | Monday start of sales week. |
| `product_id` | String (Categorical) | Feature & Key | Series Key | Series Key | Unique product SKU identifier. |
| `store_id` | String (Categorical) | Feature & Key | Grouping Key | Series Key | Retail store identifier (14 stores). |
| `sales_channel` | String (Categorical) | Feature & Key | Grouping Key | Series Key | `offline`, `online`, `omnichannel`. |
| `category` | String (Categorical) | Feature | Filter / Group | Metadata | Broad category (e.g., Smartphones, Laptops). |
| `subcategory` | String (Categorical) | Feature | - | - | Granular category division. |
| `brand` | String (Categorical) | Feature | Filter / Group | Metadata | Manufacturer (Samsung, Apple, Vivo, etc.). |
| `region` | String (Categorical) | Feature | - | - | Geographical region (North, South, West, East). |
| `season` | String (Categorical) | Feature | - | - | Winter, Summer, Monsoon, Festive. |
| `festival_name` | String (Categorical) | Feature | - | - | Diwali, Dussehra, Eid, None. |
| `promotion_type` | String (Categorical) | Feature | - | - | BOGO, Flat Discount, Cashback, None. |
| `month` | Integer (1–12) | Numerical Feature | - | - | Calendar month seasonality. |
| `week_number` | Integer (1–52) | Numerical Feature | - | - | Annual week index. |
| `holiday_flag` | Binary (0 / 1) | Numerical Feature | - | - | Flag indicating public/national holiday. |
| `promotion_flag` | Binary (0 / 1) | Numerical Feature | - | - | Active price promotion flag. |
| `unit_price` | Float (INR) | Numerical Feature | - | - | Selling price per unit. |
| `discount_pct` | Float (%) | Numerical Feature | - | - | Percentage discount applied. |
| `opening_stock` | Integer | Numerical Feature | - | - | Inventory available at start of week. |
| `stock_received` | Integer | Numerical Feature | - | - | Supply replenishment arrived that week. |
| `supplier_lead_time_days` | Integer | Numerical Feature | - | - | Days required to restock from supplier. |
| `lag_1_sales` | Float | Engineered Feature | - | - | Sales from 1 week prior (within series). |
| `lag_4_sales` | Float | Engineered Feature | - | - | Sales from 4 weeks prior (within series). |
| `rolling_4_week_avg` | Float | Engineered Feature | - | - | Moving mean of past 4 weeks (within series). |

---

## 5. Model Training, Convergence & Verification

### How Models are Trained

1. **CatBoost Training Process (`train_cb.py`)**:
   - Data is sorted chronologically by `series_key` (`product_id, store_id, sales_channel`) and `week_start_date`.
   - Lag features (`lag_1_sales`, `lag_4_sales`, `rolling_4_week_avg`) are calculated using `shift()` strictly inside series groups.
   - Rows lacking sufficient historical window for lags are cleanly dropped.
   - Data is split chronologically: training split before the validation cutoff; validation split after the cutoff.
   - Categorical missing values are imputed with `"UNKNOWN"` and explicitly passed via `cat_features=CAT_FEATURES`.
   - Model fits for up to 1,200 iterations with early stopping (stops if validation MAE does not improve for 100 consecutive rounds).

2. **ARIMA Batch Training (`train_2arima.py`)**:
   - Pulls `product_id, week_start_date, units_sold` from MySQL.
   - Groups weekly demand per product across all stores.
   - Splits series: train on all weeks up to $T-4$, test on the last 4 weeks.
   - Fits `statsmodels.tsa.arima.model.ARIMA(order=(1, 1, 1))` in parallel using Python's `ProcessPoolExecutor`.
   - Generates 4-step out-of-sample predictions, clips negative values at 0, calculates MAE, RMSE, WAPE, and inserts records into MySQL table `forecast_results`.

### How Do We Confirm Models are Trained?

| Verification Method | CatBoost Regressor | ARIMA Pipeline | Evaluator Benchmark |
|---|---|---|---|
| **File Artifacts** | `artifacts/catboost_demand.cbm` (binary weights file, ~44 MB) and `artifacts/model_metadata.json`. | Generates `artifacts/arima/duplicate_series_weeks.csv`. | Writes `artifacts/evaluation/eval_*.json` and `forecast_run_*.json`. |
| **Database Records** | Metadata logged in `app_settings` / `evaluation_runs`. | Thousands of rows written to MySQL table `forecast_results`. | Logged in MySQL `evaluation_runs` and `evaluation_metrics`. |
| **Python Test Check** | `from catboost import CatBoostRegressor`<br>`m = CatBoostRegressor()`<br>`m.load_model("artifacts/catboost_demand.cbm")`<br>`print(m.get_param('iterations'))` | Run `SELECT COUNT(*) FROM forecast_results;` in MySQL to verify entries. | Check latest timestamp in `artifacts/evaluation/`. |

---

## 6. Model Evaluation Methodology & Error Metrics

### 6.1. Evaluation Protocol (Rolling-Origin Backtesting)

To simulate true production forecasting without future lookahead bias:
- **Held-Out Test Window**: The last 8 weeks of historical data are held out strictly for testing.
- **Rolling Origins**: The model generates 4-week forecasts from multiple chronological cutoff dates.
- **Recursive Autoregression**: For multi-step forecasting ($h=1, 2, 3, 4$), the predicted demand from step $h=1$ is dynamically fed into `lag_1_sales` for step $h=2$, simulating real operational conditions.

### 6.2. Mathematical Error Metrics

The platform evaluates models across three complementary metrics defined in [`src/evaluation/metrics.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/metrics.py):

#### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
- Measures the average absolute error in physical unit sales.
- Directly interpretable by inventory managers (e.g., "We are off by an average of 3.2 units per product per week").

#### 2. Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
- Penalizes large errors disproportionately due to the squaring term.
- Identifies models that make catastrophic mistakes during unexpected demand surges.

#### 3. Weighted Absolute Percentage Error (WAPE)
$$\text{WAPE} = 100 \times \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} |y_i|}$$
- **Why WAPE instead of MAPE?** In retail, low-volume or zero-sales weeks cause standard MAPE ($\frac{|y - \hat{y}|}{y}$) to divide by zero or explode to infinity. WAPE sums all errors across the portfolio and divides by total actual sales, providing a robust, volume-weighted percentage error.

### 6.3. Series Reliability Auditing & Volatility Flags

In [`src/evaluation/reliability.py`](file:///d:/AI_PROJECTS/retail__data/src/evaluation/reliability.py), the system audits all 30,000+ series and flags products with reliability warnings:
- `INSUFFICIENT_HISTORY`: Products with fewer than 8 weeks of data.
- `EXTREME_VOLATILITY`: Coefficient of Variation ($CV = \frac{\sigma}{\mu} > 1.5$) or heavy intermittent zero sales.
- `DEMAND_SPIKE`: Single-week sales exceeding $3\sigma$ above the product's historical mean.
- `STOCKOUT_DISTORTION`: Weeks where `opening_stock == 0` or `stockout_flag == 1`, suppressing true consumer demand.

---

## 7. The AI Business Assistant (Google Gemini)

### 7.1. Which Model is Attached to the Assistant?
- **Cognitive / Reasoning Model**: **Google Gemini 2.5 Flash** (with automatic fallback to **Gemini 1.5 Flash** via `google-genai` SDK in [`src/gemini/client.py`](file:///d:/AI_PROJECTS/retail__data/src/gemini/client.py)).
- **Predictive ML Engine**: **CatBoost Regressor** (`artifacts/catboost_demand.cbm`).
- **Relationship**: Gemini **does not** generate numbers or predict demand from its own weights. Gemini acts as an **orchestrator and natural language interface** that translates user questions into tool calls, queries the CatBoost forecasts and MySQL database, and synthesizes executive answers.

```
+--------------------+        +-----------------------+        +--------------------------+
|    User Prompt     | -----> |  Google Gemini Flash  | -----> | Function Call:           |
| "Top Samsung sales |        |  (Reasoning Engine)   |        | get_demand_aggregates()  |
|  in Maharashtra?"  |        +-----------------------+        +------------+-------------+
+--------------------+                                                      |
                                                                            v
+--------------------+        +-----------------------+        +------------+-------------+
| Verified Executive | <----- | Synthesis & Grounding | <----- | Python/MySQL Dispatcher: |
| Answer with Tables |        | Validation            |        | Units: 507, Rev: ₹1.14 Cr|
+--------------------+        +-----------------------+        +--------------------------+
```

### 7.2. How the Assistant Generates Answers

1. **Prompt Ingestion**: User inputs questions via the AI Assistant chat interface.
2. **Context & Reinforcement Injection**: The orchestrator injects `system_instruction` containing:
   - Retail business domain rules (state/store lookup, whole-integer units, monetary rupee values).
   - In-context reinforcement directives derived from user rewards and penalties.
3. **Intent Detection & Function Calling**: Gemini determines whether the question requires quantitative retail data. If yes, it emits a structured function call instead of guessing.
4. **Deterministic Dispatch**: The backend [`Dispatcher`](file:///d:/AI_PROJECTS/retail__data/src/analytics/dispatcher.py) executes the matching Python analytical tool in [`src/analytics/tools.py`](file:///d:/AI_PROJECTS/retail__data/src/analytics/tools.py).
5. **Evidence Return**: The exact numerical results are formatted as JSON and sent back to Gemini as a tool response.
6. **Natural Language Generation**: Gemini converts the verified numbers into an executive summary, Markdown tables, and actionable business insights.
7. **Grounding Validation Guard**: [`GroundingValidator`](file:///d:/AI_PROJECTS/retail__data/src/gemini/grounding.py) cross-examines the output text against the tool's raw data to confirm zero hallucinated metrics.
8. **UI Presentation**: The chat renders the response, appends the collapsible **"N queries executed"** accordion for tool inspection, and attaches **Copy**, **Good Response (Thumbs Up)**, and **Bad Response (Thumbs Down)** buttons.

### 7.3. Reinforcement Learning & Feedback Loop

File: [`src/gemini/learning.py`](file:///d:/AI_PROJECTS/retail__data/src/gemini/learning.py)

The assistant features an active feedback and learning mechanism:
- **Positive Reward (`👍 Good Response`)**:
  - Signals preferred response format, verified product names, and clear business metrics.
  - Adds the conversation pair to high-quality exemplars in `artifacts/assistant_learning.json` and database table `gemini_feedback`.
  - Injects exemplars into subsequent Gemini system prompts.
- **Negative Penalty (`👎 Bad Response`)**:
  - Signals low quality, vague answers, or ungrounded statements.
  - Generates explicit negative constraints (e.g., "Do not provide generic prose without exact numerical data from tools").

---

## 8. Tool & Function Calling Protocol (Zero-Hallucination)

### 8.1. What is Tool Calling?
Tool calling (or Function Calling) is an API protocol where the LLM is provided with strict JSON schemas describing available functions. The LLM cannot execute code directly; instead, it outputs a structured JSON object specifying:
```json
{
  "name": "get_demand_aggregates",
  "args": {
    "brand": "Samsung",
    "category": "Smartphones",
    "group_by": "state",
    "period": "last_4_weeks"
  }
}
```
The server runs the function locally, queries MySQL or forecast artifacts, and returns the result. This guarantees that **all calculations, aggregations, and forecasts are 100% deterministic, auditable, and mathematically exact.**

### 8.2. Catalog of Declared Tools

Configured in [`src/gemini/function_declarations.py`](file:///d:/AI_PROJECTS/retail__data/src/gemini/function_declarations.py):

| Tool Name | Parameters | Purpose |
|---|---|---|
| `get_demand_aggregates` | `group_by`, `brand`, `category`, `state`, `region`, `city`, `period`, `top_n` | Aggregates actual historical demand and revenue across states, cities, regions, and brands. |
| `get_product_forecast` | `product_id`, `weeks` | Retrieves the 4-week CatBoost demand projection and projected rupee revenue for a specific SKU. |
| `get_top_demand_changes` | `direction` (`highest`, `increase`, `decrease`, `highest_revenue`), `top_n` | Identifies top growth, decline, or volume products based on model forecasts. |
| `search_products` | `query`, `category`, `top_n` | Searches the catalog by product name, brand, or model to map keywords to canonical product IDs. |
| `get_revenue_and_pricing_overview` | `group_by`, `brand`, `category`, `state`, `region`, `period`, `top_n` | Comprehensive monetary revenue, discount percentage, and pricing analytics across tiers. |
| `compare_forecast_models` | *(none)* | Compares overall MAE and WAPE across CatBoost, Naive, Moving Average, ARIMA, and Prophet. |
| `get_product_accuracy` | `product_id` | Looks up out-of-sample backtest accuracy and error rates for a specific SKU. |
| `get_unreliable_products` | `top_n` | Lists products with data warnings, stockouts, demand spikes, or extreme intermittency. |
| `get_forecast_summary` | `run_id` | High-level summary of the latest model evaluation run. |
| `get_data_quality_report` | *(none)* | Returns database row counts, null audit, and series coverage. |

### 8.3. Grounding Validation Layer
Implemented in [`src/gemini/grounding.py`](file:///d:/AI_PROJECTS/retail__data/src/gemini/grounding.py):
- Every number appearing in the assistant's written answer is extracted using regular expressions.
- The numbers are cross-referenced against the values present in the tool execution result JSON.
- If ungrounded numbers appear, the system logs a validation warning or triggers an automatic re-generation with a strict grounding reminder.

---

## 9. MySQL Database Connection & Data Architecture

The application connects to MySQL via SQLAlchemy and PyMySQL (`mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major`).

### Key Database Tables

| Table Name | Schema & Columns | Operational Purpose |
|---|---|---|
| `retail` | `record_id, product_id, product_name, category, brand, store_id, city, state, region, sales_channel, week_start_date, units_sold, unit_price, discount_pct, revenue, opening_stock, closing_stock...` | The primary historical transactions table containing all 1,024,772 weekly retail records. |
| `forecast_results` | `product_id, forecast_date, actual_units, forecast_units, mae, rmse, wape, model_name` | Stores product-level forecasts and backtest metrics generated by the batch ARIMA pipeline. |
| `evaluation_runs` | `run_id, model_name, eval_cutoff_date, test_start_date, test_end_date, config_json, created_at` | Tracks benchmark evaluation metadata and execution history. |
| `evaluation_metrics` | `run_id, model_type, scope_type, scope_value, horizon_week, mae, wape, n_observations` | Stores granular MAE and WAPE metrics across models, categories, and horizon weeks. |
| `reliability_flags` | `run_id, product_id, store_id, sales_channel, flag_type, flag_reason, severity` | Stores audit flags for volatile, stockout-distorted, or intermittent series. |
| `gemini_interactions` | `user_question, tools_called, grounding_status, response_text, created_at` | Audit log of every question asked, tools invoked, and assistant responses. |
| `gemini_feedback` | `message_id, user_query, response_text, feedback_type, feedback_score, comment, created_at` | Stores human feedback (rewards/penalties) to reinforce AI assistant accuracy over time. |

---

## 10. Model Status & Production Summary Matrix

| Question | Clear Answer | Details / Implementation |
|---|---|---|
| **Are all models connected and trained properly?** | **YES** | - **CatBoost**: Fully trained with weights saved in `artifacts/catboost_demand.cbm` (~44 MB); actively generates production forecasts.<br>- **ARIMA**: Implemented both as a standalone batch pipeline (`train_2arima.py`) writing to MySQL `forecast_results` and inside the 5-model evaluation engine.<br>- **Prophet**: Connected inside `src/evaluation/prophet_forecast.py` and benchmarked on cohorts.<br>- **Naive & Moving Average**: Connected and evaluated across all series in `src/evaluation/baselines.py`. |
| **Which model's prediction is being used for prediction?** | **CatBoost Regressor** | CatBoost recursive multi-step autoregression produces the active 4-week forward predictions stored in `artifacts/evaluation/forecast_run_*.json` and displayed in the Retailer Studio. |
| **Which model is attached to the AI Assistant?** | **Google Gemini Flash + CatBoost Engine** | Google Gemini provides natural language reasoning and tool invocation; CatBoost provides the underlying demand predictions via tool execution. |
| **How are error rates given?** | **MAE, RMSE, WAPE** | Calculated via chronologically held-out rolling-origin backtests. MAE provides unit error, RMSE catches outliers, and WAPE provides volume-weighted percentage error without division-by-zero errors. |
| **How do we confirm if models are trained?** | **Artifacts & Database** | Check for `artifacts/catboost_demand.cbm`, inspect `artifacts/model_metadata.json`, query `forecast_results` in MySQL, or review `artifacts/evaluation/eval_*.json`. |

---
*Document generated for RetailForecaster — Enterprise Demand Analytics Platform.*
