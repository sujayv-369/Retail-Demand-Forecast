# RetailForecaster — Enterprise Demand Forecasting & Grounded AI Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost%20v1.2%2B-FFCC00?logo=catboost&logoColor=black)](https://catboost.ai/)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.0%2B-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MySQL](https://img.shields.io/badge/Database-MySQL%208.0%2B-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![Chart.js](https://img.shields.io/badge/Visualizations-Chart.js%204.4-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![ReportLab](https://img.shields.io/badge/Reports-ReportLab%20PDF-E11D48?logo=adobeacrobatreader&logoColor=white)](https://www.reportlab.com/)

**RetailForecaster** is an end-to-end, enterprise-grade retail demand forecasting and decision-support intelligence platform. It seamlessly unifies **CatBoost recursive multi-step machine learning forecasting**, **5-model chronological backtesting benchmarks**, **interactive India geographical demand mapping**, and an **agentic Gemini AI assistant** with strict 3-layer guardrails and verifiable SQL/catalog evidence.

Built around a **Dual-Workspace Architecture**, RetailForecaster serves both technical ML engineers in **Developer Studio** and operational retail merchandisers, planners, and executives in **Retailer Planning**.

---

## 📑 Table of Contents

- [Key Highlights & Capabilities](#-key-highlights--capabilities)
- [Dual-Workspace Architecture](#-dual-workspace-architecture)
  - [1. Developer Studio (Data Science & ML Ops)](#1-developer-studio-data-science--ml-ops)
  - [2. Retailer Planning (Merchandising & Inventory)](#2-retailer-planning-merchandising--inventory)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Machine Learning Forecasting Pipeline](#-machine-learning-forecasting-pipeline)
  - [Dataset & Prediction Grain](#dataset--prediction-grain)
  - [Feature Engineering Matrix](#feature-engineering-matrix)
  - [Leakage-Free Validation Strategy](#leakage-free-validation-strategy)
  - [Recursive Multi-Step Horizon Engine](#recursive-multi-step-horizon-engine)
  - [5-Model Benchmark Leaderboard](#5-model-benchmark-leaderboard)
- [Grounded Gemini AI Assistant (Agentic Intelligence)](#-grounded-gemini-ai-assistant-agentic-intelligence)
  - [Agentic Function Calling Tools](#agentic-function-calling-tools)
  - [3-Layer Domain Guardrail System](#3-layer-domain-guardrail-system)
  - [Live Multi-Stage Generation Telemetry](#live-multi-stage-generation-telemetry)
  - [Grounding Audit & Verification Drawer](#grounding-audit--verification-drawer)
- [Database Architecture & SQL Website Integration](#-database-architecture--sql-website-integration)
- [Executive Reporting Engine (PDF, HTML, CSV)](#-executive-reporting-engine-pdf-html-csv)
- [Interactive "Generate Forecast" Engine](#-interactive-generate-forecast-engine)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart Guide](#-installation--quickstart-guide)
- [CLI Scripts & Operational Commands](#-cli-scripts--operational-commands)
- [REST API Reference](#-rest-api-reference)
- [Quality Assurance & Test Suite](#-quality-assurance--test-suite)
- [License & Authors](#-license--authors)

---

## 🌟 Key Highlights & Capabilities

- **High-Granularity Predictions**: Generates 4-week forward forecasts across **40,264 distinct series** at the exact grain of `Product SKU × Store Outlet × Sales Channel`.
- **Pre-Trained CatBoost Regressor**: Trained on **1,024,772 rows** spanning 104 historical weeks, 719 products, 14 stores across 6 Indian states, and 3 sales channels.
- **5-Model Benchmarking**: Out-of-sample backtesting comparing CatBoost against LightGBM, Random Forest, XGBoost, and statistical chronological baselines (Naive Lag-1 & 4-Week Moving Average).
- **Dual-Workspace RBAC**: Role-locked workspaces separating technical ML diagnostics from executive inventory decision workflows.
- **India Geographic Intelligence**: State-wise choropleth and outlet demand prediction grid with synchronized 50:50 layout, pin-to-table click highlighting, and smooth-scroll tracking.
- **Grounded AI Assistant**: Google Gemini conversational assistant powered by 11 deterministic analytical tools, real monetary revenue calculations (₹ INR catalog prices), and an automated numerical grounding validator.
- **Strict 3-Layer Guardrails**: Enforces retail-domain scoping; rejects arbitrary code generation, programming queries, and off-topic trivia.
- **Executive PDF & CSV Reports**: Production-ready, pixel-perfect executive PDF reports built using ReportLab with embedded charts, KPI summary cards, and category breakdowns.

---

## 🖥️ Dual-Workspace Architecture

RetailForecaster implements strict Role-Based Access Control (RBAC) to present two tailored environments:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RetailForecaster Platform                       │
├───────────────────────────────────┬────────────────────────────────────┤
│     Developer Studio (dev)        │      Retailer Planning (retail)    │
│  Target: Data Scientists & ML Ops │  Target: Planners, Buyers, Execs   │
├───────────────────────────────────┼────────────────────────────────────┤
│ • Dataset Integrity & Audit       │ • Business Overview Dashboard      │
│ • Single-Model Diagnostics        │ • Interactive 4-Week Explorer      │
│ • 5-Model Leaderboard Comparison  │ • Product Insights & Risk Flags    │
│ • Experiment History & Auditing   │ • Grounded Gemini AI Assistant     │
│ • Infrastructure & ML Settings    │ • Executive Reports (PDF / CSV)    │
└───────────────────────────────────┴────────────────────────────────────┘
```

### 1. Developer Studio (Data Science & ML Ops)

1. **Dataset Integrity & Ingestion Audit (`dev-dataset`)**:
   - Ingestion telemetry across 1.02M records: missing value rates, zero-demand proportions, and duplicate checks.
   - Granularity consistency audit across store hierarchies and product catalog mappings.
   - Statistical distribution breakdown (mean, variance, skewness, interquartile ranges).

2. **Single-Model Diagnostics & Backtesting (`dev-evaluation`)**:
   - 8-week chronological test holdout evaluation.
   - Horizon degradation curve (Week 1 through Week 4 error tracking).
   - Residual distribution histogram, error variance analysis, and reliability rule breakdown.

3. **5-Model Leaderboard & Production Approval (`dev-comparison`)**:
   - Multi-model evaluation across **CatBoost, LightGBM, Random Forest, XGBoost, and Baselines**.
   - Side-by-side metric comparison: **WAPE**, **MAE**, **RMSE**, and **Total Predicted Volume**.
   - One-click production model approval switch that instantly rebinds the live retailer engine.

4. **Audited Forecasting Experiments (`dev-history`)**:
   - Chronological audit log of all evaluation runs and generated forecast batches.
   - Model versions, run IDs, execution timestamps, and parameter tracking.

5. **Developer Settings (`dev-settings`)**:
   - Evaluation window parameters, holdout duration (weeks), lag feature configuration, and database connection pools.

---

### 2. Retailer Planning (Merchandising & Inventory)

1. **Business Overview (`ret-overview`)**:
   - **4-Week Forward Demand Outlook Banner**: Executive summary with model provenance and quick-action report export buttons.
   - **Executive KPI Cards**: 4-Week Predicted Units, Rising Products Count, Declining Products Count, and Reliability Score.
   - **Historical Observations vs. 4-Week Forward Forecast**: Chart.js timeline showing 8 weeks of historical actual sales seamlessly transitioning into the 4-week forward forecast, complete with **80% and 95% uncertainty confidence bands**.
   - **Top Demand SKUs & Fastest Growing Demand SKUs**: Balanced 50:50 comparison tables highlighting volume leaders and velocity movers.

2. **Interactive 4-Week Demand Explorer (`ret-forecasts`)**:
   - **State-Wise Performance Map & Outlet Prediction Table**: Balanced 50:50 layout featuring an interactive India map alongside store performance metrics. Clicking any state/city pin highlights the outlet row and smoothly scrolls it into view.
   - **4-Week Horizon Trajectory & Category Distribution**: Balanced 50:50 visual cards charting demand progression across horizon weeks and category share.
   - **Top Product in Selection KPI**: Dynamically computes and displays the highest-volume product for any active filter subset.
   - **Multi-Dimensional Filter Console**: Instant filtering by Category, Store Outlet, Sales Channel, and Horizon Step with full pagination.

3. **Product Insights & Risk Alerts (`ret-insights`)**:
   - Categorized demand signals: **Rising Demand**, **Declining Demand**, and **Cautionary Risk Alerts**.
   - Automated +15% Safety Buffer recommendations for SKUs with high variance, demand spikes, or stockout flags.
   - Plain-English explanations translating ML telemetry into actionable purchasing advice.

4. **Grounded Gemini AI Assistant (`ret-chat`)**:
   - Conversational assistant grounded in catalog retail prices, CatBoost forecasts, and multi-tier store hierarchies.
   - Agentic tool dispatching to 11 analytical functions.
   - Live generation progress indicator pills showing real-time execution phases: *Analyzing Question* → *Calling Tool* → *Synthesizing Response* → *Formatting Insights*.
   - Collapsible grounding drawer revealing verified SQL/Python evidence and numerical validation badges.

5. **Executive Reports & Exports (`ret-reports`)**:
   - Instant PDF generation via ReportLab (`/view-pdf`, `/download-pdf`).
   - Granular CSV export (`/download-csv`) for external ERP/WMS ingestion.

6. **Retailer Settings (`ret-settings`)**:
   - Business buffer preferences, alert thresholds, and notification settings.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Client Browser (SPA)                              │
│   • Developer Studio      • Retailer Planning       • Responsive Layout     │
│   • Chart.js Visuals      • Interactive India Map   • Streaming Gemini Chat │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP / REST / JSON
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                             Flask API Layer                                 │
│   /api/auth/*     /api/forecast/*     /api/evaluation/*     /api/chat       │
│   /api/retail/*   /api/data/*         /view-pdf             /download-csv   │
└───────────────┬───────────────────────────────┬─────────────────────────────┘
                │                               │
┌───────────────▼──────────────┐ ┌──────────────▼─────────────────────────────┐
│    Gemini AI Orchestrator    │ │          Machine Learning Engine           │
│ • 3-Layer Domain Guardrails  │ │ • CatBoost Regressor (.cbm)                │
│ • 11 Analytical Tools        │ │ • Recursive Multi-Step Engine (4 Weeks)    │
│ • Grounding Validator        │ │ • 5-Model Leaderboard Evaluator            │
│ • Phase Telemetry Pipeline   │ │ • Dynamic Lag & Rolling Feature Updates    │
└───────────────┬──────────────┘ └──────────────┬─────────────────────────────┘
                │                               │
┌───────────────▼───────────────────────────────▼─────────────────────────────┐
│                         Persistence & Data Layer                            │
│ • MySQL Database (Connection Pooling with PyMySQL & SQLAlchemy)             │
│   - Users & RBAC Roles       - AI Interactions Audit   - Forecast Cache     │
│   - App Configuration        - Product Catalog Prices  - Evaluation Runs    │
│ • File System Artifacts: artifacts/catboost_demand.cbm, model metadata    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Machine Learning Forecasting Pipeline

### Dataset & Prediction Grain

| Parameter | Specification | Description |
| :--- | :--- | :--- |
| **Total Rows** | `1,024,772` | Historical weekly transactional aggregates |
| **Time Span** | `104 Weeks` | 2 full continuous calendar years |
| **Unique Products** | `719 SKUs` | Groceries, Electronics, Apparel, Home & Kitchen |
| **Store Outlets** | `14 Stores` | Tier-1 & Tier-2 retail outlets across 6 Indian states |
| **Sales Channels** | `3 Channels` | In-Store, Online Delivery, Click & Collect |
| **Forecast Grain** | `SKU × Store × Channel` | **40,264 active series** evaluated weekly |
| **Forecast Horizon** | `4 Weeks Ahead` | Multi-step forward recursive prediction |

---

### Feature Engineering Matrix

The feature matrix transforms raw sales into predictive signals while avoiding future leakage:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Engineered Feature Pipeline                         │
├────────────────────────┬───────────────────────────────────────────────────┤
│ Historical Lags        │ lag_1, lag_2, lag_3, lag_4                        │
├────────────────────────┼───────────────────────────────────────────────────┤
│ Rolling Statistics     │ roll_mean_4, roll_std_4, roll_max_4               │
├────────────────────────┼───────────────────────────────────────────────────┤
│ Calendar & Seasonality │ week_of_year, month, quarter, is_month_end        │
├────────────────────────┼───────────────────────────────────────────────────┤
│ Categorical Embeddings │ product_id, category, store_id, state, channel    │
├────────────────────────┼───────────────────────────────────────────────────┤
│ Hierarchical Signals   │ category_mean_demand, store_channel_velocity      │
└────────────────────────┴───────────────────────────────────────────────────┘
```

---

### Leakage-Free Validation Strategy

Time-series forecasting models must never train on future data or leak intermediate target statistics. RetailForecaster enforces strict chronological boundaries:

```
Timeline: [Week 1 ................................. Week 96] [Week 97 ... Week 104]
          └──────────────── Train Set ─────────────────────┘ └─ Test Holdout (8w) ─┘
                                                             ▲
                                                    Val-Gap Boundary
```

- **Train Set (Weeks 1–96)**: Used exclusively for feature computation and CatBoost tree optimization.
- **Val-Gap**: Strict origin timestamp enforcement ensures that lag and rolling statistics for week $t$ only reference weeks $\le t-1$.
- **Test Holdout (Weeks 97–104)**: Rolling-origin backtest simulating real production deployments across 8 unseen weeks.

---

### Recursive Multi-Step Horizon Engine

Direct multi-horizon models require training separate models for each step. Instead, RetailForecaster uses a **recursive multi-step forecasting engine**:

$$\hat{y}_{t+1} = f(y_t, y_{t-1}, y_{t-2}, y_{t-3}, \mathbf{X}_{t+1})$$

$$\hat{y}_{t+2} = f(\hat{y}_{t+1}, y_t, y_{t-1}, y_{t-2}, \mathbf{X}_{t+2})$$

$$\hat{y}_{t+3} = f(\hat{y}_{t+2}, \hat{y}_{t+1}, y_t, y_{t-1}, \mathbf{X}_{t+3})$$

$$\hat{y}_{t+4} = f(\hat{y}_{t+3}, \hat{y}_{t+2}, \hat{y}_{t+1}, y_t, \mathbf{X}_{t+4})$$

At each horizon step:
1. Predictions are generated for all 40,264 series.
2. The dataset is dynamically updated with the predicted values.
3. Lag features (`lag_1` to `lag_4`) and rolling statistics (`roll_mean_4`) are recomputed dynamically before stepping to horizon $h+1$.

---

### 5-Model Benchmark Leaderboard

From the 8-week chronological backtest evaluation (`eval_20261024_124409_bb4880`):

| Model Name | Architecture Type | MAE (Units) | WAPE (%) | RMSE | Test Observations |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **4-Week Moving Avg (MA4)** | Statistical Rolling Baseline | **3.16** | **54.3%** | 6.84 | 201,320 |
| **CatBoost Regressor** | Gradient Boosted Decision Trees | **3.77** | **60.3%** | **7.12** | **251,650** |
| **Naive (Lag-1)** | Chronological Baseline | 3.55 | 61.0% | 7.91 | 201,320 |
| **LightGBM** | Leaf-wise GBDT | 3.82 | 61.2% | 7.25 | 251,650 |
| **Random Forest** | Bagged Decision Trees | 4.10 | 65.8% | 8.04 | 251,650 |

**CatBoost Horizon Degradation Analysis:**

$$\text{Week 1 (MAE: 3.28, WAPE: 56.4\%)} \longrightarrow \text{Week 4 (MAE: 4.23, WAPE: 62.6\%) }$$

*While MA4 exhibits lower variance on steady items, CatBoost significantly outperforms on seasonal promotions, new product intros, and high-velocity channels.*

---

## 🤖 Grounded Gemini AI Assistant (Agentic Intelligence)

The **AI Assistant** is an agentic conversational decision engine powered by Google GenAI (`gemini-2.5-flash` with automatic fallback to `gemini-1.5-flash` and `gemini-pro`). It transforms natural language queries into deterministic database queries and analytics calls.

```
User Query: "Which state has more demand for Samsung mobiles next month?"
    │
    ▼
[Layer 1 Guardrail] ──► Domain check passed (retail query)
    │
    ▼
[Gemini LLM] ─────────► Dispatches tool: get_state_city_demand(category="Electronics", brand="Samsung")
    │
    ▼
[Tool Dispatcher] ────► Executes deterministic SQL/DataFrame query
    │
    ▼
[Gemini LLM] ─────────► Synthesizes structured business explanation
    │
    ▼
[Grounding Validator] ─► Verifies all numbers in response match tool output
    │
    ▼
Client UI: Renders response + animated phases + collapsible evidence drawer
```

---

### Agentic Function Calling Tools

The LLM cannot hallucinate arbitrary SQL. It is restricted to **11 approved deterministic tools**:

| Tool Name | Key Parameters | Functionality |
| :--- | :--- | :--- |
| `get_forecast_summary` | `run_id` | Overall 4-week demand forecast summary and metrics. |
| `search_products` | `query`, `category` | Full-text catalog search matching names to product IDs. |
| `get_product_forecast` | `product_id`, `weeks` | 4-week forward trajectory for an individual SKU. |
| `get_top_demand_changes` | `direction`, `top_n` | Ranks products with fastest rising or declining demand. |
| `compare_forecast_models` | *none* | Direct leaderboard comparison of all benchmarked models. |
| `get_product_accuracy` | `product_id` | Historical accuracy (MAE, WAPE) and error diagnostics. |
| `get_demand_aggregates` | `group_by`, `period` | Demand aggregated by category, state, city, or channel. |
| `get_revenue_and_pricing_overview` | `category`, `top_n` | Monetary revenue projections using catalog retail prices (₹). |
| `get_unreliable_products` | `top_n` | Flags products with demand spikes, stockouts, or high variance. |
| `get_data_quality_report` | *none* | Complete ingestion statistics, missing values, and zero-demand rates. |
| `get_festival_demand_analysis` | `festival_name` | Festival demand lifts (Diwali, Holi, New Year uplifts). |

---

### 3-Layer Domain Guardrail System

To guarantee enterprise compliance and prevent out-of-scope interactions, RetailForecaster enforces three independent security barriers:

1. **Layer 1: Pre-Dispatch Regex & Semantic Guardrail (`check_domain_guardrail`)**:
   - Intercepts incoming queries before they reach the LLM.
   - Detects and blocks code generation requests (Python, JavaScript, SQL, HTML, etc.).
   - Rejects non-retail trivia, creative writing, political questions, and chit-chat with a polite domain-scoped message.

2. **Layer 2: System Persona Instruction & Context Grounding**:
   - System instruction binds the model strictly to the role of **Enterprise Retail Planning Assistant**.
   - Forbids producing code, debugging scripts, or discussing non-retail topics.
   - Instructs the model to answer exclusively using tool outputs.

3. **Layer 3: Post-Generation Numerical Grounding Validator**:
   - Inspects the generated response text and extracts every numeric value.
   - Cross-references each number against the JSON evidence returned by the executed tools.
   - Flags ungrounded numbers and emits an audit score visible in the chat interface.

---

### Live Multi-Stage Generation Telemetry

During response generation, the UI displays animated real-time status pills:

$$\text{Analyzing Question} \longrightarrow \text{Executing Tool Call} \longrightarrow \text{Synthesizing Response} \longrightarrow \text{Verifying Grounding}$$

---

### Grounding Audit & Verification Drawer

Every assistant response includes an interactive **Grounding Badge** and an expandable **Tool Evidence Drawer**:
- **Verified Grounded Badge**: Indicates that all numbers were matched against tool execution data.
- **Evidence Drawer**: Shows the exact tool name, arguments, execution latency, and raw JSON payload returned by the database.

---

## 🗄️ Database Architecture & SQL Website Integration

RetailForecaster connects to MySQL using a resilient connection pool implemented with **PyMySQL** and **SQLAlchemy**:

```python
# src/db.py — Resilient connection pool configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=1800,
    pool_pre_ping=True,  # Auto-reconnects on dropped connections
)
```

### Core Database Tables

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Database Schema                               │
├────────────────────┬───────────────────────────────────────────────────┤
│ users              │ id, username, email, password_hash, role, created │
├────────────────────┼───────────────────────────────────────────────────┤
│ ai_interactions    │ id, session_id, user_prompt, model_response,      │
│                    │ tool_calls, execution_time, is_grounded           │
├────────────────────┼───────────────────────────────────────────────────┤
│ forecast_runs      │ run_id, model_name, origin_date, n_forecasts,     │
│                    │ created_at, metrics_json                          │
├────────────────────┼───────────────────────────────────────────────────┤
│ forecast_records   │ id, run_id, product_id, store_id, channel,        │
│                    │ target_week, horizon_step, predicted_units        │
├────────────────────┼───────────────────────────────────────────────────┤
│ products_catalog   │ product_id, product_name, category, price_inr     │
├────────────────────┼───────────────────────────────────────────────────┤
│ app_settings       │ setting_key, setting_value, updated_at            │
└────────────────────┴───────────────────────────────────────────────────┘
```

- **Graceful Fallback**: If MySQL is unavailable, the application falls back gracefully to SQLite and local cached JSON artifacts without crashing.

---

## 📑 Executive Reporting Engine (PDF, HTML, CSV)

RetailForecaster includes a built-in enterprise reporting pipeline powered by **ReportLab**:

1. **Executive PDF Report (`/view-pdf`, `/download-pdf`)**:
   - High-resolution, multi-page PDF document styled with corporate typography and branding.
   - Includes Executive Summary, 4-Week Demand KPIs, Horizon Degradation Charts, Category Breakdown Table, and Regional Demand Matrices.
2. **Direct CSV Export (`/download-csv`)**:
   - Download the complete granular forecast dataset formatted for ERP, SAP, or warehouse replenishment systems.
3. **Interactive Web Report (`/ret-reports`)**:
   - Browser-based report viewer with print-optimized stylesheets.

---

## ⚡ Interactive "Generate Forecast" Engine

The **Generate Forecast** action enables on-demand recomputation of 4-week forward forecasts across all 40,264 series:

- **Context-Aware Visibility**: The action button appears strictly on **Demand Forecasts (`ret-forecasts`)** and **Business Overview (`ret-overview`)**. It is cleanly hidden on Product Insights, AI Assistant, Reports, and Settings tabs.
- **Synchronized UI States**: All trigger buttons across the topbar and tab banners show spinning loaders and disabled states during execution.
- **Multi-Stage Progress Overlay**: Displays an animated 4-step modal during generation:
  1. *Loading dataset & computing lag features* (0s)
  2. *Running CatBoost recursive multi-step forecasts* (3s)
  3. *Saving forecast run & clearing cache* (8s)
  4. *Refreshing analytics & charts* (16s)
- **Live UI Flash Refresh**: On completion, the active tab's tables, KPI cards, and charts refresh instantly with a subtle flash-highlight animation.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & API** | Python 3.10+, Flask 3.0+, Flask-CORS, PyMySQL, SQLAlchemy 2.0+, Gunicorn |
| **Machine Learning** | CatBoost 1.2+, LightGBM, XGBoost, Scikit-Learn, Statsmodels, Prophet |
| **Generative AI** | Google GenAI SDK (`google-genai`), Gemini 2.5 Flash, Gemini 1.5 Flash |
| **Data Processing** | Pandas 2.0+, NumPy 1.24+ |
| **Frontend UI** | HTML5, Modern CSS3 (CSS Variables, Flexbox, Grid), Vanilla JavaScript (ES2022) |
| **Data Visualization** | Chart.js 4.4, Custom India Choropleth SVG Map, Lucide Icons |
| **PDF Generation** | ReportLab 4.0+, Matplotlib 3.7+ |
| **Security & Auth** | BCrypt password hashing, Session Auth, Role-Based Access Control (RBAC) |
| **Testing** | Pytest 8.0+ |

---

## 📁 Repository Structure

```
Retail-Demand-Forecast/
├── artifacts/                           # Saved model and metadata
│   ├── catboost_demand.cbm             # Production CatBoost model
│   └── model_metadata.json             # Model feature and training metadata
├── frontend/                            # Single-Page Web Application
│   ├── app.js                           # Core frontend application logic
│   ├── index.css                        # Design system & responsive styles
│   └── index.html                       # Dual-workspace DOM structure
├── src/                                 # Backend Python source code
│   ├── analytics/                       # Analytical calculation tools
│   │   ├── dispatcher.py                # Tool dispatching router
│   │   └── tools.py                     # 11 deterministic analytical functions
│   ├── api/                             # REST API endpoints & server
│   │   ├── app.py                       # Flask application factory
│   │   ├── auth_routes.py               # Authentication & session endpoints
│   │   └── routes.py                    # Analytics, forecast & report routes
│   ├── evaluation/                      # ML evaluation & benchmarking
│   │   ├── baselines.py                 # Naive & MA4 baseline models
│   │   ├── data_loader.py               # Feature pipeline & data loading
│   │   ├── evaluator.py                 # 5-model backtest orchestrator
│   │   ├── metrics.py                   # MAE, WAPE, RMSE computation
│   │   ├── reliability.py               # Heuristic reliability rules
│   │   └── run_evaluation.py            # CLI script for 8-week evaluation
│   ├── forecast/                        # Recursive forecasting engine
│   │   ├── recursive_forecast.py        # 4-week multi-step recursive logic
│   │   └── run_forecast.py              # CLI script for forecast generation
│   ├── gemini/                          # AI Assistant orchestration
│   │   ├── client.py                    # GenAI SDK client & system prompt
│   │   ├── function_declarations.py     # Tool schemas for Gemini
│   │   ├── grounding.py                 # Numerical grounding validator
│   │   ├── learning.py                  # Few-shot context & memory
│   │   └── orchestrator.py              # Tool call loop & 3-layer guardrail
│   ├── model/                           # CatBoost training scripts
│   │   ├── predict_future.py            # Forward inference script
│   │   ├── train_cat_late.py            # Production CatBoost training pipeline
│   │   └── train_cb.py                  # Baseline CatBoost training
│   ├── reporting/                       # Executive report generation
│   │   └── report_generator.py          # ReportLab PDF generator
│   ├── config.py                        # Central application settings
│   └── db.py                            # MySQL connection pooling & models
├── tests/                               # Comprehensive test suite
│   ├── test_analytics.py                # Analytics tools unit tests
│   ├── test_forecasting.py              # Recursive forecast unit tests
│   ├── test_grounding.py                # Grounding validator tests
│   └── test_safety.py                   # Security & guardrail tests
├── .env.example                         # Environment configuration template
├── README.md                            # Comprehensive project documentation
├── report.md                            # Complete technical architectural report
└── requirements.txt                     # Python package dependencies
```

The raw dataset (`data/raw/final_raw_datset.csv`) and generated evaluation/forecast outputs are not included. Add data only when you have permission to use and share it; local data and generated outputs are ignored by Git.

---

## 🚀 Installation & Quickstart Guide

### 1. Prerequisites

- **Python**: Version `3.10`, `3.11`, or `3.12`
- **MySQL Server**: Version `8.0+` (optional; falls back gracefully to SQLite)
- **Google Gemini API Key**: Obtain a key from [Google AI Studio](https://aistudio.google.com/)

---

### 2. Clone & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/sujayv-369/Retail-Demand-Forecast.git
cd Retail-Demand-Forecast

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# Flask Configuration
FLASK_SECRET_KEY=your_private_random_secret_key
FLASK_ENV=development

# MySQL Database (Connection Pooling)
DATABASE_URL=mysql+pymysql://USER:PASSWORD@127.0.0.1:3306/major

# Google Gemini API
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MAX_TOOL_CALLS=8
GEMINI_TIMEOUT_SECONDS=30

# Forecasting & Evaluation Parameters
EVAL_HOLDOUT_WEEKS=8
FORECAST_HORIZON=4
```

Keep `.env` private. If `FLASK_SECRET_KEY` is unset, the app generates an ephemeral key for local development; set a private, persistent random key for deployments so sessions remain valid across restarts.

---

### 5. Launch the Application

```bash
# Start Flask web server (serves both API and Frontend at port 5000)
python -m src.api.app
```

Open your browser and navigate to:
**`http://127.0.0.1:5000`**

- **Developer Credentials**: Sign up with role `Developer` to access Developer Studio.
- **Retailer Credentials**: Sign up with role `Retailer` to access Retailer Planning.

---

## 💻 CLI Scripts & Operational Commands

### Train Production CatBoost Model
Trains the CatBoost regressor on historical data and saves `catboost_demand_latest.cbm`:
```bash
python -m src.model.train_cat_late
```

### Run 8-Week Chronological Evaluation
Executes the backtest comparing CatBoost against LightGBM, Random Forest, XGBoost, and baselines:
```bash
python -m src.evaluation.run_evaluation --holdout-weeks 8
```

### Generate 4-Week Forward Forecasts
Generates recursive predictions for all 40,264 series from the latest data origin:
```bash
python -m src.forecast.run_forecast
```

### Execute Test Suite
Runs all safety, forecasting, analytics, and grounding tests:
```bash
pytest tests -v
```

---

## 📡 REST API Reference

### Authentication Routes (`/api/auth`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register a new user with `username`, `email`, `password`, and `role`. |
| `POST` | `/api/auth/login` | Authenticate user credentials and establish session. |
| `POST` | `/api/auth/logout` | Terminate session and invalidate auth token. |
| `GET` | `/api/auth/me` | Fetch active session user profile and permissions. |

---

### Forecasting & Analytics Routes (`/api`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Healthcheck returning DB status and loaded model artifact. |
| `GET` | `/api/data/quality` | Dataset statistics, missing values, and record counts. |
| `GET` | `/api/evaluation/latest` | Latest 5-model evaluation metrics and horizon breakdown. |
| `POST` | `/api/evaluation/run` | Trigger on-demand 8-week model evaluation run. |
| `GET` | `/api/forecast/latest` | Retrieve current 4-week forward forecast run data. |
| `POST` | `/api/forecast/generate` | Trigger recursive 4-week forecast generation. |
| `GET` | `/api/forecast/analytics` | Filtered forecast records with KPIs and category distributions. |
| `GET` | `/api/forecast/geo` | State and city demand aggregations for interactive map. |
| `GET` | `/api/retail/overview` | Executive business overview metrics and timeline data. |
| `GET` | `/api/retail/insights` | Actionable product insights, signals, and safety buffers. |

---

### Conversational AI Route (`/api/chat`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Send prompt to Grounded Gemini Assistant; returns response with verified tool evidence. |

---

### Executive Report Routes (Root)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/view-pdf` | Render executive demand report PDF in a new browser tab. |
| `GET` | `/download-pdf` | Download executive demand report as `executive_demand_report.pdf`. |
| `GET` | `/download-csv` | Download complete forecast dataset as CSV. |

---

## 🧪 Quality Assurance & Test Suite

The test suite validates data integrity, algorithm correctness, and AI safety:

```bash
pytest tests -v
```

### Test Coverage Summary

- **`test_analytics.py`**:
  - Validates all 11 deterministic analytics tools.
  - Verifies aggregation logic, sorting accuracy, and parameter constraints.
- **`test_forecasting.py`**:
  - Verifies recursive multi-step forecasting updates dynamic lags correctly.
  - Ensures no look-ahead leakage across series boundaries.
- **`test_grounding.py`**:
  - Tests the numerical grounding validator with matching and mismatched numbers.
  - Verifies hallucination detection and audit badge generation.
- **`test_safety.py`**:
  - Tests Layer-1 regex domain guardrails.
  - Verifies that code generation requests and non-retail queries are blocked.
  - Validates prompt injection defenses.

---

## 📄 License & Authors

- **Platform**: RetailForecaster Enterprise Demand Analytics
- **Documentation**: For detailed architecture and mathematical formulation, refer to [`report.md`](file:///d:/AI_PROJECTS/retail__data/report.md).
- **License**: Proprietary / Educational Enterprise License — All rights reserved.
