# CatBoost Demand Forecasting Evaluation & Model Assessment Report

**Run ID**: `eval_20261024_124409_bb4880`  
**Evaluation Date**: 2026-10-24  
**Test Window**: August 24, 2026 to October 12, 2026 (8 Weeks Held-Out)  
**Dataset**: 1,024,772 rows across 104 weeks (719 Products, 14 Stores, 3 Sales Channels, 30 Categories)

---

## 1. Executive Summary

This report provides a rigorous chronological evaluation of the CatBoost demand forecasting model versus standard retail benchmarks (**Naive Lag-1** and **4-Week Moving Average**). 

The evaluation was performed strictly on held-out out-of-sample data with rolling evaluation origins and recursive lag updates to simulate real production forecasting.

### Key Takeaways
1. **Baseline Superiority for Regular Demand**: The **4-Week Moving Average (MA4)** achieved the lowest overall error (MAE = 3.16, WAPE = 54.3%), outperforming both CatBoost (MAE = 3.77, WAPE = 60.3%) and Naive (MAE = 3.55, WAPE = 61.0%).
2. **Error Accumulation across Horizons**: CatBoost accuracy degraded systematically from Week 1 (WAPE = 56.4%) to Week 4 (WAPE = 62.6%) due to error propagation in recursive autoregressive predictions.
3. **Severe Intermittent & Volatile Series**: Over 22.7% of all weekly records contain zero sales, and 8,626 series instances showed extreme demand spikes (>3 standard deviations above mean), which heavily skewed tree-based loss functions.

---

## 2. Model Performance Benchmarks

| Model | MAE (Units) | WAPE (%) | Observations | Delta vs CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **4-Week Moving Average (MA4)** | **3.16** | **54.29%** | 201,320 | **9.9% lower WAPE** |
| **CatBoost (Recursive ML)** | **3.77** | **60.29%** | 251,650 | *Reference Model* |
| **Naive Baseline (Lag-1)** | **3.55** | **61.03%** | 201,320 | 1.2% higher WAPE |

*Observation count differences: CatBoost evaluated all rolling origins up to the final week, while lag baselines require historical lookback buffers.*

---

## 3. Forecast Accuracy by Horizon (CatBoost)

Recursive multi-step forecasting exhibits compounding error as predicted values are fed back into subsequent lag features:

| Forecast Step | Lead Time | MAE (Units) | WAPE (%) | Sample Size |
| :---: | :---: | :---: | :---: | :---: |
| **Step 1** | Week 1 | **3.28** | **56.35%** | 70,462 |
| **Step 2** | Week 2 | **3.78** | **61.27%** | 70,462 |
| **Step 3** | Week 3 | **3.94** | **61.31%** | 60,396 |
| **Step 4** | Week 4 | **4.23** | **62.62%** | 50,330 |

---

## 4. Reliability Rule Diagnostics

Out of 10,066 active series analyzed:
- **719 series** were flagged with data reliability warnings:
  - **Demand Spike Alert**: 8,626 instances where weekly demand exceeded 3 standard deviations above historical mean.
  - **Sparse Demand Alert**: 1,436 series where more than 50% of historical weeks recorded zero demand.
  - **High Error Alert**: 282 series where out-of-sample WAPE exceeded 80%.
  - **Stockout Distortion Alert**: 3 series where documented out-of-stock events compressed observed sales below true unconstrained customer demand.

---

## 5. Failure Case Analysis

### Case 1: Unmodeled Promotional Demand Spike (Product P0618)
- **Actual Units Sold**: 214 units
- **Predicted Units**: 28.4 units (Absolute Error: 185.6 units)
- **Root Cause**: Product P0618 experienced an unannounced promotional surge. Historical baseline mean was 18.2 units ($\sigma = 14.5$). The model, relying heavily on historical lag averages, anchored near standard velocity and severely under-forecasted the spike.

### Case 2: Intermittent Zero-Demand (Product P0084)
- **Actual Units Sold**: 0 units
- **Predicted Units**: 4.8 units
- **Root Cause**: Sparse demand product with zero sales in 65% of weeks. Regression models trained with continuous loss functions (MAE/RMSE) struggle to output exact zeros, consistently over-predicting small non-zero values on slow-moving inventory.

---

## 6. Recommendations for Supply Chain & Inventory Planners

1. **Hybrid Ensembling**: Use the 4-Week Moving Average baseline for stable, non-promoted staple goods (where MA4 achieves 54.3% WAPE), and deploy CatBoost selectively for items with high seasonal or promotional elasticity.
2. **Horizon-Dependent Safety Stock**: Scale inventory buffers dynamically: lead times requiring 4-week forecasts require ~11% higher safety stock than 1-week replenishment cycles to offset horizon degradation.
3. **Two-Stage Hurdle Model for Intermittent SKUs**: Implement a classification stage (predicting whether demand will be zero vs. non-zero) prior to the quantity regression step to eliminate baseline drift on sparse items.
