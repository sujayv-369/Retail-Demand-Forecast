import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Connect to MySQL
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

# 2. Fetch Data (Focusing on a few products to avoid days of computing time)
# Replace 'retail_sales' with your actual table name
# Updated query to bypass MySQL LIMIT restrictions in subqueries
query = """
    SELECT 
        rs.product_id, 
        rs.week_start_date, 
        rs.units_sold, 
        rs.unit_price, 
        rs.discount_pct, 
        rs.promotion_flag, 
        rs.holiday_flag ,
        rs.festival_name
    FROM retail rs
    INNER JOIN (
        SELECT product_id 
        FROM retail 
        GROUP BY product_id 
        ORDER BY SUM(units_sold) DESC 
        LIMIT 3
    ) top_products ON rs.product_id = top_products.product_id
"""
df = pd.read_sql(query, con=engine)
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

# 3. Define Exogenous Features (External signals influencing sales)
exog_cols = ['unit_price', 'discount_pct', 'promotion_flag', 'holiday_flag']
forecast_horizon = 4

results = []

# 4. Loop through each product and train an individual Auto-SARIMAX model
for product in df['product_id'].unique():
    print(f"\n--- Training Model for Product: {product} ---")
    
    # Filter for the product AND aggregate across all stores/channels
    prod_df = df[df['product_id'] == product].groupby('week_start_date').agg({
        'units_sold': 'sum',           # Total sales across all stores
        'unit_price': 'mean',          # Average price across stores
        'discount_pct': 'mean',        # Average discount
        'promotion_flag': 'max',       # If promoted in ANY store, count as 1
        'holiday_flag': 'max'          # If holiday, count as 1
    }).reset_index()
    
    # Now set the index and enforce the weekly sequence
    prod_df.set_index('week_start_date', inplace=True)
    prod_df = prod_df.asfreq('W-MON').fillna(0)
    
    # Train/Test Split
    train = prod_df.iloc[:-forecast_horizon]
    test = prod_df.iloc[-forecast_horizon:]
    
    # Separate Target (y) and Features (X)
    y_train = train['units_sold']
    X_train = train[exog_cols]
    
    y_test = test['units_sold']
    X_test = test[exog_cols]
    
    # 5. Auto-ARIMA (Automatically finds the best math for this specific product)
    print("Finding optimal parameters (this may take a minute)...")
    model = pm.auto_arima(
        y=y_train,
        X=X_train,
        seasonal=True,
        m=52,          # 52 weeks in a year (captures annual seasonality)
        stepwise=True, # Uses a smart search to speed up the process
        suppress_warnings=True,
        error_action="ignore"
    )
    
    print(model.summary())
    
    # 6. Predict the holdout period using the test set's exogenous features
    predictions = model.predict(n_periods=forecast_horizon, X=X_test)
    
    # 7. Evaluate
    wmape = np.sum(np.abs(y_test.values - predictions.values)) / np.sum(y_test.values) * 100
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Product {product} | WMAPE: {wmape:.2f}% | MAE: {mae:.2f}")
    
    # Store results for review
    eval_df = pd.DataFrame({
        'Product': product,
        'Date': test.index,
        'Actual': y_test.values,
        'Predicted': predictions.values
    })
    results.append(eval_df)

print("\n--- Final Consolidated Forecast vs Actuals ---")
final_results = pd.concat(results)
print(final_results)