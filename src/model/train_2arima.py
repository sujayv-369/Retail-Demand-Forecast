import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load Environment Variables & Connect to MySQL
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env file")

print("Connecting to database...")
engine = create_engine(DATABASE_URL)

# 2. Fetch Data and Count Records
# Replace 'retail_sales' with your actual table name
query = """
    SELECT week_start_date, units_sold 
    FROM retail 
"""
print("Fetching dataset...")
df = pd.read_sql(query, con=engine)

total_records = len(df)
print(f"Total records fetched: {total_records:,}")

# 3. Preprocess for ARIMA (Convert to Single Time Series)
# Ensure dates are datetime objects
df['week_start_date'] = pd.to_datetime(df['week_start_date'])

# Aggregate sales globally by week (since ARIMA requires a single series)
# If you want a specific product, replace this with: df = df[df['product_id'] == 'XYZ']
ts_data = df.groupby('week_start_date')['units_sold'].sum().reset_index()
ts_data = ts_data.sort_values('week_start_date')
ts_data.set_index('week_start_date', inplace=True)
# Ensure weekly frequency (Monday start)
ts_data = ts_data.asfreq('W-MON') 
ts_data['units_sold'] = ts_data['units_sold'].fillna(0) # Handle any missing weeks

print(f"Aggregated time series length: {len(ts_data)} weeks")

# 4. Train/Test Split
# We are forecasting 4 weeks out, so we hold out the last 4 weeks for evaluation
forecast_horizon = 4
train = ts_data.iloc[:-forecast_horizon]
test = ts_data.iloc[-forecast_horizon:]

# 5. Train the ARIMA Model
# Order (p,d,q) = (Autoregressive lags, Differencing, Moving Average lags)
# (4, 1, 1) is a common starting point for weekly data, but requires tuning.
print("Training ARIMA model...")
model = ARIMA(train['units_sold'], order=(4, 1, 1))
fitted_model = model.fit()

print(fitted_model.summary())

# 6. Evaluation (Metrics)
print("\n--- Evaluation on Holdout Test Set ---")
# Predict the exact timeframe of our test set
predictions = fitted_model.forecast(steps=forecast_horizon)

mae = mean_absolute_error(test['units_sold'], predictions)
rmse = np.sqrt(mean_squared_error(test['units_sold'], predictions))

# Calculate WMAPE (Weighted Mean Absolute Percentage Error) - Standard retail metric
wmape = np.sum(np.abs(test['units_sold'].values - predictions.values)) / np.sum(test['units_sold'].values) * 100

print(f"Mean Absolute Error (MAE): {mae:,.2f} units")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} units")
print(f"WMAPE: {wmape:.2f}%")

# Display Test vs Predicted
eval_df = pd.DataFrame({
    'Actual': test['units_sold'],
    'Predicted': predictions
})
print("\nForecast vs Actual (Holdout Period):")
print(eval_df)

# 7. Future Prediction (Next 4 weeks beyond the dataset)
print("\n--- Forecasting Unknown Future ---")
# Retrain on the FULL dataset to get the most up-to-date future forecast
final_model = ARIMA(ts_data['units_sold'], order=(4, 1, 1))
final_fitted = final_model.fit()

future_forecast = final_fitted.forecast(steps=forecast_horizon)

print("Next 4 Weeks Forecast:")
print(future_forecast)