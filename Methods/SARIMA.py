import pmdarima as pm
import pandas as pd

# Load the data
file_path = "../Portugal/Portugal_Model_training.csv"
data = pd.read_csv(file_path)

# Convert 'Date' to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)

# Extract the time series data (Daily_new_confirmed_cases)
series = data['Daily_new_confirmed_cases']

# Use auto_arima to find the optimal parameters for SARIMA
model = pm.auto_arima(series,
                      start_p=1, start_q=1,  # Initial p and q values
                      max_p=3, max_q=3,      # Maximum p and q values
                      d=None,                # Automatically determine differencing order d
                      seasonal=True,         # Enable seasonality
                      start_P=0, start_Q=0,  # Initial P and Q values
                      max_P=2, max_Q=2,      # Maximum P and Q values
                      D=1,                   # Seasonal differencing order
                      m=7,                   # Assuming a weekly seasonality
                      trace=True,            # Print detailed information for each iteration
                      error_action='ignore', # Ignore errors
                      suppress_warnings=True, # Suppress warnings
                      stepwise=True)          # Use stepwise search to reduce computation time

# Print the summary of the best model
print(model.summary())

# Fit the model to the series
model.fit(series)

# Predict the next 30 days (or any period you desire)
forecast = model.predict(n_periods=30)
forecast[forecast < 0] = 0


# Print the forecasted results
print(forecast)

