import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Load the data
file_path = "../Germany/Germany_Model_training.csv"
data = pd.read_csv(file_path)

# Convert 'Date' to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)

# Extract the time series data (Daily_new_confirmed_cases)
series = data['Daily_new_confirmed_cases']

# Calculate the ACF values
acf_values = acf(series, nlags=50)

# Identify the first significant peak (ignoring lag 0)
lag_threshold = 0.2  # Set a threshold for significance (this can be adjusted)
significant_lags = [i for i in range(1, len(acf_values)) if acf_values[i] > lag_threshold]

# Automatically determine the periodicity based on the first significant peak
if significant_lags:
    detected_period = significant_lags[0]
else:
    detected_period = 1  # Default to 1 if no significant peak is found

# Print the detected period
print(f"Detected period: {detected_period}")

# Plot the ACF values
plt.figure(figsize=(10, 5))
plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
plt.title('Auto-Correlation Function (ACF) Plot')
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.grid(True)
plt.show()
