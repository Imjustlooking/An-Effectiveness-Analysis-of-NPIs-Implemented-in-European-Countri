import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the CSV file
file_path = '../Portugal/Portugal_plot.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

# Plotting both the actual and prediction values for comparison
plt.figure(figsize=(12, 6))

# Plot actual values across the entire time period in blue
plt.plot(data['Date'], data['Actual'], label='Actual', color='blue')

# Plot prediction values where available, in red with a dashed line
plt.plot(data['Date'], data['Prediction'], label='Prediction', color='red', linestyle='--')

# Plot the implementaion date and the terminal data point
plt.axvline(datetime(2021, 1, 15), color='black', linestyle='--', label='NPI Implementation Date (2021-1-15)')
plt.axvline(datetime(2021, 1, 18), color='g', linestyle='--', label='Terminal Data Point (2021-1-18)')
plt.legend(loc='upper left')

# Adding labels for axes, a title for the plot, and a legend
plt.xlabel('Date')  # X-axis label
plt.ylabel('Daily_new_confirmed_cases')  # Y-axis label
plt.title('Actual vs Prediction(Portugal)')  # Plot title
plt.legend()  # Show legend to differentiate lines


# Display the plot
plt.show()

