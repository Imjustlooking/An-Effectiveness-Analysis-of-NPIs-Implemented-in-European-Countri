import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

# Load the CSV file
file_path = '../Portugal/Portugal_test.csv'
data = pd.read_csv(file_path)

# Convert the date column to datetime type for easier manipulation
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')

# Setting up the dataset for Interrupted Time Series Analysis (ITSA)
# Define intervention point
intervention_date = datetime(2021, 1, 15)

# Add variables for ITSA: time, intervention dummy, and interaction term
data['time'] = range(len(data))  # Time variable
data['post_intervention'] = (data['Date'] >= intervention_date).astype(int)  # Indicator for post-intervention period
data['time_post_intervention'] = data['time'] * data['post_intervention']  # Interaction term for post-intervention trend

# Prepare the dependent variable (y) and independent variables (X) for the regression
y = data['Daily_new_confirmed_cases']
X = sm.add_constant(data[['time', 'post_intervention', 'time_post_intervention']])  # Add constant for the intercept

# Run OLS regression with heteroscedasticity and autocorrelation consistent (HAC) covariance
itsa_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':7})

# Display the regression summary to assess the causal impact of the NPI
print(itsa_model.summary())
