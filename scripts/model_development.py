#!/usr/bin/env python
# coding: utf-8

# # Model Development
# This notebook covers the model development process for predicting stock prices. We will load the enhanced datasets, select features, train multiple models, and evaluate their performance.
# 

# In[2]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set global settings
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


# In[7]:


# List of assets
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'GLD', 'BTC-USD']

# Load the enhanced datasets
data = {}
for asset in assets:
    file_path = f'../data/{asset}_enhanced_data.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    data[asset] = df

# # Display the first few rows of the datasets
# for asset, df in data.items():
#     print(f"\n{asset} data:")
#     print(df.head())

# Prepare a single dataset for modeling (for simplicity, let's use AAPL as an example)
df = data['AAPL']
df.dropna(inplace=True)  # Drop any remaining NaN values

# Select features and target
features = ['Volume', 'Daily Return', 'MA50', 'RSI', 'MACD', 'Signal Line', 'Volatility']
target = 'Close'

# Create feature and target datasets
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[4]:


# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(kernel='rbf')
}

# Train and validate models using cross-validation
results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)


# In[5]:


# Plot the results for comparison
results_df.plot(kind='bar', figsize=(14, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Error')
plt.show()

# Evaluate the best-performing model
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]

# Predict and plot the results
y_pred_best = best_model.predict(X_test_scaled)
plt.figure(figsize=(14, 8))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred_best, label='Predicted')
plt.title(f'{best_model_name} Predictions vs Actual')
plt.legend()
plt.show()


# In[6]:


# Save the best model to disk
joblib.dump(best_model, f'../models/{best_model_name}_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')


# # Conclusion
# In this notebook, we developed and evaluated multiple models for stock price prediction. The best-performing model was saved for future use. Next steps include further hyperparameter tuning and incorporating additional data sources.
# 

# In[ ]:




