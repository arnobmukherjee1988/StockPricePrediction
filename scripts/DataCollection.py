#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import reqired libraries
import numpy as np
import yfinance as yf
import pandas as pd
import os


# In[2]:


# List of diverse stock symbols and ETFs
# AAPL: Apple Inc.
# MSFT: Microsoft Corporation
# GOOGL: Alphabet Inc. (Google)
# AMZN: Amazon.com Inc.
# TSLA: Tesla Inc.
# SPY: SPDR S&P 500 ETF Trust (an ETF that tracks the S&P 500)
# GLD: SPDR Gold Shares (an ETF that tracks the price of gold)
# BTC-USD: Bitcoin in USD (Cryptocurrency)
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'GLD', 'BTC-USD']

# Define the time period
start_date = '2010-01-01'
end_date = '2023-01-01'


# In[3]:


# Create a directory to save the data if it doesn't exist
os.makedirs('../data', exist_ok=True)


# 1. **Daily Returns**:
#    $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
# 
#    where $R_t$ is the daily return, $P_t$ is the adjusted closing price at time $t$, and $P_{t-1}$ is the adjusted closing price at time $t-1$.
# 
# 2. **Moving Averages**:
#    - 20-day Moving Average (MA20):
#      $MA_{20} = \frac{1}{20} \sum_{i=0}^{19} P_{t-i}$
# 
#    - 50-day Moving Average (MA50):
#      $MA_{50} = \frac{1}{50} \sum_{i=0}^{49} P_{t-i}$
# 
# 3. **Volatility**:
#    - 20-day Volatility (standard deviation of daily returns):
#      $\sigma_{20} = \sqrt{\frac{1}{20} \sum_{i=0}^{19} (R_{t-i} - \mu)^2}$
# 
#      where $\sigma_{20}$ is the 20-day volatility, $R_{t-i}$ is the daily return at time $t-i$, and $\mu$ is the mean daily return over the past 20 days.
# 

# In[4]:


def add_features(df):
  """
  Adds additional features to the stock data DataFrame.
  Parameters --> : df (DataFrame) containing stock data.
  Returns --> DataFrame with additional features.
  """
  # Calculate daily returns as the percentage change of the adjusted close price
  df['Daily Return'] = df['Adj Close'].pct_change()
  
  # Calculate the 20-day moving average of the adjusted close price
  df['MA20'] = df['Adj Close'].rolling(window=20).mean()
  
  # Calculate the 50-day moving average of the adjusted close price
  df['MA50'] = df['Adj Close'].rolling(window=50).mean()
  
  # Calculate the volatility (standard deviation of daily returns) over a 20-day window
  df['Volatility'] = df['Daily Return'].rolling(window=20).std()
  
  # Drop any rows with NaN values resulting from the calculations above
  df.dropna(inplace=True)
  
  return df


# In[5]:


# Download historical stock data for the given symbol and time period
for asset in assets:
  stock_data = yf.download(asset, start=start_date, end=end_date)
  
  # Save the raw data to a CSV file
  stock_data.to_csv(f'../data/{asset}_historical_data.csv')
  
  # Add features
  stock_data = add_features(stock_data)
  
  # Save the enhanced data to a new CSV file
  stock_data.to_csv(f'../data/{asset}_enhanced_data.csv')
  
  # Display the first few rows of the enhanced data
  print(f"Enhanced data for {asset}:")
  print(stock_data.head())
  print("\n")


# In[ ]:




