# StockPricePrediction

# Stock Price Prediction using Monte Carlo Simulations and Machine Learning

## Project Overview

This project aims to predict stock prices by combining historical market data with advanced data processing techniques. We collect data for a diverse set of assets, including stocks, ETFs, commodities, and cryptocurrencies, and enrich this data with additional features. The ultimate goal is to create a robust dataset that can be used for quantitative analysis, trading strategies, and financial modeling.

## Project Structure
StockPricePrediction/

├── data/ # Directory to store raw and enhanced data files
├── notebooks/ # Jupyter notebooks for data collection and analysis
├── scripts/ # Python scripts for data processing
├── .gitignore # Git ignore file
├── README.md # Project documentation
├── requirements.txt # List of Python libraries required for the project


## Data Collection and Feature Engineering

### Assets

We collect data for the following diverse assets:

- **AAPL**: Apple Inc.
- **MSFT**: Microsoft Corporation
- **GOOGL**: Alphabet Inc. (Google)
- **AMZN**: Amazon.com Inc.
- **TSLA**: Tesla Inc.
- **SPY**: SPDR S&P 500 ETF Trust (an ETF that tracks the S&P 500)
- **GLD**: SPDR Gold Shares (an ETF that tracks the price of gold)
- **BTC-USD**: Bitcoin in USD (Cryptocurrency)

### Time Period

The data is collected for the period from January 1, 2010, to January 1, 2023.

### Data Processing

1. **Daily Returns**: 

   \begin{equation*}
   \text{Daily Return} = \frac{\text{Adj Close}_t - \text{Adj Close}_{t-1}}{\text{Adj Close}_{t-1}}
   \end{equation*}

2. **Moving Averages**:
   - 20-day Moving Average (MA20):

     \[
     \text{MA20} = \frac{1}{20} \sum_{i=0}^{19} \text{Adj Close}_{t-i}
     \]

   - 50-day Moving Average (MA50):

     \[
     \text{MA50} = \frac{1}{50} \sum_{i=0}^{49} \text{Adj Close}_{t-i}
     \]

3. **Volatility**:
   - 20-day Volatility (standard deviation of daily returns):

     \[
     \text{Volatility}_{20} = \sqrt{\frac{1}{20} \sum_{i=0}^{19} (\text{Daily Return}_{t-i} - \bar{r})^2}
     \]

     where \( \bar{r} \) is the mean daily return over the past 20 days.


## Data Collection Script

The main script for collecting and processing data is provided in the notebooks directory under DataCollection.ipynb. This script performs the following steps:

    Downloads historical data for each asset.
    Adds extra features.
    Saves both raw and enhanced data to CSV files.
