import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import DataReader as web

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

########################################################################################################################
# DATA DICTIONARY
########################################################################################################################
# 1. Dependent Variable
# eqprem = Equity Risk Premium (for each index): index return - 3-month government bond yield
#
# 2. Indices
# SP500 = S&P 500 (USA)
# FTSE100 = FTSE 100 (UK)
# ASX200 = S&P/ASX 200 (Australia)
# N225 = Nikkei 225 (Japan)
# DAX = DAX (Germany)
# CAC40 = CAC 30 (France)
#
# 2. Factors (all percentage changes)
# Rf = 3-month government bond yield
# GDP = Quarterly GDP % change (OECD)
# GFCF = Gross fixed capital formation (OECD)
# Infl = Inflation less food and energy
# RHousing = Real Housing Prices
# Oil = Global Brent Crude oil prices

########################################################################################################################
# KEY ASSUMPTIONS, SET DATES, IMPORT INDEX AND RISK-FREE DATA, AND CALCULATE EXCESS RETURNS
########################################################################################################################

# Dates (Q1 1993 - Q3 2023)
start_date = '1993-01-01'
end_date = '2023-10-01'

# Indices (note: first 3 are from market-based economies, last 3 are from bank-based economies)
index_list = ['^GSPC', '^FTSE', '^AXJO', '^N225', '^GDAXI', '^FCHI']
index_names = ['SP500', 'FTSE100', 'ASX200', 'N225', 'DAX', 'CAC40']
log_qtr_returns = pd.DataFrame()

# Risk-free rates (will eventually include in factor-specific file for each country)
# NOTE: ONLY HAS USA DATA FOR NOW, WILL GET OTHER DATA IN ON BLOOMBERG (note to self - change to end of period)
risk_free_rates = pd.read_csv('RiskFreeRates.csv', index_col='Date',parse_dates=True)

# Ensure risk-free rate DataFrame's index is in the same datetime format as returns dataframe
risk_free_rates.index = pd.to_datetime(risk_free_rates.index)
risk_free_rates.iloc[:, :] = risk_free_rates.iloc[:, :] / 100 # convert risk-free rates to decimal form
risk_free_rates.index = risk_free_rates.index - pd.Timedelta(days=1) # due to way data is imported offset by one day
risk_free_rates = risk_free_rates.iloc[1:] # eliminate first row to sync up dates
log_risk_free_rates = np.log(1 + risk_free_rates) # convert to log risk-free

ticker_count = 0 #Iterator for renaming columns

for ticker in index_list:
    # Import index price data from Yahoo Finance, keeping only the close prices
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)

    # Convert prices to log returns - log(R) = log(Pt / Pt-1)
    log_returns = np.log(data/data.shift(1))

    #Resample log returns to quarterly time frame
    log_qtr_returns_data = log_returns.resample('Q').sum()
    log_qtr_returns_data.rename(columns={'Close': index_names[ticker_count]}, inplace=True) #rename dataframe columns

    # If the quarterly_returns DataFrame is empty, initialize it with the data from the first ticker
    if log_qtr_returns.empty:
        log_qtr_returns = log_qtr_returns_data
    else:
        # Otherwise, join the new data with the existing DataFrame on the date index
        log_qtr_returns = log_qtr_returns.join(log_qtr_returns_data, how='outer')

    ticker_count += 1

excess_returns = log_qtr_returns.copy()

# Calculate excess index returns
for ticker in index_names:
    rf_column = f'Rf_{ticker}'
    if rf_column in log_risk_free_rates.columns:
        excess_returns[ticker] = excess_returns[ticker] - log_risk_free_rates[rf_column]

# Test prints
print(excess_returns)
print(log_qtr_returns)
print(risk_free_rates)

########################################################################################################################
# IMPORT MACRO FACTORS FOR EACH COUNTRY
########################################################################################################################

# Currently building out Excel file that includes macro factors for each economy.
# Refer to Country Data.xlsx for current progress
# The final version of data will be in the following form:
# Date | CountryCode_Factor | ... for all factors and countries. Will iterate through or divide

# Read in factors that apply to all countries (note to self: will compile these in Excel given the nature of data)
# 1. Exchange rates (EUR, USD, JPY, AUD)
# 2. Commodity prices (ex. Brent crude, metal prices)
# 3. Fed Funds Rate

# Read in factors that differ per country (will do via Excel) - a lot of these factors I sourced from APT research paper
# 1. Dividend Yield
# 2. Industrial production
# 3. GDP
# 4. GFCF
# 5. Real housing prices
# 6. Unemployment
# 7. ECB rates
# 8. Earnings
# 9. Current Account balance
# 10. Inflation (less food and energy)
# 11. Real consumption
# 12. Money Supply
# 13. Retail Prices
# 14. Capital Flows
# 15. Wages
# 16. Export Prices
# 17. Domestic National Product
# 18. Imports / Exports
# 19. ...and more as I find 'em

########################################################################################################################
# BACKUP CODE DUMP (will delete later)
########################################################################################################################
# for ticker in index_list:
#     data = yf.download(ticker, start=start_date, end=end_date)
#     data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)
#
#     quarterly_data = data.pct_change()
#     quarterly_data = quarterly_data.resample('Q').agg(lambda x: (x+1).prod() -1)
#     quarterly_data.rename(columns={'Close': ticker}, inplace=True)
#
#     # If the quarterly_returns DataFrame is empty, initialize it with the data from the first ticker
#     if quarterly_returns.empty:
#         quarterly_returns = quarterly_data
#     else:
#         # Otherwise, join the new data with the existing DataFrame on the date index
#         quarterly_returns = quarterly_returns.join(quarterly_data, how='outer')
#
# print(quarterly_returns.head())


# index_ticker = '^GSPC'
# data = yf.download(index_ticker, start = start_date, end = end_date)
# data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)
# data = data.pct_change()
# data = data.resample('M').agg(lambda x: (x+1).prod() - 1) #WORKS
#
# print(data)
