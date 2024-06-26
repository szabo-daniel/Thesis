import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas_datareader import DataReader as web

import tensorflow as tf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

########################################################################################################################
# DATA DICTIONARY
########################################################################################################################
# 1. Dependent Variable
# eqprem = Equity Risk Premium (for each index): index return - 3-month government bond yield (or 2 year)
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
# start_date = '1993-01-01'
# end_date = '2023-07-01'
# end_date = '2021-10-01' #Note: set this as the end date if using GW factor data

# Indices (note: first 3 are from market-based economies, last 3 are from bank-based economies)
# May replace with MSCI data if I can get it
# index_list = ['^GSPC', '^FTSE', '^AXJO', '^N225', '^GDAXI', '^FCHI']
# index_names = ['SP500', 'FTSE100', 'ASX200', 'N225', 'DAX', 'CAC40']
# log_qtr_returns = pd.DataFrame()

# Import quarterly factor data from Goyal and Welch (2008) paper
GW_df = pd.read_excel('Country Data.xlsx', sheet_name='GW Data', index_col='Date', parse_dates=True)
print(GW_df)

# # Risk-free rates (will eventually include in factor-specific file for each country)
# # NOTE: ONLY HAS USA DATA FOR NOW, WILL GET OTHER DATA IN ON BLOOMBERG (note to self - change to end of period)
# risk_free_rates = pd.read_csv('RiskFreeRates.csv', index_col='Date',parse_dates=True) # ERROR if risk-free rate = 0 since we take log version
#
# # Ensure risk-free rate DataFrame's index is in the same datetime format as returns dataframe
# risk_free_rates.index = pd.to_datetime(risk_free_rates.index)
# # risk_free_rates.iloc[:, :] = risk_free_rates.iloc[:, :] / 100 # convert risk-free rates to decimal form
# risk_free_rates.index = risk_free_rates.index - pd.Timedelta(days=1) # due to way data is imported offset by one day
# risk_free_rates = risk_free_rates.iloc[1:] # eliminate first row to sync up dates
# log_risk_free_rates = np.log(1 + risk_free_rates) # convert to log risk-free

# ticker_count = 0 # Iterator for renaming columns
#
# for ticker in index_list:
#     # Import index price data from Yahoo Finance, keeping only the close prices
#     data = yf.download(ticker, start=start_date, end=end_date)
#     data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)
#
#     # Convert prices to log returns - log(R) = log(Pt / Pt-1)
#     log_returns = np.log(data/data.shift(1))
#
#     # Resample log returns to quarterly time frame
#     log_qtr_returns_data = log_returns.resample('Q').sum()
#     log_qtr_returns_data.rename(columns={'Close': index_names[ticker_count]}, inplace=True) #rename dataframe columns
#
#     # If the quarterly_returns DataFrame is empty, initialize it with the data from the first ticker
#     if log_qtr_returns.empty:
#         log_qtr_returns = log_qtr_returns_data
#     else:
#         # Otherwise, join the new data with the existing DataFrame on the date index
#         log_qtr_returns = log_qtr_returns.join(log_qtr_returns_data, how='outer')
#
#     ticker_count += 1
#
# excess_returns = log_qtr_returns.copy()
# excess_returns['SP500'] = excess_returns['SP500'] + np.log(GW_df['D12']).values[1:]

# # Calculate excess index returns
# for ticker in index_names:
#     rf_column = f'Rf_{ticker}'
#     if rf_column in log_risk_free_rates.columns:
#         excess_returns[ticker] = excess_returns[ticker] - log_risk_free_rates[rf_column]


# # Test prints
# print('INITIAL PRINTS')
# print(excess_returns)
# print(log_qtr_returns)
# print(risk_free_rates)
########################################################################################################################
# IMPORT MACRO FACTORS FOR EACH COUNTRY
########################################################################################################################
# Organize dataframes for analysis - beginning with excess returns

# United States
# Line up quarterly dates for appending
US_factors = pd.read_excel('Country Data.xlsx', sheet_name='US Data', index_col='Date', parse_dates=True)
# US_factors.index = US_factors.index - pd.Timedelta(days=1)

# Create list of all columns to iterate through
US_factor_list = US_factors.columns.tolist()
print('factor list')
print(US_factor_list)

US_data = pd.DataFrame(index=GW_df.index)
US_data['ER'] = GW_df['EqPrem']
# print(US_data['ER'])
# US_data['ER'] = (np.log(GW_df['Index'] + GW_df['D12']) - np.log(GW_df['Rfree'])).values[1:]

# Read in all factors from list of given factors into one dataframe
for factor in US_factor_list:
    US_data[factor] = US_factors[factor]

# Read in GW factors (comment out if not used)
US_data['div_price'] = (np.log(GW_df['D12']) - np.log(GW_df['Index'])).values # makes sure that indexes aren't an issue, an pastes only values over. Start at 1 to omit header as a value
US_data['div_yield'] = (np.log(GW_df['D12']) - np.log(GW_df['Index'].shift(1))).values #check logic on this, should work
US_data['earnings_price'] = (np.log(GW_df['E12']) - np.log(GW_df['Index'])).values
US_data['dividend_payout'] = (np.log(GW_df['D12']) - np.log(GW_df['E12'])).values
US_data['term_spread'] = (GW_df['lty'] - GW_df['tbl']).values
US_data['default_yield_spread'] = (GW_df['BAA'] - GW_df['AAA']).values
US_data['default_return_spread'] = (GW_df['corpr'] - GW_df['ltr']).values
US_data['book_to_market'] = GW_df['b/m']
US_data['stock_variance'] = GW_df['svar']
US_data['investment_to_capital'] = GW_df['ik']

print('US DATA')
print(US_data)
print('')

# Note: Read in sheet containing index and risk-free return data for other countries, since no longer using Goyal and Welch data.
index_data = pd.read_excel('Country Data.xlsx', sheet_name='Index Data',index_col='Date',parse_dates=True)

# United Kingdom
UK_factors = pd.read_excel('Country Data.xlsx', sheet_name='UK Data',index_col='Date',parse_dates=True)
# UK_factors.index = UK_factors.index - pd.Timedelta(days=1)

UK_factor_list = UK_factors.columns.tolist()

UK_data = pd.DataFrame()
UK_data['EqPrem'] = index_data['UK_EqPrem']

for factor in UK_factor_list:
    UK_data[factor] = UK_factors[factor]

print('UK DATA')
print(UK_data)
print('')

# # Australia
# AU_factors = pd.read_excel('Country Data.xlsx', sheet_name='AU Data',index_col='Date',parse_dates=True)
# # AU_factors.index = AU_factors.index - pd.Timedelta(days=1)
#
# AU_factor_list = AU_factors.columns.tolist()
#
# AU_data = pd.DataFrame()
#
# for factor in AU_factor_list:
#     AU_data[factor] = AU_factors[factor]

# Germany
# DE_factors = pd.read_excel('Country Data.xlsx', sheet_name='DE Data',index_col='Date',parse_dates=True)
# DE_factors.index = DE_factors.index - pd.Timedelta(days=1)
#
# DE_factor_list = DE_factors.columns.tolist()
#
# DE_data = pd.DataFrame()
# DE_data['EqPrem'] = index_data['EqPrem']
#
# for factor in DE_factor_list:
#     DE_data[factor] = DE_factors[factor]
#
# print('GERMANY DATA')
# print(DE_data)
# print('')

# # France
# FR_factors = pd.read_excel('Country Data.xlsx', sheet_name='FR Data',index_col='Date',parse_dates=True)
# FR_factors.index = FR_factors.index - pd.Timedelta(days=1)
#
# FR_factor_list = FR_factors.columns.tolist()
#
# FR_data = pd.DataFrame()
# FR_data['EqPrem'] = index_data['EqPrem']
#
# for factor in FR_factor_list:
#     FR_data[factor] = FR_factors[factor]

# # Japan
# JP_factors = pd.read_excel('Country Data.xlsx', sheet_name='JP Data',index_col='Date',parse_dates=True)
# JP_factors.index = JP_factors.index - pd.Timedelta(days=1)
#
# JP_factor_list = JP_factors.columns.tolist()
#
# JP_data = pd.DataFrame()
# # JP_data['ER'] = excess_returns['N225']
#
# for factor in JP_factor_list:
#     JP_data[factor] = JP_factors[factor]

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

# This matches CRSP returns exactly
# index_ticker = '^GSPC'
# data = yf.download(index_ticker, start = start_date, end = end_date)
# data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)
# data = data.pct_change()
# data = data.resample('M').agg(lambda x: (x+1).prod() - 1)
# print(data)

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
# 19. ...and more as I find them