import numpy as np
import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)

# all_data = pd.read_excel('Country Data.xlsx',sheet_name='Macro Data',header=1,index_col=0)
#
# print(all_data)

########################################################################################################################
# KEY ASSUMPTIONS, SET DATES, IMPORT INDEX DATA
########################################################################################################################

# Dates
start_date = '1993-01-01'
end_date = '2023-12-31'

# Indices (note: first 3 are from market-based economies, last 3 are from bank-based economies)
index_list = ['^GSPC', '^FTSE','^AXJO', '^N225', '^GDAXI', '^FCHI']
index_names = ['SP500', 'FTSE100','ASX200','N225','DAX','CAC40']
log_qtr_returns = pd.DataFrame()

ticker_count = 0

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

print(log_qtr_returns)




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
