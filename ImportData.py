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
# KEY ASSUMPTIONS, INPUTS, AND IMPORTS
########################################################################################################################

# Dates
start_date = '1993-01-01'
end_date = '2023-12-31'

index_list = ['^GSPC', '^FTSE','^AXJO', '^N225', '^GDAXI', '^FCHI']
quarterly_returns = pd.DataFrame()

for ticker in index_list:
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)

    quarterly_data = data.pct_change()
    quarterly_data = quarterly_data.resample('Q').agg(lambda x: (x+1).prod() -1)
    quarterly_data.rename(columns={'Close': ticker}, inplace=True)

    # If the quarterly_returns DataFrame is empty, initialize it with the data from the first ticker
    if quarterly_returns.empty:
        quarterly_returns = quarterly_data
    else:
        # Otherwise, join the new data with the existing DataFrame on the date index
        quarterly_returns = quarterly_returns.join(quarterly_data, how='outer')

print(quarterly_returns.head())


# index_ticker = '^GSPC'
# data = yf.download(index_ticker, start = start_date, end = end_date)
# data = data.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis=1)
# data = data.pct_change()
# data = data.resample('M').agg(lambda x: (x+1).prod() - 1) #WORKS
#
# print(data)
