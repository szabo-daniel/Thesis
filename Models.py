import matplotlib.pyplot as plt
import numpy as np

from ImportData import *
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from math import sqrt
from sklearn.metrics import mean_squared_error

US_data = US_data[:-1]
countries = [US_data]
print(US_data)

for country_data in countries:
    # n_factors = len(US_factor_list)

    #Plot correlations of factors
    # sb.heatmap(factors.corr(), annot=True, cbar=False)
    # plt.title('Correlation Matrix - All Factors')
    # plt.show()
    #
    # sb.heatmap(factors.corr() > 0.9, annot=True, cbar=False)
    # plt.title('Correlation Matrix - All Factors Above 0.9 Correlation')
    # plt.show()
    #
    # sb.heatmap(factors.corr() < -0.9, annot=True, cbar=False)
    # plt.title('Correlation Matrix - All Factors Below -0.9 Correlation')
    # plt.show()

    print(country_data.describe())
    print(country_data.median())

    # Split into training and test sets
    y = country_data.iloc[:, 0]
    X = country_data.iloc[:, 1:].shift(1) # lag factor data back by one period to prevent look-ahead bias
    X = X.iloc[1:,:] # omit first row due to NaN record
    y = y.iloc[1:] # omit first row to create same number of rows for models

    # MODEL 1 - OLS REGRESSION
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

    OLS = LinearRegression()
    OLS.fit(X_train, y_train)
    print(f'Coefficients: {OLS.coef_}')
    print(f'R2: {OLS.score(X_train, y_train)}')

    print(X_test.tail())
    # Predict with OLS
    y_pred = OLS.predict(X_test)
    performance = pd.DataFrame({'Predictions': y_pred, 'Actual values': y_test})
    performance['Error'] = performance['Actual values'] - performance['Predictions']
    print(performance.head())

    rmse_OLS = sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {rmse_OLS}')

    # Plot errors
    performance.reset_index(drop=True, inplace=True)
    performance.reset_index(inplace=True)

    fig = plt.figure(figsize=(10,5))
    plt.bar('index', 'Error', data=performance, color='black', width=0.3)
    plt.show()

    # Statsmodels OLS model - clearer output
    X_train = sm.add_constant(X_train) # adds constant equal to 1 for training dataset
    OLS_model = sm.OLS(y_train, X_train).fit()
    # OLS_model2 = sm.OLS(y_test, X_test).fit()
    print(OLS_model.summary())
    # print(OLS_model2.summary())

    # Machine learning data setup
    factors = country_data.iloc[:, 1:].shift(1) # lag factor data back by one period to prevent look-ahead bias
    factors = factors.iloc[1:,:]
    targets = country_data.iloc[:, 0]
    targets = targets.iloc[1:]

    # print('FACTORS')
    # print(factors)
    # print(targets)

    # Rescale data to be between 0 and 1 for use in machine learning models
    scaler = MinMaxScaler()
    factors_rescaled = scaler.fit_transform(factors)
    factors_rescaled = pd.DataFrame(factors_rescaled, columns=factors.columns, index=factors.index)
    # print(factors_rescaled)

    train_size = int(0.80 * factors.shape[0])
    train_factors = factors[:train_size]
    train_targets = targets[:train_size]
    test_factors = factors[train_size:]
    test_targets = targets[train_size:]

    # reshape targets to match keras expectations
    train_targets = train_targets.values.reshape(-1, 1)
    test_targets = test_targets.values.reshape(-1, 1)

    train_factors = np.reshape(train_factors, (train_factors.shape[0], train_factors.shape[1], 1))
    test_factors = np.reshape(test_factors, (test_factors.shape[0], test_factors.shape[1], 1))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train model
    model.fit(train_factors, train_targets, epochs = 100, verbose = 0)

    # Prediction & evaluation
    predictions = model.predict(test_factors)
    rmse = sqrt(mean_squared_error(test_targets, predictions))
    print(f'RMSE: {rmse}')

    #










