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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.linear_model import Ridge, Lasso
import pmdarima as pm
from statsmodels.tsa.arima_model import ARIMA

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False) # Split into training and test

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
    oos_r2_OLS = r2_score(y_test, y_pred)

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
    # scaler = MinMaxScaler()
    # factors_rescaled = scaler.fit_transform(factors)
    # factors_rescaled = pd.DataFrame(factors_rescaled, columns=factors.columns, index=factors.index)
    # targets_rescaled = scaler.fit_transform(targets)
    # targets_rescaled = pd.DataFrame(targets_rescaled, columns=targets.columns, index=targets.index)

    train_size = int(0.80 * factors.shape[0])
    train_factors = factors[:train_size]
    train_targets = targets[:train_size]
    test_factors = factors[train_size:]
    test_targets = targets[train_size:]

    train_factors = scale(train_factors)
    test_factors = scale(test_factors)
    # print('RESCALED FACTORS')
    # print(factors_rescaled)
    # print('RESCALED TARGETS')
    # print(targets_rescaled)

    # reshape targets to match keras expectations
    train_targets = train_targets.values.reshape(-1, 1)
    test_targets = test_targets.values.reshape(-1, 1)

    train_factors = np.reshape(train_factors, (train_factors.shape[0], train_factors.shape[1], 1))
    test_factors = np.reshape(test_factors, (test_factors.shape[0], test_factors.shape[1], 1))

    # LSTM model
    model = Sequential() # initialize the neural network
    model.add(LSTM(100, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    # model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=True))
    model.add(LSTM(50, activation='relu', input_shape=(train_factors.shape[1], 1), return_sequences=False))
    model.add(Dense(50, activation='relu', input_shape=(train_factors.shape[1], 1)))
    model.add(Dense(1))
    model.add(Dropout(0.2))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Train model
    model.fit(train_factors, train_targets, epochs = 100, verbose = 0)
    # model.fit(train_factors, train_targets, epochs=100)

    # Prediction & evaluation
    predictions = model.predict(test_factors)
    rmse = sqrt(mean_squared_error(test_targets, predictions))
    oos_r2 = r2_score(test_targets, predictions)

    # Random forest model

    # print('FEATURE COUNT', len(country_data.columns))
    total_feature_count = int(len(country_data.columns))
    test_scores_rf = []

    train_factors = factors[:train_size]
    train_targets = targets[:train_size]
    test_factors = factors[train_size:]
    test_targets = targets[train_size:]

    grid_rf = {'n_estimators': [50, 100, 150, 200],
               'max_depth': [3, 4, 5, 6, 7],
               'max_features': [total_feature_count-5, total_feature_count-4, total_feature_count-3, total_feature_count-2, total_feature_count-1],
               'random_state': [42]}

    rf_model = RandomForestRegressor()

    for g in ParameterGrid(grid_rf):
        rf_model.set_params(**g)
        rf_model.fit(train_factors, train_targets)
        test_scores_rf.append(rf_model.score(test_factors, test_targets))
        print(f'Iterating through parameter grid: {g}...')

    best_index = np.argmax(test_scores_rf)
    print(test_scores_rf[best_index], ParameterGrid(grid_rf)[best_index])

    # Create random forest model with best parameters from grid:
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=15, random_state=42)
    rf_model.fit(train_factors, train_targets)

    y_pred_rf = rf_model.predict(test_factors)

    # Evaluate model performance
    rf_train_score = rf_model.score(train_factors, train_targets)
    rf_test_score = rf_model.score(test_factors, test_targets)
    oos_r2_rf = r2_score(test_targets, y_pred_rf)

    # Decision Tree
    dt_model = DecisionTreeRegressor()
    dt_model.fit(train_factors, train_targets)

    y_pred_dt = dt_model.predict(test_factors)
    oos_r2_dt = r2_score(test_targets, y_pred_dt)

    # KNN Model
    knn_train_scores = []
    knn_test_scores = []
    cv_scores = []
    max_test_score = -np.inf
    max_test_metrics = None
    max_test_neighbors = None

    for i in range(2, 80):
        print(f'Building model for {i} neighbors...')
        knn_model = KNeighborsRegressor(n_neighbors=i)

        scores = cross_val_score(knn_model, train_factors, train_targets, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())

        knn_model.fit(train_factors, train_targets)
        y_pred_knn = knn_model.predict(test_factors)

    optimal_n = np.argmin(cv_scores) + 2
    print()
    print(f'Optimal number of neighbors is {optimal_n}')
    knn_model_opt = KNeighborsRegressor(n_neighbors=optimal_n)
    knn_model_opt.fit(train_factors, train_targets)

    y_pred_knn_opt = knn_model_opt.predict(test_factors)
    oos_r2_knn_opt = r2_score(test_targets, y_pred_knn_opt)

    # Model - LASSO regression
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    lasso_model = Lasso(alpha=2)
    lasso_model.fit(X_train_const, y_train)
    y_pred_lasso = lasso_model.predict(X_test_const)

    rmse_lasso = sqrt(mean_squared_error(y_test, y_pred_lasso))
    oos_r2_lasso = r2_score(y_test, y_pred_lasso)

    # Model - Ridge regression
    ridge_model = Ridge(alpha=5)
    ridge_model.fit(X_train_const, y_train)
    y_pred_ridge = ridge_model.predict(X_test_const)

    rmse_ridge = sqrt(mean_squared_error(y_test, y_pred_ridge))
    oos_r2_ridge = r2_score(y_test, y_pred_ridge)

    # Report key model statistics:
    print('OLS Regression Metrics')
    print(f'RMSE: {rmse_OLS}')
    print(f'OOS R2 {oos_r2_OLS}')
    print('')

    print('Optimized Random Forest Metrics')
    print(f'RMSE: {sqrt(mean_squared_error(test_targets, y_pred_rf))}')
    print(f'OOS R2: {oos_r2_rf}')
    print('')

    print('Decision Tree Metrics')
    print(f'RMSE: {sqrt(mean_squared_error(test_targets, y_pred_dt))}')
    print(f'OOS R2: {oos_r2_dt}')
    print('')

    print('Optimized KNN Metrics')
    print(f'RMSE: {sqrt(mean_squared_error(test_targets, y_pred_knn_opt))}')
    print(f'OOS R2: {oos_r2_knn_opt}')
    print('')

    print('Lasso Regression Metrics')
    print(f'RMSE: {rmse_lasso}')
    print(f'OOS R2: {oos_r2_lasso}')
    print('')

    print('Ridge Regression Metrics')
    print(f'RMSE: {rmse_ridge}')
    print(f'OOS R2: {oos_r2_ridge}')
    print('')

    print('LSTM Metrics')
    print(f'RMSE: {rmse}')
    print(f'OOS R2: {oos_r2}')

    # Model - ARIMA
    # X_train_reshaped = X_train.ravel()
    # arima_model = pm.auto_arima(X_train_reshaped,
    #                             seasonal=False,
    #                             stepwise=True,
    #                             suppress_warnings=True,
    #                             error_action='ignore',
    #                             max_order=None,
    #                             trace=True)
    #
    # y_pred_arima = arima_model.predict(n_periods=len(y_test))

    # arima_model = ARIMA()

    # rmse_arima = sqrt(mean_squared_error(y_test, y_pred_arima))
    # oos_r2_arima = r2_score(y_test, y_pred_arima)
    #
    # print('ARIMA Model Metrics')
    # print(f'RMSE: {rmse_arima}')
    # print(f'OOS R2: {oos_r2_arima}')

    # # Model - Scaled LASSO regression - same results as non-scaled version
    # scaled_lasso_model = Lasso(alpha=2)
    # scaled_lasso_model.fit(train_factors, train_targets)
    # y_pred_lasso_scaled = scaled_lasso_model.predict(test_factors)
    #
    # rmse_lasso_scaled = sqrt(mean_squared_error(test_targets, y_pred_lasso_scaled))
    # oos_r2_lasso_scaled = r2_score(test_targets, y_pred_lasso_scaled)
    #
    # print('Scaled Lasso Regression Metrics')
    # print(f'RMSE: {rmse_lasso_scaled}')
    # print(f'OOS R2: {oos_r2_lasso_scaled}')
    # print('')
    #



    # Model - Scaled LASSO regression
    # lasso_model_scaled = Lasso(alpha=0.01)
    # lasso_model_scaled.fit()


















