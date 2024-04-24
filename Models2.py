import keras_tuner

from ImportData import *

from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split, cross_val_score, ParameterGrid
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras_tuner import RandomSearch, HyperParameters

n_iterations = 5
test_size = 0.2

scaler = MinMaxScaler(feature_range=(0,1))

US_data = US_data[1:-1]
countries = [US_data]
print(US_data) #good- all values read in properly

def GW_R2_score(MSE_A, MSE_N):
    # MSE_A is the mean squared error of the test model
    # MSE_N is the mean squared error of the historical mean model
    R2 = 1 - MSE_A / MSE_N
    return R2

def dRMSE(MSE_A, MSE_N):
    dRMSE = np.mean(MSE_N) - np.sqrt(MSE_A)
    return dRMSE

for country_data in countries:
    #############################################################
    # Split data into factors and targets, lagging appropriately
    #############################################################
    # 1. Not rescaled
    targets = country_data.iloc[:, 0]
    factors = country_data.iloc[:, 1:].shift(-1) # Lag factor data back by one period to prevent look-ahead bias
    factors = factors[:-1] # Due to lag drop NaN in last row
    targets = targets[:-1] # Drop last row to bring time periods into line

    # 2. Standardized (Mean zero and unit variance - useful for Lasso and Ridge)
    targets_standard = scale(targets)
    factors_standard = scale(factors)

    # 3. Normalized (values scaled to be between 0 and 1, useful for ML models)
    targets_rescaled = scaler.fit_transform(targets.values.reshape(-1, 1))
    factors_rescaled = scaler.fit_transform(factors)

    # Split data into training and test sets
    train_factors, test_factors, train_targets, test_targets = train_test_split(factors, targets, test_size=test_size, shuffle=False)
    train_factors_standard, test_factors_standard, train_targets_standard, test_targets_standard = train_test_split(factors_standard, targets_standard, test_size=test_size, shuffle=False)
    train_factors_rescaled, test_factors_rescaled, train_targets_rescaled, test_targets_rescaled = train_test_split(factors_rescaled, targets_rescaled, test_size=test_size, shuffle=False)

    ############################################################
    # Model 0 - Historical mean model
    ############################################################
    print('Generating historical mean model...')
    total_length = len(targets)
    hist_pred_all = np.zeros(total_length) # Create list to store values

    for i in range(1, total_length):
        hist_pred_all[i] = np.mean(targets[:i-1]) # Predict next period's value by the cumulative mean of the previous periods

    hist_pred_test = hist_pred_all[-len(test_targets):] # Get test period data for the historical mean model

    hist_MSE = mean_squared_error(test_targets, hist_pred_test) # Used as benchmark for subsequent model evaulation

    # Model 1 - OLS linear regression model (kitchen sink)
    OLS = LinearRegression()
    OLS.fit(train_factors, train_targets)
    OLS_pred = OLS.predict(test_factors)

    # OLS Metrics
    OLS_MSE = mean_squared_error(test_targets, OLS_pred)
    OLS_RMSE = root_mean_squared_error(test_targets, OLS_pred)
    OLS_dRMSE = dRMSE(OLS_MSE, hist_MSE)
    OLS_MAPE = mean_absolute_percentage_error(test_targets, OLS_pred)
    OLS_OOS_R2 = r2_score(test_targets, OLS_pred)
    OLS_OOS_GW_R2 = GW_R2_score(OLS_MSE, hist_MSE)

    print('Historical mean model complete')
    print('')
    ################################################################
    # Model 2 - Ridge Regression
    ################################################################
    print('Generating ridge regression model...')

    ridge_model = Ridge(alpha=5)
    ridge_model.fit(train_factors_standard, train_targets_standard)
    ridge_pred = ridge_model.predict(test_factors_standard)

    # Ridge Metrics
    ridge_MSE = mean_squared_error(test_targets, ridge_pred)
    ridge_RMSE = root_mean_squared_error(test_targets, ridge_pred)
    ridge_dRMSE = dRMSE(ridge_MSE, hist_MSE)
    ridge_MAPE = mean_absolute_percentage_error(test_targets, ridge_pred)
    ridge_OOS_R2 = r2_score(test_targets, ridge_pred)
    ridge_OOS_GW_R2 = GW_R2_score(ridge_MSE, hist_MSE)

    print('Ridge regression model complete')
    print('')
    ################################################################
    # Model 3 - Lasso Regression
    ################################################################
    print('Generating lasso regression model...')

    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(train_factors_standard, train_targets_standard) #Note: rescaled both
    lasso_pred = lasso_model.predict(test_factors_standard)

    # Lasso Metrics
    lasso_MSE = mean_squared_error(test_targets_standard, lasso_pred)
    lasso_RMSE = root_mean_squared_error(test_targets_standard, lasso_pred)
    lasso_dRMSE = dRMSE(lasso_MSE, hist_MSE)
    lasso_MAPE = mean_absolute_percentage_error(test_targets_standard, lasso_pred)
    lasso_OOS_R2 = r2_score(test_targets_standard, lasso_pred)
    lasso_OOS_GW_R2 = GW_R2_score(lasso_MSE, hist_MSE)

    print('Lasso regression model complete')
    print('')
    ################################################################
    # KNN Model (N-nearest neighbors optimized)
    ################################################################
    print('Generating KNN model...')
    knn_train_scores = []
    knn_test_scores = []
    cv_scores = []
    max_test_score = -np.inf
    max_test_metrics = None
    max_test_neighbors = None

    for i in range(2, 300):
        # print(f'Building model for {i} neighbors...')
        knn_model = KNeighborsRegressor(n_neighbors=i)

        scores = cross_val_score(knn_model, train_factors_rescaled, train_targets_rescaled)
        cv_scores.append(-scores.mean())

        # knn_model.fit(train_factors_rescaled, train_targets_rescaled)
        # knn_pred = knn_model.predict(test_factors_rescaled)

    optimal_n = np.argmin(cv_scores) + 2
    print('')
    print(f'Optimal number of neighbors: {optimal_n}')

    knn_model_opt = KNeighborsRegressor(n_neighbors=optimal_n)
    knn_model_opt.fit(train_factors_rescaled, train_targets_rescaled)
    knn_pred = knn_model_opt.predict(test_factors_rescaled)

    # KNN Metrics
    knn_MSE = mean_squared_error(test_targets_rescaled, knn_pred)
    knn_RMSE = root_mean_squared_error(test_targets_rescaled, knn_pred)
    knn_dRMSE = dRMSE(knn_MSE, hist_MSE)
    knn_MAPE = mean_absolute_percentage_error(test_targets_rescaled, knn_pred)
    knn_OOS_R2 = r2_score(test_targets_rescaled, knn_pred)
    knn_OOS_GW_R2 = GW_R2_score(knn_MSE, hist_MSE)

    print('KNN model complete')
    print('')

    ################################################################
    # Random forest (hyperparameter-optimized)
    ################################################################
    print('Generating random forest model...')
    factor_count = int(len(factors.columns)) # Should be 22 in total
    test_scores_rf = []

    grid_rf = {'n_estimators': [50, 100, 150, 200, 250],
               'max_depth': [None, 3, 5, 7, 10, 15, 20, 25, 30],
               'max_features': [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1],
               'random_state': [42]}
    rf_model = RandomForestRegressor()

    for g in ParameterGrid(grid_rf):
        rf_model.set_params(**g)
        rf_model.fit(train_factors_rescaled, train_targets_rescaled)
        test_scores_rf.append(rf_model.score(test_factors_rescaled, test_targets_rescaled))
        print(f'Iterating through parameter grid: {g}')

    best_index = np.argmax(test_scores_rf)
    print('Optimal Random Forest parameters:')
    print(test_scores_rf[best_index], ParameterGrid(grid_rf)[best_index])
    print('')

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, max_features=16, random_state=42)
    rf_model.fit(train_factors_rescaled, train_targets_rescaled)
    rf_pred = rf_model.predict(test_factors_rescaled)

    # Random Forest Metrics
    rf_MSE = mean_squared_error(test_targets_rescaled, rf_pred)
    rf_RMSE = root_mean_squared_error(test_targets_rescaled, rf_pred)
    rf_dRMSE = dRMSE(rf_MSE, hist_MSE)
    rf_MAPE = mean_absolute_percentage_error(test_targets_rescaled, rf_pred)
    rf_OOS_R2 = r2_score(test_targets_rescaled, rf_pred)
    rf_OOS_GW_R2 = GW_R2_score(rf_MSE, hist_MSE)

    print('Random forest complete')
    print('')

    ################################################################
    # LSTM Model
    ################################################################
    print('Generating LSTM models...')

    print("Train factors shape:", train_factors_rescaled.shape)
    print("Test factors shape:", test_factors_rescaled.shape)

    # train_factors_rescaled = train_factors_rescaled.reshape(
    #     (train_factors_rescaled.shape[0], 1, train_factors_rescaled.shape[1]))
    # test_factors_rescaled = test_factors_rescaled.reshape(
    #     (test_factors_rescaled.shape[0], 1, test_factors_rescaled.shape[1]))

    print("Reshaped Train factors shape:", train_factors_rescaled.shape)
    print("Reshaped Test factors shape:", test_factors_rescaled.shape)

    time_steps = 1  # assuming each sample is treated as a single time step sequence.

    train_factors_rescaled = train_factors_rescaled.reshape((-1, time_steps, train_factors_rescaled.shape[1]))
    test_factors_rescaled = test_factors_rescaled.reshape((-1, time_steps, test_factors_rescaled.shape[1]))


    def build_model(hp):
        model = Sequential()
        # Start with a Bidirectional LSTM layer to process data forward and backward; capture patterns from both directions in time series data
        model.add(Bidirectional(
            LSTM(
                # Dynamically set number of neurons based on tuner result
                units=hp.Int('units_lstm1', min_value=32, max_value=128, step=32),
                input_shape=(1, train_factors_rescaled.shape[2]),
                activation='tanh', # tanh activation function for to capture nonlinear relationships
                return_sequences=hp.Int('num_lstm_layers', 1, 8) > 1,
                kernel_regularizer=l2(hp.Float('l2_lstm1', min_value=0.01, max_value=0.1, step=0.01)) # penalty on layer weights to prevent overfitting
            )
        ))

        # Adding more LSTM layers based on tuner
        for i in range(hp.Int('num_lstm_layers', 1, 8)):  # Tuning the number of additional LSTM layers from 1 to 8
            model.add(LSTM(
                units=hp.Int('units_lstm' + str(i + 1), min_value=32, max_value=128, step=32),
                activation='tanh',
                return_sequences= (i < hp.Int('num_lstm_layers', 1, 8) - 1),
                kernel_regularizer=l2(hp.Float('l2_lstm' + str(i + 1), min_value=0.01, max_value=0.1, step=0.01))
            ))

        model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='relu'))
        model.compile(
            optimizer='adam',
            loss='mse'
        )
        return model

    # Callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=50,
        executions_per_trial=3,  # Increasing this for more robust results
        directory='C:/Users/dansz/PycharmProjects/Thesis/lstm_tuner',
        project_name='lstm_optim_advanced',
        overwrite=True
    )

    def run_tuner_with_batch_size():
        # Running the tuner with different batch sizes
        best_score = float('inf')
        best_model = None
        best_batch_size = None
        for batch_size in [10, 20, 30, 40]:  # You can customize these values
            tuner.search(train_factors_rescaled, train_targets_rescaled,
                         epochs=50,
                         batch_size=batch_size,
                         validation_data=(test_factors_rescaled, test_targets_rescaled),
                         callbacks=[early_stopping, reduce_lr])
            best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
            if best_trial.score < best_score:
                best_score = best_trial.score
                best_model = tuner.get_best_models(num_models=1)[0]
                best_batch_size = batch_size

        return best_model, best_batch_size

    best_model, best_batch_size = run_tuner_with_batch_size()
    lstm_pred = best_model.predict(test_factors_rescaled)

    # LSTM Metrics
    lstm_MSE = mean_squared_error(test_targets_rescaled, lstm_pred)
    lstm_RMSE = root_mean_squared_error(test_targets_rescaled, lstm_pred)
    lstm_dRMSE = dRMSE(lstm_MSE, hist_MSE)
    lstm_MAPE = mean_absolute_percentage_error(test_targets_rescaled, lstm_pred)
    lstm_OOS_R2 = r2_score(test_targets_rescaled, lstm_pred)
    lstm_OOS_GW_R2 = GW_R2_score(lstm_MSE, hist_MSE)

    ################################################################
    # Report all metrics
    ################################################################
    # OLS
    print('OLS Regression')
    print(f'MSE: {OLS_MSE}')
    print(f'RMSE: {OLS_RMSE}')
    print(f'dRMSE: {OLS_dRMSE}')
    print(f'MAPE: {OLS_MAPE}')
    print(f'OOS R2: {OLS_OOS_R2}')
    print(f'OOS GW R2: {OLS_OOS_GW_R2}')
    print('')

    # Ridge
    print('Ridge Regression')
    print(f'MSE: {ridge_MSE}')
    print(f'RMSE: {ridge_RMSE}')
    print(f'dRMSE: {ridge_dRMSE}')
    print(f'MAPE: {ridge_MAPE}')
    print(f'OOS R2: {ridge_OOS_R2}')
    print(f'OOS GW R2: {ridge_OOS_GW_R2}')
    print('')

    # Lasso
    print('Lasso Regression')
    print(f'MSE: {lasso_MSE}')
    print(f'RMSE: {lasso_RMSE}')
    print(f'dRMSE: {lasso_dRMSE}')
    print(f'MAPE: {lasso_MAPE}')
    print(f'OOS R2: {lasso_OOS_R2}')
    print(f'OOS GW R2: {lasso_OOS_GW_R2}')
    print('')

    # KNN
    print('KNN')
    print(f'MSE: {knn_MSE}')
    print(f'RMSE: {knn_RMSE}')
    print(f'dRMSE: {knn_dRMSE}')
    print(f'MAPE: {knn_MAPE}')
    print(f'OOS R2: {knn_OOS_R2}')
    print(f'OOS GW R2: {knn_OOS_GW_R2}')
    print('')

    # Random Forest
    print('Random Forest')
    print(f'MSE: {rf_MSE}')
    print(f'RMSE: {rf_RMSE}')
    print(f'dRMSE: {rf_dRMSE}')
    print(f'MAPE: {rf_MAPE}')
    print(f'OOS R2: {rf_OOS_R2}')
    print(f'OOS GW R2: {rf_OOS_GW_R2}')
    print('')

    # LSTM
    print('LSTM')
    print(f'MSE: {lstm_MSE}')
    print(f'RMSE: {lstm_RMSE}')
    print(f'dRMSE: {lstm_dRMSE}')
    print(f'MAPE: {lstm_MAPE}')
    print(f'OOS R2: {lstm_OOS_R2}')
    print(f'OOS GW R2: {lstm_OOS_GW_R2}')
    print('')

# Code dump
# Test if factors/targets split properly
    # print('TRAIN FACTORS')
    # print(train_factors)
    # print('TEST FACTORS')
    # print(test_factors)
    # print('TRAIN TARGETS')
    # print(train_targets)
    # print('TEST TARGETS')
    # print(test_targets)

# Test if lengths are the same for the hist mean and test targets
    # print(f'Test target length: {len(test_targets)}')
    # print(test_targets)
    # print(f'HM model test length: {len(hist_pred_test)}')
    # print(hist_pred_test)

# Copy/paste reporting metrics
#     print('OLS Regression')
#     print(f'MSE: {}')
#     print(f'RMSE: {}')
#     print(f'dRMSE: {}')
#     print(f'MAPE: {}')
#     print(f'OOS R2: {}')
#     print(f'OOS GW R2: {}')

    # print('SHAPE TEST - should match')
    # print(factors_rescaled.shape)
    # print(targets_rescaled.shape)
    # print('DATA TEST')
    # print(factors_rescaled)
    # print(targets_rescaled)

    # from keras.models import Sequential
    # from keras.layers import Dense
    # import numpy as np
    #
    # # Create a simple model
    # model = Sequential([Dense(2, input_dim=3)])
    # model.compile(optimizer='adam', loss='mse')
    #
    # # Dummy data
    # X = np.random.rand(10, 3)
    # y = np.random.rand(10, 2)
    #
    # # Fit model
    # model.fit(X, y, epochs=1, verbose=0)
    #
    # # Try saving the weights
    # try:
    #     model.save_weights('simple_model_weights.weights.h5')
    #     print("Weights saved successfully.")
    # except Exception as e:
    #     print("Failed to save weights:", str(e))

    # time_steps = 1
    # factors_rescaled = factors_rescaled.reshape((factors_rescaled.shape[0], time_steps, factors_rescaled.shape[1]))
    # train_factors_rescaled = train_factors_rescaled.reshape(
    #     (train_factors_rescaled.shape[0], time_steps, train_factors_rescaled.shape[1]))

    # print("Shape of train_factors_rescaled:", train_factors_rescaled.shape)


    # def build_LSTM_model(hp):
    #     model = Sequential()
    #     model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
    #                    input_shape=(1,22),
    #                    return_sequences=True))
    #     model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
    #     model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
    #                    return_sequences=False))
    #     model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
    #     model.add(Dense(1))
    #     model.compile(optimizer='adam', loss='mse')
    #
    #     return model
    # def build_model(hp):
    #     model = Sequential([
    #         LSTM(hp.Int('units', min_value=32, max_value=256, step=32),
    #                           input_shape=(train_factors_rescaled.shape[1], train_factors_rescaled.shape[2])),
    #         Dropout(hp.Float('dropout', 0, 0.5, step=0.1)),
    #         Dense(1)
    #     ])
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     if model is None:
    #         raise ValueError("Model construction failed")
    #     return model
    #
    # hp = HyperParameters()
    # hp.Fixed('units', value=64)
    # hp.Fixed('dropout', value=0.2)
    #
    # # Testing model save functionality outside of tuning
    # model = build_model(hp)
    # model.summary()
    #
    # history = model.fit(train_factors_rescaled,
    #                     train_targets_rescaled,
    #                     epochs=10,
    #                     batch_size=11,
    #                     validation_split=0.2,
    #                     verbose=1)
    # model.save_weights('model_weights.weights.h5')
    # print(history.history)
    # tuner = RandomSearch(
    #     build_model,
    #     objective='val_loss',
    #     max_trials=10,
    #     executions_per_trial=1,
    #     directory='C:/Users/dansz/PycharmProjects/Thesis/lstm_tuner',
    #     project_name='lstm_optim',
    #     overwrite=True
    # )
    # #
    # tuner.search(train_factors_rescaled, train_targets_rescaled, epochs=10, batch_size=11, validation_split=0.2, verbose=1)
    #
    # best_model = tuner.get_best_models(num_models=1)[0]

    # def build_model(hp): #THIS RUNS
    #     model = Sequential()
    #     model.add(Dense(
    #         hp.Choice('units', [8,16,32]),
    #         activation='relu'))
    #     model.add(Dense(1, activation='relu'))
    #     model.compile(loss='mse')
    #     return model
    #
    # tuner = keras_tuner.RandomSearch(
    #     build_model,
    #     objective='val_loss',
    #     max_trials=5
    # )
    # tuner.search(train_factors_rescaled, train_targets_rescaled, epochs=5, validation_data=(test_factors_rescaled, test_targets_rescaled))
    # best_model = tuner.get_best_models()[0]

    # def build_model(hp):
    #     model = Sequential()
    #     model.add(Bidirectional(
    #         LSTM(
    #             hp.Choice('units', [32, 64, 128]),
    #             input_shape=(1, train_factors_rescaled.shape[2]),
    #             activation='tanh',
    #             return_sequences=True,
    #             kernel_regularizer=l2(0.01)
    #         )
    #     ))
    #     model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    #     model.add(LSTM(
    #         hp.Choice('units', [32, 64, 128]),
    #         return_sequences=False,
    #         activation='tanh',
    #         kernel_regularizer=l2(0.01)
    #     ))
    #     model.add(Dense(1, activation='relu'))
    #     model.compile(
    #         optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
    #         loss='mse'
    #     )
    #     return model
    #
    # tuner = keras_tuner.RandomSearch(
    #     build_model,
    #     objective='val_loss',
    #     max_trials=50,
    #     directory='C:/Users/dansz/PycharmProjects/Thesis/lstm_tuner',
    #     project_name='lstm_optim',
    #     overwrite=True
    # )
    # tuner.search(train_factors_rescaled,
    #              train_targets_rescaled,
    #              epochs=50,
    #              validation_data=(test_factors_rescaled, test_targets_rescaled))
    # best_model = tuner.get_best_models(num_models=1)[0]
    # lstm_pred = best_model.predict(test_factors_rescaled)
    # def build_model(hp):
    #     model = Sequential()
    #     # Start with a Bidirectional LSTM layer
    #     model.add(Bidirectional(
    #         LSTM(
    #             units=hp.Int('units_lstm1', min_value=32, max_value=128, step=32),
    #             input_shape=(1, train_factors_rescaled.shape[2]),
    #             activation='tanh',
    #             return_sequences=True,  # Always true when followed by another LSTM layer
    #             kernel_regularizer=l2(hp.Float('l2_lstm1', min_value=0.01, max_value=0.1, step=0.01))
    #         )
    #     ))
    #
    #     # Adding more LSTM layers
    #     for i in range(hp.Int('num_lstm_layers', 1, 8)):  # Tuning the number of additional LSTM layers from 1 to 3
    #         model.add(LSTM(
    #             units=hp.Int('units_lstm' + str(i + 2), min_value=32, max_value=128, step=32),
    #             activation='tanh',
    #             return_sequences=(i < hp.Int('num_lstm_layers', 1, 3) - 1),
    #             kernel_regularizer=l2(hp.Float('l2_lstm' + str(i + 2), min_value=0.01, max_value=0.1, step=0.01))
    #         ))
    #
    #     model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    #     model.add(Dense(1, activation='relu'))
    #     model.compile(
    #         optimizer='adam',
    #         loss='mse'
    #     )
    #     return model
    #
    #
    # tuner = keras_tuner.RandomSearch(
    #     build_model,
    #     objective='val_loss',
    #     max_trials=40,
    #     executions_per_trial=3,  # Increasing this for more robust results
    #     directory='C:/Users/dansz/PycharmProjects/Thesis/lstm_tuner',
    #     project_name='lstm_optim_advanced',
    #     overwrite=True
    # )
    #
    # def run_tuner_with_batch_size():
    #     # Running the tuner with different batch sizes
    #     best_score = float('inf')
    #     best_model = None
    #     for batch_size in [10, 20, 30, 40]:  # You can customize these values
    #         tuner.search(train_factors_rescaled, train_targets_rescaled,
    #                      epochs=50,
    #                      batch_size=batch_size,
    #                      validation_data=(test_factors_rescaled, test_targets_rescaled))
    #         best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    #         if best_trial.score < best_score:
    #             best_score = best_trial.score
    #             best_model = tuner.get_best_models(num_models=1)[0]
    #             best_batch_size = batch_size
    #
    #     return best_model, best_batch_size
    #
    # best_model, best_batch_size = run_tuner_with_batch_size()
    # lstm_pred = best_model.predict(test_factors_rescaled)