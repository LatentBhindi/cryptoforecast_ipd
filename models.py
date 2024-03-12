import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU
from datetime import datetime, timedelta
from tcn import TCN
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


def lstm(df_scaled, X_train, y_train):
    

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model


def gru(df_scaled, X_train, y_train):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(GRU(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model

def tcn(df_scaled, X_train, y_train):
    model = Sequential()
    model.add(TCN(input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(TCN(return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model

def cnn_lstm(df_scaled, X_train, y_train):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    return model

def xg(df_scaled, X_train, y_train):
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    return model


def svr(df_scaled, X_train, y_train):
    model = SVR(kernel='rbf')  
    model.fit(X_train, y_train)

    return model

def arima(df_scaled, X_train, y_train):
    p = 2
    d = 1
    q = 1
    model = ARIMA(y_train, order=(p, d, q))
    model_fit = model.fit()

    return model_fit

def garch(df_scaled, X_train, y_train):
    model = arch_model(y_train, vol='GARCH', p=1, q=1)
    model_fit = model.fit(disp='off')

    return model_fit

def metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {mse,mae,r2}

def metrics_garch(model, X_test, y_test):
    y_pred = model.forecast(horizon=len(y_test))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {mse,mae,r2}

def metrics_arima(model, X_test, y_test):
    y_pred = model.forecast(steps=len(y_test))
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {mse,mae,r2}




