import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="USD–GBP Forecasting App", layout="wide")

st.title("USD–GBP Exchange Rate Forecasting")
st.write("OLS, ARIMA and GARCH based interactive forecasting tool")

st.cache_data
def load_data():
    data = pd.read_csv("/Users/sshub/GP/GBP_USD.csv")
    data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")

    data.set_index('Date', inplace=True)
    data = data[['Price']].dropna()
    data.columns = ['rate']
    return data

data = load_data()
st.subheader("Raw Exchange Rate Data")
st.line_chart(data)

train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

model_choice = st.selectbox("Choose Model", ["OLS", "ARIMA", "GARCH"])
forecast_horizon = st.slider("Forecast Horizon", 5, 50, 20)

if model_choice == "OLS":
    st.subheader("OLS Model")

    train_lag = train.copy()
    train_lag['lag1'] = train_lag['rate'].shift(1)
    train_lag.dropna(inplace=True)

    X = sm.add_constant(train_lag['lag1'])
    y = train_lag['rate']

    ols_model = sm.OLS(y, X).fit()
    st.text(ols_model.summary())

    last_val = train['rate'].iloc[-1]
    forecasts = []

    for _ in range(forecast_horizon):
        pred = ols_model.predict([1, last_val])[0]
        forecasts.append(pred)
        last_val = pred

    forecast_index = pd.date_range(
        start=test.index[0],
        periods=forecast_horizon,
        freq='B'
    )

    forecast_series = pd.Series(forecasts, index=forecast_index)

elif model_choice == "ARIMA":
    st.subheader("ARIMA Model")
    p = st.slider("AR order (p)", 0, 5, 1)
    d = st.slider("Difference (d)", 0, 2, 1)
    q = st.slider("MA order (q)", 0, 5, 1)
    
    
    arima_model = ARIMA(train['rate'], order=(p, d, q)).fit()
    st.text(arima_model.summary())
    
    
    forecast = arima_model.forecast(steps=forecast_horizon)
    forecast_series = forecast

elif model_choice == "GARCH":
    st.subheader("GARCH Model")
    returns = 100 * np.log(train['rate'] / train['rate'].shift(1)).dropna()
    
    
    garch = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp="off")
    st.text(garch_fit.summary())
    
    
    forecasts = garch_fit.forecast(horizon=forecast_horizon)
    vol_forecast = np.sqrt(forecasts.variance.values[-1])
    forecast_series = pd.Series(vol_forecast, name="Forecasted Volatility")

st.subheader("Forecast Results")
fig, ax = plt.subplots()
ax.plot(train.index, train['rate'], label='Train')
ax.plot(test.index, test['rate'], label='Test')
if model_choice != "GARCH":
    ax.plot(forecast_series.index, forecast_series.values, label='Forecast')
    ax.legend()
    st.pyplot(fig)