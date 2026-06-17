if __name__ == "__main__":
print("Running Standalone USD–GBP Forecasting Script")


data = load_data()
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]


train_lag = train.copy()
train_lag['lag1'] = train_lag['rate'].shift(1)
train_lag.dropna(inplace=True)


X = sm.add_constant(train_lag['lag1'])
y = train_lag['rate']
ols = sm.OLS(y, X).fit()


preds = []
last_val = train['rate'].iloc[-1]
for _ in range(len(test)):
p = ols.predict([1, last_val])[0]
preds.append(p)
last_val = p


rmse = np.sqrt(mean_squared_error(test['rate'][:len(preds)], preds))
print("OLS RMSE:", rmse)


arima = ARIMA(train['rate'], order=(1,1,1)).fit()
arima_forecast = arima.forecast(steps=len(test))
rmse_arima = np.sqrt(mean_squared_error(test['rate'], arima_forecast))
print("ARIMA RMSE:", rmse_arima)


returns = 100 * np.log(data['rate'] / data['rate'].shift(1)).dropna()
garch = arch_model(returns, vol='Garch', p=1, q=1)
garch_fit = garch.fit(disp='off')
print(garch_fit.summary())


print("Script execution completed successfully")