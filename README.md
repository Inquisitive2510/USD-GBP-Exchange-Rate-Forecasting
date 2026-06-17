# USD-GBP-Exchange-Rate-Forecasting
Interactive econometric forecasting application for USD–GBP exchange rates using OLS, ARIMA, and GARCH models with Streamlit.
# USD–GBP Exchange Rate Forecasting Using OLS, ARIMA, and GARCH

## Overview

This project develops an interactive econometric forecasting application for USD–GBP exchange rates using:

- Ordinary Least Squares (OLS)
- Autoregressive Integrated Moving Average (ARIMA)
- Generalized Autoregressive Conditional Heteroskedasticity (GARCH)

The application is built using Streamlit and allows users to:

- Visualize exchange rate data
- Select forecasting models
- Tune model parameters
- Generate forecasts
- Analyze volatility

---

## Features

### OLS Forecasting
- Lag-based autoregressive forecasting
- Recursive future predictions

### ARIMA Forecasting
- User-defined p, d, q parameters
- Time-series forecasting

### GARCH Forecasting
- Volatility estimation
- Conditional variance forecasting

### Interactive Dashboard
- Built with Streamlit
- Dynamic visualizations
- Adjustable forecast horizon

---

## Dataset

Historical GBP/USD exchange rate data obtained from Investing.com.

---

## Project Structure

```text
USD-GBP-Exchange-Rate-Forecasting/
│
├── app.py
├── standalone.py
├── GBP_USD.csv
├── requirements.txt
├── README.md
├── .gitignore
│
├── report/
│   └── Group_Project.pdf
