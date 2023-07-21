import streamlit as st
import pandas as pd
import yfinance as yf
import wbdata
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Function to fetch historical stock prices
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']

# Function to fetch historical stock prices for a list of tickers
def get_portfolio_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

# Function to fetch GDP and Inflation data from World Bank API for a country
def fetch_economic_data(country_code, start_date, end_date):
    indicators = {"NY.GDP.MKTP.CD": "GDP", "FP.CPI.TOTL.ZG": "Inflation Rate"}
    data_date_range = {"date": (start_date, end_date)}

    data = wbdata.get_dataframe(indicators=indicators, data_date=data_date_range, country=country_code, convert_date=True)

    return data[["GDP", "Inflation Rate"]].reset_index()

#Function to calculate future portfolio returns and risk
def calculate_portfolio_returns_risk(returns, weights, future_returns):
    # Calculate portfolio returns
    portfolio_returns = future_returns.dot(weights)

    # Calculate portfolio risk
    portfolio_cov_matrix = returns.cov()
    portfolio_risk = np.sqrt(weights.T.dot(portfolio_cov_matrix).dot(weights))

    return portfolio_returns, portfolio_risk

# Function to train and predict using Linear Regression
def predict_linear_regression(stock_data, num_years):
    X = pd.DataFrame({'Days': range(len(stock_data))})
    y = stock_data.values
    model = LinearRegression()
    model.fit(X, y)
    future_days = len(stock_data) + int(num_years * 365)
    future_X = pd.DataFrame({'Days': range(len(stock_data), future_days)})
    future_prices = model.predict(future_X)
    return pd.Series(future_prices, index=future_X.index)

# Function to train and predict using Random Forest
def predict_random_forest(stock_data, num_years):
    X = pd.DataFrame({'Days': range(len(stock_data))})
    y = stock_data.values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    future_days = len(stock_data) + int(num_years * 365)
    future_X = pd.DataFrame({'Days': range(len(stock_data), future_days)})
    future_prices = model.predict(future_X)
    return pd.Series(future_prices, index=future_X.index)

# Function to train and predict using Gradient Boosting
def predict_gradient_boosting(stock_data, num_years):
    X = pd.DataFrame({'Days': range(len(stock_data))})
    y = stock_data.values
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    future_days = len(stock_data) + int(num_years * 365)
    future_X = pd.DataFrame({'Days': range(len(stock_data), future_days)})
    future_prices = model.predict(future_X)
    return pd.Series(future_prices, index=future_X.index)

# Function to train and predict using Support Vector Regression
def predict_svr(stock_data, num_years):
    X = pd.DataFrame({'Days': range(len(stock_data))})
    y = stock_data.values
    model = SVR(kernel='linear')
    model.fit(X, y)
    future_days = len(stock_data) + int(num_years * 365)
    future_X = pd.DataFrame({'Days': range(len(stock_data), future_days)})
    future_prices = model.predict(future_X)
    return pd.Series(future_prices, index=future_X.index)

# Function to determine the best model based on R-squared value
def determine_best_model(stock_data, num_years):
    models = {
        'Linear Regression': predict_linear_regression(stock_data, num_years),
        'Random Forest': predict_random_forest(stock_data, num_years),
        'Gradient Boosting': predict_gradient_boosting(stock_data, num_years),
        'Support Vector Regression': predict_svr(stock_data, num_years)
    }

    r_squared_values = {}
    for model_name, predictions in models.items():
        X = pd.DataFrame({'Days': range(len(stock_data))})
        y = stock_data.values
        model = LinearRegression()
        model.fit(X, y)
        r_squared_values[model_name] = model.score(X, y)

    best_model = max(r_squared_values, key=r_squared_values.get)
    return best_model, models[best_model]

# App Setup
st.title('Financial Model Web App')
st.write('Welcome to the Financial Model Web App!')

# User Input
st.sidebar.title('User Input')

# Input the stock tickers and allocation percentages
selected_stocks = []
allocation_percentages = []

for i in range(1, 6):
    stock_ticker = st.sidebar.text_input(f'Enter Stock Ticker {i}', key=f'stock_ticker_{i}')
    allocation = st.sidebar.number_input(f'Enter Allocation Percentage for {stock_ticker}', min_value=0.0, max_value=1.0, value=0.2, key=f'allocation_{i}')
    
    if stock_ticker:
        selected_stocks.append(stock_ticker.upper())
        allocation_percentages.append(allocation)

# Normalize allocation percentages to sum to 1.0
allocation_percentages = np.array(allocation_percentages)
allocation_percentages /= allocation_percentages.sum()

# Calculate the start date and end date for one year from the current month back
current_date = datetime.now()
start_date = current_date - timedelta(days=365)
end_date = current_date

# Fetch historical stock prices for selected stocks in the portfolio
if not selected_stocks:
    st.write('Please enter at least one stock ticker.')
else:
    portfolio_data = get_portfolio_data(selected_stocks, start_date, end_date)

    # Check if portfolio_data contains data for the selected stock(s)
    if portfolio_data.empty:
        st.write('No data found for the selected stock(s). Please check your input.')

    else:
        # Show Historical Stock Prices for the selected stocks in the portfolio and the portfolio
        st.header('Historical Stock Prices for the Portfolio and the Portfolio')
        plt.figure(figsize=(10, 6))
        for stock in selected_stocks:
            plt.plot(portfolio_data.index, portfolio_data[stock], label=stock)
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.title('Historical Stock Prices')
        plt.legend()
        st.pyplot()

# Input GDP and Inflation Values
gdp_value = st.sidebar.number_input('Enter GDP Value', value=3.0, step=0.1)
inflation_value = st.sidebar.number_input('Enter Inflation Rate', value=2.0, step=0.1)

# Calculate the number of years
num_years = (end_date - start_date).days / 365.0

# Fetch historical stock prices for selected stocks in the portfolio
if not selected_stocks:
    st.write('Please select at least one stock for the portfolio.')
else:
    portfolio_data = get_portfolio_data(selected_stocks, start_date, end_date)

    # Check if portfolio_data contains data for the selected stock(s)
    if portfolio_data.empty:
        st.write('No data found for the selected stock(s). Please check your input.')

    else:
        # Show Historical Stock Prices for the selected stocks in the portfolio
        st.header('Historical Stock Prices for the Portfolio')
        st.line_chart(portfolio_data)

# Function to calculate future returns based on the number of years
def calculate_future_returns(stock_data, years):
    returns = stock_data.pct_change().dropna()
    future_returns = (1 + returns.mean()) ** years - 1
    return future_returns

# Show Historical Stock Prices for the selected stocks
st.header('Historical Stock Prices')
st.line_chart(portfolio_data)

# Model Training and Best Model Selection
best_model_name, best_model_predictions = determine_best_model(portfolio_data[selected_stocks[0]], num_years)

# Show Future Returns and Best Model for the selected stocks
st.header('Future Returns and Best Model Based on Selected Number of Years')
returns_df = pd.DataFrame()
for stock in selected_stocks:
    future_returns = calculate_future_returns(portfolio_data[stock], num_years)
    returns_df[stock] = future_returns

st.bar_chart(returns_df)
st.write(f'Best Model for {selected_stocks[0]}: {best_model_name}')

# Calculate Portfolio Returns and Risk
returns_df['Portfolio'] = returns_df.mean(axis=1)
portfolio_returns, portfolio_risk = calculate_portfolio_returns_risk(returns_df, weights)

# Display Portfolio Cumulative Return, Risk, and Allocation
st.header('Portfolio Metrics as at the End Date')
st.write(f'Portfolio Cumulative Return: {portfolio_returns.sum():.2%}')
st.write(f'Portfolio Risk: {portfolio_risk:.2%}')

# Check if weights are defined before displaying allocation
if 'weights' in locals():
    st.write(f'Portfolio Allocation: {", ".join([f"{stock}: {weight:.2%}" for stock, weight in zip(selected_stocks, weights)])}')
else:
    st.write('Please set portfolio allocation using the sliders.')


# Visualization of GDP and Inflation
st.header('GDP and Inflation Visualization')
gdp_inflation_df = pd.DataFrame({'GDP': [gdp_value], 'Inflation Rate': [inflation_value]})
st.bar_chart(gdp_inflation_df)

# Visualize Best Model Predictions
st.header('Model Predictions for Future Stock Prices (Best Model)')
st.line_chart(pd.DataFrame({'Predictions': best_model_predictions}))