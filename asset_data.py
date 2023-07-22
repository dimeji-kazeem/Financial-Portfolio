import yfinance as yf
import pandas as pd
import wbdata

# Function to fetch historical stock data from Yahoo Finance and save to CSV
def download_asset_data(asset_symbols, start_date, end_date, output_csv):
    asset_data = pd.DataFrame()
    for symbol in asset_symbols:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        asset_data[symbol] = stock_data['Adj Close']

    asset_data.to_csv('asset_data.csv', index=True)

# Function to fetch World Bank data and save to CSV
def download_world_bank_data(country, indicators, start_date, end_date, worldbank_csv):
    indicators_data = wbdata.get_dataframe(indicators=indicators, data_date=start_date, end_date=end_date, country=country)
    indicators_data.to_csv('worldbank.csv', index=True)

# Example usage:
asset_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NAT', 'AMD', 'AACT', 'META', 'AMZN']
start_date = '2018-01-01'
end_date = '2022-12-31'
output_asset_csv = 'data/assets_data.csv'
download_asset_data(asset_symbols, start_date, end_date, output_asset_csv)

country = 'USA'
indicators = {'NY.GDP.MKTP.CD': 'GDP', 'FP.CPI.TOTL.ZG': 'Inflation', 'FR.INR.RINR': 'Interest_Rate'}
output_world_bank_csv = 'data/world_bank_data.csv'
download_world_bank_data(country, indicators, start_date, end_date, output_world_bank_csv)