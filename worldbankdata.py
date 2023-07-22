import yfinance as yf
import pandas as pd
import wbdata
from datetime import datetime, timedelta

# Function to fetch World Bank data and save to CSV for multiple countries
def download_world_bank_data(countries, indicators, start_date, end_date):
    for country in countries:
        # Fetch the data for the desired end date
        data_date = datetime(end_date.year, end_date.month, end_date.day)

        indicators_data = wbdata.get_dataframe(indicators=indicators, data_date=data_date, country=country)
        # Filter the dataframe to include data up to the specified end_date
        indicators_data = indicators_data[indicators_data.index <= end_date]

        output_csv = f"world_bank_data_{country}.csv"
        indicators_data.to_csv(output_csv, index=True)

countries = ['USA', 'UK', 'Canada', 'China', 'Germany', 'Hong Kong']
start_date = '2018-01-01'
end_date = datetime(2022, 12, 31)

indicators = {'NY.GDP.MKTP.CD': 'GDP', 'FP.CPI.TOTL.ZG': 'Inflation', 'FR.INR.RINR': 'Interest_Rate'}
download_world_bank_data(countries, indicators, start_date, end_date)