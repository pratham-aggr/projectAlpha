import yfinance as yf
import pandas as pd
import os

class OptionsDataFetcher:
    def __init__(self, ticker: str):
        """
        Initialize the data fetcher with a specific stock ticker.
        """
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def fetch_stock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data between start_date and end_date.
        
        Parameters:
            start_date (str): Format 'YYYY-MM-DD'
            end_date (str): Format 'YYYY-MM-DD'

        Returns:
            DataFrame: Historical OHLCV data
        """
        return self.stock.history(start=start_date, end=end_date)

    def fetch_options_chain(
        self,
        calls_only: bool = False,
        puts_only: bool = False
    ) -> pd.DataFrame:
        """
        Fetch the options chain (calls, puts, or both) across all expiration dates.

        Parameters:
            calls_only (bool): If True, fetch only call options
            puts_only (bool): If True, fetch only put options

        Returns:
            DataFrame: Combined options data based on selection
        """
        expirations = self.stock.options
        options_data = []

        for exp in expirations:
            chain = self.stock.option_chain(exp)

            if calls_only:
                calls = chain.calls.copy()
                calls['type'] = 'call'
                calls['expiration'] = exp
                if 'lastPrice' in calls.columns:
                    calls.rename(columns={'lastPrice': 'lastOptionPrice'}, inplace=True)
                options_data.append(calls)

            elif puts_only:
                puts = chain.puts.copy()
                puts['type'] = 'put'
                puts['expiration'] = exp
                if 'lastPrice' in puts.columns:
                    puts.rename(columns={'lastPrice': 'lastOptionPrice'}, inplace=True)
                options_data.append(puts)

            else:
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls['type'] = 'call'
                puts['type'] = 'put'
                calls['expiration'] = exp
                puts['expiration'] = exp
                if 'lastPrice' in calls.columns:
                    calls.rename(columns={'lastPrice': 'lastOptionPrice'}, inplace=True)
                if 'lastPrice' in puts.columns:
                    puts.rename(columns={'lastPrice': 'lastOptionPrice'}, inplace=True)
                options_data.append(pd.concat([calls, puts]))

        return pd.concat(options_data).reset_index(drop=True)

    def save(
        self,
        stock_data: pd.DataFrame = None,
        options_data: pd.DataFrame = None,
        path: str = ".",
        save_stock: bool = True,
        save_options: bool = True
    ):
        """
        Save stock and/or options data to CSV files in the given directory.

        Parameters:
            stock_data (DataFrame): Historical stock data
            options_data (DataFrame): Options chain data
            path (str): Directory path to save files
            save_stock (bool): Whether to save stock data
            save_options (bool): Whether to save options data
        """
        os.makedirs(path, exist_ok=True)

        if save_stock and stock_data is not None:
            stock_file = os.path.join(path, f"{self.ticker}_stock_data.csv")
            stock_data.to_csv(stock_file)
            print(f"Stock data saved to: {stock_file}")

        if save_options and options_data is not None:
            options_file = os.path.join(path, f"{self.ticker}_options_chain.csv")
            options_data.to_csv(options_file)
            print(f"Options data saved to: {options_file}")

        if not save_stock and not save_options:
            print("Nothing was saved. Both save_stock and save_options were set to False.")
