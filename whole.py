import yfinance as yf
import seaborn as sns
import pandas  as pd
import numpy as np
import pandas_market_calendars as mcal
import pytz

from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from FMNM.BS_pricer import BS_pricer
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process

import scipy.stats as ss
from scipy.integrate import quad
from typing import List, Optional
from scipy.stats import lognorm
from scipy.stats import norm
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

%matplotlib inline

from utils.gbm import GBMAnalyzer
from utils.dataFetcher import OptionsDataFetcher as odf

fetcher = odf('TSLA')
stock_data = fetcher.fetch_stock_data("2022-01-01", "2025-05-18")
opts_data = fetcher.fetch_options_chain()
opts_data_p = fetcher.fetch_options_chain(puts_only = True)
opts_data_c = fetcher.fetch_options_chain(calls_only = True)
# fetcher.save(stock_data, opts_data, path="retrivedData", save_stock=True, save_options=True)

nyse = mcal.get_calendar('NYSE')
ny_tz = pytz.timezone('America/New_York')

def trading_day_after_n_days(start_date, n_days):
    """
    Returns the first NYSE trading day on or after (or on or before, if n_days < 0)
    the date shifted by n_days. Always returns a trading day >= or <= the shifted date.
    """
    # Shift the date by n_days
    target_date = (start_date + pd.Timedelta(days=n_days)).normalize()

    # Define search window (inclusive)
    if n_days >= 0:
        search_start = target_date
        search_end = target_date + pd.Timedelta(days=15)
    else:
        search_start = target_date - pd.Timedelta(days=15)
        search_end = target_date

    # Get valid trading days in the range
    schedule = nyse.valid_days(start_date=search_start, end_date=search_end)

    if not schedule.empty:
        selected_date = schedule[0] if n_days >= 0 else schedule[-1]
        return selected_date.tz_convert(ny_tz)
    else:
        raise ValueError(f"No trading day found in the expected window around {target_date}")


end_date_train = pd.Timestamp(2023, 1, 3, tz="America/New_York")
t = 1
days = 365
start_date_train = trading_day_after_n_days(end_date_train, -t*days)
start_date_test = trading_day_after_n_days(end_date_train,0)
end_date_test = trading_day_after_n_days(end_date_train,t*days + 1)


train_data = stock_data.loc[start_date_train:end_date_train]
test_data = stock_data.loc[end_date_train:end_date_test]

prices = np.array(train_data['Close'])
actual = np.array(test_data['Close'])
dates = np.array(test_data.index)
pred = predicted(prices, t)
sims = simulate(prices, t, 1000)


def predicted_gbm(S0, mu, sigma, t_days):
    """
    Simulate GBM path starting from S0 for t_days days.
    Returns a vector of simulated prices.
    """
    dt = 1/252
    prices = [S0]
    for _ in range(t_days):
        Z = np.random.normal()
        S_prev = prices[-1]
        S_new = S_prev * np.exp((mu - 0.5 * sigma**2)*dt + sigma * np.sqrt(dt) * Z)
        prices.append(S_new)
    return np.array(prices[1:])  # exclude initial price

def simulate_gbm_paths(S0, mu, sigma, t_days, n_paths):
    """
    Simulate n_paths GBM paths over t_days.
    Returns a (n_paths x t_days) array.
    """
    paths = np.zeros((n_paths, t_days))
    for i in range(n_paths):
        paths[i] = predicted_gbm(S0, mu, sigma, t_days)
    return paths

log_returns = np.log(prices[1:] / prices[:-1])
mu_daily = np.mean(log_returns)
sigma_daily = np.std(log_returns)

mu = mu_daily * 252
sigma = sigma_daily * np.sqrt(252)

S0 = prices[-1]  # last known price in training data
t_days = len(test_data)  # number of days to simulate ahead

# simulate
paths = simulate_gbm_paths(S0, mu, sigma, t_days, 1000)

sim_mean = np.mean(paths, axis=0)

plt.plot(dates, actual, label='Actual')
for path in paths:
    plt.plot(dates, path, color='grey', alpha=0.1)
plt.plot(dates, sim_mean, label='Mean Simulation')
plt.legend()
plt.show()