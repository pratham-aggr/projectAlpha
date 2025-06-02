import gymnasium as gym
from gymnasium import spaces
from datetime import timedelta
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import norm
from utils.dataFetcher import OptionsDataFetcher as odf


class OptionTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, asset_ticker, portfolio, end_date_str, yrs):
        super().__init__()
        fetcher = odf(asset_ticker)

        start_date = pd.Timestamp(end_date_str) - timedelta(days=yrs * 365)
        start_date_str = start_date.strftime("%Y-%m-%d")

        stock_data = fetcher.fetch_stock_data(start_date_str, end_date_str)
        opts_data = fetcher.fetch_options_chain()
        opts_data_p = fetcher.fetch_options_chain(puts_only=True)
        opts_data_c = fetcher.fetch_options_chain(calls_only=True)

        if 'Date' in stock_data.columns and not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            stock_data.sort_index(inplace=True)

        stock_data['log_return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
        stock_data['hist_vol'] = stock_data['log_return'].rolling(window=20).std() * np.sqrt(252)
        stock_data.dropna(inplace=True)

        self.stock_data = stock_data
        self.options_data = opts_data
        self.options_data_p = opts_data_p
        self.options_data_c = opts_data_c

        self.current_step = 0
        self.max_steps = min(252, len(stock_data)-1)
        self.steps_left = self.max_steps

        self.asset_ticker = asset_ticker
        self.portfolio = portfolio

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.state = self._get_state()

    def _get_state(self) -> List[float]:
        return self.get_observation(self.current_step)

    def get_observation(self, step) -> List[float]:
        if step >= len(self.stock_data):
            step = len(self.stock_data) - 1

        row = self.stock_data.iloc[step]
        current_price = row['Close']
        current_volatility = row['hist_vol']
        time_remaining = (self.steps_left) / self.max_steps

        options_today = self.options_data.copy()
        options_today['moneyness'] = abs(options_today['strike'] - current_price)
        atm_option = options_today.sort_values(by='moneyness').iloc[0]

        expiration = pd.to_datetime(atm_option['expiration']).tz_localize(None)
        today = self.stock_data.index[step]
        if today.tzinfo is not None:
            today = today.tz_localize(None)

        T = max((expiration - today).days / 252, 1/252)

        K = atm_option['strike']
        sigma = current_volatility
        r = 0.043
        option_type = atm_option['type'].lower()

        delta, gamma, vega, theta = self.compute_greeks(current_price, K, T, r, sigma, option_type)

        return [
            current_price,
            current_volatility,
            time_remaining,
            delta,
            gamma,
            vega,
            theta
        ]

    def compute_greeks(self, S, K, T, r, sigma, option_type='call', q=0.0):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2)
                     + q * S * np.exp(-q * T) * norm.cdf(d1))
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1))

        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        return delta, gamma, vega, theta

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        row = self.stock_data.iloc[self.current_step]
        current_price = row['Close']

        options_today = self.options_data.copy()
        options_today['moneyness'] = abs(options_today['strike'] - current_price)
        atm_option = options_today.sort_values(by='moneyness').iloc[0]
        strike = atm_option['strike']

        reward = 0.0
        if action == 0:
            reward = 0.0  # Hold
        elif action == 1:
            reward = self.buy_option('call', current_price, strike)
        elif action == 2:
            reward = self.sell_option('call', current_price, strike)
        elif action == 3:
            reward = self.buy_option('put', current_price, strike)
        elif action == 4:
            reward = self.sell_option('put', current_price, strike)

        self.current_step += 1
        self.steps_left -= 1

        terminated = self.steps_left == 0 or self.current_step >= len(self.stock_data)
        truncated = False  # You can add truncation logic here if needed

        self.state = self._get_state()

        info = {}

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.steps_left = self.max_steps
        self.state = self._get_state()
        info = {}
        return np.array(self.state, dtype=np.float32), info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: {self.state}")

    def buy_option(self, option_type, spot, strike):
        cost = abs(spot - strike) * 0.1  # Simplified cost
        return -cost

    def sell_option(self, option_type, spot, strike):
        income = abs(spot - strike) * 0.1  # Simplified premium
        return income

    # ===== Getters =====
    def get_state(self):
        return self.state

    def get_state_with_labels(self):
        labels = [
            "Current Stock Price",
            "Historical Volatility (20-day annualized)",
            "Normalized Time Remaining",
            "Delta",
            "Gamma",
            "Vega",
            "Theta"
        ]
        return dict(zip(labels, self.state))

    def get_stock_data(self):
        return self.stock_data

    def get_options_data(self):
        return self.options_data

    def get_puts_data(self):
        return self.options_data_p

    def get_calls_data(self):
        return self.options_data_c

    def get_stock_dates(self):
        return self.stock_data.index

    def get_option_expirations(self):
        return pd.to_datetime(self.options_data['expiration']).drop_duplicates().sort_values()

    def get_current_date(self):
        if self.current_step < len(self.stock_data):
            return self.stock_data.index[self.current_step]
        return None

    def get_current_step(self):
        return self.current_step

    def get_max_steps(self):
        return self.max_steps

    def get_steps_left(self):
        return self.steps_left

    def get_asset_ticker(self):
        return self.asset_ticker

    def get_portfolio(self):
        return self.portfolio
