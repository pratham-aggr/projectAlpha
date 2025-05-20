import numpy as np
import pandas as pd

class GBMAnalyzer:
    """
    A class to analyze historical stock prices using Geometric Brownian Motion (GBM).

    This class computes expected log-price statistics and confidence intervals 
    under the GBM assumption, based on historical closing prices.

    Parameters
    ----------
    stock_data : pandas.DataFrame
        DataFrame with a datetime index and a 'Close' column containing daily prices.

    Raises
    ------
    ValueError
        If 'Close' column is missing from the input DataFrame.
    """

    def __init__(self, stock_data: pd.DataFrame):
        if 'Close' not in stock_data.columns:
            raise ValueError("DataFrame must contain a 'Close' column.")
        self.stock_data = stock_data.sort_index()

    def _get_prices(self, start_date, end_date):
        """
        Extract closing prices for a given date range.

        Parameters
        ----------
        start_date : str or pandas.Timestamp
            Start date of the analysis window (inclusive).
        end_date : str or pandas.Timestamp
            End date of the analysis window (inclusive).

        Returns
        -------
        np.ndarray
            Array of closing prices in the selected range.

        Raises
        ------
        ValueError
            If fewer than 2 data points are found in the range.
        """
        prices = self.stock_data.loc[start_date:end_date, 'Close'].values
        if len(prices) < 2:
            raise ValueError("Not enough data points in selected period.")
        return prices

    @staticmethod
    def log_return(prices):
        """
        Compute daily log returns.

        Parameters
        ----------
        prices : np.ndarray
            Array of daily closing prices.

        Returns
        -------
        np.ndarray
            Array of log returns: log(P_t / P_{t-1}).
        """
        return np.log(prices[1:] / prices[:-1])

    @staticmethod
    def mean_log_return(prices):
        """
        Compute the average daily log return.

        Parameters
        ----------
        prices : np.ndarray
            Array of daily closing prices.

        Returns
        -------
        float
            Mean of the daily log returns.
        """
        return np.mean(GBMAnalyzer.log_return(prices))

    @staticmethod
    def sigma(prices):
        """
        Compute the sample standard deviation of daily log returns.

        Parameters
        ----------
        prices : np.ndarray
            Array of daily closing prices.

        Returns
        -------
        float
            Sample standard deviation of the log returns.
        """
        return np.std(GBMAnalyzer.log_return(prices), ddof=1)

    def mean_log_s(self, prices, t, S0):
        """
        Compute expected log-price E[log(S(t))] under GBM.

        Parameters
        ----------
        prices : np.ndarray
            Array of daily closing prices.
        t : float
            Time horizon in years.
        S0 : float
            Initial stock price at time 0.

        Returns
        -------
        float
            Expected log-price at time t.
        """
        mu_daily = self.mean_log_return(prices)
        sigma_daily = self.sigma(prices)
        mu_annual = mu_daily * 252
        sigma_annual = sigma_daily * np.sqrt(252)
        drift = (mu_annual - 0.5 * sigma_annual**2) * t
        return np.log(S0) + drift

    def std_log_s(self, prices, t):
        """
        Compute the standard deviation of log-price under GBM.

        Parameters
        ----------
        prices : np.ndarray
            Array of daily closing prices.
        t : float
            Time horizon in years.

        Returns
        -------
        float
            Standard deviation of log(S(t)).
        """
        sigma_daily = self.sigma(prices)
        sigma_annual = sigma_daily * np.sqrt(252)
        return sigma_annual * np.sqrt(t)

    def gbm_log_stats(self, start_date, end_date):
        """
        Compute expected value and standard deviation of log-price under GBM.

        Parameters
        ----------
        start_date : str or pandas.Timestamp
            Start date of the analysis window (inclusive).
        end_date : str or pandas.Timestamp
            End date of the analysis window (inclusive).

        Returns
        -------
        mean : float
            Expected log-price at time t.
        std : float
            Standard deviation of log-price at time t.

        Raises
        ------
        ValueError
            If fewer than 2 data points are in the selected period.
        """
        prices = self._get_prices(start_date, end_date)
        S0 = prices[0]
        t_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        t_years = t_days / 365.0
        mean = self.mean_log_s(prices, t_years, S0)
        std = self.std_log_s(prices, t_years)
        return mean, std

    def confidence_interval(self, start_date, end_date, num_std=2):
        """
        Compute a confidence interval for the terminal price under GBM.

        The interval is based on log-normal distribution assumptions and is
        given by [exp(mean - k*std), exp(mean + k*std)] where k = num_std.

        Parameters
        ----------
        start_date : str or pandas.Timestamp
            Start date of the analysis window (inclusive).
        end_date : str or pandas.Timestamp
            End date of the analysis window (inclusive).
        num_std : float, optional
            Number of standard deviations for the confidence interval (default is 2).

        Returns
        -------
        tuple of float
            (Lower bound, Upper bound) of the confidence interval for S(t).
        """
        mean, std = self.gbm_log_stats(start_date, end_date)
        lower = mean - num_std * std
        upper = mean + num_std * std
        return np.exp(lower), np.exp(upper)
