import pandas as pd
import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    @staticmethod
    def summary_statistics(df, price_col="close"):
        return {
            "mean": df[price_col].mean(),
            "median": df[price_col].median(),
            "std": df[price_col].std(),
            "min": df[price_col].min(),
            "max": df[price_col].max(),
            "skewness": df[price_col].skew(),
            "kurtosis": df[price_col].kurtosis(),
            "q1": df[price_col].quantile(0.25),
            "q3": df[price_col].quantile(0.75),
            "volatility": df[price_col].pct_change().std() * np.sqrt(252),
        }

    @staticmethod
    def normality_test(series):
        stat, p_value = stats.shapiro(series.dropna().sample(min(5000, len(series))))
        return {"statistic": stat, "p_value": p_value, "is_normal": p_value > 0.05}

    @staticmethod
    def stationarity_test(series, price_col="close"):
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series[price_col].dropna())
        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05,
            "critical_values": result[4],
        }

    @staticmethod
    def moving_average_convergence(df, short=20, long=50, price_col="close"):
        df = df.copy()
        df["sma_short"] = df[price_col].rolling(window=short).mean()
        df["sma_long"] = df[price_col].rolling(window=long).mean()
        df["ma_crossover"] = np.where(df["sma_short"] > df["sma_long"], "bullish", "bearish")
        return df

    @staticmethod
    def detect_outliers(series, method="iqr", threshold=3):
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            return (series < lower) | (series > upper)
        elif method == "zscore":
            z = np.abs(stats.zscore(series.dropna()))
            return pd.Series(z > threshold, index=series.dropna().index)
        return pd.Series([False] * len(series))

    @staticmethod
    def calculate_var(returns, confidence_level=0.95):
        return {
            "historical_var": returns.quantile(1 - confidence_level),
            "parametric_var": returns.mean() + returns.std() * stats.norm.ppf(1 - confidence_level),
        }

    @staticmethod
    def calculate_cvar(returns, confidence_level=0.95):
        var = returns.quantile(1 - confidence_level)
        return returns[returns <= var].mean()
