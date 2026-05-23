import pandas as pd
import numpy as np
from scipy import stats


class CorrelationAnalyzer:
    @staticmethod
    def pearson_correlation(df1, df2, col1="close", col2="avg_compound"):
        merged = pd.merge(
            df1[["date", col1]],
            df2[["date", col2]],
            on="date", how="inner", suffixes=("_stock", "_sentiment")
        )
        corr, p_value = stats.pearsonr(merged[col1], merged[col2])
        return {
            "correlation": corr,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "n_observations": len(merged),
        }

    @staticmethod
    def cross_correlation(df1, df2, col1="close", col2="avg_compound", max_lag=10):
        merged = pd.merge(
            df1[["date", col1]],
            df2[["date", col2]],
            on="date", how="inner"
        )
        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shifted = merged[col1].shift(-lag)
            elif lag > 0:
                shifted = merged[col1].shift(lag)
            else:
                shifted = merged[col1]
            valid = pd.concat([shifted, merged[col2]], axis=1).dropna()
            if len(valid) > 30:
                corr, _ = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                correlations[lag] = corr
        return correlations

    @staticmethod
    def sentiment_return_correlation(daily_sentiment, stock_returns, sent_col="avg_compound", ret_col="daily_return"):
        merged = pd.merge(
            daily_sentiment[["date", sent_col]],
            stock_returns[["date", ret_col]],
            on="date", how="inner"
        )
        merged["next_day_return"] = merged[ret_col].shift(-1)
        merged = merged.dropna()

        results = {}
        for label, col in [("same_day", ret_col), ("next_day", "next_day_return")]:
            corr, p_val = stats.pearsonr(merged[sent_col], merged[col])
            results[label] = {"correlation": corr, "p_value": p_val}

        results["n_samples"] = len(merged)
        return results

    @staticmethod
    def sector_correlation_matrix(ticker_data, price_col="close"):
        pivot = ticker_data.pivot_table(
            index="date", columns="ticker", values=price_col
        )
        returns = pivot.pct_change().dropna()
        return returns.corr()
