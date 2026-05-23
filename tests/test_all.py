import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from analysis.sentiment import SentimentAnalyzer
from analysis.statistics import StatisticalAnalyzer
from analysis.correlation import CorrelationAnalyzer
from models.predictor import PricePredictor
from visualization.charts import ChartBuilder
from tests.conftest import make_sample_stock_data, make_sample_news_data


class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_textblob_positive(self):
        result = self.analyzer.analyze_textblob("This is great and amazing!")
        assert result["polarity"] > 0

    def test_textblob_negative(self):
        result = self.analyzer.analyze_textblob("This is terrible and awful.")
        assert result["polarity"] < 0

    def test_vader_returns_all_keys(self):
        result = self.analyzer.analyze_vader("This is a test.")
        for key in ["compound", "positive", "negative", "neutral"]:
            assert key in result

    def test_analyze_news_adds_columns(self):
        df = make_sample_news_data()
        result = self.analyzer.analyze_news(df)
        for col in ["tb_polarity", "vader_compound", "sentiment_label"]:
            assert col in result.columns

    def test_aggregate_daily(self):
        df = make_sample_news_data()
        analyzed = self.analyzer.analyze_news(df)
        daily = self.analyzer.aggregate_daily(analyzed)
        assert "date" in daily.columns
        assert "avg_compound" in daily.columns
        assert len(daily) > 0


class TestStatisticalAnalyzer:
    def test_summary_statistics(self):
        df = make_sample_stock_data()
        stats = StatisticalAnalyzer.summary_statistics(df)
        for key in ["mean", "median", "std", "min", "max"]:
            assert key in stats

    def test_stationarity_test(self):
        df = make_sample_stock_data()
        result = StatisticalAnalyzer.stationarity_test(df)
        assert "adf_statistic" in result
        assert "p_value" in result

    def test_var_calculation(self):
        returns = pd.Series(np.random.randn(1000) * 0.02)
        var = StatisticalAnalyzer.calculate_var(returns)
        assert "historical_var" in var
        assert "parametric_var" in var
        assert var["historical_var"] < 0


class TestCorrelationAnalyzer:
    def test_pearson_correlation(self):
        df1 = make_sample_stock_data()
        dates = df1["date"].iloc[:30].reset_index(drop=True)
        df2 = pd.DataFrame({"date": dates, "avg_compound": np.random.randn(30) * 0.5})
        result = CorrelationAnalyzer.pearson_correlation(df1.iloc[:30], df2)
        assert "correlation" in result
        assert -1 <= result["correlation"] <= 1

    def test_sector_correlation_matrix(self):
        df1 = make_sample_stock_data()
        df1["ticker"] = "AAPL"
        df2 = make_sample_stock_data()
        df2["ticker"] = "MSFT"
        multi = pd.concat([df1, df2], ignore_index=True)
        corr = CorrelationAnalyzer.sector_correlation_matrix(multi)
        assert "AAPL" in corr.columns
        assert "MSFT" in corr.columns


class TestPricePredictor:
    def test_linear_trend(self):
        df = make_sample_stock_data()
        predictor = PricePredictor()
        result = predictor.linear_trend_prediction(df)
        assert "slope" in result
        assert "r_squared" in result
        assert "predictions" in result

    def test_moving_average_forecast(self):
        series = pd.Series(np.random.randn(100) + 100,
                           index=pd.date_range("2024-01-01", periods=100, freq="D"))
        predictor = PricePredictor()
        forecast = predictor.moving_average_prediction(series, forecast_steps=10)
        assert len(forecast) == 10


class TestChartBuilder:
    def test_price_chart_returns_figure(self):
        df = make_sample_stock_data()
        fig = ChartBuilder.price_chart(df, "TEST")
        assert fig is not None

    def test_volume_chart_returns_figure(self):
        df = make_sample_stock_data()
        fig = ChartBuilder.volume_chart(df, "TEST")
        assert fig is not None

    def test_returns_distribution_returns_figure(self):
        df = make_sample_stock_data()
        fig = ChartBuilder.returns_distribution(df)
        assert fig is not None
