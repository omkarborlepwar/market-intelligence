import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collector import StockDataCollector
from data.news import NewsFetcher
from analysis.sentiment import SentimentAnalyzer
from analysis.statistics import StatisticalAnalyzer
from analysis.correlation import CorrelationAnalyzer
from models.predictor import PricePredictor
from visualization.charts import ChartBuilder

st.set_page_config(
    page_title="Market Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding: 0 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; }
    h1, h2, h3 { font-weight: 600; }
    .metric-card { background: #f8fafc; border-radius: 8px; padding: 1rem; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=1800)
def load_news(ticker):
    fetcher = NewsFetcher()
    analyzer = SentimentAnalyzer()
    news_df = fetcher.fetch_news()
    if len(news_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    ticker_news = fetcher.filter_by_ticker(news_df, ticker)
    fallback = ticker_news if len(ticker_news) > 5 else news_df.head(30)
    analyzed = analyzer.analyze_news(fallback)
    daily = analyzer.aggregate_daily(analyzed)
    return analyzed, daily


st.title("📈 Market Intelligence Platform")
st.markdown("Real-time stock analysis, sentiment tracking, and price predictions.")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V"]
periods = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y"}

col1, col2 = st.sidebar.columns(2)
with col1:
    ticker = st.selectbox("Ticker", tickers, index=0)
with col2:
    period_label = st.selectbox("Period", list(periods.keys()), index=2)
period = periods[period_label]

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Price Analysis", "📰 Sentiment", "🔗 Correlations", "🔮 Predictions", "ℹ️ Company Info"]
)

try:
    collector = StockDataCollector()
    df = collector.fetch_historical(ticker, period)
    df = collector.compute_returns(df)
    df = collector.add_technical_indicators(df)
    company_info = collector.get_company_info(ticker)
except Exception:
    from pathlib import Path as _P
    expected_csv = f"{ticker}_{period}_1d.csv"
    expected_json = f"{ticker}_info.json"
    cwd = _P.cwd()
    fallback_paths = [
        cwd / "data" / "cache",
        _P(__file__).resolve().parent.parent / "data" / "cache",
    ]
    diag_parts = [f"Looking for: {expected_csv}"]
    for p in fallback_paths:
        exists = p.exists()
        files_list = list(p.iterdir()) if exists else []
        diag_parts.append(f"  {p} exists={exists}, files={len(files_list)}")
        diag_parts.append(f"  has_csv={ (p / expected_csv).exists() }, has_json={ (p / expected_json).exists() }")

    st.error(f"⚠️ **Yahoo Finance API rate limit** — no cached data for **{ticker} ({period})**.")
    with st.expander("Diagnostic info"):
        st.code("\n".join(diag_parts))
    if st.button("🔄 Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# Warn if data is stale (older than 1 trading day)
last_date = df["date"].max()
if isinstance(last_date, pd.Timestamp):
    age = pd.Timestamp.now() - last_date
    if age > timedelta(days=2):
        st.warning(f"Showing cached data from {last_date.strftime('%Y-%m-%d')}. Yahoo Finance API is currently unavailable.")

news_df, daily_sentiment = load_news(ticker)

with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader(f"{ticker} — Price & Indicators")
        chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True, label_visibility="collapsed")
        if chart_type == "Candlestick":
            st.plotly_chart(ChartBuilder.candlestick_chart(df, ticker), use_container_width=True)
        else:
            st.plotly_chart(ChartBuilder.price_chart(df, ticker, add_indicators=True), use_container_width=True)

        vol_col, rsi_col = st.columns(2)
        with vol_col:
            st.plotly_chart(ChartBuilder.volume_chart(df, ticker), use_container_width=True)
        with rsi_col:
            if "rsi_14" in df.columns and df["rsi_14"].notna().sum() > 0:
                st.plotly_chart(ChartBuilder.rsi_chart(df), use_container_width=True)

        if "macd" in df.columns:
            st.plotly_chart(ChartBuilder.macd_chart(df), use_container_width=True)

    with col_right:
        st.subheader("Key Statistics")
        stats = StatisticalAnalyzer.summary_statistics(df)
        st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}",
                  f"{(df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100:.2f}%")
        st.metric("Mean", f"${stats['mean']:.2f}")
        st.metric("Median", f"${stats['median']:.2f}")
        st.metric("Volatility (Annual)", f"{stats['volatility'] * 100:.2f}%")
        st.metric("Std Dev", f"${stats['std']:.2f}")

        st.divider()
        try:
            stationarity = StatisticalAnalyzer.stationarity_test(df)
            st.metric("ADF Statistic", f"{stationarity['adf_statistic']:.4f}")
            st.metric("p-value", f"{stationarity['p_value']:.6f}")
            st.markdown(f"**Stationary:** {'✅' if stationarity['is_stationary'] else '❌'}")
        except Exception:
            st.info("Insufficient data for stationarity test.")

        st.divider()
        var = StatisticalAnalyzer.calculate_var(df["close"].pct_change().dropna())
        st.metric("VaR (95%)", f"{var['historical_var'] * 100:.2f}%")

        st.divider()
        st.plotly_chart(ChartBuilder.returns_distribution(df), use_container_width=True)

with tab2:
    col1, col2 = st.columns([1.5, 1])

    with col1:
        if len(daily_sentiment) > 0:
            st.subheader("Sentiment Over Time")
            st.plotly_chart(ChartBuilder.sentiment_chart(daily_sentiment), use_container_width=True)

            if len(news_df) > 0:
                st.subheader("Recent News")
                for _, row in news_df.head(10).iterrows():
                    sentiment = row.get("sentiment_label", "neutral")
                    emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
                    st.markdown(f"{emoji.get(sentiment, '⚪')} **{row['title']}**")
                    st.caption(f"{row['source']} · {row['published'].strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('published')) else ''}")
                    st.markdown("---")
        else:
            st.info("No recent news found for this ticker. Try a different stock.")

    with col2:
        st.subheader("Sentiment Metrics")
        if len(daily_sentiment) > 0:
            avg_sent = daily_sentiment["avg_compound"].mean()
            st.metric("Avg Sentiment Score", f"{avg_sent:.3f}",
                      "Positive" if avg_sent > 0 else ("Negative" if avg_sent < 0 else "Neutral"))
            st.metric("Total Articles", int(daily_sentiment["article_count"].sum()))
            total = daily_sentiment[["positive_count", "negative_count", "neutral_count"]].sum()
            if total.sum() > 0:
                st.metric("Positive %", f"{total['positive_count'] / total.sum() * 100:.1f}%")
                st.metric("Negative %", f"{total['negative_count'] / total.sum() * 100:.1f}%")

            st.divider()
            st.subheader("Distribution")
            sentiment_counts = daily_sentiment["sentiment_label"].value_counts() if "sentiment_label" in daily_sentiment.columns else pd.Series()
            if len(sentiment_counts) > 0:
                st.plotly_chart(ChartBuilder.sentiment_vs_price(daily_sentiment, df), use_container_width=True)
        else:
            st.warning("No sentiment data available yet. News feeds may take a moment.")

with tab3:
    st.subheader("Stock Correlation Matrix")
    st.markdown("Correlation of daily returns across selected stocks (last 6 months)")
    collector = StockDataCollector()
    try:
        multi_df = collector.fetch_multiple(tickers, "6mo")
    except Exception:
        multi_df = pd.DataFrame()
        st.error("Failed to fetch correlation data due to API rate limits. Try again later.")
    if len(multi_df) > 0:
        corr_matrix = CorrelationAnalyzer.sector_correlation_matrix(multi_df)
        st.plotly_chart(ChartBuilder.correlation_heatmap(corr_matrix), use_container_width=True)

        st.divider()
        st.subheader("Sentiment vs Returns")
        if len(daily_sentiment) > 0:
            try:
                sent_return_corr = CorrelationAnalyzer.sentiment_return_correlation(
                    daily_sentiment, df, "avg_compound", "daily_return"
                )
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Same-Day Correlation",
                           f"{sent_return_corr['same_day']['correlation']:.3f}",
                           f"p={sent_return_corr['same_day']['p_value']:.3f}")
                sc2.metric("Next-Day Correlation",
                           f"{sent_return_corr['next_day']['correlation']:.3f}",
                           f"p={sent_return_corr['next_day']['p_value']:.3f}")
                sc3.metric("Samples", sent_return_corr["n_samples"])
            except Exception:
                st.info("Not enough overlapping data for sentiment-return correlation.")
    else:
        st.warning("Unable to fetch correlation data.")

with tab4:
    st.subheader(f"{ticker} — Price Prediction")
    st.markdown("ARIMA-based time series forecasting with confidence analysis.")

    forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        predictor = PricePredictor()
        price_series = df.set_index("date")["close"]
        try:
            arima_result = predictor.arima_forecast(price_series, forecast_steps=forecast_days)
            st.plotly_chart(
                ChartBuilder.forecast_chart(price_series, arima_result["forecast"], ticker),
                use_container_width=True
            )
        except Exception:
            st.warning("Insufficient data for ARIMA forecasting. Try a longer time period.")
            arima_result = None
    with col_b:
        if arima_result is None:
            st.metric("ARIMA Order", "N/A")
            st.metric("AIC", "N/A")
            st.metric("BIC", "N/A")
        else:
            st.metric("ARIMA Order", str(arima_result.get("order", "(5,1,0)")))
            st.metric("AIC", f"{arima_result['aic']:.2f}")
            st.metric("BIC", f"{arima_result['bic']:.2f}")
            if "test_mae" in arima_result:
                st.metric("Test MAE", f"${arima_result['test_mae']:.2f}")
                st.metric("Test RMSE", f"${arima_result['test_rmse']:.2f}")
                st.metric("Test MAPE", f"{arima_result['test_mape']:.2f}%")
            forecast_end = arima_result["forecast"].iloc[-1]
            current = price_series.iloc[-1]
            change = ((forecast_end / current) - 1) * 100
            st.metric("Forecast (End)", f"${forecast_end:.2f}", f"{change:+.2f}%")

        st.divider()
        st.subheader("Trend Analysis")
        try:
            trend = predictor.linear_trend_prediction(df)
            st.metric("Trend Direction", trend["trend"].title())
            st.metric("R² Score", f"{trend['r_squared']:.4f}")
            st.metric("Daily Slope", f"{trend['slope']:.4f}")
        except Exception:
            st.info("Insufficient data for trend analysis.")

with tab5:
    st.subheader(f"{company_info['name']} ({ticker})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sector", company_info["sector"])
    c1.metric("Industry", company_info["industry"])
    c2.metric("Market Cap", f"${company_info['market_cap'] / 1e9:.2f}B" if company_info["market_cap"] else "N/A")
    c2.metric("P/E Ratio", f"{company_info['pe_ratio']:.2f}" if company_info["pe_ratio"] else "N/A")
    c3.metric("52W High", f"${company_info['52w_high']:.2f}" if company_info["52w_high"] else "N/A")
    c3.metric("52W Low", f"${company_info['52w_low']:.2f}" if company_info["52w_low"] else "N/A")

    st.divider()
    st.subheader("Recent Price History")
    st.dataframe(
        df[["date", "open", "high", "low", "close", "volume"]].tail(20)
        .style.format({"open": "${:.2f}", "high": "${:.2f}", "low": "${:.2f}", "close": "${:.2f}", "volume": "{:,.0f}"}),
        use_container_width=True, hide_index=True
    )

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.info(
    "This platform demonstrates end-to-end data analysis:\n\n"
    "• **Data Collection**: yfinance API + RSS news feeds\n"
    "• **Analysis**: Technical indicators, statistical tests\n"
    "• **NLP**: VADER + TextBlob sentiment analysis\n"
    "• **Modeling**: ARIMA forecasting, trend analysis\n"
    "• **Visualization**: Interactive Plotly charts\n\n"
    "Built with Python, Streamlit, scikit-learn & statsmodels"
)
