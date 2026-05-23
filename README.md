# Market Intelligence Platform

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://market-intelligence-k69phg2f8tgkzgjxbrgz4g.streamlit.app)

**End-to-end data analysis project** — Real-time stock analysis, NLP sentiment tracking, statistical modeling, and price prediction, all wrapped in an interactive Streamlit dashboard.

## Features

- **📊 Stock Analysis** — Historical prices, candlestick charts, technical indicators (SMA, RSI, MACD, Bollinger Bands)
- **📰 Sentiment Analysis** — Real-time news aggregation from RSS feeds, VADER + TextBlob NLP sentiment scoring
- **🔗 Correlation Analysis** — Cross-asset return correlations, sentiment vs price relationships, lag analysis
- **🔮 Price Prediction** — ARIMA time series forecasting, linear trend analysis, moving average models
- **📈 Interactive Dashboard** — Built with Streamlit + Plotly for rich, responsive visualizations

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Collection | yfinance API, RSS feeds (feedparser) |
| Analysis | pandas, numpy, scipy, statsmodels |
| NLP | NLTK (VADER), TextBlob |
| Modeling | ARIMA (statsmodels), scikit-learn |
| Visualization | Plotly, Streamlit |
| Testing | pytest |

## Quick Start

```bash
# Clone and enter directory
git clone <your-repo-url>
cd market-intelligence

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py
```

Open **http://localhost:8501** in your browser.

## Project Structure

```
market-intelligence/
├── app/
│   └── dashboard.py            # Streamlit dashboard
├── src/
│   ├── data/
│   │   ├── collector.py        # Stock data (yfinance)
│   │   └── news.py             # News aggregation (RSS)
│   ├── analysis/
│   │   ├── sentiment.py        # NLP sentiment (VADER + TextBlob)
│   │   ├── statistics.py       # Statistical analysis
│   │   └── correlation.py      # Correlation analysis
│   ├── models/
│   │   └── predictor.py        # Price prediction (ARIMA)
│   └── visualization/
│       └── charts.py           # Plotly chart builders
├── tests/
│   ├── conftest.py             # Test fixtures
│   └── test_all.py             # Unit tests
└── requirements.txt
```

## Key Capabilities

- **Technical Analysis**: SMA crossover signals, RSI overbought/oversold, MACD divergence
- **Statistical Tests**: ADF stationarity, Shapiro-Wilk normality, VaR/CVaR risk metrics
- **Sentiment Pipeline**: Fetches live financial news → filters by ticker → scores with VADER + TextBlob → aggregates daily
- **Cross-Asset Correlation**: Full correlation matrix across any basket of stocks
- **Forecasting**: Automated ARIMA order selection, walk-forward validation, trend decomposition

## Example Dashboard Sections

1. **Price Analysis** — Interactive line/candlestick charts with volume, RSI, and MACD panels
2. **Sentiment** — Real-time news feed with sentiment labels, daily aggregation charts
3. **Correlations** — Returns correlation heatmap, sentiment vs price overlay
4. **Predictions** — ARIMA forecast with accuracy metrics, linear trend analysis
5. **Company Info** — Fundamental data, recent price history table

## Running Tests

```bash
pytest tests/ -v
```
