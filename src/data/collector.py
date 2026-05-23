import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class StockDataCollector:
    def __init__(self):
        self.cache = {}

    def fetch_historical(self, ticker, period="6mo", interval="1d"):
        cache_key = f"{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        }, inplace=True)
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        self.cache[cache_key] = df
        return df

    def fetch_multiple(self, tickers, period="6mo", interval="1d"):
        frames = []
        for t in tickers:
            df = self.fetch_historical(t, period, interval)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_company_info(self, ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }

    def compute_returns(self, df):
        df = df.copy()
        df["daily_return"] = df["close"].pct_change()
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def add_technical_indicators(self, df):
        df = df.copy()
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        bb_mid = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        return df


import numpy as np
