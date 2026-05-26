import random
import time
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = None


def _rate_limit_safe_request(fn, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            is_rate_limit = (
                YFRateLimitError is not None
                and isinstance(exc, YFRateLimitError)
            ) or "rate" in str(exc).lower()
            if attempt >= max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            if is_rate_limit:
                delay = max(delay, 10)
            time.sleep(delay)


class StockDataCollector:
    def __init__(self):
        self.cache = {}
        self.info_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def _get_ticker(self, ticker):
        stock = yf.Ticker(ticker, session=self.session)
        return stock

    def fetch_historical(self, ticker, period="6mo", interval="1d"):
        cache_key = f"{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        stock = self._get_ticker(ticker)

        def _fetch():
            return stock.history(period=period, interval=interval)

        df = _rate_limit_safe_request(_fetch)
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
        for i, t in enumerate(tickers):
            df = self.fetch_historical(t, period, interval)
            frames.append(df)
            if i < len(tickers) - 1:
                time.sleep(1.5)
        return pd.concat(frames, ignore_index=True)

    def get_company_info(self, ticker):
        if ticker in self.info_cache:
            return self.info_cache[ticker]
        stock = self._get_ticker(ticker)

        def _info():
            return stock.info

        info = _rate_limit_safe_request(_info)
        result = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
        self.info_cache[ticker] = result
        return result

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
