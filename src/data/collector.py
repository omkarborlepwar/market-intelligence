import random
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = None

_BASE = Path(__file__).resolve().parent.parent.parent
CACHE_DIRS = [
    _BASE / "data" / "cache",
    Path.cwd() / "data" / "cache",
]
CACHE_DIR = CACHE_DIRS[0]


def _rate_limit_safe_request(fn, max_retries=5, base_delay=3):
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
            delay = base_delay * (3 ** attempt) + random.uniform(0, 2)
            if is_rate_limit:
                delay = max(delay, 15)
            time.sleep(delay)


class StockDataCollector:
    def __init__(self):
        self.cache = {}
        self.info_cache = {}
        self._use_cache = True
        for d in CACHE_DIRS:
            d.mkdir(parents=True, exist_ok=True)

    def _get_ticker(self, ticker):
        stock = yf.Ticker(ticker)
        return stock

    def _resolve_path(self, filename):
        for d in CACHE_DIRS:
            p = d / filename
            if p.exists():
                return p
        return CACHE_DIRS[0] / filename

    def _load_cached_df(self, ticker, period, interval="1d"):
        path = self._resolve_path(f"{ticker}_{period}_{interval}.csv")
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df["ticker"] = ticker
            return df
        return None

    def _save_cached_df(self, df, ticker, period, interval="1d"):
        path = CACHE_DIRS[0] / f"{ticker}_{period}_{interval}.csv"
        df.to_csv(path, index=False)

    def _load_cached_info(self, ticker):
        path = self._resolve_path(f"{ticker}_info.json")
        if path.exists():
            return pd.read_json(path, typ="series").to_dict()
        return None

    def _save_cached_info(self, info, ticker):
        path = CACHE_DIRS[0] / f"{ticker}_info.json"
        pd.Series(info).to_json(path)

    def fetch_historical(self, ticker, period="6mo", interval="1d"):
        cache_key = f"{ticker}_{period}_{interval}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self._use_cache:
            cached = self._load_cached_df(ticker, period, interval)
            if cached is not None:
                self.cache[cache_key] = cached
                return cached
        stock = self._get_ticker(ticker)

        def _fetch():
            return stock.history(period=period, interval=interval)

        try:
            df = _rate_limit_safe_request(_fetch)
        except Exception:
            cached = self._load_cached_df(ticker, period, interval)
            if cached is not None:
                return cached
            raise

        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        }, inplace=True)
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        self.cache[cache_key] = df
        self._save_cached_df(df, ticker, period, interval)
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
        if self._use_cache:
            cached = self._load_cached_info(ticker)
            if cached is not None:
                self.info_cache[ticker] = cached
                return cached
        stock = self._get_ticker(ticker)

        def _info():
            return stock.info

        try:
            info = _rate_limit_safe_request(_info)
        except Exception:
            cached = self._load_cached_info(ticker)
            if cached is not None:
                return cached
            raise

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
        self._save_cached_info(result, ticker)
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
