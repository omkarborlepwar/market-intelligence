import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def make_sample_stock_data():
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    price = 100 * (1 + np.random.randn(100).cumsum() * 0.02)
    return pd.DataFrame({
        "date": dates,
        "open": price * (1 + np.random.randn(100) * 0.005),
        "high": price * (1 + abs(np.random.randn(100) * 0.01)),
        "low": price * (1 - abs(np.random.randn(100) * 0.01)),
        "close": price,
        "volume": np.random.randint(1000000, 10000000, 100),
        "ticker": "TEST",
    })


def make_sample_news_data():
    return pd.DataFrame({
        "title": ["Apple announces new product", "Market hits record high",
                  "Supply chain concerns grow", "Strong earnings report",
                  "Tech stocks rally today"],
        "summary": ["Apple announced a revolutionary new product lineup",
                     "Stock markets reached all-time highs amid optimism",
                     "Supply chain disruptions may impact quarterly results",
                     "Company reported better than expected earnings",
                     "Technology stocks surged on positive sentiment"],
        "published": [datetime.now() - timedelta(days=i) for i in range(5)],
        "source": ["test"] * 5,
    })
