import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.collector import StockDataCollector

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V"]
PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]

collector = StockDataCollector()

print("Seeding cache for all tickers and periods...")
for ticker in TICKERS:
    for period in PERIODS:
        print(f"  Fetching {ticker} ({period})...")
        try:
            df = collector.fetch_historical(ticker, period)
            info = collector.get_company_info(ticker)
            print(f"    OK — {len(df)} rows")
        except Exception as e:
            print(f"    FAILED — {e}")

print("\nCache seeding complete!")
print(f"Files saved in: {Path(__file__).parent.parent / 'data' / 'cache'}")
