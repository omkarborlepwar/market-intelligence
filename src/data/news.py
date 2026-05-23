import feedparser
import pandas as pd
from datetime import datetime, timedelta
import re


class NewsFetcher:
    def __init__(self):
        self.rss_feeds = {
            "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
            "cnbc_top": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories",
            "google_business": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
        }

    def fetch_news(self, source=None, max_articles=50):
        sources = [source] if source else self.rss_feeds.keys()
        articles = []
        for s in sources:
            url = self.rss_feeds.get(s)
            if not url:
                continue
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_articles]:
                    articles.append({
                        "source": s,
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", entry.get("description", "")),
                        "published": self._parse_date(entry),
                        "link": entry.get("link", ""),
                    })
            except Exception as e:
                print(f"Failed to fetch {s}: {e}")
        return pd.DataFrame(articles)

    def filter_by_ticker(self, df, ticker):
        company = self._ticker_to_company(ticker)
        pattern = re.compile(rf"\b{re.escape(ticker)}\b|\b{re.escape(company)}\b", re.IGNORECASE)
        mask = df["title"].str.contains(pattern, na=False) | df["summary"].str.contains(pattern, na=False)
        return df[mask].copy()

    def _parse_date(self, entry):
        try:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                return datetime(*published[:6])
        except Exception:
            pass
        return datetime.now()

    def _ticker_to_company(self, ticker):
        mapping = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Alphabet",
            "GOOG": "Alphabet",
            "AMZN": "Amazon",
            "META": "Meta",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan",
            "V": "Visa",
        }
        return mapping.get(ticker.upper(), ticker)
