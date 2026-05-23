import nltk
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import timedelta

try:
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_textblob(self, text):
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }

    def analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def analyze_news(self, df, text_column="title"):
        result = df.copy()
        textblob_data = result[text_column].apply(self.analyze_textblob)
        vader_data = result[text_column].apply(self.analyze_vader)
        result["tb_polarity"] = textblob_data.apply(lambda x: x["polarity"])
        result["tb_subjectivity"] = textblob_data.apply(lambda x: x["subjectivity"])
        result["vader_compound"] = vader_data.apply(lambda x: x["compound"])
        result["vader_pos"] = vader_data.apply(lambda x: x["positive"])
        result["vader_neg"] = vader_data.apply(lambda x: x["negative"])
        result["sentiment_label"] = result["vader_compound"].apply(
            lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
        )
        return result

    def aggregate_daily(self, df, date_column="published"):
        df = df.copy()
        df["date"] = pd.to_datetime(df[date_column]).dt.date
        daily = df.groupby("date").agg(
            avg_polarity=("tb_polarity", "mean"),
            avg_compound=("vader_compound", "mean"),
            avg_subjectivity=("tb_subjectivity", "mean"),
            article_count=("title", "count"),
            positive_count=("sentiment_label", lambda x: (x == "positive").sum()),
            negative_count=("sentiment_label", lambda x: (x == "negative").sum()),
            neutral_count=("sentiment_label", lambda x: (x == "neutral").sum()),
        ).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily["sentiment_score"] = (
            (daily["positive_count"] - daily["negative_count"]) / daily["article_count"]
        )
        return daily
