import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np


class ChartBuilder:
    @staticmethod
    def price_chart(df, ticker=None, add_indicators=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["close"],
            mode="lines", name="Close Price",
            line=dict(color="#2563eb", width=2)
        ))
        if add_indicators:
            for col, color, name in [
                ("sma_20", "#f59e0b", "SMA 20"),
                ("sma_50", "#ef4444", "SMA 50"),
            ]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["date"], y=df[col],
                        mode="lines", name=name,
                        line=dict(color=color, width=1, dash="dash")
                    ))
        title = f"{ticker} Stock Price" if ticker else "Stock Price"
        fig.update_layout(
            title=title, xaxis_title="Date", yaxis_title="Price (USD)",
            template="plotly_white", hovermode="x unified",
            height=500, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def candlestick_chart(df, ticker=None):
        fig = go.Figure(data=[go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name=ticker or ""
        )])
        fig.update_layout(
            title=f"{ticker} Candlestick" if ticker else "Candlestick Chart",
            template="plotly_white", xaxis_rangeslider_visible=False,
            height=500, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def volume_chart(df, ticker=None):
        colors = ["#22c55e" if c >= o else "#ef4444"
                  for c, o in zip(df["close"], df["open"])]
        fig = go.Figure(data=[go.Bar(
            x=df["date"], y=df["volume"],
            marker_color=colors, name="Volume"
        )])
        fig.update_layout(
            title=f"{ticker} Volume" if ticker else "Trading Volume",
            template="plotly_white", yaxis_title="Volume",
            height=300, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def rsi_chart(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["rsi_14"],
            mode="lines", name="RSI (14)",
            line=dict(color="#8b5cf6", width=2)
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444",
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e",
                      annotation_text="Oversold (30)")
        fig.update_layout(
            title="Relative Strength Index (RSI)",
            template="plotly_white", yaxis_title="RSI",
            height=300, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def macd_chart(df):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Scatter(x=df["date"], y=df["close"],
                                 mode="lines", name="Close",
                                 line=dict(color="#2563eb")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd"],
                                 mode="lines", name="MACD",
                                 line=dict(color="#8b5cf6")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"],
                                 mode="lines", name="Signal",
                                 line=dict(color="#f59e0b")), row=2, col=1)
        fig.update_layout(title="MACD Indicator", template="plotly_white",
                          height=500, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    @staticmethod
    def sentiment_chart(daily_sentiment):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=daily_sentiment["date"], y=daily_sentiment["avg_compound"],
            mode="lines+markers", name="Sentiment Score",
            line=dict(color="#8b5cf6", width=2)
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=daily_sentiment["date"], y=daily_sentiment["article_count"],
            name="Article Count", marker_color="rgba(37, 99, 235, 0.3)"
        ), secondary_y=True)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="News Sentiment Over Time",
                          template="plotly_white", hovermode="x unified",
                          height=400, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Articles", secondary_y=True)
        return fig

    @staticmethod
    def correlation_heatmap(corr_matrix):
        fig = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Stock Returns Correlation Matrix"
        )
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    @staticmethod
    def forecast_chart(historical, forecast, ticker=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical.index, y=historical.values,
            mode="lines", name="Historical",
            line=dict(color="#2563eb", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index, y=forecast.values,
            mode="lines", name="Forecast",
            line=dict(color="#ef4444", width=2, dash="dash")
        ))
        fig.update_layout(
            title=f"{ticker} Price Forecast" if ticker else "Price Forecast",
            template="plotly_white", hovermode="x unified",
            height=500, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def returns_distribution(df, price_col="close"):
        returns = df[price_col].pct_change().dropna()
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns, nbinsx=50,
            marker_color="#2563eb", opacity=0.7,
            name="Daily Returns"
        ))
        fig.update_layout(
            title="Returns Distribution",
            template="plotly_white", xaxis_title="Daily Return",
            yaxis_title="Frequency", bargap=0.1,
            height=400, margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @staticmethod
    def sentiment_vs_price(daily_sentiment, price_df, price_col="close"):
        merged = pd.merge(
            daily_sentiment[["date", "avg_compound", "sentiment_score"]],
            price_df[["date", price_col]],
            on="date", how="inner"
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged[price_col],
            mode="lines", name="Stock Price",
            line=dict(color="#2563eb", width=2)
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=merged["date"], y=merged["sentiment_score"],
            mode="lines+markers", name="Sentiment",
            line=dict(color="#ef4444", width=1)
        ), secondary_y=True)
        fig.update_layout(title="Sentiment vs Stock Price",
                          template="plotly_white", hovermode="x unified",
                          height=450, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        return fig
