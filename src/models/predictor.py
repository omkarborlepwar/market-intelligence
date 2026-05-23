import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")


class PricePredictor:
    def __init__(self):
        self.models = {}

    def arima_forecast(self, series, order=(5, 1, 0), forecast_steps=30):
        series = series.dropna()
        train = series[:-forecast_steps] if len(series) > forecast_steps * 2 else series
        test = series[-forecast_steps:] if len(series) > forecast_steps * 2 else None

        model = ARIMA(train, order=order)
        fitted = model.fit()

        forecast = fitted.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_steps, freq="D"
        )

        result = {
            "model": fitted,
            "forecast": pd.Series(forecast.values, index=forecast_index[:len(forecast)]),
            "aic": fitted.aic,
            "bic": fitted.bic,
        }

        if test is not None:
            pred = fitted.forecast(steps=len(test))
            result["test_mae"] = mean_absolute_error(test, pred)
            result["test_rmse"] = np.sqrt(mean_squared_error(test, pred))
            result["test_mape"] = np.mean(np.abs((test - pred) / test)) * 100

        return result

    def arima_order_selection(self, series, max_p=5, max_d=2, max_q=5):
        series = series.dropna()
        best_aic, best_order = float("inf"), None
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        return {"best_order": best_order, "best_aic": best_aic}

    def linear_trend_prediction(self, df, price_col="close", forecast_days=30):
        df = df.copy().reset_index(drop=True)
        df["day"] = np.arange(len(df))
        X = df[["day"]].values
        y = df[price_col].values

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
        predictions = model.predict(future_days)

        return {
            "model": model,
            "slope": model.coef_[0],
            "intercept": model.intercept_,
            "r_squared": model.score(X, y),
            "predictions": predictions,
            "trend": "uptrend" if model.coef_[0] > 0 else "downtrend",
        }

    def moving_average_prediction(self, series, window=20, forecast_steps=30):
        last_values = series.dropna().tail(window)
        forecast = pd.Series(
            [last_values.mean()] * forecast_steps,
            index=pd.date_range(
                start=series.dropna().index[-1] + pd.Timedelta(days=1),
                periods=forecast_steps, freq="D"
            )
        )
        return forecast
