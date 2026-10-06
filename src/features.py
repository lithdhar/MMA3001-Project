"""Build the model inputs (features) for the price forecast.

Golden rule: a forecast made at time T may only use information known at T.
For a forecast `horizon` hours ahead, price information must be at least
`horizon` hours old. Weather and demand for the target hour stand in for
forecasts of them (project assumption).
"""
import numpy as np
import pandas as pd

TARGET = "price_mwh"
WEATHER_DEMAND = ["temp_c", "wind_10m", "wind_100m",
                  "solar_wm2", "direct_wm2", "diffuse_wm2", "demand_mw"]
PRICE_COLUMNS = ["price_mwh", "price_max_mwh", "price_min_mwh", "price_kwh"]


def allowed_lags(horizon, candidates=(24, 48, 72, 96, 120, 144, 168, 336)):
    """Price lags (in hours) that are old enough to be known at forecast time."""
    return [lag for lag in candidates if lag >= horizon]


def build_features(master, horizon):
    """Return (X, y) for a forecast `horizon` hours ahead.

    master : table indexed by hour_end with the columns in WEATHER_DEMAND and price_mwh.
    """
    price = master[TARGET]
    start = master.index - pd.Timedelta(hours=1)      # start of each hour
    X = pd.DataFrame(index=master.index)

    # Calendar features (always known in advance)
    X["hour"] = start.hour
    X["hour_sin"] = np.sin(2 * np.pi * start.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * start.hour / 24)
    X["day_of_week"] = start.dayofweek
    X["is_weekend"] = (start.dayofweek >= 5).astype(int)
    X["month"] = start.month

    # Weather and demand for the target hour (stand-ins for forecasts)
    for col in WEATHER_DEMAND:
        X[col] = master[col]

    # Past prices: only lags at least as old as the horizon
    lags = [lag for lag in (horizon, 168, 336) if lag in allowed_lags(horizon)]
    assert all(lag >= horizon for lag in lags)
    for lag in lags:
        X[f"price_lag_{lag}"] = price.shift(lag)

    # Rolling averages of price, ending at the moment the forecast is made
    X["price_mean_24h"] = price.rolling(24).mean().shift(horizon)
    X["price_mean_7d"] = price.rolling(168).mean().shift(horizon)

    X = X.dropna()
    y = price.loc[X.index]
    return X, y