"""A second way of testing the price forecast: one fixed train/test split.

The main test (notebook 04) is a walk-forward test: the models are retrained
before every 28-day block of the test year. This module provides the stricter
alternative. Each model is trained once on the data before the test year, with
a gap of one horizon, and then forecasts the whole test year without ever being
updated. Comparing the two shows how much of the result depends on the way the
forecast is validated.
"""
import numpy as np
import pandas as pd

from .metrics import mae, rmse
from .models import make_models


def split_train_test(df, test_start, horizon):
    """Split `df` into a training part and a test part.

    The training part ends `horizon` hours before `test_start`, because the most
    recent `horizon` hours of prices would not yet be known when the first test
    forecast is made. The last column of `df` is the target, the others are inputs.
    """
    test_start = pd.Timestamp(test_start)
    train = df[df.index <= test_start - pd.Timedelta(hours=horizon)]
    test = df[df.index >= test_start]
    return train, test


def fixed_split_forecast(df, horizon, test_start="2025-10-01"):
    """Fit each model once on the past and forecast the whole test period.

    Returns a table indexed by hour with the actual price and one column of
    forecasts per model ($/MWh).
    """
    target = df.columns[-1]
    inputs = list(df.columns[:-1])
    train, test = split_train_test(df, test_start, horizon)
    out = pd.DataFrame({"actual": test[target]}, index=test.index)
    for name, model in make_models().items():
        model.fit(train[inputs], train[target])
        out[name] = model.predict(test[inputs])
    return out


def score_table(forecasts, columns):
    """MAE, RMSE and bias ($/MWh) of each forecast column against `actual`."""
    rows = {}
    for column in columns:
        error = forecasts[column] - forecasts["actual"]
        rows[column] = {"MAE": mae(forecasts["actual"], forecasts[column]),
                        "RMSE": rmse(forecasts["actual"], forecasts[column]),
                        "Bias": float(np.mean(error))}
    return pd.DataFrame(rows).T


def monthly_mae(forecasts, column):
    """MAE of one forecast column for each calendar month."""
    return forecasts.groupby(forecasts.index.to_period("M")).apply(
        lambda g: mae(g["actual"], g[column]))
