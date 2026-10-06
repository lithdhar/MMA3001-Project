"""Forecast error measures."""
import numpy as np


def mae(actual, forecast):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecast))))


def rmse(actual, forecast):
    """Root mean square error."""
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2)))
