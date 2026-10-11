"""Tests for src/metrics.py and src/models.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import mae, rmse
from src.models import make_models, scaled_svr


def test_mae_and_rmse_on_a_hand_calculated_example():
    actual, forecast = [1.0, 2.0, 3.0], [2.0, 2.0, 5.0]       # errors 1, 0, 2
    assert mae(actual, forecast) == pytest.approx(1.0)
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(5 / 3))


def test_perfect_forecast_has_zero_error():
    x = np.array([5.0, -3.0, 10.0])
    assert mae(x, x) == 0.0 and rmse(x, x) == 0.0


def test_rmse_is_never_below_mae_and_neither_cares_about_sign():
    rng = np.random.default_rng(0)
    a, f = rng.normal(size=200), rng.normal(size=200)
    assert rmse(a, f) >= mae(a, f)
    assert mae(a, f) == pytest.approx(mae(f, a))
    assert rmse(a, f) == pytest.approx(rmse(f, a))


def test_metrics_accept_pandas_series():
    a, f = pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 6.0])
    assert mae(a, f) == pytest.approx(1.0)


def test_make_models_returns_fresh_unfitted_models_each_call():
    first, second = make_models(), make_models()
    assert list(first) == ["Linear regression", "Decision tree", "SVR"]
    for name in first:
        assert first[name] is not second[name]


def test_every_model_fits_and_predicts_finite_values():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=300)
    for name, model in make_models().items():
        model.fit(X[:200], y[:200])
        pred = model.predict(X[200:])
        assert pred.shape == (100,) and np.isfinite(pred).all(), name


def test_linear_regression_recovers_an_exact_line():
    X = np.arange(50, dtype=float).reshape(-1, 1)
    model = make_models()["Linear regression"].fit(X, 2 * X[:, 0] + 1)
    assert model.predict([[100.0]])[0] == pytest.approx(201.0)


def test_scaled_svr_predicts_in_original_units():
    # Without target scaling an SVR with epsilon = 0.1 would be hopeless at prices near 1000.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 3))
    y = 1000 + 50 * X[:, 0]
    pred = scaled_svr(C=1.0).fit(X, y).predict(X)
    assert abs(pred.mean() - 1000) < 50