"""Tests for src/numerics.py: Euler stability."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.numerics import (amplification_factor, euler_unclamped, exact_solution,
                          stability_limit)


def perturbation(k, dt, n_steps=10, eps=0.1, forcing=2.0):
    """Difference between two runs that start eps apart and share the same forcing."""
    a = euler_unclamped(5.0, k, dt, n_steps, forcing)
    b = euler_unclamped(5.0 + eps, k, dt, n_steps, forcing)
    return b - a


def test_stability_limit_for_battery_leak():
    assert stability_limit(0.001) == pytest.approx(2000.0)


def test_amplification_factor_value():
    assert amplification_factor(0.5, 1.0) == pytest.approx(0.5)
    assert amplification_factor(2.5, 1.0) == pytest.approx(-1.5)


def test_stable_case_error_shrinks():
    e = np.abs(perturbation(k=0.5, dt=1.0))
    assert np.all(np.diff(e) < 0)


def test_unstable_case_error_grows_by_predicted_factor():
    e = perturbation(k=2.5, dt=1.0)
    assert abs(e[-1]) == pytest.approx(0.1 * 1.5 ** 10)


def test_boundary_case_error_neither_grows_nor_shrinks():
    e = np.abs(perturbation(k=2.0, dt=1.0))
    assert np.allclose(e, 0.1)


def test_real_battery_leak_is_stable_and_accurate():
    n = 24 * 30
    euler = euler_unclamped(10.0, 0.001, 1.0, n)
    exact = exact_solution(10.0, 0.001, np.arange(n + 1))
    assert euler[-1] == pytest.approx(exact[-1], rel=1e-3)
    assert abs(perturbation(0.001, 1.0, n_steps=n)[-1]) < 0.1