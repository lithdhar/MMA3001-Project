"""Tests for src/numerics.py: Euler stability, convergence and solver agreement."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BATTERY_EFF
from src.numerics import (amplification_factor, amplification_factor_implicit,
                          euler_hourly, euler_unclamped, exact_hourly,
                          exact_solution, forcing_from_power, solve_ivp_hourly,
                          stability_limit)

DAY = np.zeros(24)
DAY[9:13] = 2.0
DAY[18:22] = -1.5
FORCING = forcing_from_power(DAY)


def perturbation(k, dt, n_steps=10, eps=0.1, forcing=2.0):
    """Difference between two runs that start eps apart and share the same forcing."""
    a = euler_unclamped(5.0, k, dt, n_steps, forcing)
    b = euler_unclamped(5.0 + eps, k, dt, n_steps, forcing)
    return b - a


def final_error(substeps, scheme, k=0.001):
    got = euler_hourly(2.0, FORCING, k, substeps, scheme)[-1]
    return abs(got - exact_hourly(2.0, FORCING, k)[-1])


# ---- Step 16: stability ---------------------------------------------------
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


# ---- Step 17: convergence and alternatives --------------------------------
@pytest.mark.parametrize("scheme", ["explicit", "implicit"])
def test_euler_is_first_order(scheme):
    # halving the step should roughly halve the error
    ratio = final_error(10, scheme) / final_error(20, scheme)
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_implicit_is_stable_where_explicit_is_not():
    assert abs(amplification_factor_implicit(2.5, 1.0)) < 1
    assert abs(amplification_factor(2.5, 1.0)) > 1


def test_solve_ivp_matches_exact_solution():
    exact = exact_hourly(2.0, FORCING, 0.001)
    for method in ["BDF", "Radau", "LSODA"]:
        assert np.allclose(solve_ivp_hourly(2.0, FORCING, 0.001, method), exact, atol=1e-7)


def test_no_leak_matches_hand_calculation():
    # k tiny: charging 2 kW for 4 hours stores eta * 2 * 4
    out = euler_hourly(0.0, forcing_from_power([2.0] * 4), 1e-12, substeps=1)
    assert out[-1] == pytest.approx(BATTERY_EFF * 8.0)