"""Numerical methods for the battery energy equation dE/dt = forcing - k * E.

Stability (Step 16)
    With no charging or discharging, one explicit Euler step is
    E_next = (1 - k * dt) * E, so a small error is multiplied by (1 - k * dt)
    each step. The method is stable only when |1 - k * dt| < 1, i.e. dt < 2 / k.

Convergence and alternatives (Step 17)
    Explicit Euler, implicit (backward) Euler and scipy's solve_ivp are compared
    with the exact solution for a forcing that is constant within each hour.
"""
import time

import numpy as np
from scipy.integrate import solve_ivp

from .config import BATTERY_EFF


# --------------------------------------------------------------------------
# Step 16: stability
# --------------------------------------------------------------------------
def amplification_factor(k, dt):
    """Factor by which one explicit Euler step multiplies a small error."""
    return 1.0 - k * dt


def amplification_factor_implicit(k, dt):
    """Backward Euler on dE/dt = -k*E: E_next = E / (1 + k*dt). Stable for every dt."""
    return 1.0 / (1.0 + k * dt)


def stability_limit(k):
    """Largest time step (hours) for which explicit Euler stays stable."""
    return 2.0 / k


def euler_unclamped(e0, k, dt, n_steps, forcing=0.0):
    """Explicit Euler for dE/dt = forcing - k * E, with NO capacity or power limits.

    The limits in src/battery.py would hide any blow-up, so stability is
    studied here on the bare equation. Returns an array of n_steps + 1 values.
    """
    energy = np.empty(n_steps + 1)
    energy[0] = e0
    for n in range(n_steps):
        energy[n + 1] = energy[n] + dt * (forcing - k * energy[n])
    return energy


def exact_solution(e0, k, t, forcing=0.0):
    """Exact solution of dE/dt = forcing - k * E at times t (hours)."""
    steady = forcing / k
    return steady + (e0 - steady) * np.exp(-k * np.asarray(t))


# --------------------------------------------------------------------------
# Step 17: convergence, implicit Euler and a professional solver
# --------------------------------------------------------------------------
def forcing_from_power(p_kw, eta_c=BATTERY_EFF, eta_d=BATTERY_EFF):
    """Rate of change of stored energy (kWh per hour) caused by the power.

    Positive p_kw charges (stores eta_c * p), negative discharges (removes |p| / eta_d).
    """
    p = np.asarray(p_kw, dtype=float)
    return eta_c * np.maximum(p, 0.0) - np.maximum(-p, 0.0) / eta_d


def euler_hourly(e0, forcing, k, substeps=1, scheme="explicit"):
    """Integrate dE/dt = forcing - k*E with a forcing that is constant in each hour.

    Each hour is split into `substeps` equal Euler steps (dt = 1/substeps hours).
    scheme is "explicit" (forward Euler) or "implicit" (backward Euler).
    Returns the energy at the end of every hour. No capacity or power limits.
    """
    dt = 1.0 / substeps
    energy = e0
    out = np.empty(len(forcing))
    for n, u in enumerate(forcing):
        for _ in range(substeps):
            if scheme == "explicit":
                energy = energy + dt * (u - k * energy)
            else:
                energy = (energy + dt * u) / (1.0 + k * dt)
        out[n] = energy
    return out


def exact_hourly(e0, forcing, k):
    """Exact energy at the end of each hour for hourly-constant forcing."""
    decay = np.exp(-k)
    energy = e0
    out = np.empty(len(forcing))
    for n, u in enumerate(forcing):
        energy = energy * decay + (u / k) * (1.0 - decay)
        out[n] = energy
    return out


def solve_ivp_hourly(e0, forcing, k, method="BDF", rtol=1e-10, atol=1e-12):
    """Same problem solved hour by hour with scipy's solve_ivp (BDF, Radau or LSODA)."""
    energy = e0
    out = np.empty(len(forcing))
    for n, u in enumerate(forcing):
        sol = solve_ivp(lambda t, e: u - k * e, (0.0, 1.0), [energy],
                        method=method, rtol=rtol, atol=atol)
        energy = sol.y[0, -1]
        out[n] = energy
    return out


def time_it(function, repeats=5):
    """Average wall-clock seconds for one call of `function()`."""
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats