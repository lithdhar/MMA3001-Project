"""Stability of the explicit Euler method on the battery energy equation.

Test problem (battery with no charging or discharging):  dE/dt = -k * E.
One explicit Euler step is  E_next = E + dt * (-k * E) = (1 - k * dt) * E.
A small error eps in E is therefore multiplied by (1 - k * dt) every step.
The method is stable only when |1 - k * dt| < 1, that is dt < 2 / k.
"""
import numpy as np


def amplification_factor(k, dt):
    """Factor by which one Euler step multiplies a small error."""
    return 1.0 - k * dt


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