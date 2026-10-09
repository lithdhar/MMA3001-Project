"""Home battery: stored energy solved with the explicit Euler method.

The battery energy E (kWh) obeys

    dE/dt = eta_c * P_ch - P_dis / eta_d - k * E

and one explicit Euler step of length dt hours is

    E_next = E + dt * (eta_c * P_ch - P_dis / eta_d - k * E).

Powers are measured on the household side (kW). The controller asks for a signed
power: positive charges the battery, negative discharges it.
"""
import numpy as np
import pandas as pd

from .config import (BATTERY_CAPACITY_KWH, BATTERY_EFF, BATTERY_POWER_KW,
                     BATTERY_SELF_DISCHARGE, BATTERY_START_SOC)


def euler_step(energy, p_request, dt=1.0, capacity=BATTERY_CAPACITY_KWH,
               p_max=BATTERY_POWER_KW, eta_c=BATTERY_EFF, eta_d=BATTERY_EFF,
               k=BATTERY_SELF_DISCHARGE):
    """Advance the battery by one Euler step of dt hours.

    p_request is the requested power in kW (positive = charge, negative =
    discharge). The request is cut back if it exceeds the power limit, or if
    the battery would overfill or run empty. Returns (new_energy, p_actual),
    where p_actual is the power that really flowed (same sign convention).
    """
    kept = energy * (1.0 - k * dt)            # energy left after self-discharge
    if p_request >= 0:
        room = max(capacity - kept, 0.0)      # kWh of space left
        p_actual = min(p_request, p_max, room / (eta_c * dt))
        new_energy = kept + dt * eta_c * p_actual
    else:
        p_actual = -min(-p_request, p_max, kept * eta_d / dt)
        new_energy = kept + dt * p_actual / eta_d
    return float(np.clip(new_energy, 0.0, capacity)), p_actual


def simulate(p_requests, energy0=None, dt=1.0, capacity=BATTERY_CAPACITY_KWH, **kwargs):
    """Run the battery through a list of requested powers (one per time step).

    Returns a DataFrame with the energy at the end of each step, the state of
    charge (0 to 1) and the power that actually flowed. The index of
    p_requests is kept if it is a pandas Series.
    """
    index = p_requests.index if isinstance(p_requests, pd.Series) else None
    requests = np.asarray(p_requests, dtype=float)
    energy = BATTERY_START_SOC * capacity if energy0 is None else energy0
    energies, flows = np.empty(len(requests)), np.empty(len(requests))
    for n, request in enumerate(requests):
        energy, flows[n] = euler_step(energy, request, dt, capacity=capacity, **kwargs)
        energies[n] = energy
    return pd.DataFrame({"energy_kwh": energies, "soc": energies / capacity,
                         "p_actual_kw": flows}, index=index)