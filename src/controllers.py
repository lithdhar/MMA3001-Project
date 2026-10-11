"""The four battery controllers.

A  no battery            the battery never moves
B  simple rule           store surplus solar, cover the house from the battery,
                         never charge from the grid
C  forecast-driven       plan with FORECAST prices, then run the plan, bill at ACTUAL prices
D  perfect foresight     plan with ACTUAL prices (the best the planner can do)

C and D use the same planner (a small linear programme). They differ only in
the prices they are given. Solar and household use are treated as known in
both, so the only difference between C and D is the price forecast error.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .battery import euler_step
from .config import (BATTERY_CAPACITY_KWH, BATTERY_EFF, BATTERY_POWER_KW,
                     BATTERY_SELF_DISCHARGE, BATTERY_START_SOC, TARIFF)
from .dispatch import buy_price, feed_in_price, grid_flows


def plan_battery(pv, load, buy, fit, energy0, capacity=BATTERY_CAPACITY_KWH,
                 p_max=BATTERY_POWER_KW, eta=BATTERY_EFF, k=BATTERY_SELF_DISCHARGE):
    """Cheapest battery schedule over the next len(pv) hours (one-hour steps).

    Minimises  sum(buy * import - fit * export)  subject to the household energy
    balance and the battery energy equation (the same equation as src/battery.py).
    Returns (p, planned_cost): p[t] is the signed battery power in kW
    (positive charges, negative discharges) and planned_cost is the objective in $.
    """
    pv, load = np.asarray(pv, float), np.asarray(load, float)
    buy, fit = np.asarray(buy, float), np.asarray(fit, float)
    n = len(pv)
    # Variables, each of length n:  charge c, discharge d, import i, export e, stored energy E
    c, d, i, e, E = (slice(j * n, (j + 1) * n) for j in range(5))
    cost = np.zeros(5 * n)
    cost[i], cost[e] = buy, -fit

    a_eq = np.zeros((2 * n, 5 * n))
    b_eq = np.zeros(2 * n)
    rows = np.arange(n)
    # Balance:  c - d - i + e = pv - load
    a_eq[rows, rows] = 1.0                      # c
    a_eq[rows, n + rows] = -1.0                 # d
    a_eq[rows, 2 * n + rows] = -1.0             # i
    a_eq[rows, 3 * n + rows] = 1.0              # e
    b_eq[:n] = pv - load
    # Energy:  E_t - (1-k) E_{t-1} - eta c_t + d_t / eta = 0
    a_eq[n + rows, 4 * n + rows] = 1.0
    a_eq[n + rows[1:], 4 * n + rows[:-1]] = -(1.0 - k)
    a_eq[n + rows, rows] = -eta
    a_eq[n + rows, n + rows] = 1.0 / eta
    b_eq[n] = (1.0 - k) * energy0

    bounds = ([(0, p_max)] * n + [(0, p_max)] * n + [(0, None)] * n + [(0, None)] * n
              + [(0, capacity)] * n)
    result = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"planner failed: {result.message}")
    x = result.x
    return x[c] - x[d], float(result.fun)


def _run(pv, load, p_requests, energy0=None, capacity=BATTERY_CAPACITY_KWH, **battery_kwargs):
    """Run requested battery powers through the battery model and the household balance."""
    energy = BATTERY_START_SOC * capacity if energy0 is None else energy0
    p_actual = np.empty(len(p_requests))
    stored = np.empty(len(p_requests))
    for n, request in enumerate(p_requests):
        energy, p_actual[n] = euler_step(energy, request, capacity=capacity, **battery_kwargs)
        stored[n] = energy
    imp, exp = grid_flows(pv, load, p_actual)
    return pd.DataFrame({"pv_kwh": np.asarray(pv, float), "load_kwh": np.asarray(load, float),
                         "p_battery_kw": p_actual, "energy_kwh": stored,
                         "soc": stored / capacity, "import_kwh": imp, "export_kwh": exp},
                        index=getattr(pv, "index", None))


def controller_a(pv, load, **kwargs):
    """No battery: solar serves the house, the surplus is exported, the rest is imported."""
    return _run(pv, load, np.zeros(len(pv)), **kwargs)


def controller_b(pv, load, **kwargs):
    """Simple rule: surplus solar charges the battery, a shortfall is covered by it.

    The request is solar minus load: positive charges (never from the grid),
    negative discharges. The battery model cuts it back to the power and
    energy limits.
    """
    return _run(pv, load, np.asarray(pv, float) - np.asarray(load, float), **kwargs)


def controller_plan(pv, load, plan_spot_mwh, tariff=TARIFF, window=48, step=24,
                    capacity=BATTERY_CAPACITY_KWH, energy0=None, **battery_kwargs):
    """Controllers C and D: replan every `step` hours using prices `plan_spot_mwh`.

    At the start of each block the planner looks `window` hours ahead, starting from the
    battery's real energy level, and the first `step` hours of its plan are carried out.
    Give it forecast prices for C and actual prices for D. The bill is always
    calculated afterwards from the actual prices.

    The buy price used for planning is never allowed below the feed-in price. Otherwise
    a negative buy price would let the planner earn money by importing and exporting
    the same energy, which no real meter allows.
    """
    pv_a, load_a = np.asarray(pv, float), np.asarray(load, float)
    buy = buy_price(plan_spot_mwh, tariff)
    fit = feed_in_price(plan_spot_mwh, tariff)
    buy = np.maximum(buy, fit)

    energy = BATTERY_START_SOC * capacity if energy0 is None else energy0
    p_actual = np.empty(len(pv_a))
    stored = np.empty(len(pv_a))
    for start in range(0, len(pv_a), step):
        stop = min(start + window, len(pv_a))
        plan, _ = plan_battery(pv_a[start:stop], load_a[start:stop], buy[start:stop],
                               fit[start:stop], energy, capacity=capacity)
        for n in range(start, min(start + step, len(pv_a))):
            energy, p_actual[n] = euler_step(energy, plan[n - start], capacity=capacity,
                                             **battery_kwargs)
            stored[n] = energy
    imp, exp = grid_flows(pv_a, load_a, p_actual)
    return pd.DataFrame({"pv_kwh": pv_a, "load_kwh": load_a, "p_battery_kw": p_actual,
                         "energy_kwh": stored, "soc": stored / capacity,
                         "import_kwh": imp, "export_kwh": exp},
                        index=getattr(pv, "index", None))