"""Household energy balance and electricity bill.

Energy balance in every time step (all in kWh):

    solar + battery discharge + grid import = load + battery charge + grid export

The battery power p is measured on the household side: positive charges the
battery, negative discharges it (the same convention as src/battery.py).
"""
import numpy as np
import pandas as pd

from .config import TARIFF


def grid_flows(pv_kwh, load_kwh, p_battery_kw, dt=1.0):
    """Grid import and export (kWh per step) that make the balance above hold.

    Returns (import_kwh, export_kwh). A step never has both import and export.
    """
    net = (np.asarray(load_kwh, dtype=float) + np.asarray(p_battery_kw, dtype=float) * dt
           - np.asarray(pv_kwh, dtype=float))
    return np.maximum(net, 0.0), np.maximum(-net, 0.0)


def buy_price(spot_mwh, tariff=TARIFF):
    """Price paid for each kWh imported ($/kWh, including GST).

    (spot + network and other charges) * (1 + GST); spot is converted from $/MWh.
    """
    spot = np.asarray(spot_mwh, dtype=float) / 1000.0
    return (spot + tariff["network_other_c_kwh"] / 100.0) * (1.0 + tariff["gst"])


def feed_in_price(spot_mwh, tariff=TARIFF):
    """Price received for each kWh exported ($/kWh)."""
    spot = np.asarray(spot_mwh, dtype=float) / 1000.0
    if tariff["feed_in_mode"] == "fixed":
        return np.full_like(spot, tariff["feed_in_fixed_c_kwh"] / 100.0)
    return np.maximum(spot, 0.0) if tariff["feed_in_floor_zero"] else spot


def energy_cost(import_kwh, export_kwh, spot_mwh, tariff=TARIFF):
    """Cost of energy in each step ($): imports bought minus exports sold."""
    return (np.asarray(import_kwh) * buy_price(spot_mwh, tariff)
            - np.asarray(export_kwh) * feed_in_price(spot_mwh, tariff))


def annual_bill(import_kwh, export_kwh, spot_mwh, n_days, tariff=TARIFF):
    """Bill summary ($) over n_days: import cost, export revenue, supply charge, total."""
    import_cost = float(np.sum(np.asarray(import_kwh) * buy_price(spot_mwh, tariff)))
    export_revenue = float(np.sum(np.asarray(export_kwh) * feed_in_price(spot_mwh, tariff)))
    supply = n_days * tariff["supply_per_day"]
    return pd.Series({"import_cost": import_cost, "export_revenue": export_revenue,
                      "supply_charge": supply, "total": import_cost - export_revenue + supply})
