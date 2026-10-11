"""Tests for src/controllers.py: the four battery controllers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery import euler_step
from src.config import BATTERY_POWER_KW
from src.controllers import (controller_a, controller_b, controller_plan, plan_battery)
from src.dispatch import annual_bill, grid_flows


def synthetic_days(days=20, seed=0):
    """A small made-up household: sunny middays, evening peak, cheap midday, dear evening."""
    rng = np.random.default_rng(seed)
    n = 24 * days
    idx = pd.date_range("2026-01-01 01:00", periods=n, freq="h")
    h = idx.hour
    pv = pd.Series(np.maximum(0, 4 * np.sin((h - 6) / 12 * np.pi)) * rng.uniform(0.6, 1, n), index=idx)
    load = pd.Series(0.5 + 0.8 * ((h >= 17) & (h <= 22)) + rng.uniform(0, 0.3, n), index=idx)
    spot = 60 + 120 * ((h >= 17) & (h <= 21)) - 80 * ((h >= 10) & (h <= 15)) + rng.normal(0, 20, n)
    forecast = spot + rng.normal(15, 40, n)
    return pv, load, np.asarray(spot), np.asarray(forecast)


def energy_balance_error(run):
    charge = run["p_battery_kw"].clip(lower=0)
    discharge = (-run["p_battery_kw"]).clip(lower=0)
    return (run["pv_kwh"] + discharge + run["import_kwh"]
            - (run["load_kwh"] + charge + run["export_kwh"])).abs().max()


@pytest.fixture(scope="module")
def runs():
    pv, load, spot, forecast = synthetic_days()
    return {"A": controller_a(pv, load), "B": controller_b(pv, load),
            "C": controller_plan(pv, load, forecast), "D": controller_plan(pv, load, spot),
            "spot": spot, "days": len(pv) / 24}


@pytest.mark.parametrize("name", ["A", "B", "C", "D"])
def test_every_controller_balances_energy_and_respects_limits(runs, name):
    run = runs[name]
    assert energy_balance_error(run) < 1e-9
    assert run["soc"].between(0, 1).all()
    assert run["p_battery_kw"].abs().max() <= BATTERY_POWER_KW + 1e-9
    assert (run["import_kwh"] >= 0).all() and (run["export_kwh"] >= 0).all()


def test_controller_a_never_uses_the_battery(runs):
    assert (runs["A"]["p_battery_kw"] == 0).all()


def test_controller_b_never_charges_from_the_grid(runs):
    run = runs["B"]
    surplus = (run["pv_kwh"] - run["load_kwh"]).clip(lower=0)
    assert (run["p_battery_kw"].clip(lower=0) <= surplus + 1e-9).all()


def test_perfect_foresight_is_no_worse_than_the_others_on_this_example(runs):
    totals = {k: annual_bill(runs[k]["import_kwh"], runs[k]["export_kwh"], runs["spot"], runs["days"])["total"]
              for k in "ABCD"}
    assert totals["D"] <= totals["C"] and totals["D"] <= totals["B"] and totals["D"] <= totals["A"]


def test_planner_is_optimal_for_its_window():
    pv, load, spot, _ = synthetic_days(days=2)
    fit = np.maximum(spot / 1000, 0)
    buy = np.maximum((spot / 1000 + 0.15) * 1.1, fit)
    plan, planned = plan_battery(pv.to_numpy(), load.to_numpy(), buy, fit, energy0=2.0)
    energy, flows = 2.0, []
    for p in plan:
        energy, actual = euler_step(energy, p)
        flows.append(actual)
    imp, exp = grid_flows(pv, load, flows)
    assert float((imp * buy - exp * fit).sum()) == pytest.approx(planned, abs=1e-6)
    for alternative in (np.zeros(len(pv)), (pv - load).to_numpy()):
        energy, flows = 2.0, []
        for p in alternative:
            energy, actual = euler_step(energy, p)
            flows.append(actual)
        imp, exp = grid_flows(pv, load, flows)
        assert float((imp * buy - exp * fit).sum()) >= planned - 1e-6


def test_planner_charges_cheap_and_discharges_dear():
    n = 24
    price = np.where(np.arange(n) < 12, 0.05, 0.50)
    plan, _ = plan_battery(np.zeros(n), np.full(n, 1.0), price, np.zeros(n), energy0=0.0)
    assert plan[:12].max() > 0.5     # charges in the cheap hours
    assert plan[12:].sum() < 0       # discharges more than it charges in the dear hours


def test_planner_does_nothing_when_prices_are_flat_and_battery_empty():
    n = 24
    plan, _ = plan_battery(np.zeros(n), np.full(n, 1.0), np.full(n, 0.30), np.zeros(n), energy0=0.0)
    assert np.abs(plan).max() < 1e-6


def test_planner_copes_with_a_very_negative_price():
    pv, load, spot, _ = synthetic_days(days=3)
    spot = spot.copy()
    spot[30:34] = -300.0
    run = controller_plan(pv, load, spot)
    assert energy_balance_error(run) < 1e-9