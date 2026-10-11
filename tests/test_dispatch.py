"""Tests for src/dispatch.py: energy balance and bill."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TARIFF
from src.dispatch import annual_bill, buy_price, energy_cost, feed_in_price, grid_flows


def test_energy_balance_holds_every_step():
    rng = np.random.default_rng(0)
    pv = rng.uniform(0, 5, 1000)
    load = rng.uniform(0.2, 3, 1000)
    p = rng.uniform(-5, 5, 1000)                       # signed battery power
    imp, exp = grid_flows(pv, load, p)
    charge, discharge = np.maximum(p, 0), np.maximum(-p, 0)
    assert np.allclose(pv + discharge + imp, load + charge + exp)


def test_never_import_and_export_in_the_same_step():
    rng = np.random.default_rng(1)
    imp, exp = grid_flows(rng.uniform(0, 5, 500), rng.uniform(0, 3, 500), rng.uniform(-5, 5, 500))
    assert np.all((imp == 0) | (exp == 0))
    assert np.all(imp >= 0) and np.all(exp >= 0)


def test_no_sun_no_battery_imports_the_load():
    imp, exp = grid_flows([0.0, 0.0], [1.2, 0.8], [0.0, 0.0])
    assert np.allclose(imp, [1.2, 0.8]) and np.allclose(exp, 0)


def test_surplus_solar_is_exported():
    imp, exp = grid_flows([4.0], [1.0], [0.0])
    assert imp[0] == 0 and exp[0] == pytest.approx(3.0)


def test_buy_price_formula():
    # spot $100/MWh = 10 c/kWh; plus 15 c network; times 1.10 GST = 27.5 c/kWh
    assert buy_price(100.0) == pytest.approx(0.275)


def test_feed_in_is_floored_at_zero_by_default_and_can_go_negative():
    assert feed_in_price(-50.0) == 0.0
    assert feed_in_price(80.0) == pytest.approx(0.08)
    assert feed_in_price(-50.0, {**TARIFF, "feed_in_floor_zero": False}) == pytest.approx(-0.05)


def test_fixed_feed_in_ignores_spot():
    t = {**TARIFF, "feed_in_mode": "fixed", "feed_in_fixed_c_kwh": 5.0}
    assert np.allclose(feed_in_price([-50.0, 300.0], t), 0.05)


def test_bill_by_hand():
    # import 10 kWh at spot 100 -> 2.75 ; export 4 kWh at spot 100 -> 0.40 ; supply 2 days
    bill = annual_bill([10.0], [4.0], [100.0], n_days=2)
    assert bill["import_cost"] == pytest.approx(2.75)
    assert bill["export_revenue"] == pytest.approx(0.40)
    assert bill["total"] == pytest.approx(2.75 - 0.40 + 2 * TARIFF["supply_per_day"])


def test_energy_cost_sums_to_bill_without_supply():
    imp, exp, spot = np.array([1.0, 0.0, 2.0]), np.array([0.0, 3.0, 0.0]), np.array([50.0, -20.0, 300.0])
    total = annual_bill(imp, exp, spot, n_days=0)["total"]
    assert energy_cost(imp, exp, spot).sum() == pytest.approx(total)