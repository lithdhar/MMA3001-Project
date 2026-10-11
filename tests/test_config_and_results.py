"""Sanity checks on the assumptions, and a guard on the headline results."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.controllers import controller_a, controller_b, controller_plan, plan_battery
from src.dispatch import annual_bill
from src.numerics import time_it


def test_one_way_efficiency_squares_to_round_trip_efficiency():
    assert config.BATTERY_EFF ** 2 == pytest.approx(config.BATTERY_ROUNDTRIP_EFF)


def test_assumptions_are_physically_sensible():
    assert 0 < config.BATTERY_ROUNDTRIP_EFF <= 1
    assert 0 <= config.BATTERY_START_SOC <= 1
    assert 0 <= config.BATTERY_SELF_DISCHARGE < 0.1
    assert config.BATTERY_CAPACITY_KWH > 0 and config.BATTERY_POWER_KW > 0
    assert config.TARIFF["supply_per_day"] > 0


def test_planner_reports_failure_instead_of_returning_nonsense():
    # Starting energy far above the capacity cannot be brought back within limits in one hour.
    with pytest.raises(RuntimeError):
        plan_battery([0.0] * 3, [1.0] * 3, [0.3] * 3, [0.05] * 3, energy0=100.0)


def test_time_it_returns_a_positive_time():
    assert time_it(lambda: sum(range(1000)), repeats=3) > 0


DATA = ROOT / "data" / "processed"
needs_data = pytest.mark.skipif(not (DATA / "pv_hourly.csv").exists()
                                or not (ROOT / "results" / "forecasts.csv").exists(),
                                reason="processed data not available")


@pytest.fixture(scope="module")
def year():
    fc = pd.read_csv(ROOT / "results" / "forecasts.csv", index_col=0, parse_dates=True)
    pv = pd.read_csv(DATA / "pv_hourly.csv", index_col=0, parse_dates=True)["pv_kwh"]
    load = pd.read_csv(DATA / "household_load_hourly.csv", index_col=0, parse_dates=True)["load_kwh"]
    return fc, pv, load


@needs_data
def test_headline_bills_have_not_changed(year):
    """If this fails, the model or its data changed. Check that it was on purpose,
    then update the numbers here and in the report."""
    fc, pv, load = year
    spot, days = fc["price_actual_mwh"].values, len(fc) / 24
    bill = lambda r: annual_bill(r["import_kwh"], r["export_kwh"], spot, days)["total"]
    assert bill(controller_a(pv, load)) == pytest.approx(1179.84, abs=0.01)
    assert bill(controller_b(pv, load)) == pytest.approx(643.46, abs=0.01)
    assert bill(controller_plan(pv, load, fc["price_forecast_mwh"].values)) == pytest.approx(590.75, abs=0.01)
    assert bill(controller_plan(pv, load, spot)) == pytest.approx(546.13, abs=0.01)


@needs_data
def test_controller_ordering_on_real_data(year):
    fc, pv, load = year
    spot, days = fc["price_actual_mwh"].values, len(fc) / 24
    bill = lambda r: annual_bill(r["import_kwh"], r["export_kwh"], spot, days)["total"]
    a, b = bill(controller_a(pv, load)), bill(controller_b(pv, load))
    c = bill(controller_plan(pv, load, fc["price_forecast_mwh"].values))
    d = bill(controller_plan(pv, load, spot))
    assert a > b > c > d