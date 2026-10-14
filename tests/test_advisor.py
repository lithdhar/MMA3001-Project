"""Tests for src/advisor.py: the daily battery advice."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.advisor import PRICE_INPUTS, action_labels, advise, day_cost, fit_price_model
from src.config import TARIFF
from src.day_profile import build_typical_days, make_day
from test_day_profile import synthetic_master


class CheapMiddayModel:
    """Stand-in price model: cheap in the middle of the day, dear in the evening."""

    def predict(self, X):
        angle = np.arctan2(X["hour_sin"], X["hour_cos"]) % (2 * np.pi)
        h = np.round(angle / (2 * np.pi) * 24).astype(int) % 24
        return 80 + 150 * ((h >= 17) & (h <= 21)) - 90 * ((h >= 10) & (h <= 15))


def household_profile(kwh=0.4):
    """Made-up household: a steady use with an evening peak, for every month and day type."""
    rows = [(m, w, minute, kwh + 0.4 * (17 * 60 <= minute <= 22 * 60))
            for m in range(1, 13) for w in (False, True) for minute in range(0, 1440, 30)]
    return pd.DataFrame(rows, columns=["month", "is_weekend", "minute_of_day", "kwh"])


@pytest.fixture(scope="module")
def library():
    return build_typical_days(synthetic_master()[0])


@pytest.fixture(scope="module")
def result(library):
    day = make_day(library, 1, "Clear", 16, 32, 4.0)
    return advise(day, CheapMiddayModel(), household_profile(), start_soc=0.5)


def test_every_hour_balances_and_the_battery_stays_in_range(result):
    h = result["hourly"]
    charge = h["battery_kw"].clip(lower=0)
    discharge = (-h["battery_kw"]).clip(lower=0)
    balance = h["solar_kwh"] + discharge + h["import_kwh"] - (h["load_kwh"] + charge + h["export_kwh"])
    assert balance.abs().max() < 1e-9
    assert h["battery_pct_C"].between(0, 100).all() and h["battery_pct_B"].between(0, 100).all()


def test_costs_are_ordered_no_battery_then_rule_then_plan(result):
    for table in (result["costs_repeating"], result["costs_today"]):
        assert table.loc["A", "total"] >= table.loc["B", "total"] - 1e-9
        assert table.loc["B", "total"] >= table.loc["C", "total"] - 0.05


def test_a_battery_saves_money_on_a_sunny_day(result):
    table = result["costs_repeating"]
    assert table.loc["A", "total"] - table.loc["B", "total"] > 0.5
    assert table.loc["B", "total"] - table.loc["C", "total"] > 0.0


def test_total_is_imports_minus_exports_plus_supply(result):
    row = result["costs_today"].loc["C"]
    assert row["total"] == pytest.approx(row["import_cost"] - row["export_income"] + row["supply_charge"])
    assert row["supply_charge"] == pytest.approx(TARIFF["supply_per_day"])


def test_day_cost_hand_calculation():
    run = pd.DataFrame({"import_kwh": [1.0, 0.0], "export_kwh": [0.0, 2.0]})
    spot = np.array([100.0, 50.0])          # $/MWh
    tariff = {**TARIFF, "supply_per_day": 1.0}
    cost = day_cost(run, spot, tariff)
    assert cost["import_cost"] == pytest.approx((0.100 + 0.15) * 1.10)
    assert cost["export_income"] == pytest.approx(2 * 0.050)
    assert cost["total"] == pytest.approx(cost["import_cost"] - cost["export_income"] + 1.0)


def test_action_names_cover_each_situation():
    def run(p, imp, exp):
        return pd.DataFrame({"p_battery_kw": p, "import_kwh": imp, "export_kwh": exp})
    r = run([2.0, 2.0, -2.0, -2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert action_labels(r) == ["Charge battery from solar", "Charge battery from the grid",
                                "Discharge battery to the home", "Discharge battery to the grid",
                                "Use grid power", "Export surplus solar", "Solar covers the home"]


def test_supplied_prices_override_the_forecast(library):
    day = make_day(library, 1, "Clear", 16, 32, 4.0)
    flat = advise(day, None, household_profile(), prices=np.full(24, 100.0))
    assert np.allclose(flat["prices"], 100.0)


def test_bigger_battery_never_costs_more_when_the_day_repeats(library):
    day = make_day(library, 1, "Clear", 16, 32, 4.0)
    totals = [advise(day, CheapMiddayModel(), household_profile(), capacity=c,
                     )["costs_repeating"].loc["C", "total"] for c in (2.0, 10.0, 20.0)]
    assert totals[0] >= totals[1] - 1e-6 >= totals[2] - 2e-6


def test_price_model_learns_from_a_small_table():
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.normal(size=(300, len(PRICE_INPUTS))), columns=PRICE_INPUTS)
    table["price_mwh"] = 80 + 20 * table["temp_c"]
    model = fit_price_model(table)
    assert np.isfinite(model.predict(table[PRICE_INPUTS])).all()
