"""Tests for src/battery.py: the Euler battery model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.battery import euler_step, simulate
from src.config import BATTERY_CAPACITY_KWH, BATTERY_EFF


def test_constant_charge_matches_hand_calculation():
    # No self-discharge: charging 2 kW for 3 hours from empty stores eta * 2 * 3 kWh
    out = simulate(np.full(3, 2.0), energy0=0.0, k=0.0)
    assert out["energy_kwh"].iloc[-1] == pytest.approx(BATTERY_EFF * 2.0 * 3)


def test_discharge_matches_hand_calculation():
    out = simulate(np.full(2, -1.0), energy0=5.0, k=0.0)
    assert out["energy_kwh"].iloc[-1] == pytest.approx(5.0 - 2 * 1.0 / BATTERY_EFF)


def test_self_discharge_matches_euler_formula():
    energy, _ = euler_step(8.0, 0.0, dt=1.0, k=0.01)
    assert energy == pytest.approx(8.0 * (1 - 0.01))


def test_cannot_overcharge_or_overdischarge():
    out = simulate(np.concatenate([np.full(20, 5.0), np.full(20, -5.0)]))
    assert out["energy_kwh"].between(0, BATTERY_CAPACITY_KWH).all()
    assert out["soc"].between(0, 1).all()


def test_power_limit_is_respected():
    energy, p = euler_step(0.0, 50.0, p_max=5.0)
    assert p == pytest.approx(5.0)
    energy, p = euler_step(10.0, -50.0, p_max=5.0)
    assert p == pytest.approx(-5.0)


def test_empty_battery_cannot_discharge_and_full_cannot_charge():
    assert euler_step(0.0, -3.0)[1] == 0.0
    assert euler_step(BATTERY_CAPACITY_KWH, 3.0, k=0.0)[1] == pytest.approx(0.0)


def test_energy_balance_every_step():
    # stored change = eta_c * charge - discharge / eta_d - self-discharge loss
    k, dt = 0.001, 1.0
    out = simulate(np.array([3.0, 5.0, -4.0, -5.0, 2.0]), energy0=4.0, k=k)
    e_prev = 4.0
    for e, p in zip(out["energy_kwh"], out["p_actual_kw"]):
        charge, discharge = max(p, 0), max(-p, 0)
        expected = e_prev + dt * (BATTERY_EFF * charge - discharge / BATTERY_EFF - k * e_prev)
        assert e == pytest.approx(expected)
        e_prev = e


def test_series_index_is_kept():
    s = pd.Series([1.0, -1.0], index=pd.date_range("2026-01-01", periods=2, freq="h"))
    assert simulate(s).index.equals(s.index)