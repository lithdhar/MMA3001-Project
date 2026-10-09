"""Tests for src/pv.py: the solar model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pv import pv_power


def test_zero_sunlight_gives_zero_output():
    assert pv_power(0.0, 20.0) == 0.0


def test_standard_conditions_equal_rated_times_performance_ratio():
    # cell temperature is 25 C when T + 0.03 * 1000 = 25, i.e. air at -5 C
    assert pv_power(1000.0, -5.0, rated_kw=6.6, performance_ratio=0.8) == pytest.approx(6.6 * 0.8)


def test_hotter_air_reduces_output():
    assert pv_power(800.0, 35.0) < pv_power(800.0, 15.0)


def test_output_rises_with_sunlight():
    g = np.array([100.0, 300.0, 600.0, 900.0])
    assert np.all(np.diff(pv_power(g, 20.0)) > 0)


def test_output_is_never_negative():
    assert np.all(pv_power(np.array([0.0, 5.0, 50.0]), np.array([45.0, 45.0, 45.0])) >= 0)


def test_series_in_series_out():
    s = pd.Series([0.0, 500.0], index=pd.date_range("2026-01-01", periods=2, freq="h"))
    out = pv_power(s, pd.Series([20.0, 20.0], index=s.index))
    assert isinstance(out, pd.Series) and out.index.equals(s.index)


def test_zero_at_night_and_sensible_yearly_yield():
    w = pd.read_csv(ROOT / "data" / "processed" / "master_hourly.csv", index_col=0, parse_dates=True)
    year = w.loc["2025-10-01":"2026-09-24"]
    p = pv_power(year["solar_wm2"], year["temp_c"], rated_kw=1.0)
    assert (p[year["solar_wm2"] == 0] == 0).all()
    yield_per_kw = p.sum() * 365 / (len(p) / 24)       # kWh per kW per year
    assert 900 < yield_per_kw < 1800