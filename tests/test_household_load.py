"""Tests for src/load.py: the hourly household load."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.load import hourly_load, load_profile

RBEE = ROOT / "data" / "raw" / "rbee extract by hour and state.csv"


@pytest.fixture(scope="module")
def profile():
    return load_profile(RBEE)


def day_total(profile, month, weekend):
    rows = profile[(profile["month"] == month) & (profile["is_weekend"] == weekend)]
    return rows["kwh"].sum()


def one_day(date):
    """Hour-ending timestamps 01:00 of `date` to 00:00 of the next day."""
    start = pd.Timestamp(date)
    return pd.date_range(start + pd.Timedelta(hours=1), periods=24, freq="h")


def test_weekday_day_total_matches_profile(profile):
    load = hourly_load(one_day("2026-01-14"), profile)   # a Wednesday
    assert load.sum() == pytest.approx(day_total(profile, 1, False))


def test_weekend_day_total_matches_profile(profile):
    load = hourly_load(one_day("2026-01-17"), profile)   # a Saturday
    assert load.sum() == pytest.approx(day_total(profile, 1, True))


def test_no_missing_values_over_a_year(profile):
    index = pd.date_range("2025-10-01 01:00", "2026-10-01 00:00", freq="h")
    assert not hourly_load(index, profile).isna().any()


def test_scaling_is_linear(profile):
    index = one_day("2026-07-08")
    assert np.allclose(hourly_load(index, profile, 1.4), 1.4 * hourly_load(index, profile))