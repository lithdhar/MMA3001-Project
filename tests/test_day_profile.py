"""Tests for src/day_profile.py: from a simple description of a day to hourly weather."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.day_profile import SKY_CLASSES, build_typical_days, classify_days, make_day


def synthetic_master(days=800, seed=0):
    """Two years of made-up hourly weather with a known cloudiness for every day."""
    rng = np.random.default_rng(seed)
    hour_end = pd.date_range("2024-12-01 01:00", periods=24 * days, freq="h")
    start = hour_end - pd.Timedelta(hours=1)
    h = start.hour.to_numpy()
    sun = np.maximum(0, np.sin((h - 6) / 12 * np.pi)) * 900
    factor = rng.choice([1.0, 0.7, 0.5, 0.2], size=days, p=[0.4, 0.2, 0.2, 0.2]).repeat(24)
    solar = sun * factor
    temp = 15 + 8 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.2, len(h))
    master = pd.DataFrame({"temp_c": temp, "wind_10m": rng.uniform(2, 6, len(h)),
                           "wind_100m": 0.0, "solar_wm2": solar,
                           "direct_wm2": solar * 0.7, "diffuse_wm2": solar * 0.3}, index=hour_end)
    return master, factor[::24]


def test_days_are_sorted_into_the_expected_sky_classes():
    master, factors = synthetic_master()
    days, _ = classify_days(master)
    expected = {1.0: "Clear", 0.7: "Partly cloudy", 0.5: "Mostly cloudy", 0.2: "Overcast"}
    by_day = dict(zip(pd.date_range("2024-12-01", periods=len(factors)), factors))
    agree = [days.loc[d, "sky"] == expected[by_day[d]] for d in days.index if d in by_day]
    assert len(agree) > 700 and np.mean(agree) > 0.95


def test_library_has_every_month_and_sky_combination():
    library = build_typical_days(synthetic_master()[0])
    cells = library.groupby(["month", "sky"]).size()
    assert (cells == 24).all()
    assert set(library["sky"]) == set(SKY_CLASSES)


def test_make_day_returns_24_hours_for_the_right_day_type():
    library = build_typical_days(synthetic_master()[0])
    weekday = make_day(library, 3, "Clear", 10, 24, 4.0)
    weekend = make_day(library, 3, "Clear", 10, 24, 4.0, weekend=True)
    assert len(weekday) == 24 and list(weekday["hour"]) == list(range(24))
    assert weekday["is_weekend"].eq(0).all() and weekend["is_weekend"].eq(1).all()
    assert (weekday.index[0] - pd.Timedelta(hours=1)).dayofweek == 2
    assert (weekend.index[0] - pd.Timedelta(hours=1)).dayofweek == 5
    assert (weekday.index[0] - pd.Timedelta(hours=1)).month == 3


def test_clear_days_have_more_sun_than_overcast_days():
    library = build_typical_days(synthetic_master()[0])
    clear = make_day(library, 6, "Clear", 5, 14, 4.0)["solar_wm2"].sum()
    overcast = make_day(library, 6, "Overcast", 5, 14, 4.0)["solar_wm2"].sum()
    assert clear > 3 * overcast


def test_temperature_follows_the_requested_range_and_wind_scales():
    library = build_typical_days(synthetic_master()[0])
    mild = make_day(library, 1, "Clear", 15, 25, 4.0)
    hot = make_day(library, 1, "Clear", 15, 35, 4.0)
    assert mild["temp_c"].min() >= 15 - 1e-9 and mild["temp_c"].max() <= 25 + 1e-9
    assert (hot["temp_c"] >= mild["temp_c"] - 1e-9).all()
    windy = make_day(library, 1, "Clear", 15, 25, 8.0)
    assert windy["wind_10m"].mean() == pytest.approx(2 * mild["wind_10m"].mean())
    assert (windy["wind_100m"] > windy["wind_10m"]).all()


def test_impossible_requests_are_refused():
    library = build_typical_days(synthetic_master()[0])
    with pytest.raises(ValueError):
        make_day(library, 1, "Clear", 30, 20, 4.0)           # highest below lowest
    with pytest.raises(ValueError):
        make_day(library, 1, "Hailstorm", 10, 20, 4.0)       # unknown sky
