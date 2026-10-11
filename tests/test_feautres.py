"""Tests for src/features.py, above all the golden rule: no information from the future."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import WEATHER_DEMAND, allowed_lags, build_features

HORIZONS = [72, 96, 120]


def make_master(hours=1000, seed=0):
    """A made-up table with the same columns as master_hourly.csv."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01 01:00", periods=hours, freq="h", name="hour_end")
    master = pd.DataFrame(rng.normal(size=(hours, len(WEATHER_DEMAND))),
                          index=index, columns=WEATHER_DEMAND)
    master["price_mwh"] = rng.normal(80, 30, hours)
    return master


def test_allowed_lags_are_never_newer_than_the_horizon():
    for horizon in [24, 48, 72, 96, 120, 168]:
        assert all(lag >= horizon for lag in allowed_lags(horizon))
    assert allowed_lags(72) == [72, 96, 120, 144, 168, 336]
    assert allowed_lags(168) == [168, 336]


@pytest.mark.parametrize("horizon", HORIZONS)
def test_price_lag_columns_are_old_enough(horizon):
    X, _ = build_features(make_master(), horizon)
    lags = [int(c.split("_")[-1]) for c in X.columns if c.startswith("price_lag_")]
    assert lags and min(lags) >= horizon


@pytest.mark.parametrize("horizon", HORIZONS)
def test_features_do_not_change_when_recent_prices_change(horizon):
    """Scramble every price newer than `horizon` hours before a target hour.
    The features for that target hour must not move at all."""
    master = make_master()
    X, _ = build_features(master, horizon)
    target_position = 700
    target = master.index[target_position]
    scrambled = master.copy()
    cut = target_position - horizon + 1           # first price that is too recent to know
    scrambled.iloc[cut:, scrambled.columns.get_loc("price_mwh")] = 1e6
    X2, _ = build_features(scrambled, horizon)
    pd.testing.assert_series_equal(X.loc[target], X2.loc[target])


@pytest.mark.parametrize("horizon", HORIZONS)
def test_lag_and_rolling_values_are_what_they_claim(horizon):
    master = make_master()
    X, y = build_features(master, horizon)
    j = 700
    target = master.index[j]
    price = master["price_mwh"].to_numpy()
    assert X.loc[target, f"price_lag_{horizon}"] == pytest.approx(price[j - horizon])
    assert X.loc[target, "price_lag_168"] == pytest.approx(price[j - 168])
    assert X.loc[target, "price_mean_24h"] == pytest.approx(price[j - horizon - 23:j - horizon + 1].mean())
    assert X.loc[target, "price_mean_7d"] == pytest.approx(price[j - horizon - 167:j - horizon + 1].mean())
    assert y.loc[target] == pytest.approx(price[j])


def test_target_is_aligned_and_complete():
    X, y = build_features(make_master(), 72)
    assert X.index.equals(y.index)
    assert not X.isna().any().any() and not y.isna().any()


def test_calendar_features_use_the_start_of_the_hour():
    # An hour ending 00:00 is the last hour of the previous day, starting at 23:00.
    X, _ = build_features(make_master(hours=1000), 72)
    row = X.loc[X.index[X.index.hour == 0][0]]
    assert row["hour"] == 23
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 23 / 24))
    start = X.index[X.index.hour == 0][0] - pd.Timedelta(hours=1)
    assert row["day_of_week"] == start.dayofweek
    assert row["is_weekend"] == int(start.dayofweek >= 5)