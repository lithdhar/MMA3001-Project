"""Time-alignment tests.

All data in this project must be on one clock: NEM time (UTC+10, no daylight
saving), with each timestamp marking the END of its hour.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def master():
    path = ROOT / "data" / "processed" / "master_hourly.csv"
    return pd.read_csv(path, parse_dates=["hour_end"]).set_index("hour_end")


def test_hourly_steps_with_no_gaps_or_duplicates(master):
    gaps = master.index.to_series().diff().dropna()
    assert master.index.is_monotonic_increasing
    assert not master.index.duplicated().any()
    assert (gaps == pd.Timedelta(hours=1)).all()


def test_no_daylight_saving_jumps(master):
    # Every full day has exactly 24 hours, including the changeover days.
    counts = master.groupby(master.index.date).size()
    full_days = counts.iloc[1:-1]
    assert (full_days == 24).all()


def test_weather_solar_peak_matches_the_nem_clock(master):
    # Melbourne solar noon is about 12:05 to 12:35 NEM time. With hour-ending
    # labels, the daily centre of mass of solar radiation should sit near 12.5 to 13.1.
    for month, group in master.groupby(master.index.month):
        by_hour = group.groupby(group.index.hour)["solar_wm2"].mean()
        centre = (by_hour.index * by_hour).sum() / by_hour.sum()
        assert 12.4 <= centre <= 13.2, f"Month {month}: centre {centre:.2f}"


def test_load_profile_lines_up_with_market_demand(master):
    path = ROOT / "data" / "raw" / "vic_5mlp_hourly.csv"
    profile = pd.read_csv(path, parse_dates=["hour_end"]).set_index("hour_end")["VIC_TOTAL"]
    joined = pd.concat([profile, master["demand_mw"]], axis=1, join="inner")
    same_hour = joined["VIC_TOTAL"].corr(joined["demand_mw"])
    shifted_back = joined["VIC_TOTAL"].corr(joined["demand_mw"].shift(1))
    shifted_forward = joined["VIC_TOTAL"].corr(joined["demand_mw"].shift(-1))
    assert same_hour > shifted_back
    assert same_hour > shifted_forward