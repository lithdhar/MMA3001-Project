"""Hourly household electricity use built from the rbee typical-day profile."""
import pandas as pd


def load_profile(path, state="Vic"):
    """Read the rbee file and return one row per month, day type and half-hour."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw[raw["State"] == state].copy()
    t = pd.to_datetime(raw["Time"], format="%d/%m/%Y %I:%M:%S %p")
    raw["minute_of_day"] = t.dt.hour * 60 + t.dt.minute
    raw["is_weekend"] = raw["Day of the Week"].eq("Weekend")
    raw = raw.rename(columns={"Month": "month", "Total Use": "kwh"})
    return raw[["month", "is_weekend", "minute_of_day", "kwh"]]


def hourly_load(index, profile, scale=1.0):
    """Hourly household energy (kWh) for hour-ending timestamps in `index`.

    Hour ending 13:00 covers 12:00 to 13:00, so it adds the half-hours labelled
    12:00 and 12:30. Month and weekday come from the START of the hour, so the
    00:00 row (the last hour of the previous day) gets the previous day's values.
    """
    index = pd.DatetimeIndex(index)
    start = index - pd.Timedelta(hours=1)
    key = profile.set_index(["month", "is_weekend", "minute_of_day"])["kwh"]

    def look(extra_minutes):
        idx = pd.MultiIndex.from_arrays(
            [start.month, start.dayofweek >= 5, start.hour * 60 + extra_minutes])
        return key.reindex(idx).to_numpy()

    return pd.Series((look(0) + look(30)) * scale, index=index, name="load_kwh")