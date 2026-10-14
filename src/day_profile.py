"""Turn a simple description of a day into a full hourly weather profile.

A user describes a day with a month, a sky condition, the lowest and highest
temperature and an average wind speed. The profile library built here holds,
for every month and sky condition, the typical hourly shape of sunshine and
temperature, measured from the real weather record. `make_day` then scales
that shape to the temperatures and wind the user asked for.

Hours are labelled by their START (0 to 23), as in src/features.py.
"""
import numpy as np
import pandas as pd

SKY_CLASSES = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]
SKY_THRESHOLDS = (0.85, 0.65, 0.40)       # clearness index limits between the four classes
WIND_100M_RATIO = 1.6                      # typical 100 m wind speed / 10 m wind speed
MIN_DAYS = 3                               # fewest days allowed in one library cell


def classify_days(master):
    """One row per complete day: month, sky class, temperature range and mean wind.

    The clearness index is the day's total sunshine divided by that of a clear day
    (the 95th percentile of each hour in that month).
    """
    start = master.index - pd.Timedelta(hours=1)
    table = master[["temp_c", "wind_10m", "solar_wm2", "direct_wm2", "diffuse_wm2"]].copy()
    table["day"], table["hour"], table["month"] = start.normalize(), start.hour, start.month
    envelope = table.groupby(["month", "hour"])["solar_wm2"].quantile(0.95)
    table["clear"] = envelope.reindex(
        pd.MultiIndex.from_arrays([table["month"], table["hour"]])).to_numpy()
    days = table.groupby("day").agg(month=("month", "first"), hours=("hour", "size"),
                                    solar=("solar_wm2", "sum"), clear=("clear", "sum"),
                                    tmin=("temp_c", "min"), tmax=("temp_c", "max"),
                                    wind=("wind_10m", "mean"))
    days = days[days["hours"] == 24].copy()
    days["clearness"] = days["solar"] / days["clear"]
    high, mid, low = SKY_THRESHOLDS
    days["sky"] = np.select([days["clearness"] >= high, days["clearness"] >= mid,
                             days["clearness"] >= low], SKY_CLASSES[:3], SKY_CLASSES[3])
    return days, table


def build_typical_days(master):
    """Library of typical hourly shapes for each (month, sky class).

    A cell pools the month before, the month itself and the month after, so that
    rare combinations (an overcast day in January) still have enough days behind them.
    """
    days, table = classify_days(master)
    table = table[table["day"].isin(days.index)].copy()
    table = table.join(days[["sky", "tmin", "tmax", "wind"]], on="day")
    span = (table["tmax"] - table["tmin"]).replace(0, np.nan)
    table["temp_shape"] = (table["temp_c"] - table["tmin"]) / span
    table["wind_shape"] = table["wind_10m"] / table["wind"].replace(0, np.nan)
    rows = []
    for month in range(1, 13):
        window = [(month - 2) % 12 + 1, month, month % 12 + 1]
        for sky in SKY_CLASSES:
            sample = table[table["month"].isin(window) & (table["sky"] == sky)]
            if sample["day"].nunique() < MIN_DAYS:
                continue
            mean = sample.groupby("hour")[["solar_wm2", "direct_wm2", "diffuse_wm2",
                                           "temp_shape", "wind_shape"]].mean()
            mean["month"], mean["sky"], mean["days"] = month, sky, sample["day"].nunique()
            rows.append(mean.reset_index())
    return pd.concat(rows, ignore_index=True)


def make_day(library, month, sky, tmin, tmax, wind_10m, weekend=False):
    """Hourly inputs for the price model, for the day described.

    Returns a table of 24 rows indexed by hour-ending time on a reference date in
    the chosen month (a Wednesday, or a Saturday when `weekend` is True).
    """
    cell = library[(library["month"] == month) & (library["sky"] == sky)].sort_values("hour")
    if len(cell) != 24:
        raise ValueError(f"no typical day recorded for month {month} with sky '{sky}'")
    if tmax < tmin:
        raise ValueError("highest temperature must not be below the lowest")
    wanted = 5 if weekend else 2
    first = pd.Timestamp(2025, month, 1)
    date = first + pd.Timedelta(days=(wanted - first.dayofweek) % 7)
    hour_end = date + pd.to_timedelta(cell["hour"].to_numpy() + 1, unit="h")
    wind = np.clip(wind_10m * cell["wind_shape"].to_numpy(), 0, None)
    out = pd.DataFrame({
        "hour": cell["hour"].to_numpy(),
        "hour_sin": np.sin(2 * np.pi * cell["hour"].to_numpy() / 24),
        "hour_cos": np.cos(2 * np.pi * cell["hour"].to_numpy() / 24),
        "day_of_week": wanted, "is_weekend": int(weekend), "month": month,
        "temp_c": tmin + (tmax - tmin) * cell["temp_shape"].to_numpy(),
        "wind_10m": wind, "wind_100m": wind * WIND_100M_RATIO,
        "solar_wm2": cell["solar_wm2"].to_numpy(),
        "direct_wm2": cell["direct_wm2"].to_numpy(),
        "diffuse_wm2": cell["diffuse_wm2"].to_numpy()}, index=hour_end)
    out.index.name = "hour_end"
    return out
