# MMA3001-Project
Electricity Price Predictor

# Data sources

Data used for the MMA3001 project: forecasting Victorian electricity prices 3 to 5 days ahead and testing a solar-battery controller for a household.

All files in `data/raw` are original and must not be edited. Cleaned data goes in `data/processed`.

## Files

| File | Contents | Period | Resolution | Source |
| --- | --- | --- | --- | --- |
| `aemo_vic_raw_price_demand.csv` | Victorian (VIC1) regional reference price (RRP, $/MWh) and total demand (MW) | 1 Dec 2024 to 1 Oct 2026 | 5-minute, 192,672 rows | AEMO, Aggregated Price and Demand data, monthly VIC1 files |
| `aemo_vic_hourly_price_demand.csv` | Hourly version of the file above: mean, max and min price, mean demand, and price in $/kWh | 1 Dec 2024 to 1 Oct 2026 | Hourly, 16,056 rows | Derived from the raw AEMO file (see below) |
| `weather_melbourne_hourly.csv` | Melbourne temperature, 10 m and 100 m wind speed, shortwave, direct and diffuse radiation | 1 Jan 2021 to 24 Sep 2026 | Hourly, 50,232 rows | Open-Metro Historical Forecast API |
| `vic_5mlp_hourly.csv` | Victorian network load profile for five distribution areas plus a total | 29 Dec 2024 to 28 Dec 2025 | Hourly, 8,736 rows | AEMO net system load profile (NSLP) weekly files for 2025, aggregated to hourly |
| `rbee extract by hour and state.csv` | Average household electricity use by state, month, weekday or weekend, and time of day | Typical day (no calendar year) | Half-hourly | Michael Ambrose - Pricipal Research Scientist at CSIRO |

## Data kept outside the repository

These are too large for GitHub and are listed in `.gitignore`.

| Folder | Contents | Source |
| --- | --- | --- |
| `pd7day_raw` | 180 AEMO 7-day pre-dispatch price forecast files (PD7DAY), about 3 per day, 1 Aug to 30 Sep 2026 | AEMO |
| `nslp_raw` and `nslp_extracted` | 52 weekly AEMO frozen load profile files for 2025, zipped and extracted | AEMO |

The PD7DAY files are AEMO's own forecasts, not actual prices. They are used only to benchmark my forecast models for August and September 2026.

## How the AEMO price and demand data was obtained

Downloaded on 6 October 2026 using a Google Colab script. The script requests the monthly VIC1 files from AEMO (December 2024 to September 2026) and joins them. It then averages the 5-minute values to hourly values.

- Timestamps mark the end of each interval, in National Electricity Market (NEM) time, which is UTC+10 all year with no daylight saving.
- Hourly values use hour-ending timestamps: 01:00 covers 00:05 to 01:00. This matches the load-profile file.
- Every hour contains all 12 five-minute readings. There are no missing values and no duplicates.
- Price is in $/MWh. The column `rrp_per_kwh` divides by 1,000 to give $/kWh.

## Checks done on the raw data

- Weather and load-profile timestamps follow the same fixed clock as AEMO. There is no daylight-saving jump, so no time shift is needed.
- The load profile and AEMO market demand have a correlation of 0.946 when aligned, which supports the alignment.
- 3,638 hours (about 23%) have a negative average price. These are real and are kept.
- Price spikes reach $19,070/MWh. They are real and are kept.
- Weather ends on 24 Sep 2026, so the master table is trimmed to that date.

## Assumptions

- Actual weather stands in for forecast weather.
- The tariff is spot-linked, with network charges, GST and feed-in rate set out in the report.

## Time conventions

All data in this project is aligned to a single clock: **National Electricity Market (NEM) time**. This is Australian Eastern Standard Time (UTC+10) throughout the year, with no daylight saving.

- Every timestamp marks the **end** of its interval (hour-ending). For example, 13:00 covers 12:00 to 13:00.
- All timestamps are stored without a time-zone label and are interpreted as NEM time.
- Hourly values are formed from AEMO's 5-minute data. Row 01:00 averages the readings from 00:05 to 01:00.

### Alignment evidence

| File | Evidence that it is on NEM time |
| --- | --- |
| AEMO price and demand | AEMO's published convention. The hourly series has consecutive one-hour steps over 669 days, with no 23-hour or 25-hour days. |
| Weather (Melbourne) | The daily centre of mass of solar radiation stays between about 12.6 and 13.1 all year, with no daylight-saving jump. Solar noon in Melbourne is about 12:05 to 12:35 NEM time, which matches hour-ending labels. |
| Victorian load profile (2025) | Correlation with AEMO demand is highest at zero lag (0.946) and lower when shifted by one hour. |
| AEMO price forecasts (PD7DAY) | Published in NEM time. |
| Household load (rbee) | The household file contains only a time of day. Values are assumed to be kWh per half-hour, and each time label is treated as the start of the half-hour. The data supplier has been asked to confirm the units, the label convention and whether the clock includes daylight saving. A one-hour shift is tested as a sensitivity case. Public holidays are treated as ordinary weekdays or weekends. |

### Open item

The household load file contains only a time of day. If it is on local clock time with daylight saving, evening values in summer would sit one hour later than NEM time. This will be confirmed with the data supplier. If it cannot be confirmed, a one-hour shift will be tested as a sensitivity case.

### Automated check

`tests/test_time_alignment.py` fails if the merged table contains gaps, duplicate hours, daylight-saving jumps, a solar peak inconsistent with NEM time, or a misaligned load profile.