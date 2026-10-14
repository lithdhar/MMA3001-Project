"""Daily battery advice for a household, from a simple description of the day.

Steps: describe the day (src/day_profile.py) -> forecast hourly prices from the
weather and time of day -> work out solar output and household use -> let the
battery planner (src/controllers.py) choose the cheapest schedule -> compare the
cost with the simpler alternatives.

All costs are EXPECTED costs: they use the forecast prices, because the real
prices for the day are not known in advance.
"""
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import BATTERY_CAPACITY_KWH, BATTERY_POWER_KW, BATTERY_ROUNDTRIP_EFF, TARIFF
from .controllers import controller_a, controller_b, controller_plan
from .dispatch import buy_price, feed_in_price
from .load import hourly_load
from .pv import pv_power

PRICE_INPUTS = ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "month", "temp_c",
                "wind_10m", "wind_100m", "solar_wm2", "direct_wm2", "diffuse_wm2"]
STRATEGIES = {"A": "No battery (solar, then grid)",
              "B": "Simple rule (store surplus solar, never charge from the grid)",
              "C": "Forecast-driven plan (recommended)"}
THRESHOLD_KW = 0.05          # flows below this are treated as zero when naming an action


def fit_price_model(features, C=0.1):
    """SVR that forecasts the hourly price ($/MWh) from weather and time of day only.

    `features` is a table like data/processed/features_h72.csv (price in the last column).
    """
    model = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=C, epsilon=0.1)),
        transformer=StandardScaler())
    return model.fit(features[PRICE_INPUTS], features[features.columns[-1]])


def action_labels(run):
    """Plain-English action for each hour of a controller run."""
    p, imp, exp = run["p_battery_kw"], run["import_kwh"], run["export_kwh"]
    labels = []
    for charge, grid_in, grid_out in zip(p, imp, exp):
        if charge > THRESHOLD_KW:
            labels.append("Charge battery from the grid" if grid_in > THRESHOLD_KW
                          else "Charge battery from solar")
        elif charge < -THRESHOLD_KW:
            labels.append("Discharge battery to the grid" if grid_out > THRESHOLD_KW
                          else "Discharge battery to the home")
        elif grid_in > THRESHOLD_KW:
            labels.append("Use grid power")
        elif grid_out > THRESHOLD_KW:
            labels.append("Export surplus solar")
        else:
            labels.append("Solar covers the home")
    return labels


def day_cost(run, spot_mwh, tariff):
    """Expected cost of one day ($): imports bought, exports sold, plus the supply charge."""
    bought = float(np.sum(run["import_kwh"] * buy_price(spot_mwh, tariff)))
    sold = float(np.sum(run["export_kwh"] * feed_in_price(spot_mwh, tariff)))
    supply = tariff["supply_per_day"]
    return {"import_cost": bought, "export_income": sold,
            "supply_charge": supply, "total": bought - sold + supply}


def advise(day, price_model, profile, scale=1.0, pv_kw=None, capacity=BATTERY_CAPACITY_KWH,
           power_kw=BATTERY_POWER_KW, roundtrip_eff=BATTERY_ROUNDTRIP_EFF, start_soc=0.5,
           tariff=TARIFF, prices=None):
    """Advice and cost comparison for one day.

    day         : 24-row table from day_profile.make_day
    price_model : fitted model from fit_price_model (ignored if `prices` is given)
    profile     : household profile from src.load.load_profile
    prices      : optional list of 24 hourly spot prices ($/MWh) to use instead of the forecast

    Returns a dict with 'hourly' (today's schedule), 'costs_repeating' (cost of a day like
    this when it repeats), 'costs_today' (cost of today from the starting charge) and 'prices'.
    """
    if prices is not None:
        spot = np.asarray(prices, float)
    else:
        spot = price_model.predict(day[PRICE_INPUTS])
    pv_options = {} if pv_kw is None else {"rated_kw": pv_kw}
    solar = pv_power(day["solar_wm2"].to_numpy(), day["temp_c"].to_numpy(), **pv_options)
    load = hourly_load(day.index, profile, scale).to_numpy()
    eta = roundtrip_eff ** 0.5
    energy0 = start_soc * capacity
    battery = dict(capacity=capacity, p_max=power_kw)

    def run_all(days, start):
        """Run the three strategies over `days` identical days, starting at `start` kWh."""
        pv, use, price = np.tile(solar, days), np.tile(load, days), np.tile(spot, days)
        return {
            "A": controller_a(pv, use),
            "B": controller_b(pv, use, eta_c=eta, eta_d=eta, energy0=start, **battery),
            "C": controller_plan(pv, use, price, tariff=tariff, eta=eta, energy0=start, **battery),
        }

    # Today: plan two days ahead (tomorrow assumed the same) so that the plan does not
    # drain the battery just because the window ended; only the first 24 hours are carried out.
    today = {k: r.iloc[:24] for k, r in run_all(2, energy0).items()}
    # A day like this when it repeats: the third of four identical days, so the answer does
    # not depend on today's starting charge and every plan still sees a full day ahead.
    steady = {k: r.iloc[48:72] for k, r in run_all(4, energy0).items()}

    def cost_table(runs):
        table = pd.DataFrame({k: day_cost(r, spot, tariff) for k, r in runs.items()}).T
        table.insert(0, "strategy", [STRATEGIES[k] for k in table.index])
        table["battery_end_pct"] = [np.nan if k == "A" else 100 * runs[k]["soc"].iloc[-1]
                                    for k in table.index]
        return table
        """Cost table for the strategies in `runs`, with the battery level at the end of the day."""
        
    hourly = pd.DataFrame({"hour": day["hour"].to_numpy(), "price_mwh": spot,
                           "solar_kwh": solar, "load_kwh": load}, index=day.index)
    for k in ("B", "C"):
        hourly[f"battery_pct_{k}"] = 100 * today[k]["soc"].to_numpy()
    hourly["action"] = action_labels(today["C"])
    hourly["battery_kw"] = today["C"]["p_battery_kw"].to_numpy()
    hourly["import_kwh"] = today["C"]["import_kwh"].to_numpy()
    hourly["export_kwh"] = today["C"]["export_kwh"].to_numpy()
    return {"hourly": hourly, "costs_repeating": cost_table(steady),
            "costs_today": cost_table(today), "prices": spot, "runs_today": today}
