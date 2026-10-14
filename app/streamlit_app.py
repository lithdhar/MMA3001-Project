"""Interactive battery advisor: describe a day, get a recommendation and a cost comparison.

Run locally with:  python -m streamlit run app/streamlit_app.py
"""
import calendar
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.advisor import advise, fit_price_model
from src.config import TARIFF
from src.day_profile import SKY_CLASSES, build_typical_days, make_day
from src.load import load_profile

MONTHS = list(calendar.month_name)[1:]
SKY_HELP = {"Clear": "almost no cloud", "Partly cloudy": "some cloud, sunny spells",
            "Mostly cloudy": "mostly cloud, little sun", "Overcast": "thick cloud all day"}

st.set_page_config(page_title="Solar battery advisor", layout="wide")


@st.cache_resource(show_spinner="Preparing the price model (first run only)...")
def load_engine():
    """Read the data once, build the typical-day library and fit the price model."""
    master = pd.read_csv(ROOT / "data" / "processed" / "master_hourly.csv",
                         index_col=0, parse_dates=True)
    features = pd.read_csv(ROOT / "data" / "processed" / "features_h72.csv",
                           index_col=0, parse_dates=True)
    profile = load_profile(ROOT / "data" / "raw" / "rbee extract by hour and state.csv")
    return build_typical_days(master), fit_price_model(features), profile


def sidebar_inputs():
    """Collect every input from the sidebar and return them as a dict."""
    s = st.sidebar
    s.header("1. Describe the day")
    month = s.selectbox("Month", MONTHS, index=0)
    weekend = s.radio("Day type", ["Weekday", "Weekend"], horizontal=True) == "Weekend"
    sky = s.selectbox("Sky", SKY_CLASSES, index=0, format_func=lambda x: f"{x} ({SKY_HELP[x]})")
    tmin, tmax = s.slider("Lowest and highest temperature (°C)", -5, 45, (14, 30))
    wind = s.slider("Average wind speed at 10 m (m/s)", 0.0, 15.0, 4.0, 0.5)

    s.header("2. Your home")
    pv_kw = s.slider("Solar panels (kW)", 0.0, 15.0, 6.6, 0.1)
    size = s.slider("Household use relative to the typical Victorian home", 0.25, 3.0, 1.0, 0.05)
    capacity = s.slider("Battery capacity (kWh)", 1.0, 30.0, 10.0, 0.5)
    power = s.slider("Battery maximum power (kW)", 1.0, 15.0, 5.0, 0.5)
    start_pct = s.slider("Battery charge at midnight (%)", 0, 100, 50, 5)

    s.header("3. Tariff")
    network = s.number_input("Network and other charges (c/kWh, before GST)", 0.0, 60.0, 15.0, 0.5)
    supply = s.number_input("Daily supply charge ($)", 0.0, 5.0, float(TARIFF["supply_per_day"]),
                            0.05)
    fixed_fit = s.checkbox("My feed-in tariff is a fixed rate (not the spot price)")
    fit_rate = s.number_input("Fixed feed-in rate (c/kWh)", 0.0, 30.0, 5.0, 0.5,
                              disabled=not fixed_fit)
    tariff = {**TARIFF, "network_other_c_kwh": network, "supply_per_day": supply,
              "feed_in_mode": "fixed" if fixed_fit else "spot", "feed_in_fixed_c_kwh": fit_rate}
    return dict(month=MONTHS.index(month) + 1, weekend=weekend, sky=sky, tmin=tmin, tmax=tmax,
                wind=wind, pv_kw=pv_kw, size=size, capacity=capacity, power=power,
                start_soc=start_pct / 100, tariff=tariff)


def show_headline(costs, label):
    """Three cost tiles and the saving that the plan achieves."""
    a, b, c = (costs.loc[k, "total"] for k in "ABC")
    cols = st.columns(3)
    cols[0].metric("A. No battery", f"${a:.2f}")
    cols[1].metric("B. Simple rule", f"${b:.2f}", f"{b - a:+.2f} vs A", delta_color="inverse")
    cols[2].metric("C. Forecast plan", f"${c:.2f}", f"{c - a:+.2f} vs A", delta_color="inverse")
    st.caption(f"{label} Totals include the daily supply charge. A negative change is a saving.")


def main():
    st.title("Solar battery advisor")
    st.write("Describe a day, and the advisor forecasts the electricity prices from the weather, "
             "plans the cheapest battery schedule and compares its cost with two simpler "
             "approaches. It is a university research prototype.")
    try:
        library, price_model, profile = load_engine()
    except FileNotFoundError as error:
        st.error(f"A data file is missing: {error.filename}")
        st.stop()
    inputs = sidebar_inputs()
    try:
        day = make_day(library, inputs["month"], inputs["sky"], inputs["tmin"], inputs["tmax"],
                       inputs["wind"], inputs["weekend"])
    except ValueError as error:
        st.warning(str(error))
        st.stop()
    result = advise(day, price_model, profile, scale=inputs["size"], pv_kw=inputs["pv_kw"],
                    capacity=inputs["capacity"], power_kw=inputs["power"],
                    start_soc=inputs["start_soc"], tariff=inputs["tariff"])
    hourly = result["hourly"]

    st.subheader("Expected cost of the day")
    tab_repeat, tab_today = st.tabs(["A day like this, repeated", "Today only"])
    with tab_repeat:
        show_headline(result["costs_repeating"],
                      "Cost of one day when every day looks like this one; independent of the "
                      "starting charge.")
    with tab_today:
        show_headline(result["costs_today"],
                      "Cost of today alone, starting from the charge you chose. Strategy C may end "
                      "the day with less stored energy, which is shown in the table below.")
    cols = ["strategy", "import_cost", "export_income", "supply_charge", "total", "battery_end_pct"]
    names = {"strategy": "Strategy", "import_cost": "Bought ($)", "export_income": "Sold ($)",
             "supply_charge": "Supply ($)", "total": "Total ($)",
             "battery_end_pct": "Battery at end (%)"}
    with st.expander("Cost breakdown"):
        st.write("A day like this, repeated")
        st.dataframe(result["costs_repeating"][cols].rename(columns=names).round(2))
        st.write("Today only")
        st.dataframe(result["costs_today"][cols].rename(columns=names).round(2))

    st.subheader("What to do, hour by hour (strategy C)")
    hour = st.slider("Hour of the day (start of the hour)", 0, 23, 12)
    row = hourly.iloc[hour]
    st.info(f"**{hour:02d}:00 to {hour + 1:02d}:00: {row['action']}.** "
            f"Forecast price ${row['price_mwh']:.0f}/MWh, solar {row['solar_kwh']:.2f} kWh, "
            f"home use {row['load_kwh']:.2f} kWh, battery {row['battery_pct_C']:.0f}% full "
            f"at the end of the hour.")
    left, right = st.columns(2)
    with left:
        st.write("Forecast price ($/MWh)")
        st.line_chart(hourly.set_index("hour")["price_mwh"])
        st.write("Battery charge (%)")
        st.line_chart(hourly.set_index("hour")[["battery_pct_B", "battery_pct_C"]]
                      .rename(columns={"battery_pct_B": "B. Simple rule",
                                       "battery_pct_C": "C. Forecast plan"}))
    with right:
        st.write("Solar output and home use (kWh per hour)")
        st.line_chart(hourly.set_index("hour")[["solar_kwh", "load_kwh"]]
                      .rename(columns={"solar_kwh": "Solar", "load_kwh": "Home use"}))
        st.write("Battery power (kW; positive is charging)")
        st.bar_chart(hourly.set_index("hour")["battery_kw"])
    with st.expander("Full hourly schedule"):
        st.dataframe(hourly.round(2))

    st.divider()
    st.caption(
        "Limitations. Costs are expected values at forecast prices; real prices on the day will "
        "differ. The day is a typical shape for the chosen month and sky, scaled to your "
        "temperatures and wind. The price model uses weather and time only, so it cannot see "
        "outages, bidding behaviour or demand events. Battery: 90% round trip, grid-connected. "
        "This is not financial advice.")


main()
