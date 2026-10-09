"""Assumptions used throughout Stage 2. Change them here only."""

# Household size scaling applied to the typical household load
HOUSEHOLD_SCALE = {"small": 0.6, "average": 1.0, "large": 1.4}

# Rooftop solar system
PV_RATED_KW = 6.6            # system size (kW)
PV_PERFORMANCE_RATIO = 0.8   # inverter, cable and dirt losses
PV_TEMP_COEFF = 0.004        # fractional power loss per degree C above 25 C
PV_CELL_HEATING = 0.03       # cell temperature rise per W/m2 of sunlight (C)

# Home battery
BATTERY_CAPACITY_KWH = 10.0       # usable energy capacity (kWh)
BATTERY_POWER_KW = 5.0            # maximum charge or discharge power (kW)
BATTERY_ROUNDTRIP_EFF = 0.90      # energy out / energy in over a full cycle
BATTERY_EFF = BATTERY_ROUNDTRIP_EFF ** 0.5   # one-way efficiency (charge = discharge)
BATTERY_SELF_DISCHARGE = 0.001    # fraction of stored energy lost per hour
BATTERY_START_SOC = 0.5           # state of charge at the start of the simulation