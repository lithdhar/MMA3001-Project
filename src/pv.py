"""Rooftop solar (PV) output from sunlight and air temperature."""
import numpy as np

from .config import (PV_CELL_HEATING, PV_PERFORMANCE_RATIO, PV_RATED_KW,
                     PV_TEMP_COEFF)


def pv_power(radiation_wm2, temp_c, rated_kw=PV_RATED_KW,
             performance_ratio=PV_PERFORMANCE_RATIO,
             temp_coeff=PV_TEMP_COEFF, cell_heating=PV_CELL_HEATING):
    """Return PV output in kW.

    P = P_rated * (G / 1000) * PR * [1 - c * (T + h * G - 25)]

    G is sunlight in W/m2, T the air temperature in degrees C, PR the
    performance ratio, c the temperature coefficient and h the cell heating
    factor. Output is never negative. Inputs may be numbers, arrays or pandas
    Series; a Series comes back as a Series. Because the time step is one hour,
    kW and kWh per hour are numerically equal.
    """
    cell_temp = temp_c + cell_heating * radiation_wm2
    power = (rated_kw * (radiation_wm2 / 1000.0) * performance_ratio
             * (1.0 - temp_coeff * (cell_temp - 25.0)))
    return np.maximum(power, 0.0)