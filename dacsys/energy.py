"""Local renewable supply profiles and conversion technologies.

Screening-level models:
  * wind: generic 3.x MW onshore turbine power curve at 100 m, density-corrected, 12 % losses
  * PV:   single-axis tracking, capacity factor from ERA5 global horizontal irradiance with a
          tracking gain and temperature derate
  * heat pump COP from a Carnot fraction; electric boiler efficiency 0.99
"""
from __future__ import annotations

import numpy as np

# generic IEC class II/III turbine power curve (m/s -> fraction of rated)
_V = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 25.01, 40])
_P = np.array([0, 0, .04, .10, .19, .31, .46, .63, .80, .93, 1.0, 1.0, 0, 0])


def wind_cf(u100, v100, T0, P, losses=0.12):
    v = np.hypot(u100, v100)
    rho = P / (287.05 * T0)
    v_eq = v * (rho / 1.225) ** (1 / 3)                      # density-equivalent wind speed
    return np.interp(v_eq, _V, _P) * (1 - losses)


def pv_cf(ssrd, T0, tracking_gain=1.25, pr=0.85):
    """ssrd: W/m2 global horizontal (hour mean). Returns AC capacity factor (per kWp)."""
    g = ssrd / 1000.0
    T_cell = (T0 - 273.15) + 25.0 * g
    derate = 1 - 0.0035 * (T_cell - 25.0)
    return np.clip(g * tracking_gain * pr * derate, 0, 1)


def heat_pump_cop(T_source_K, T_sink_K, carnot_frac=0.45, cop_max=5.0):
    lift = np.maximum(T_sink_K - T_source_K, 5.0)
    return np.clip(carnot_frac * T_sink_K / lift, 1.0, cop_max)
