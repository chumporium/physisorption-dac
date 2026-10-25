"""Ammonia-water mixture properties.

Correlations of Patek & Klomfar (1995), "Simple functions for fast calculations
of selected thermodynamic properties of the ammonia-water system",
Int. J. Refrig. 18(4), 228-234.

Units: T [K], P [kPa], x/y = ammonia mass fraction [-], h [kJ/kg].
Enthalpy reference: Patek-Klomfar (liquid water at 273.16 K ~ 0).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

# molar masses (kg/kmol)
M_NH3 = 17.03026
M_H2O = 18.01528

# --- Eq. (1) bubble-point temperature T(p, x) -------------------------------
_T_BUB = np.array([
    (0, 0, +0.322302e1), (0, 1, -0.384206e0), (0, 2, +0.460965e-1),
    (0, 3, -0.378945e-2), (0, 4, +0.135610e-3), (1, 0, +0.487755e0),
    (1, 1, -0.120108e0), (1, 2, +0.106154e-1), (2, 3, -0.533589e-3),
    (4, 0, +0.785041e1), (5, 0, -0.115941e2), (5, 1, -0.523150e-1),
    (6, 0, +0.489596e1), (13, 1, +0.421059e-1),
])
# --- Eq. (2) vapour in equilibrium y(p, x) ----------------------------------
_Y_EQ = np.array([
    (0, 0, +1.98022017e1), (0, 1, -1.18092669e1), (0, 6, +2.77479980e1),
    (0, 7, -2.88634277e1), (1, 0, -5.91616608e1), (2, 1, +5.78091305e2),
    (2, 2, -6.21736743e0), (3, 2, -3.42198402e3), (4, 3, +1.19403127e4),
    (5, 4, -2.45413777e4), (6, 5, +2.91591865e4), (7, 6, -1.84782290e4),
    (7, 7, +2.34819434e1), (8, 7, +4.80310617e3),
])
# --- Eq. (3) dew-point temperature T(p, y) ----------------------------------
_T_DEW = np.array([
    (0, 0, +0.324004e1), (0, 1, -0.395920e0), (0, 2, +0.435624e-1),
    (0, 3, -0.218943e-2), (1, 0, -0.143526e1), (1, 1, +0.105256e1),
    (1, 2, -0.719281e-1), (2, 0, +0.122362e2), (2, 1, -0.224368e1),
    (3, 0, -0.201780e2), (3, 1, +0.110834e1), (4, 0, +0.145399e2),
    (4, 2, +0.644312e0), (5, 0, -0.221246e1), (5, 2, -0.756266e0),
    (6, 0, -0.135529e1), (7, 2, +0.183541e0),
])
# --- Eq. (4) liquid enthalpy h_l(T, x) --------------------------------------
_H_LIQ = np.array([
    (0, 1, -0.761080e1), (0, 4, +0.256905e2), (0, 8, -0.247092e3),
    (0, 9, +0.325952e3), (0, 12, -0.158854e3), (0, 14, +0.619084e2),
    (1, 0, +0.114314e2), (1, 1, +0.118157e1), (2, 1, +0.284179e1),
    (3, 3, +0.741609e1), (5, 3, +0.891844e3), (5, 4, -0.161309e4),
    (5, 5, +0.622106e3), (6, 2, -0.207588e3), (6, 4, -0.687393e1),
    (8, 0, +0.350716e1),
])
# --- Eq. (5) vapour enthalpy h_g(T, y) --------------------------------------
_H_VAP = np.array([
    (0, 0, +0.128827e1), (1, 0, +0.125247e0), (2, 0, -0.208748e1),
    (3, 0, +0.217696e1), (0, 2, +0.235687e1), (1, 2, -0.886987e1),
    (2, 2, +0.102635e2), (3, 2, -0.237440e1), (0, 3, -0.670515e1),
    (1, 3, +0.164508e2), (2, 3, -0.936849e1), (0, 4, +0.842254e1),
    (1, 4, -0.858807e1), (0, 5, -0.277049e1), (4, 6, -0.961248e0),
    (2, 7, +0.988009e0), (1, 10, +0.308482e0),
])

_P0 = 2000.0  # kPa


def _mass_to_mole(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    n_a = x / M_NH3
    return n_a / (n_a + (1.0 - x) / M_H2O)


def _mole_to_mass(xm: float) -> float:
    xm = min(max(xm, 0.0), 1.0)
    m_a = xm * M_NH3
    return m_a / (m_a + (1.0 - xm) * M_H2O)


def T_bubble(P: float, x: float) -> float:
    """Bubble-point temperature [K] of liquid with NH3 mass fraction x."""
    xm = _mass_to_mole(x)
    L = math.log(_P0 / P)
    m, n, a = _T_BUB.T
    return 100.0 * float(np.sum(a * (1.0 - xm) ** m * L ** n))


def T_dew(P: float, y: float) -> float:
    """Dew-point temperature [K] of vapour with NH3 mass fraction y."""
    ym = _mass_to_mole(y)
    L = math.log(_P0 / P)
    m, n, a = _T_DEW.T
    return 100.0 * float(np.sum(a * (1.0 - ym) ** (m / 4.0) * L ** n))


def y_equilibrium(P: float, x: float) -> float:
    """NH3 mass fraction of vapour in equilibrium with saturated liquid x."""
    xm = _mass_to_mole(x)
    if xm >= 1.0:
        return 1.0
    m, n, a = _Y_EQ.T
    s = float(np.sum(a * (P / _P0) ** m * xm ** (n / 3.0)))
    ym = 1.0 - math.exp(math.log(1.0 - xm) * s)
    return _mole_to_mass(ym)


def h_liquid(T: float, x: float) -> float:
    """Saturated/subcooled liquid enthalpy [kJ/kg]."""
    xm = _mass_to_mole(x)
    m, n, a = _H_LIQ.T
    return 100.0 * float(np.sum(a * (T / 273.16 - 1.0) ** m * xm ** n))


def h_vapour(T: float, y: float) -> float:
    """Saturated/superheated vapour enthalpy [kJ/kg] (ideal-mixture approx.)."""
    ym = _mass_to_mole(y)
    m, n, a = _H_VAP.T
    base = max(1.0 - ym, 0.0)
    return 1000.0 * float(np.sum(a * (1.0 - T / 324.0) ** m * base ** (n / 4.0)))


def x_bubble(P: float, T: float) -> float:
    """Liquid NH3 mass fraction that boils at (P, T)."""
    return brentq(lambda x: T_bubble(P, x) - T, 0.0, 1.0, xtol=1e-10)


def y_dew(P: float, T: float) -> float:
    """Vapour NH3 mass fraction that condenses at (P, T)."""
    return brentq(lambda y: T_dew(P, y) - T, 0.0, 1.0, xtol=1e-12)


def P_bubble(T: float, x: float) -> float:
    """Saturation pressure [kPa] of liquid x at temperature T."""
    return brentq(lambda P: T_bubble(P, x) - T, 1.0, 20000.0, xtol=1e-8)


def flash(P: float, T: float, z: float):
    """Two-phase flash at (P, T) for overall composition z.

    Returns (quality, x_liquid, y_vapour, h_mixture). Quality is clipped to
    [0, 1]; outside the two-phase dome the single-phase enthalpy is returned.
    """
    Tb, Td = T_bubble(P, z), T_dew(P, z)
    if T <= Tb:
        return 0.0, z, y_equilibrium(P, z), h_liquid(T, z)
    if T >= Td:
        return 1.0, z, z, h_vapour(T, z)
    xl, yv = x_bubble(P, T), y_dew(P, T)
    q = (z - xl) / (yv - xl)
    return q, xl, yv, q * h_vapour(T, yv) + (1.0 - q) * h_liquid(T, xl)


def specific_volume_liquid(T: float, x: float) -> float:
    """Rough liquid specific volume [m3/kg] (ideal mixing of pure liquids)."""
    rho_nh3 = 638.6 - 1.6 * (T - 293.15)   # approx. saturated NH3 liquid
    rho_h2o = 998.0 - 0.3 * (T - 293.15)
    return x / rho_nh3 + (1.0 - x) / rho_h2o
