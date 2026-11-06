"""Claude hydrogen liquefaction cycle (states 1-17 of the paper's Fig. 1).

Hydrogen properties from CoolProp (normal hydrogen, real fluid).

Model:
* Compressor - paper's isothermal model (Table 2):
      W_com = m R T0 ln(P4/P3) / eta_com,   T4 = T3.
* RHE2 hot side 4 -> 5 with T5 fixed (Table 1).
* Turbine 15 -> 16 with isentropic efficiency, P16 = P0.
* TV3 isenthalpic, separator gives saturated liquid 10 and vapour 11.
* For a given warm-end approach d = T4 - T17 the cold-box energy balance gives
  the liquid yield yL in closed form. RHE3/RHE4 are split so that the
  streams meeting at mixer 13 have equal temperature (h12 = h16), which
  maximises the smaller of the two approaches at T7.
* The warm-end approach of RHE2 is DT_MIN. Every heat exchanger is checked
  along its length (hydrogen cp varies strongly at cryogenic temperature).
  - Params.strict_pinch = False ("paper" mode): the turbine fraction x_tur
    of Table 7 is kept and any approach < DT_MIN is only reported.
  - Params.strict_pinch = True: x_tur is lowered until all approaches are
    >= DT_MIN, giving the maximum thermodynamically feasible yield.
* Steady state: liquid product m10 == feed m1 (set by the GAX evaporator).

Note: the paper's Table 7 contains a temperature cross (T12 = 247.6 K while
T7 = 131.2 K in RHE4); use strict_pinch=True for a thermodynamically feasible
cold box.
"""
from __future__ import annotations

import math

import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.optimize import brentq

from .gax import Infeasible
from .params import Params

FLUID = "Hydrogen"
R_H2 = 8.314462618 / 2.01588  # kJ/kgK
DT_MIN = 2.0                  # K minimum approach in RHE2-4 (assumed)
N_SEG = 8                     # segments for the internal-pinch check


def _h(T, P):  # K, kPa -> kJ/kg
    return PropsSI("H", "T", T, "P", P * 1e3, FLUID) / 1e3


def _s(T, P):
    return PropsSI("S", "T", T, "P", P * 1e3, FLUID) / 1e3


def _T_hP(h, P):
    return PropsSI("T", "H", h * 1e3, "P", P * 1e3, FLUID)


def _s_hP(h, P):
    return PropsSI("S", "H", h * 1e3, "P", P * 1e3, FLUID) / 1e3


def _h_sP(s, P):
    return PropsSI("H", "S", s * 1e3, "P", P * 1e3, FLUID) / 1e3


def _min_approach(h_hot_in, h_hot_out, P_hot, m_hot, h_cold_in, P_cold, m_cold):
    """Smallest hot-cold temperature difference along a counter-flow HX."""
    Q = m_hot * (h_hot_in - h_hot_out)
    q = np.linspace(0.0, Q, N_SEG + 1)
    dts = []
    for qi in q:
        Th = _T_hP(h_hot_out + qi / m_hot, P_hot)
        Tc = _T_hP(h_cold_in + qi / m_cold + 1e-6, P_cold)
        dts.append(Th - Tc)
    return min(dts)


class _Consts:
    def __init__(self, p: Params):
        P0, PH = p.P0, p.P_high
        self.h5 = _h(p.T5, PH)
        self.h16 = self.h5 - p.eta_tur * (self.h5 - _h_sP(_s(p.T5, PH), P0))
        self.T_sat = PropsSI("T", "P", P0 * 1e3, "Q", 0, FLUID)
        self.hL = PropsSI("H", "P", P0 * 1e3, "Q", 0, FLUID) / 1e3
        self.hV = PropsSI("H", "P", P0 * 1e3, "Q", 1, FLUID) / 1e3


def _cold_box(p: Params, c: _Consts, h2: float, T2: float, d: float):
    """Solve the cycle (per kg/s of feed) for warm-end approach d [K]."""
    P0, PH = p.P0, p.P_high
    r = p.x_tur / (1.0 - p.x_tur)
    h11 = c.hV

    def at(T3):
        h4 = _h(T3, PH)
        h17 = _h(T3 - d, P0)
        A = c.h5 + h11 + r * c.h16 - (1.0 + r) * h17 + (h4 - c.h5) / (1.0 - p.x_tur)
        yL = (c.hV - A) / (h17 - c.hL)
        if not 1e-4 < yL <= 1.0:
            return None
        m6 = 1.0 / yL
        m5 = m6 / (1.0 - p.x_tur)
        m13 = m5 - 1.0
        # mixer 3 residual: m5 h3 - m1 h2 - m13 h17 (m1 = 1)
        return h4, h17, A, yL, m5, m6, m13, m5 * _h(T3, P0) - h2 - m13 * h17

    # Solve the mixer-3 closure for T3 exactly (fixed-point iteration converges
    # very slowly when the recycle flow is much larger than the feed).
    T_hi = T2
    if at(T_hi) is None:
        return None
    T_lo = T_hi
    while True:
        T_lo -= 10.0
        if T_lo <= p.T5 + d + 0.5:
            return None
        v = at(T_lo)
        if v is None:
            return None
        if v[-1] < 0:
            break
    T3 = brentq(lambda T: at(T)[-1], T_lo, T_hi, xtol=1e-9)
    h4, h17, A, yL, m5, m6, m13, _ = at(T3)
    m15, m11 = m5 - m6, m6 - 1.0
    h8 = A + yL * (h17 - h11)
    h14 = h17 - m5 * (h4 - c.h5) / m13
    h12 = c.h16                                   # equal-temperature mixing at 13
    h7 = h8 + m11 * (h12 - h11) / m6
    h13 = c.h16
    if not (h8 < h7 < c.h5):
        return None
    dt = min(
        _min_approach(h7, h8, PH, m6, h11, P0, m11),          # RHE4
        _min_approach(c.h5, h7, PH, m6, h13, P0, m13),        # RHE3
        _min_approach(h4, c.h5, PH, m5, h14, P0, m13),        # RHE2
    )
    return dict(dt=dt, yL=yL, T3=T3, m5=m5, m6=m6, m11=m11, m13=m13, m15=m15,
                h4=h4, h7=h7, h8=h8, h12=h12, h13=h13, h14=h14, h17=h17)


def solve_claude(p: Params, m1: float, T1: float, T2: float) -> dict:
    P0, PH = p.P0, p.P_high
    c = _Consts(p)
    h1, h2 = _h(T1, P0), _h(T2, P0)
    x_tur = p.x_tur

    def g(x):
        sol = _cold_box(p.with_(x_tur=x), c, h2, T2, DT_MIN)
        return (-1e3 if sol is None else sol["dt"] - DT_MIN + 1e-3), sol

    g0, s = g(x_tur)
    if s is None:
        raise Infeasible("Claude cold box: no liquid yield")
    if p.strict_pinch and g0 < 0:
        # largest turbine fraction that keeps every approach >= DT_MIN
        x_lo = next((x for x in np.arange(0.05, x_tur, 0.025) if g(x)[0] >= 0), None)
        if x_lo is None:
            raise Infeasible("no pinch-feasible Claude cold box")
        x_tur = brentq(lambda x: g(x)[0], x_lo, x_tur, xtol=1e-4)
        _, s = g(x_tur)
    k = m1  # everything above is per kg/s of feed
    m5, m6, m11, m13, m15 = (k * s[n] for n in ("m5", "m6", "m11", "m13", "m15"))
    T3 = s["T3"]
    h3 = _h(T3, P0)
    W_com = m5 * R_H2 * p.T0 * math.log(PH / P0) / p.eta_com
    Q_ic = W_com - m5 * (s["h4"] - h3)          # heat rejected by isothermal compression
    W_tur = m15 * (c.h5 - c.h16)

    table = [
        (1, h1, P0, m1), (2, h2, P0, m1), (3, h3, P0, m5), (4, s["h4"], PH, m5),
        (5, c.h5, PH, m5), (6, c.h5, PH, m6), (7, s["h7"], PH, m6), (8, s["h8"], PH, m6),
        (9, s["h8"], P0, m6), (10, c.hL, P0, m1), (11, c.hV, P0, m11),
        (12, s["h12"], P0, m11), (13, s["h13"], P0, m13), (14, s["h14"], P0, m13),
        (15, c.h5, PH, m15), (16, c.h16, P0, m15), (17, s["h17"], P0, m13),
    ]
    st = {}
    for n, h, P, m in table:
        if n in (10, 11):
            T = c.T_sat
            sv = PropsSI("S", "P", P0 * 1e3, "Q", 0 if n == 10 else 1, FLUID) / 1e3
        else:
            T, sv = _T_hP(h, P), _s_hP(h, P)
        st[n] = dict(T=T, P=P, h=h, s=sv, m=m)

    return dict(states=st, yL=s["yL"], m_liq=m1, W_com=W_com, W_tur=W_tur, Q_ic=Q_ic,
                Q_RHE2=m5 * (s["h4"] - c.h5), Q_RHE3=m6 * (c.h5 - s["h7"]),
                Q_RHE4=m6 * (s["h7"] - s["h8"]), min_approach=s["dt"], x_tur=x_tur,
                warm_end_dT=T3 - st[17]["T"],
                rho_L=PropsSI("D", "P", P0 * 1e3, "Q", 0, FLUID))


def dead_state(T0: float, P0: float):
    return _h(T0, P0), _s(T0, P0)


def exergy(state: dict, T0: float, h0: float, s0: float) -> float:
    """Physical exergy rate [kW] of a hydrogen state."""
    return state["m"] * ((state["h"] - h0) - T0 * (state["s"] - s0))
