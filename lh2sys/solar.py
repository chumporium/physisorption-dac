"""Parabolic-trough collector field + storage tank with Therminol VP-1."""
from __future__ import annotations

import math

from .params import Params


def cp_vp1(T: float) -> float:
    """Therminol VP-1 specific heat [kJ/kgK] (manufacturer fit, T in K)."""
    t = T - 273.15
    return 1.498 + 0.002414 * t + 5.9591e-6 * t**2 - 2.9879e-8 * t**3 + 4.4172e-11 * t**4


def h_vp1(T: float) -> float:
    """Therminol VP-1 enthalpy [kJ/kg] relative to 0 degC (integrated cp)."""
    t = T - 273.15
    return 1.498 * t + 0.002414 / 2 * t**2 + 5.9591e-6 / 3 * t**3 - 2.9879e-8 / 4 * t**4 + 4.4172e-11 / 5 * t**5


def incidence_modifier(theta_deg: float) -> float:
    """LS-2 incident angle modifier K(theta) (Dudley et al.)."""
    th = theta_deg
    return max(math.cos(math.radians(th)) + 0.000884 * th - 0.00005369 * th**2, 0.0)


def solar_exergy(Q_sol: float, T0: float, T_sun: float) -> float:
    """Petela exergy of solar radiation [kW] (paper, below Eq. 7)."""
    r = T0 / T_sun
    return Q_sol * (1.0 - 4.0 / 3.0 * r + r**4 / 3.0)


def solve_solar_loop(p: Params) -> dict:
    """Useful heat delivered to the GAX generator [kW] and loop temperatures."""
    Q_sol = p.A_col * p.Gb                        # kW
    T_st_in = p.T_gen + p.dT_gen_approach         # oil back from generator
    T_sc_in = T_st_in + 2.3                       # tank stratification (Table 7: 498.5 vs 496.2)
    T_sc_out = T_sc_in + p.dT_col
    Tm = 0.5 * (T_sc_in + T_sc_out)
    G = p.Gb * 1000.0                             # W/m2
    eta_col = p.eta_opt * incidence_modifier(p.theta) - p.U_L * (Tm - p.T0) / (p.conc_ratio * G)
    Q_u = max(eta_col, 0.0) * Q_sol

    # storage tank (vertical cylinder H = 2D) heat loss
    D = (2.0 * p.V_ST / math.pi) ** (1.0 / 3.0)
    A_st = math.pi * D * 2.0 * D + 2.0 * math.pi * D**2 / 4.0
    T_st = 0.5 * (T_sc_out + T_st_in)
    Q_loss_st = p.U_T * A_st * (T_st - p.T0) / 1000.0
    Q_gen = max(Q_u - Q_loss_st, 0.0)

    T_st_out = T_sc_out - 2.3
    dh = h_vp1(T_st_out) - h_vp1(T_st_in)
    m_oil = Q_gen / dh if dh > 0 else 0.0
    return dict(Q_sol=Q_sol, Q_u=Q_u, Q_gen=Q_gen, Q_loss_st=Q_loss_st, eta_col=eta_col,
                m_oil=m_oil, T_sc_in=T_sc_in, T_sc_out=T_sc_out,
                T_st_in=T_st_in, T_st_out=T_st_out,
                Ex_solar=solar_exergy(Q_sol, p.T0, p.T_sun))
