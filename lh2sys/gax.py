"""Generator-absorber heat exchange (GAX) ammonia-water pre-cooling cycle.

Solved per 1 kg/s of refrigerant, then scaled by the solar heat available.

Pressures:
  P_high  from condenser outlet (saturated liquid, x_rect, T_cond)
  P_low   from evaporator outlet (T_eva, vapour quality q_eva_out)
Concentrations:
  x_s (strong)  saturated liquid at (P_low, T_abs)
  x_w (weak)    saturated liquid at (P_high, T_gen)

GAX heat is found by pinch-matching the absorber heat-release curve against
the desorber heat-demand curve (counter-flow, dT >= dT_gax_pinch). The paper's
*branched* variant splits the strong solution so more of the absorber heat can
be recovered; here that effect is represented by the ideal pinch match itself
(i.e. the branch is assumed to let the full overlap be used).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from . import nh3h2o as aw
from .params import Params

N_GRID = 40


class Infeasible(Exception):
    pass


def _P_low(T_eva: float, z: float, q_target: float) -> float:
    P_q0 = brentq(lambda P: aw.T_bubble(P, z) - T_eva, 1.0, 5000.0)   # quality 0 here
    P_q1 = brentq(lambda P: aw.T_dew(P, z) - T_eva, 1.0, 5000.0)      # quality 1 here
    return brentq(lambda P: aw.flash(P, T_eva, z)[0] - q_target, P_q1 * 1.0000001, P_q0 * 0.9999999)


def _T_from_h(P: float, z: float, h: float, lo: float, hi: float) -> float:
    return brentq(lambda T: aw.flash(P, T, z)[3] - h, lo, hi)


def _gax_match(TD, QD, TA, QA, pinch):
    """Max heat transferable from absorber (QA released as T falls from TA[0])
    to desorber (QD absorbed as T rises from TD[0]) in counter-flow."""
    def feasible(Q):
        q = np.linspace(0.0, Q, 60)
        Ta = np.interp(q, QA, TA)
        Td = np.interp(Q - q, QD, TD)
        return np.all(Ta - Td >= pinch)

    hi = min(QD[-1], QA[-1])
    if hi <= 0 or not feasible(1e-9):
        return 0.0
    if feasible(hi):
        return hi
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


@lru_cache(maxsize=4096)
def _solve_unit(T_gen, T_eva, T_cond, T_abs, x_r, q_out, eps1, eta_p, pinch):
    # --- pressures & concentrations -----------------------------------------
    P_hi = aw.P_bubble(T_cond, x_r)
    P_lo = _P_low(T_eva, x_r, q_out)
    x_s = aw.x_bubble(P_lo, T_abs)
    x_w = aw.x_bubble(P_hi, T_gen)
    if x_s - x_w < 0.01:
        raise Infeasible(f"no degassing range (x_s={x_s:.3f}, x_w={x_w:.3f})")

    m_s = (x_r - x_w) / (x_s - x_w)          # strong solution per kg refrigerant
    m_w = m_s - 1.0

    # --- pump ---------------------------------------------------------------
    w_p = m_s * aw.specific_volume_liquid(T_abs, x_s) * (P_hi - P_lo) / eta_p
    h_abs_out = aw.h_liquid(T_abs, x_s)
    h_sp = h_abs_out + w_p / m_s

    # --- desorber top / rectifier ------------------------------------------
    T_ds = aw.T_bubble(P_hi, x_s)
    y_top = aw.y_equilibrium(P_hi, x_s)
    h_vtop = aw.h_vapour(T_ds, y_top)
    h_rf = aw.h_liquid(T_ds, x_s)
    m_rf = (x_r - y_top) / (y_top - x_s)
    T_rv = aw.T_dew(P_hi, x_r)
    h_rv = aw.h_vapour(T_rv, x_r)
    Q_rect = (1.0 + m_rf) * h_vtop - h_rv - m_rf * h_rf

    # --- condenser, RHE1, evaporator ---------------------------------------
    h25 = aw.h_liquid(T_cond, x_r)
    q_cond = h_rv - h25
    q28, _, _, h28 = aw.flash(P_lo, T_eva, x_r)
    h_cold_max = aw.flash(P_lo, T_cond, x_r)[3]
    Q1 = eps1 * min(h_cold_max - h28, h25 - aw.h_liquid(T_eva, x_r))
    h26 = h25 - Q1
    h27 = h26                                  # throttle valve TV2
    h29 = h28 + Q1
    q_eva = h28 - h27

    # --- desorber demand curve (T rising from pumped solution to T_gen) -----
    F = m_s + m_rf
    h_F = (m_s * h_sp + m_rf * h_rf) / F
    T_sens = np.linspace(T_abs, T_ds, 8, endpoint=False)
    Q_sens = F * (np.array([aw.h_liquid(T, x_s) for T in T_sens]) - h_F)
    T_des = np.linspace(T_ds, T_gen, N_GRID)
    x_des = np.array([aw.x_bubble(P_hi, T) for T in T_des])
    m_l = F * (y_top - x_s) / (y_top - x_des)
    Q_des = m_l * np.array([aw.h_liquid(T, x) for T, x in zip(T_des, x_des)]) \
        + (F - m_l) * h_vtop - F * h_F
    TD = np.concatenate([T_sens, T_des])
    QD = np.maximum.accumulate(np.maximum(np.concatenate([Q_sens, Q_des]), 0.0))
    Q_DG = QD[-1]

    # --- absorber release curve (T falling from weak-solution inlet) -------
    h_w = aw.h_liquid(T_gen, x_w)
    T_at = aw.T_bubble(P_lo, x_w)
    T_abs_grid = np.linspace(T_at, T_abs, N_GRID)
    x_a = np.array([aw.x_bubble(P_lo, T) for T in T_abs_grid])
    a = m_w * (x_a - x_w) / (x_r - x_a)
    Q_abs = m_w * h_w + a * h29 - (m_w + a) * np.array(
        [aw.h_liquid(T, x) for T, x in zip(T_abs_grid, x_a)])
    TA = np.concatenate([[max(T_gen, T_at)], T_abs_grid])
    QA = np.maximum.accumulate(np.maximum(np.concatenate([[0.0], Q_abs]), 0.0))
    Q_A = QA[-1]

    Q_gax = _gax_match(TD, QD, TA, QA, pinch)
    q_gen = Q_DG - Q_gax
    q_abs = Q_A - Q_gax

    # temperatures for heat-exchanger sizing
    T_d_end = float(np.interp(Q_gax, QD, TD))
    T_a_end = float(np.interp(Q_gax, QA, TA))
    T27 = _T_from_h(P_lo, x_r, h27, 180.0, T_eva)
    T29 = _T_from_h(P_lo, x_r, h29, T_eva, T_cond + 5.0)
    T26 = brentq(lambda T: aw.h_liquid(T, x_r) - h26, 150.0, T_cond)

    balance = (q_gen + q_eva + w_p) - (q_cond + Q_rect + q_abs)
    return dict(P_high=P_hi, P_low=P_lo, x_s=x_s, x_w=x_w, D_range=x_s - x_w,
                f=m_s, m_w=m_w, m_rf=m_rf, w_pump=w_p,
                q_gen=q_gen, q_eva=q_eva, q_cond=q_cond, q_rect=Q_rect, q_abs=q_abs,
                q_gax=Q_gax, q_rhe1=Q1, Q_DG=Q_DG, Q_A=Q_A,
                COP=q_eva / (q_gen + w_p), balance=balance,
                T_ds=T_ds, T_at=T_at, T_d_end=T_d_end, T_a_end=T_a_end,
                T_rv=T_rv, T26=T26, T27=T27, T29=T29)


def solve_gax(p: Params, Q_gen_available: float) -> dict:
    """Scale the per-kg GAX solution to the available generator heat [kW]."""
    u = _solve_unit(round(p.T_gen, 4), round(p.T_eva, 4), p.T_cond, p.T_abs, p.x_rect,
                    p.q_eva_out, p.eps_RHE1, p.eta_pump, p.dT_gax_pinch)
    if u["q_gen"] <= 0:
        raise Infeasible("GAX generator duty <= 0")
    m_r = Q_gen_available / u["q_gen"]
    out = dict(u)
    out.update(m_ref=m_r, m_strong=m_r * u["f"], m_weak=m_r * u["m_w"],
               Q_gen=Q_gen_available, Q_eva=m_r * u["q_eva"], Q_cond=m_r * u["q_cond"],
               Q_rect=m_r * u["q_rect"], Q_abs=m_r * u["q_abs"], Q_gax=m_r * u["q_gax"],
               Q_rhe1=m_r * u["q_rhe1"], W_pump=m_r * u["w_pump"])
    return out
