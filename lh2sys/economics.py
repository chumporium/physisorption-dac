"""Purchase-cost functions (paper Table 3) and economic indicators (Eqs. 12-19)."""
from __future__ import annotations

import math

from .params import CEPCI, Params

EX_CH_H2 = 236090.0 / 2.01588   # kJ/kg, standard chemical exergy of H2 (Szargut)


def crf(p: Params) -> float:
    i, n = p.i_rate, p.n_years
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def lmtd(dT_a: float, dT_b: float) -> float:
    dT_a, dT_b = max(dT_a, 1.0), max(dT_b, 1.0)   # floor: avoids log of <= 0
    if abs(dT_a - dT_b) < 1e-6:
        return dT_a
    return (dT_a - dT_b) / math.log(dT_a / dT_b)


def hx_area(Q: float, U: float, Th_in, Th_out, Tc_in, Tc_out) -> float:
    """Counter-flow area [m2] from Eq. (15)-(16); Q in kW, U in kW/m2K."""
    return abs(Q) / (U * lmtd(Th_in - Tc_out, Th_out - Tc_in))


def z_hx(A: float) -> float:
    return 130.0 * (A / 0.093) ** 0.78 * CEPCI[2022] / CEPCI[2005]


def z_collector(A_col: float) -> float:
    return 150.0 * A_col * CEPCI[2021] / CEPCI[2005]


def z_storage(V: float) -> float:
    # Table 3 prints "1380 x 0.4 x V_T"; the usual form is 1380 V^0.4
    return 1380.0 * V ** 0.4 * CEPCI[2021] / CEPCI[2005]


def z_pump(W: float, eta: float) -> float:
    if W <= 0:
        return 0.0
    return 3.0 * 422.0 * W ** 0.71 * 1.41 * (1.0 + 0.2 / (1.0 - eta)) * CEPCI[2022] / CEPCI[2000]


def z_valve(m: float) -> float:
    return 114.5 * m * CEPCI[2022] / CEPCI[2005]


def z_turbine(m: float, eta: float, P_in: float, P_out: float, T_in: float) -> float:
    return (479.34 * m / (0.93 - eta) * math.log(P_in / P_out)
            * (1.0 + math.exp(0.036 * T_in - 54.4)) * CEPCI[2022] / CEPCI[1994])


def z_compressor(m: float, eta: float, P_in: float, P_out: float) -> float:
    return (71.1 * m / (0.9 - eta) * (P_out / P_in) * math.log(P_out / P_in)
            * CEPCI[2022] / CEPCI[1994])


def npv_pp(p: Params, TCI: float, annual_cash: float):
    """Net present value [$], simple payback [yr] and yearly cumulative NPV."""
    cum, curve = -TCI, []
    for year in range(1, p.n_years + 1):
        cum += annual_cash / (1 + p.i_rate) ** year
        curve.append(cum)
    pp = TCI / annual_cash if annual_cash > 0 else float("inf")
    return cum, pp, curve
