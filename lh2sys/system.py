"""Integrated solar / branched-GAX / Claude liquid-hydrogen plant.

Energy chain:  Gb -> PTC field + storage tank -> GAX generator heat
               -> GAX evaporator cooling -> H2 feed pre-cooled (1 -> 2)
               -> Claude cycle liquefies the same mass flow (m10 = m1).
"""
from __future__ import annotations

from . import economics as eco
from .claude_cycle import dead_state, exergy, solve_claude, _s_hP
from .gax import Infeasible, solve_gax
from .params import Params
from .solar import solve_solar_loop


def evaluate(p: Params) -> dict:
    """Run the full model. Raises gax.Infeasible for infeasible designs."""
    T0, P0 = p.T0, p.P0
    sol = solve_solar_loop(p)
    if sol["Q_gen"] <= 0:
        raise Infeasible("no useful solar heat")
    gax = solve_gax(p, sol["Q_gen"])

    # hydrogen feed pre-cooling in the GAX evaporator
    T1, T2 = T0, p.T_eva + p.TTD_eva
    from .claude_cycle import _h
    h1, h2 = _h(T1, P0), _h(T2, P0)
    m1 = gax["Q_eva"] / (h1 - h2)
    cl = solve_claude(p, m1, T1, T2)
    st = cl["states"]
    st[9]["s"] = _s_hP(st[9]["h"], P0)

    # ---------------- energy / exergy indicators ------------------------------
    h0, s0 = dead_state(T0, P0)
    Ex = {n: exergy(s, T0, h0, s0) for n, s in st.items()}
    W_pump = gax["W_pump"]
    W_net_in = W_pump + cl["W_com"] - cl["W_tur"]
    Ex_F = sol["Ex_solar"] + W_net_in
    Ex_P = Ex[10] - Ex[1]
    eta_II = 100.0 * Ex_P / Ex_F
    COP_sys = m1 * (st[1]["h"] - st[10]["h"]) / (sol["Q_sol"] + W_net_in)
    SI = 1.0 / (1.0 - eta_II / 100.0)
    LHPR = m1 * 3600.0 / cl["rho_L"]

    T_oil = 0.5 * (sol["T_st_out"] + sol["T_st_in"])
    Ex_heat = sol["Q_gen"] * (1.0 - T0 / T_oil)
    ExD = {
        "Solar field + ST": sol["Ex_solar"] - Ex_heat,
        "GAX cycle": Ex_heat + W_pump - (Ex[2] - Ex[1]),
        "Mixer (2+17)": Ex[2] + Ex[17] - Ex[3],
        "Compressor": cl["W_com"] - (Ex[4] - Ex[3]),
        "RHE2": (Ex[4] - Ex[5]) - (Ex[17] - Ex[14]),
        "RHE3": (Ex[6] - Ex[7]) - (Ex[14] - Ex[13]),
        "RHE4": (Ex[7] - Ex[8]) - (Ex[12] - Ex[11]),
        "Turbine": (Ex[15] - Ex[16]) - cl["W_tur"],
        "Mixer (12+16)": Ex[12] + Ex[16] - Ex[13],
        "TV3": Ex[8] - Ex[9],
        "Separator": Ex[9] - Ex[10] - Ex[11],
    }
    ExD_total = sum(ExD.values())

    # ---------------- purchase costs (Table 3) --------------------------------
    U, Ur = p.U_gax, p.U_rhe
    Tw_in = T0
    Tw_out = lambda T_hot_out: min(T0 + 10.0, T_hot_out - 3.0)
    T_hot_top = max(p.T_gen, gax["T_at"])
    A = {
        "Generator": eco.hx_area(gax["Q_gen"], U, sol["T_st_out"], sol["T_st_in"],
                                 gax["T_d_end"], p.T_gen),
        "GAX (GAXA+GAXD)": eco.hx_area(gax["Q_gax"], U, T_hot_top, gax["T_a_end"],
                                       p.T_abs, gax["T_d_end"]),
        "Absorber": eco.hx_area(gax["Q_abs"], U, gax["T_a_end"], p.T_abs, Tw_in, Tw_out(p.T_abs)),
        "Condenser": eco.hx_area(gax["Q_cond"], U, gax["T_rv"], p.T_cond, Tw_in, Tw_out(p.T_cond)),
        "Rectifier": eco.hx_area(gax["Q_rect"], U, gax["T_ds"], gax["T_rv"], Tw_in, Tw_out(gax["T_rv"])),
        "RHE1": eco.hx_area(gax["Q_rhe1"], U, p.T_cond, gax["T26"], p.T_eva, gax["T29"]),
        "Evaporator": eco.hx_area(gax["Q_eva"], Ur, T1, T2, gax["T27"], p.T_eva),
        "RHE2": eco.hx_area(cl["Q_RHE2"], Ur, st[4]["T"], st[5]["T"], st[14]["T"], st[17]["T"]),
        "RHE3": eco.hx_area(cl["Q_RHE3"], Ur, st[6]["T"], st[7]["T"], st[13]["T"], st[14]["T"]),
        "RHE4": eco.hx_area(cl["Q_RHE4"], Ur, st[7]["T"], st[8]["T"], st[11]["T"], st[12]["T"]),
    }
    Z = {"Solar collector": eco.z_collector(p.A_col), "Storage tank": eco.z_storage(p.V_ST)}
    for k, a in A.items():
        if k.startswith("GAX"):
            Z["GAX absorber"] = Z["GAX desorber"] = eco.z_hx(a / 2.0)
        else:
            Z[k] = eco.z_hx(a)
    Z["Pump"] = eco.z_pump(W_pump, p.eta_pump)
    Z["TV1"] = eco.z_valve(gax["m_weak"])
    Z["TV2"] = eco.z_valve(gax["m_ref"])
    Z["TV3"] = eco.z_valve(st[8]["m"])
    Z["Turbine"] = eco.z_turbine(st[15]["m"], p.eta_tur, p.P_high, P0, p.T5)
    Z["Compressor"] = eco.z_compressor(st[3]["m"], p.eta_com, P0, p.P_high)
    PEC = sum(Z.values())
    CRF = eco.crf(p)
    Zdot = {k: CRF * p.phi_m * z / p.tau for k, z in Z.items()}      # $/h
    Zdot_total = sum(Zdot.values())

    # ---------------- product cost, NPV, PP -----------------------------------
    C_feed = p.c_feed_H2 * m1 * eco.EX_CH_H2 * 3600.0 / 1e6            # $/h
    C_el = p.c_el * W_net_in                                           # $/h
    C10 = C_feed + C_el + Zdot_total
    c_LH2 = C10 / (3600.0 * m1)                                        # $/kg
    c_F = C_el / Ex_F                                                  # $/kWh of fuel exergy
    f_sys = 100.0 * Zdot_total / (Zdot_total + c_F * ExD_total)

    TCI = p.f_TCI * PEC
    OM = (p.phi_m - 1.0) * CRF * PEC / p.tau
    annual_cash = p.tau * (p.price_LH2 * 3600.0 * m1 - C_feed - C_el - OM)
    NPV, PP, npv_curve = eco.npv_pp(p, TCI, annual_cash)

    return dict(
        # decision variables
        Gb=p.Gb, T_gen=p.T_gen, T_eva=p.T_eva, P_high=p.P_high, T0=T0,
        # headline indicators (paper Figs. 2-5, 10)
        LHPR=LHPR, COP_sys=COP_sys, eta_II=eta_II, SI=SI, Zdot=Zdot_total,
        c_LH2=c_LH2, PP=PP, NPV=NPV,
        # powers & flows
        m_H2=m1, W_pump=W_pump, W_com=cl["W_com"], W_tur=cl["W_tur"], W_net=W_net_in,
        Q_sol=sol["Q_sol"], Q_gen=sol["Q_gen"], Q_eva=gax["Q_eva"], COP_GAX=gax["COP"],
        Ex_solar=sol["Ex_solar"], Ex_F=Ex_F, Ex_P=Ex_P, ExD_total=ExD_total,
        C10=C10, C_feed=C_feed, C_el=C_el, PEC=PEC, TCI=TCI, f_sys=f_sys,
        yL=cl["yL"], x_tur=cl["x_tur"], min_approach=cl["min_approach"],
        # details
        solar=sol, gax=gax, claude=cl, states=st, Ex=Ex, ExD=ExD, areas=A, Z=Z,
        Zdot_k=Zdot, npv_curve=npv_curve,
    )


HEADLINE = ["LHPR", "COP_sys", "eta_II", "SI", "Zdot", "c_LH2", "PP", "NPV"]
UNITS = {"LHPR": "m3/h", "COP_sys": "-", "eta_II": "%", "SI": "-", "Zdot": "$/h",
         "c_LH2": "$/kg", "PP": "yr", "NPV": "$"}
