"""Integrated off-grid plant: solar PTC + tank -> GAX, PV + battery, physisorption DAC train.

One plant architecture for every research question (spec: SPESIFIKASI_SISTEM.md). Per kg/s
of dry air at nominal flow:

  ambient air -> fan -> [coil-1, +3 C, condensate] -> [dryer] -> recuperator -> [coil-2, T_ads]
              -> contactor -> recuperator -> exhaust
  heat (PTC oil tank):  GAX generator, CO2 desorption, silica regeneration, coil defrost
  electricity (PV + battery): fans, vacuum pumps, compressors (vcr), solution pumps, dry coolers

The plant runs in an hour only if the tank and the battery can cover that hour (no backup).
Operating variables (design T_ads, operating strategy, PTC / tank / PV / battery sizes) are
optimised for the lowest LCOC; architecture comes from the Case and is never changed here.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import ars_map, energy, sorbents, tvsa, vcr
from .climate_sites import YEAR
from .levers import REGEN, T_ADS, Case, Prm
from .weather import humidity_ratio, p_sat, sun

CP_AIR = 1.006        # kJ/kg K
H_FG = 2470.0         # kJ/kg condensation
H_SUB = 2834.0        # kJ/kg desublimation (frost)
H_FUS = 334.0         # kJ/kg melting (defrost)
M_CO2 = 44.01
M_H2O = 18.015
MIN_HOURS = 500
STRATEGIES = ("all", "cold40")            # run in any feasible hour / only the coldest 40 %
AREA_F = (0.8, 1.0, 1.3, 1.7, 2.2)        # PTC area x (annual heat / annual yield)
TANK_H = (6.0, 12.0, 24.0)                # tank capacity, hours of mean heat demand
PV_F = (0.8, 1.0, 1.3, 1.7, 2.2)
BATT_H = (4.0, 8.0, 14.0)
# cold-end approach of the recuperator train [K]; None = no recuperation (cold air exhausted,
# as in Kim et al.). Free cold (LNG) favours no recuperation, bought cold favours a tight approach.
DT_REC = (3.0, 8.0, 20.0, None)
LMTD_LNG = 25.0                           # K, LNG (115 -> 283 K) against air (ambient -> T_ads)


def sorbent_for(case: Case) -> sorbents.Sorbent:
    T_des, p_des, mode = REGEN[case.prm.regen]
    if case.prm.p_des is not None:
        p_des = case.prm.p_des
    return replace(sorbents.get(case.sorbent), T_des=T_des, p_des=p_des, mode=mode)


def _p_des_cases(case: Case) -> list[Case]:
    """The base vacuum regeneration is optimised over the desorption pressures in Prm.p_des_opt; any other
    regeneration (sweep, fixed sweeps in the sensitivity studies) or a fixed p_des is used as it is."""
    p = case.prm
    if p.regen != "vac100" or p.p_des is not None or not p.p_des_opt:
        return [case]
    return [replace(case, prm=replace(p, p_des=float(x))) for x in p.p_des_opt]


# ------------------------------------------------------------------ water: two-zone bed
def two_zone(sb, wcap_dry, w_feed, P, T_ads, p: Prm, n_iter: int = 4):
    """Two-zone bed at cyclic steady state (Webley; PrISMa SI eq. S1-S3). Water travels slower
    than CO2, so the bed splits into a water-loaded inlet zone (fraction alpha, CO2 capacity
    WRC x dry) and a dry zone. Per kg sorbent and cycle, the water fed equals the water desorbed:
        w_mol = (mol H2O / mol CO2 captured in the feed) = y_H2O / (y_CO2 eta_cap)
        WC_w  = q_H2O(T_ads, p_w,feed) - q_H2O(T_des, p_w,des)
        WC    = WC_dry / (1 + (w_mol WC_dry / WC_w)(1 - WRC)),  alpha = w_mol WC / WC_w
    If alpha >= 1 the whole bed is wet: WC = WRC WC_dry and only WC_w of water cycles.
    p_w,des = x_w p_des with x_w the water share of the desorbed gas (iterated).
    WRC is applied in proportion to the pore filling q_H2O,feed / q_sat (see below).
    Returns working capacity [kg CO2/kg], alpha [-], water desorbed per CO2 [mol/mol]."""
    WCd = wcap_dry / (M_CO2 * 1e-3)                                   # mol/kg
    y_w = w_feed * 28.96 / M_H2O                                      # mol H2O per mol dry air
    w_mol = y_w / (p.x_co2_ppm * 1e-6 * p.eta_cap)
    pw = w_feed * P / (0.622 + w_feed)
    q_feed = sb.q_h2o(T_ads, pw, rh=pw / p_sat(T_ads))
    # WRC is measured with water-filled pores (near-saturated feed, PrISMa); scale the penalty
    # with the pore filling so that a dry or water-repelling bed keeps its dry capacity
    fill = np.clip(q_feed / sb.h2o_qsat, 0.0, 1.0) if p.wrc_pore_scaling else 1.0
    wrc = 1.0 - (1.0 - sb.wrc) * fill
    ff = p.water_front_factor       # non-ideal front: the wet zone is ff times longer than the shock-front estimate
    r = np.minimum(w_mol, 50.0)
    for _ in range(n_iter):
        p_wdes = r / (1.0 + r) * sb.p_des * 1e5
        q_des = sb.q_h2o(sb.T_des, p_wdes, rh=p_wdes / p_sat(np.full_like(T_ads, sb.T_des)))
        WCw = np.maximum(q_feed - q_des, 0.0)
        cyc = WCw > 1e-6
        WCw_s = np.where(cyc, WCw, 1.0)
        WC_part = WCd / (1.0 + ff * (w_mol * WCd / WCw_s) * (1.0 - wrc))
        alpha = np.where(cyc, ff * w_mol * WC_part / WCw_s, np.inf)
        WC = np.where(alpha < 1.0, WC_part, wrc * WCd)
        r = np.where(cyc, np.minimum(w_mol, WCw / np.maximum(WC, 1e-12)), 0.0)
    alpha = np.where(w_mol > 0, alpha, 0.0)
    return WC * M_CO2 * 1e-3, np.minimum(alpha, 1.0), r


# ------------------------------------------------------------------ cycle time (heat transfer)
def cycles_per_h(sb, p: Prm, wcap, r_w, T_ads, c_bed):
    """Cycles per hour from the time to heat and cool the bed through its heat exchanger
    (PrISMa SI eq. S8): t = (c_bed dT + H_ads) / (U a / rho_bulk * dT_drive), per kg sorbent.
    Heating from T_ads to T_des with the medium dT_hex_heat above T_des; cooling back with the
    medium dT_hex_cool below T_ads. Adsorption takes as long as heating + cooling (two beds per
    train), so t_cycle = 2 (t_heat + t_cool). Capped, then slowed by the Arrhenius diffusion
    factor of the sorbent and the Monte Carlo cycle factor."""
    rho_bulk = sb.rho * 1e3 * (1 - p.eps_pellet) * (1 - p.eps_bed)          # kg/m3
    ua = p.U_hx * p.a_hx / rho_bulk                                          # kW/(kg K)
    dT = np.maximum(sb.T_des - T_ads, 1.0)
    n_co2 = wcap / (M_CO2 * 1e-3)                                            # mol/kg per cycle
    H = c_bed * dT + n_co2 * sb.hoa + n_co2 * r_w * (sb.h2o_dH or 0.0)       # kJ/kg
    t_heat = H / (ua * np.maximum(p.dT_hex_heat + dT / 2.0, 5.0))
    t_cool = c_bed * dT / (ua * np.maximum(dT / 2.0 + p.dT_hex_cool, 5.0))
    n = np.minimum(3600.0 / (2.0 * (t_heat + t_cool)), p.max_cycles_per_h)
    f = np.minimum(np.exp(-sb.E_diff / 8.314e-3 * (1.0 / T_ads - 1.0 / 298.15)), 1.0)
    return n * f * p.cycle_factor


# ------------------------------------------------------------------ air train (per kg/s air)
def air_train(w: dict, case: Case, T_design: float | None, dT_rec: float | None = 3.0) -> dict:
    p, sb = case.prm, sorbent_for(case)
    T0, Td, P = w["T0"], w["Td"], w["P"]
    T0C = T0 - 273.15
    w0 = humidity_ratio(Td, P)
    wc = humidity_ratio(np.full_like(T0, p.T_coil1 + 273.15), P)
    ws = humidity_ratio(np.full_like(T0, p.dewpoint_dry + 273.15), P)
    refrig = case.cold != "ambient"

    # adsorption temperature
    if refrig:
        chilled = T0 - 1.0 > T_design
        T_ads = np.where(chilled, T_design, T0 - 1.0)
    else:
        chilled = np.zeros_like(T0, bool)
        T_ads = T0.copy()
    w_sat_ads = humidity_ratio(T_ads, P)            # saturation (ice below 0 C) at T_ads
    # humidity the sorbent tolerates at its inlet. Two-zone sorbents tolerate any humidity (the
    # water is handled inside the bed); the dryers then work to their design dewpoint.
    if sb.twozone:
        w_ok = ws
    else:
        w_ok = w_sat_ads if sb.hydrophobic else np.maximum(ws, sb.rh_max * w_sat_ads)

    # ---- drying train
    coil1 = refrig and case.drying in ("cond", "cond_silica", "freeze")
    warm = coil1 & (T0C > p.T_coil1)
    cond = np.where(warm, np.maximum(w0 - np.maximum(wc, w_ok), 0.0), 0.0)
    w1 = w0 - cond                                   # after coil-1
    sil = frz = frost = 0.0 * T0
    if case.drying == "cond_silica":
        sil = np.maximum(w1 - w_ok, 0.0)
    elif case.drying == "freeze":
        # reversing regenerators freeze water out to ice saturation ~3 K below T_ads (only below
        # 0 C); the dry exhaust sublimes it back, so only freeze_loss of the heat is lost.
        # A small guard bed takes the rest; above 0 C the guard bed does all of it.
        w_fz = humidity_ratio(T_ads - 3.0, P)
        can = chilled & (T_ads < 273.15)
        frz = np.where(can, np.maximum(w1 - np.maximum(w_fz, w_ok), 0.0), 0.0)
        sil = np.maximum(w1 - frz - w_ok, 0.0)
    w2 = w1 - sil - frz
    if refrig:
        # whatever is still above saturation at T_ads is deposited on coil-2 (frost below 0 C,
        # condensate above); it is not recovered and the coil must be defrosted
        dep = np.where(chilled, np.maximum(w2 - w_sat_ads, 0.0), 0.0)
        frost = np.where(T_ads < 273.15, dep, 0.0)
        cond2 = dep - frost
        w2 = w2 - dep
    else:
        cond2 = 0.0 * T0
    feed_ok = (w2 <= w_ok * 1.0001) | sb.twozone

    # without refrigeration the heat of silica adsorption warms the feed
    if not refrig and case.drying == "cond_silica":
        T_ads = T0 + sil * p.h_ads_silica / CP_AIR
    p_co2 = p.x_co2_ppm * 1e-6 * P / 1e5
    # TVSA cycle model for vacuum regeneration; a CO2-sweep cycle (Kim et al.) keeps the legacy
    # Henry-selectivity model, which was calibrated to their measured purity
    use_tvsa = p.purity_model == "tvsa" and sb.mode == "vacuum"
    if use_tvsa:
        # 5-step TVSA with IAST (tvsa.py): product CO2 per kg sorbent, purity, recovery, vent work
        cyc = tvsa.at(sb, T_ads, float(np.mean(P)) / 1e5, p.x_co2_ppm * 1e-6, tvsa.void_volume(sb, p),
                      p.eta_vac, p.purity_min)
        wcap = cyc["prod_co2"] * M_CO2 * 1e-3
    else:
        wcap = sb.working_capacity(T_ads, p_co2)
    alpha = r_w = np.zeros_like(T0)
    if sb.twozone:
        wcap, alpha, r_w = two_zone(sb, wcap, w2, P, T_ads, p)
    elif sb.hydrophobic:
        wcap = wcap * sb.retention
    c_bed = sb.cp + p.cp_contactor                   # kJ/kg K sorbent + contactor
    if use_tvsa:
        purity, rinse = cyc["purity"], cyc["recovery"]           # rinse = CO2 recovery of the cycle
        pur_ok = cyc["ok"]
        # vented gas (VS + IHS) pumped to ambient: kJ per kg CO2 product (dry cycle basis)
        vent_kj_per_kg = cyc["work"] / np.maximum(cyc["prod_co2"] * M_CO2 * 1e-3, 1e-12)
    else:
        n2_per_co2 = (0.7808 / (p.x_co2_ppm * 1e-6)) / sb.selectivity_n2(T_ads)
        purity = 1.0 / (1.0 + n2_per_co2)
        rinse = np.where(purity < p.purity_min, np.clip(1.0 - n2_per_co2, 0.0, 1.0), 1.0)
        pur_ok = rinse > 0.05
        vent_kj_per_kg = 0.0 * T0
    x_co2 = p.x_co2_ppm * 1e-6 * M_CO2 / 28.96
    co2 = p.eta_cap * x_co2 * rinse                  # kg CO2 product / kg air
    cap_ok = (wcap > 0.1e-3 * M_CO2) & pur_ok
    wcap_s = np.where(cap_ok, wcap, 1.0)
    hoa = sb.hoa * 1e3 / M_CO2                       # kJ/kg CO2
    cycles = cycles_per_h(sb, p, wcap_s, r_w, T_ads, c_bed)

    # ---- cold loads [kW per kg/s]
    rec = dT_rec is not None
    span = np.maximum((np.minimum(T0C, p.T_coil1) if coil1 else T0C) + 273.15 - T_ads, 0.0)
    warm_sens = CP_AIR * (p.dT_warm if rec else np.maximum(T0C - p.T_coil1, 0.0))
    Q1 = (np.where(warm, warm_sens, 0.0) + cond * H_FG
          + (sil * p.h_ads_silica if refrig else 0.0))
    Q2 = np.where(chilled, CP_AIR * (np.minimum(span, dT_rec) if rec else span) + co2 * hoa
                  + frz * H_SUB * p.freeze_loss + frost * H_SUB + cond2 * H_FG
                  + co2 / wcap_s * c_bed * np.maximum(T0 + 5.0 - T_ads, 0.0), 0.0)
    Q1 = np.where(coil1, Q1, 0.0)
    # below the coil-1 temperature the ambient air is the heat sink (dry cooler, no chiller)
    Q1_free = np.where(warm, 0.0, Q1)
    Q1 = Q1 - Q1_free

    # ---- cold supply
    T_e1, T_e2 = p.T_coil1 - p.dT_evap, T_design - 273.15 - p.dT_evap if refrig else 0.0
    q_gen = el_cold = rejected = 0.0 * T0
    tgen = np.full_like(T0, np.nan)
    if case.cold == "gax":
        c1, g1 = ars_map.cop_env(T0C, np.full_like(T0C, T_e1), p.gax_T_gen_max)
        c2, g2 = ars_map.cop_env(T0C, np.full_like(T0C, T_e2), p.gax_T_gen_max)
        c1, c2 = p.gax_derate * c1, p.gax_derate * c2
        ok1 = (Q1 <= 0) | (np.isfinite(c1) & (c1 > 0.05))
        ok2 = ~chilled | (np.isfinite(c2) & (c2 > 0.05))
        q_gen = np.where(Q1 > 0, Q1 / np.where(ok1, c1, 1.0), 0) + np.where(chilled, Q2 / np.where(ok2, c2, 1.0), 0)
        rejected = Q1 + Q2 + q_gen
        el_cold = p.gax_pump_frac * q_gen + p.fan_rej_frac * rejected
        tgen = np.where(chilled, g2, np.where(Q1 > 0, g1, np.nan))
    elif case.cold == "vcr":
        c1, c2 = vcr.cop(T0C, np.full_like(T0C, T_e1)), vcr.cop(T0C, np.full_like(T0C, T_e2))
        ok1 = (Q1 <= 0) | (np.isfinite(c1) & (c1 > 0.05))
        ok2 = ~chilled | (np.isfinite(c2) & (c2 > 0.05))
        w_comp = np.where(Q1 > 0, Q1 / np.where(ok1, c1, 1.0), 0) + np.where(chilled, Q2 / np.where(ok2, c2, 1.0), 0)
        rejected = Q1 + Q2 + w_comp
        el_cold = w_comp + p.fan_rej_frac * rejected
    else:                                             # lng (free cold) or ambient
        ok1 = ok2 = np.ones_like(T0, bool)
    el_cold = el_cold + p.fan_rej_frac * Q1_free

    # ---- heat [kW_th per kg/s]
    q_des = co2 * (c_bed / wcap_s * np.maximum(sb.T_des - np.maximum(T0, T_ads), 0.0) + hoa)
    if sb.twozone:                                   # water desorbed every cycle (two-zone)
        m_w = co2 * r_w * M_H2O / M_CO2                  # kg water per kg air
        q_des = q_des + m_w * (sb.h2o_dH / (M_H2O * 1e-3) + 4.2 * np.maximum(sb.T_des - T_ads, 0.0))
    else:
        m_w = 0.0 * T0
        q_des = q_des + co2 * sb.water_ratio * (2800.0 + 4.2 * np.maximum(sb.T_des - T_ads, 0.0))
    q_sil = sil * p.h_ads_silica * p.silica_regen_factor
    q_defrost = frost * H_FUS
    hp_sil = p.silica_regen == "hp"                  # silica regenerated by an electric heat pump
    heat = q_gen + q_des + q_defrost + (0.0 * q_sil if hp_sil else q_sil)

    # ---- electricity [kW_e per kg/s]
    dP = p.dP_contactor
    if coil1:
        dP += p.dP_coil
    if refrig:
        dP += p.dP_coil + (2 * p.dP_recup if rec else 0.0)
    if case.drying in ("cond_silica", "freeze"):
        dP += p.dP_silica if case.drying == "cond_silica" else p.dP_silica * 0.3 + 2 * p.dP_regen
    rho = P / (287.05 * T0)
    fan = dP / rho / p.eta_fan / 1e3
    vac = (co2 / M_CO2 * (1.0 + r_w) * 8.314 * sb.T_des * np.log(1.0 / sb.p_des) / p.eta_vac
           if sb.mode == "vacuum" else 0.0 * T0) + co2 * vent_kj_per_kg
    elec = fan + vac + el_cold + (q_sil / p.silica_hp_cop if hp_sil else 0.0 * T0)

    feasible = feed_ok & cap_ok & ok1 & ok2
    # ice in the reversing regenerators is sublimed into the exhaust (see above): not recovered
    water = cond + cond2 + frost + sil * p.water_recovery_silica + m_w
    return dict(T_ads=T_ads, chilled=chilled, feasible=feasible, co2=co2, water=water,
                Q1=Q1, Q1_free=Q1_free, Q2=Q2, heat=heat, q_gen=q_gen, q_des=q_des, q_sil=q_sil, q_defrost=q_defrost,
                elec=elec, fan=fan, vac=vac, el_cold=el_cold, rejected=rejected, wcap=wcap,
                cycles=cycles, purity=purity, sil=sil, frz=frz, frost=frost, cond=cond + cond2,
                span=span, tgen=tgen, coil1=coil1, T_design=T_design, dT_rec=dT_rec,
                w0=w0, w_feed=w2, alpha=alpha, r_w=r_w, m_w=m_w)


# ------------------------------------------------------------------ energy block
def resources(w: dict, p: Prm) -> dict:
    H = len(w["T0"])
    cz, ci = sun(np.arange(H), f"{YEAR}-01-01 00:00", np.array([w["lat"]]), np.array([w["lon"]]))
    cz, ci = cz[:, 0], ci[:, 0]
    th = np.degrees(np.arccos(np.clip(ci, 0, 1)))
    # LS-2 incidence angle modifier (Dudley et al.); it already contains the cosine loss, so it is applied to DNI
    iam = np.clip(np.cos(np.radians(th)) + 0.000884 * th - 0.00005369 * th ** 2, 0, 1)
    dni = np.where(cz > 0.02, w["dni"], 0.0)
    gain = p.ptc_eta_opt * iam * dni / 1e3
    loss = p.ptc_loss * np.maximum(p.T_htf - (w["T0"] - 273.15), 0)
    q_sol = np.where(dni > 0, np.maximum(gain - loss, 0.0), 0.0)        # kW_th per m2
    pv = energy.pv_cf(w["ghi"], w["T0"])                                 # kW_e per kWp
    return dict(q_sol=q_sol, pv=pv)


def dispatch(want, dem_th, dem_el, q_sol, pv, A, S_th, P_pv, S_el, p: Prm):
    """Hourly operation with two stores; arrays of sizes (k,). Returns bool (H, k)."""
    H, k = len(want), len(A)
    s1, s2 = 0.5 * S_th, 0.5 * S_el
    ran = np.zeros((H, k), bool)
    for h in range(H):
        g1, g2 = q_sol[h] * A, pv[h] * P_pv
        s1 = s1 * (1 - p.storage_loss_h)
        if want[h]:
            can = (s1 + g1 >= dem_th[h]) & (s2 + g2 >= dem_el[h])
            ran[h] = can
            n1, n2 = g1 - can * dem_th[h], g2 - can * dem_el[h]
        else:
            n1, n2 = g1, g2
        s1 = np.minimum(s1 + n1, S_th)
        s2 = np.minimum(s2 + np.where(n2 > 0, p.eta_batt * n2, n2), S_el)
    return ran


def _grid(dem_th, dem_el, want, res):
    op = want.sum()
    ann_th, ann_el = dem_th[want].sum(), dem_el[want].sum()
    A0 = ann_th / max(res["q_sol"].sum(), 1e-6)
    P0 = ann_el / max(res["pv"].sum(), 1e-6)
    mth, mel = ann_th / max(op, 1), ann_el / max(op, 1)
    g = np.array([(fa * A0, th * mth, fp * P0, bh * mel)
                  for fa in AREA_F for th in TANK_H for fp in PV_F for bh in BATT_H])
    return g.T


def energy_block(dem_th, dem_el, want, res, p: Prm):
    """Optimal sizes for given hourly demands. Returns ran (H,k) and sizes/costs per combo.
    Grid mode: runs in every wanted hour; purchased energy is returned as an annualised
    'capex' equivalent so both modes share one cost formula."""
    if p.energy_mode == "grid":
        ann = (dem_th[want].sum() * 3.6e-3 / p.eta_thrm * p.c_heat_grid
               + dem_el[want].sum() / 1e3 * p.c_elec_grid)
        z = np.zeros(1)
        return want[:, None], dict(A=z, S_th=z, P_pv=z, S_el=z,
                                   capex=np.array([ann / (p.crf + p.om_energy)]))
    A, S_th, P_pv, S_el = _grid(dem_th, dem_el, want, res)
    ran = dispatch(want, dem_th, dem_el, res["q_sol"], res["pv"], A, S_th, P_pv, S_el, p)
    capex = p.c_ptc * A + p.c_tank * S_th + p.c_pv * P_pv + p.c_batt * S_el
    return ran, dict(A=A, S_th=S_th, P_pv=P_pv, S_el=S_el, capex=capex)


# ------------------------------------------------------------------ process capital
def _peak(x, op):
    return float(np.max(np.where(op, x, 0.0))) if op.any() else 0.0


def process_capex(L: dict, op, case: Case, sb) -> dict:
    p = case.prm
    span_w = max(float(np.percentile(np.where(op, L["span"], 0), 95)), 0.0)
    rec = L["dT_rec"] is not None
    ntu = ((5.0 if L["coil1"] else 0.0) + span_w / L["dT_rec"]) if rec else 0.0
    area_rec = ntu * CP_AIR / p.U_air * (p.reg_area_factor if case.drying == "freeze" else 1.0)
    Q1p, Q2p = _peak(L["Q1"], op), _peak(L["Q2"], op)
    lmtd = LMTD_LNG if case.cold == "lng" else 1.5 * p.dT_evap
    area_coil = (Q1p + Q2p) / (p.U_coil * lmtd)
    if case.cold == "gax":
        c_cold = (p.c_abs_rej + p.c_rej) * _peak(L["rejected"], op)            # per kW of heat rejected
    elif case.cold == "vcr":
        c_cold = (p.c_hp_heat + p.c_rej) * _peak(L["rejected"], op)            # per kW of heat output
    else:                                              # lng: the coils are the LNG exchangers
        c_cold = 0.0
    sorb = _peak(L["co2"] / (np.where(op, L["wcap"], np.inf) * np.maximum(L["cycles"], 1e-3)), op) * 3600
    silica = _peak(L["sil"], op) * 3600 / (p.sil_dq * p.sil_cycles)
    if p.hx_cost_model == "netl" and p.c_hx_area is None:
        # cost per UA (W/K): independent of the assumed heat transfer coefficient; exchangers in contact with LNG and
        # recuperators below 227 K (-46 C, limit of low-temperature carbon steel) in aluminium or stainless steel
        f_coil = p.f_cryo_installed if case.cold == "lng" else 1.0
        f_rec = p.f_cryo_installed if (L["T_design"] or 400.0) < 227.15 else 1.0
        ua = p.U_air * area_rec * f_rec + p.U_coil * area_coil * f_coil      # kW/K, weighted by the material factor
        hx = p.hx_cost_ua * ua * 1e3
    else:
        hx = p.hx_cost_m2 * (area_rec + area_coil)
    fan, vac = p.c_fan * _peak(L["fan"], op), p.c_vac * _peak(L["vac"], op)
    if "kim" in (p.hx_cost_model, p.fanvac_cost_model):
        # Kim et al. cost functions apply to the whole plant: scale from 1 kg/s air to kim_scale_tpy
        co2_tpy = float(np.sum(np.where(op, L["co2"], 0.0))) * 3.6            # t CO2/yr per kg/s air
        n = p.kim_scale_tpy / max(co2_tpy, 1e-9)                              # kg/s air of the plant
        if p.hx_cost_model == "kim":
            hx = 2.0 * ((area_rec + area_coil) * n) ** 0.7 * 100.0 * p.kim_f_tci / n
        if p.fanvac_cost_model == "kim":
            v_air = n / 1.2                                                    # m3/s
            fan = (v_air / 1.77) ** 0.8 * 2000.0 * p.kim_f_tci / n
            if sb.mode == "vacuum":
                r_w = float(np.mean(np.where(op, L["r_w"], 0.0)[op])) if op.any() else 0.0
                f_vac = p.kim_scale_tpy * 1e6 / 44.01 * (1.0 + r_w) / 8760.0 * 8.314 * sb.T_des / 101325.0   # m3/h
                vac = (423.9 * f_vac ** 0.653 + 30000.0) * 1.07 * p.kim_f_tci / n
            else:
                vac = 0.0
    parts = dict(cold=c_cold, hx=hx, fan=fan, vac=vac,
                 sorbent=(p.c_contactor * p.contactor_fab + sb.cost * p.sorbent_cost_factor * p.sorbent_fab) * sorb,
                 silica=p.c_silica_sys * silica
                 + (p.c_hp_heat * _peak(L["q_sil"], op) if p.silica_regen == "hp" else 0.0))
    parts["total"] = sum(parts.values())
    parts["sorbent_kg"] = sorb
    return parts


# ------------------------------------------------------------------ evaluation
def dispatch_many(want, dem_th, dem_el, q_sol, pv, A, S_th, P_pv, S_el, co2, water, p: Prm):
    """Vectorised dispatch of D designs x K sizings in one hourly loop.
    want/dem/co2/water: (H, D); sizes: (D, K). Returns hours, co2 [kg-h], water sums (D, K)."""
    H = want.shape[0]
    s1, s2 = 0.5 * S_th, 0.5 * S_el
    hours = np.zeros_like(A)
    c = np.zeros_like(A)
    wt = np.zeros_like(A)
    keep = 1.0 - p.storage_loss_h
    for h in range(H):
        g1, g2 = q_sol[h] * A, pv[h] * P_pv
        s1 = s1 * keep
        dth, dl = dem_th[h][:, None], dem_el[h][:, None]
        can = want[h][:, None] & (s1 + g1 >= dth) & (s2 + g2 >= dl)
        n1 = g1 - can * dth
        n2 = g2 - can * dl
        s1 = np.minimum(s1 + n1, S_th)
        s2 = np.minimum(s2 + np.where(n2 > 0, p.eta_batt * n2, n2), S_el)
        hours += can
        c += can * co2[h][:, None]
        wt += can * water[h][:, None]
    return hours, c, wt


def evaluate(case: Case, w: dict, res: dict | None = None, T_list=None, dT_list=None) -> dict:
    """Lowest-LCOC operation of the fixed architecture in `case` at weather `w`.

    Every design (T_ads x recuperator approach x operating strategy) and every energy-block
    sizing is dispatched hour by hour in one vectorised pass, so the optimum is exact over the
    grid. T_list overrides the design adsorption temperatures tried (validation runs only); dT_list the
    recuperator approaches (None = no recuperator), e.g. to fix the cold recovery."""
    p = case.prm
    if not case.applicable:
        return dict(status="n/a")
    res = res or resources(w, p)
    T0 = w["T0"]
    cold40 = T0 <= np.percentile(T0, 40)
    meta, cols = [], {k: [] for k in ("want", "heat", "elec", "co2", "water")}
    caps = []
    dts = tuple(dT_list or DT_REC) if case.cold != "ambient" else (None,)
    if case.drying == "freeze":
        # the reversing regenerators that freeze the water out also recover the cold of the exhaust,
        # so freeze-out is only possible with a finite approach (and the regenerators are costed)
        dts = tuple(x for x in dts if x is not None)
        if not dts:
            return dict(status="n/a")
    rh0 = np.clip(p_sat(w["Td"]) / p_sat(T0), 0.0, 1.0)
    for cp in _p_des_cases(case):
        sb = sorbent_for(cp)
        # an adsorbent whose pores fill with water above a relative humidity is not operated in those hours
        # (otherwise the equipment is sized for the pore-filled hours)
        step = getattr(sb, "h2o_rh_step", None)
        strategies = STRATEGIES + (("below_rh_step",) if step is not None else ())
        for T_design in (T_list or T_ADS[case.cold]):
            for dT_rec in dts:
                L = air_train(w, cp, T_design, dT_rec)
                for strat in strategies:
                    want = L["feasible"] & (cold40 if strat == "cold40" else
                                            (rh0 < step) if strat == "below_rh_step" else True)
                    if want.sum() < MIN_HOURS:
                        continue
                    meta.append((T_design, dT_rec, strat, cp))
                    caps.append(process_capex(L, want, cp, sb))
                    cols["want"].append(want)
                    for k in ("heat", "elec", "co2", "water"):
                        cols[k].append(np.where(want, L[k], 0.0))
    if not meta:
        return dict(status="infeasible")
    X = {k: np.stack(v, axis=1) for k, v in cols.items()}            # (H, D)
    cap_tot = np.array([c["total"] for c in caps])
    if p.energy_mode == "grid":
        hours = X["want"].sum(0)[:, None].astype(float)
        co2_h, water_h = X["co2"].sum(0)[:, None], X["water"].sum(0)[:, None]
        e_ann = (X["heat"].sum(0) * 3.6e-3 / p.eta_thrm * p.c_heat_grid
                 + X["elec"].sum(0) / 1e3 * p.c_elec_grid)[:, None]
        cap_e = e_ann / (p.crf + p.om_energy)
        sizes = np.zeros((len(meta), 1, 4))
    else:
        g = np.stack([np.stack(_grid(X["heat"][:, d], X["elec"][:, d], X["want"][:, d], res), -1)
                      for d in range(len(meta))])                      # (D, K, 4)
        A, S_th, P_pv, S_el = (g[..., i] for i in range(4))
        hours, co2_h, water_h = dispatch_many(X["want"], X["heat"], X["elec"], res["q_sol"], res["pv"],
                                              A, S_th, P_pv, S_el, X["co2"], X["water"], p)
        cap_e = p.c_ptc * A + p.c_tank * S_th + p.c_pv * P_pv + p.c_batt * S_el
        sizes = g
    co2_t, water_t = co2_h * 3.6, water_h * 3.6
    annual = cap_tot[:, None] * (p.crf + p.om_proc) + cap_e * (p.crf + p.om_energy)
    lcoc = np.where(hours >= MIN_HOURS, (annual - case.water_value * water_t) / np.maximum(co2_t, 1e-9), np.inf)
    d, k = np.unravel_index(int(np.argmin(lcoc)), lcoc.shape)
    if not np.isfinite(lcoc[d, k]):
        return dict(status="infeasible")
    T_design, dT_rec, strat, cp = meta[d]
    # re-run the chosen design once to recover the hourly operation and the breakdowns
    L = air_train(w, cp, T_design, dT_rec)
    want = X["want"][:, d]
    if p.energy_mode == "grid":
        r = want
    else:
        sz = sizes[d, k]
        r = dispatch(want, X["heat"][:, d], X["elec"][:, d], res["q_sol"], res["pv"], sz[[0]], sz[[1]],
                     sz[[2]], sz[[3]], p)[:, 0]
    cap = caps[d]
    co2 = float(co2_t[d, k])
    gj = lambda x: float((x * r).sum() * 3.6e-3 / max(co2, 1e-9))   # noqa: E731  GJ/t
    tg = L["tgen"][r & np.isfinite(L["tgen"])]
    A_, Sth_, Ppv_, Sel_ = (float(v) for v in sizes[d, k]) if p.energy_mode != "grid" else (0.0,) * 4
    return dict(
        status="ok", lcoc=float(lcoc[d, k]), T_design=T_design, p_des=float(sorbent_for(cp).p_des),
        dT_rec=np.nan if dT_rec is None else dT_rec, strategy=strat,
        hours=int(hours[d, k]), co2_t_per_kgs=co2, water_t_per_kgs=float(water_t[d, k]),
        water_per_co2=float(water_t[d, k] / max(co2, 1e-9)),
        T_ads_mean=float(L["T_ads"][r].mean()), chilled_share=float(L["chilled"][r].mean()),
        heat_GJ=gj(L["heat"]), heat_gen_GJ=gj(L["q_gen"]), heat_des_GJ=gj(L["q_des"]),
        heat_sil_GJ=gj(L["q_sil"]), heat_defrost_GJ=gj(L["q_defrost"]),
        elec_GJ=gj(L["elec"]), elec_fan_GJ=gj(L["fan"]), elec_vac_GJ=gj(L["vac"]),
        elec_cold_GJ=gj(L["el_cold"]), cold_GJ=gj(L["Q1"] + L["Q2"]),
        wcap_mmolg=float(np.nanmean(L["wcap"][r]) / M_CO2 * 1e3),
        purity=float(np.nanmean(L["purity"][r])),
        tgen_max=float(tg.max()) if case.cold == "gax" and tg.size else np.nan,
        tgen_over_165_share=float((tg > 165).mean()) if case.cold == "gax" and tg.size else np.nan,
        capex_process=cap["total"], capex_energy=float(cap_e[d, k]),
        capex_cold=cap["cold"], capex_sorbent=cap["sorbent"], sorbent_kg=cap["sorbent_kg"],
        A_ptc_m2=A_, tank_kWh=Sth_, P_pv_kW=Ppv_, batt_kWh=Sel_,
        land_m2_per_tpy=(A_ * 2.5 + Ppv_ * 15.0) / max(co2, 1e-9),   # (a) land per m2 / kWp
        feasible_hours=int(want.sum()), n_designs=len(meta),
    )


def amine(w: dict, res: dict | None = None, p: Prm = Prm()) -> dict:
    """Amine DAC (Wenzel et al. energy) on the same off-grid energy block at the same site."""
    from .process import amine_energy
    res = res or resources(w, p)
    T_mean = float(w["T0"].mean() - 273.15)
    RH = float(100 * np.clip(p_sat(w["Td"]) / p_sat(w["T0"]), 0, 1).mean())
    heat_GJ, elec_GJ = (float(x) for x in amine_energy(T_mean, RH))
    H = len(w["T0"])
    out = dict(amine_heat_GJ=heat_GJ, amine_elec_GJ=elec_GJ, T_mean_C=T_mean, RH_mean=RH)
    want = np.ones(H, bool)
    for tag, capex, om, fe in zip(("lo", "base", "hi"), p.amine_capex, p.amine_om, p.amine_energy_factor):
        dem_th = np.full(H, fe * heat_GJ / 3.6e-3)          # kW for 1 t CO2 / h
        dem_el = np.full(H, fe * elec_GJ / 3.6e-3)
        ran, eb = energy_block(dem_th, dem_el, want, res, p)
        hours = ran.sum(0)
        annual = capex * 8760 * 0.9 * (p.crf + om) + eb["capex"] * (p.crf + p.om_energy)
        lcoc = np.where(hours >= MIN_HOURS, annual / np.maximum(hours, 1), np.inf)
        k = int(np.argmin(lcoc))
        out[f"amine_lcoc_{tag}"] = float(lcoc[k])
        if tag == "base":
            out["amine_hours"] = int(hours[k])
    return out
