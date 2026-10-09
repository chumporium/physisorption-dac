"""Hourly process model, solar supply, storage dispatch, sizing and levelized costs.

Plant (per 1 kg/s dry air at nominal flow):
  ambient -> recuperator 1 (exhaust) -> ARS-1 condensing coil (T_cond, water out)
  -> silica + zeolite guard (to dewpoint_dry, heat of adsorption into air)
  -> recuperator 2 (exhaust) -> ARS-2 trim coil (T_ads) -> 13X contactor -> exhaust back.
With balanced recuperators the only cold that must be bought is
  ARS-1: warm-end approach + condensation latent heat + silica adsorption heat
  ARS-2: cold-end approach + CO2 heat of adsorption + re-cooling of the regenerated bed.
Heat (solar PTC + storage, backup gas): ARS generators, silica regeneration, 13X desorption.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ars_map, energy, sorbents
from .params import DacParams, crf
from .weather import dni_from_fdir, humidity_ratio, sun

CP_AIR = 1.006
H_FG = 2470.0
M_CO2 = 44.01


def working_capacity_13x(T_ads: float, p_co2_bar: float = 4.0e-4) -> float:
    """kg CO2 / kg 13X between p_co2 at T_ads and CO2 sweep at 473 K (kept for tests/back-compat)."""
    return float(sorbents.load()["13X"].working_capacity(T_ads, p_co2_bar))


def incidence_modifier(cos_i):
    th = np.degrees(np.arccos(np.clip(cos_i, 0, 1)))
    return np.clip(np.cos(np.radians(th)) + 0.000884 * th - 0.00005369 * th ** 2, 0, 1)


def hourly_loads(w: dict, p: DacParams) -> dict:
    """Specific loads per kg dry air for every hour and cell (arrays (H, n)).

    Adsorption runs at T_ads_eff = min(design T_ads, ambient): in cold hours the contactor
    works at ambient temperature and ARS-2 is not needed. CO2 partial pressure follows the
    surface pressure (lower at altitude).
    Heat is split by temperature level:
      heat_hi  (~150-230 C): ARS generators, sweep desorption (13X)
      heat_lo  (< 100 C)   : silica regeneration, vacuum-desorption heat (CALF-20)
    """
    sb = sorbents.get(p.sorbent)
    T0, Td, P = w["T0"], w["Td"], w["P"]
    T0C = T0 - 273.15
    w0 = humidity_ratio(Td, P)
    wc = humidity_ratio(np.full_like(T0, p.T_cond + 273.15), P)
    ws = humidity_ratio(np.full_like(T0, p.dewpoint_dry + 273.15), P)
    x_co2 = p.x_co2_ppm * 1e-6 * M_CO2 / 28.96
    co2 = p.eta_cap * x_co2 + 0 * T0                            # kg CO2 / kg air
    p_co2 = p.x_co2_ppm * 1e-6 * P / 1e5                        # bar
    T_ads = np.minimum(p.T_ads, T0 - 1.0)                       # adsorb at ambient if colder
    chilled = T0 - 1.0 > p.T_ads                                # ARS-2 needed this hour
    wcap = sb.working_capacity(T_ads, p_co2)                    # kg CO2 / kg sorbent (dry feed)
    if sb.hydrophobic:
        wcap = wcap * sb.retention                              # humid-feed capacity loss
    cycles = sb.cycles(T_ads)                                   # cycles per hour (kinetics)
    # N2 co-adsorption (assumed Henry behaviour): purity of the desorbed gas. Below the purity
    # target a CO2 rinse displaces the N2, costing product CO2 mole-for-mole.
    n2_per_co2 = (0.7808 / (p.x_co2_ppm * 1e-6)) / sb.selectivity_n2(T_ads)   # mol N2 per mol CO2
    purity = 1.0 / (1.0 + n2_per_co2)
    rinse = np.where(purity < p.purity_min, np.clip(1.0 - n2_per_co2, 0.0, 1.0), 1.0)
    co2 = co2 * rinse
    wcap_ok = (wcap > 0.1e-3 * M_CO2) & (rinse > 0.05)          # >= 0.1 mmol/g, rinse feasible
    wcap_s = np.where(wcap_ok, wcap, 1.0)                       # safe divisor
    hoa = sb.hoa * 1e3 / M_CO2                                  # kJ/kg CO2
    # drying target: dewpoint_dry for water-sensitive sorbents, or rh_max at T_ads if tolerated
    w_target = np.maximum(ws, sb.rh_max * humidity_ratio(T_ads, P))
    cond = np.maximum(w0 - np.maximum(wc, w_target), 0.0)
    sil = np.maximum(np.minimum(w0, wc) - w_target, 0.0)
    frost = 0.0 * T0
    frz = 0.0 * T0                                              # water frozen out on regenerators
    if p.drying == "freeze" and not sb.hydrophobic:
        # switching (reversing) regenerators freeze the water out down to ice saturation a few K
        # below T_ads; the dry exhaust re-sublimes it on the next switch, so only a fraction of the
        # latent+fusion heat is lost. The guard bed only handles the small remainder.
        w_fz = np.minimum(humidity_ratio(T_ads - 3.0, P), wc)
        frz = np.where(chilled, np.maximum(np.minimum(w0, wc) - np.maximum(w_fz, w_target), 0.0), 0.0)
        sil = sil - frz
    if sb.hydrophobic:
        # No drying train, but cooling the air to T_ads still condenses (above 0 C) or
        # frosts (below 0 C) whatever exceeds saturation at T_ads; that latent heat is unavoidable.
        w_ads = np.minimum(humidity_ratio(T_ads, P), w0)
        w_0C = humidity_ratio(np.full_like(T0, 273.15), P)
        cond = np.where(chilled, np.maximum(w0 - np.maximum(w_0C, w_ads), 0.0), 0.0)
        frost = np.where(chilled, np.maximum(np.minimum(w0, w_0C) - w_ads, 0.0), 0.0)
        sil = 0.0 * T0

    # Warm recuperator/condensing stage only exists when ambient is above the coil outlet.
    # The cold train is only worth a recuperator when the remaining span exceeds its approach:
    # cold to buy = min(direct cooling of the span, cold-end approach).
    warm = (T0C > p.T_cond) & (not sb.hydrophobic)
    T_train_in = (T0C if sb.hydrophobic else np.minimum(T0C, p.T_cond)) + 273.15
    span = np.maximum(T_train_in - T_ads, 0.0)
    Q1 = np.where(warm, CP_AIR * p.dT_warm, 0.0) + cond * H_FG + sil * p.h_ads_silica
    # cold hours: bed re-cooled and heat of adsorption carried away by the (cold) air itself
    Q2 = np.where(chilled, CP_AIR * np.minimum(span, p.dT_cold) + co2 * hoa + frost * 2834.0
                  + frz * 2834.0 * p.freeze_loss
                  + co2 / wcap_s * p.cp_bed * np.maximum(T0 + 5.0 - T_ads, 0), 0.0)
    cop1 = p.cop_derate * ars_map.cop(T0C, p.T_cond - p.dT_evap)
    cop2 = p.cop_derate * ars_map.cop(T0C, p.T_ads - 273.15 - p.dT_evap)
    need1 = Q1 > 0
    ok1 = ~need1 | (np.isfinite(cop1) & (cop1 > 0.05))
    ok2 = ~chilled | (np.isfinite(cop2) & (cop2 > 0.05))
    if p.free_cold:                                             # LNG/LH2 cold: nothing to buy
        ok1 = ok2 = np.ones_like(T0, bool)
        cop1 = cop2 = np.full_like(T0, np.inf)
    feasible = ok1 & ok2 & wcap_ok
    if not p.allow_chill:
        feasible &= ~chilled
    cop1 = np.where(ok1 & need1, cop1, 1.0)
    cop2 = np.where(ok2 & chilled, cop2, 1.0)
    q_ars = Q1 / cop1 + Q2 / cop2
    q_sil = sil * p.h_ads_silica * p.silica_regen_factor
    # desorption: bed warmed for free by ambient air up to ambient, bought above it
    q_des = co2 * (p.cp_bed / wcap_s * np.maximum(sb.T_des - np.maximum(T0, T_ads), 0) + hoa)
    # co-adsorbed water (tolerant sorbents) is desorbed every cycle: binding + sensible heat
    q_des = q_des + co2 * sb.water_ratio * (2800.0 + 4.2 * np.maximum(sb.T_des - T_ads, 0))
    rho = P / (287.05 * T0)
    fan = p.dP_air / rho / p.eta_fan / 1e3                      # kW per kg/s
    vac = (co2 / M_CO2 * 8.314 * sb.T_des * np.log(1.0 / sb.p_des) / p.eta_vac
           if sb.mode == "vacuum" else 0.0 * T0)                # kW per kg/s
    rejected = Q1 + Q2 + q_ars
    elec = fan + p.fan_rej_frac * rejected + vac
    hi_des = sb.mode == "sweep"
    heat_hi = q_ars + (q_des if hi_des else 0.0)
    heat_lo = q_sil + (0.0 if hi_des else q_des)
    # exergy of the purchased energy (heat at its delivery temperature, electricity as is)
    ex = (heat_hi * (1 - T0 / p.T_htf_K) + heat_lo * (1 - T0 / p.T_hp_sink)).clip(min=0) + elec
    return dict(Q1=Q1, Q2=Q2, cop1=cop1, cop2=cop2, feasible=feasible, chilled=chilled,
                heat=heat_hi + heat_lo, heat_hi=heat_hi, heat_lo=heat_lo, exergy=ex,
                q_ars=q_ars, q_sil=q_sil, q_des=q_des, vac=vac, purity=purity, rinse=rinse,
                cycles=cycles, elec=elec, fan=fan, rejected=rejected, co2=co2, T_ads=T_ads,
                water=cond + frost + frz + sil * p.water_recovery_silica, cond=cond, sil=sil,
                frost=frost, frz=frz, w0=w0, wcap=np.where(feasible, wcap, np.nan))


def solar_heat(w: dict, meta: dict, cells_lat, cells_lon, p: DacParams):
    """Useful PTC heat per m2 aperture [kW/m2] (H, n) and cos(zenith)."""
    H = w["T0"].shape[0]
    # value at index h = accumulation over the hour ending at start + h -> sun() uses h - 0.5
    cz, ci = sun(np.arange(H), meta["start"], cells_lat, cells_lon)
    dni = dni_from_fdir(w["fdir"], cz)
    gain = p.ptc_eta_opt * incidence_modifier(ci) * dni * ci / 1e3
    loss = p.ptc_loss * np.maximum(p.T_htf - (w["T0"] - 273.15), 0)
    q = np.where(dni > 0, np.maximum(gain - loss, 0.0), 0.0)
    return q, cz, dni


STRATEGIES = ("continuous", "night", "cool", "cool20", "below0", "below-15")


def strategy_mask(strat: str, feasible, cz, T0):
    """Hours in which the plant operates (seasonal / diurnal operating strategies)."""
    op = feasible.copy()
    if strat == "night":
        op &= cz <= 0
    elif strat == "cool":
        op &= T0 <= np.percentile(T0, 40, axis=0)[None, :]
    elif strat == "cool20":
        op &= T0 <= np.percentile(T0, 20, axis=0)[None, :]
    elif strat == "below0":
        op &= T0 < 273.15
    elif strat == "below-15":
        op &= T0 < 258.15
    return op
AREA_F = (0.9, 1.1, 1.35, 1.7, 2.2)
STORE_H = (6.0, 12.0, 18.0)


def dispatch(q_sol, demand, A, S_max, loss):
    """Hourly storage dispatch. q_sol (H,n); demand (H,n); A,S_max (n,k). Returns
    backup heat [kWh] and dumped solar [kWh] per (n,k)."""
    S = 0.5 * S_max
    backup = np.zeros_like(A)
    dumped = np.zeros_like(A)
    for h in range(q_sol.shape[0]):
        S = S * (1 - loss) + q_sol[h][:, None] * A - demand[h][:, None]
        over = S > S_max
        dumped += np.where(over, S - S_max, 0.0)
        S = np.where(over, S_max, S)
        under = S < 0
        backup += np.where(under, -S, 0.0)
        S = np.where(under, 0.0, S)
    return backup, dumped


SUPPLIES = ("ptc", "pv", "wind", "hybrid", "local_heat", "firm")
_AMINE = json.loads(Path(__file__).with_name("amine_benchmark.json").read_text())


def amine_energy(T_mean_C, RH_mean):
    """Amine (Lewatit TVSA) DAC energy per t CO2 from the Wenzel et al. regional data:
    heat [GJ], electricity [GJ]."""
    T, RH = np.asarray(T_mean_C, float), np.asarray(RH_mean, float)
    X = np.stack([np.ones_like(T), T, RH, T * RH, T ** 2, RH ** 2, T ** 2 * RH, T * RH ** 2,
                  (T > 15) * (RH < 35) * (T - 15) * (35 - RH)])
    heat = np.tensordot(_AMINE["coef"]["heat_GJ"], X, 1)
    elec = np.tensordot(_AMINE["coef"]["elec_GJ"], X, 1)
    return np.clip(heat, 4.0, 20.0), np.clip(elec, 0.8, 6.0)


def evaluate_cells(w: dict, meta: dict, lat, lon, p: DacParams) -> dict:
    """Cheapest (operating strategy x energy supply x sizing) per cell, ranked by the net
    levelized cost of CO2 (LCOC, $/t net CO2, water credited at p.water_value)."""
    L = hourly_loads(w, p)
    q_sol, cz, dni = solar_heat(w, meta, lat, lon, p)
    H, n = q_sol.shape
    T0 = w["T0"]
    CRF = crf(p)
    cf_pv = energy.pv_cf(w["ssrd"], T0)
    cf_w = energy.wind_cf(w["u100"], w["v100"], T0, w["P"]) if "u100" in w else np.zeros_like(T0)
    # firm electricity bought at the cell's cheaper VRE LCOE x firming factor ($/MWh)
    lcoe_pv = p.c_pv * (CRF + p.om_vre) / np.maximum(cf_pv.sum(0), 1.0) * 1e3
    lcoe_w = p.c_wind * (CRF + p.om_vre) / np.maximum(cf_w.sum(0), 1.0) * 1e3
    c_el = np.minimum(lcoe_pv, lcoe_w) * p.firming
    cop_hp = energy.heat_pump_cop(T0, p.T_hp_sink)
    r_idx = np.arange(n)
    # levelized heat prices on the firm-energy basis ($/GJ), shared with the amine benchmark
    lcoh_ptc = (p.c_ptc * (CRF + p.om_frac) / np.maximum(q_sol.sum(0) * 3.6e-3, 1e-6)) * p.firming
    c_heat_hi = np.minimum(c_el / 0.99 / 3.6, lcoh_ptc)
    c_heat_lo = np.minimum(c_el / (cop_hp.mean(0) * 3.6), c_heat_hi)
    supplies = ("firm",) if p.energy_mode == "firm" else ("ptc", "pv", "wind", "hybrid")
    if p.local_heat_price is not None:
        supplies = supplies + ("local_heat",)

    def peak(x, op):
        return np.where(op, x, 0).max(0)

    best = None
    for strat in STRATEGIES:
        op = strategy_mask(strat, L["feasible"], cz, T0)
        hours = op.sum(0)
        if (hours < 500).all():                       # fewer than ~3 weeks a year: not a plant
            continue
        op &= (hours >= 500)[None, :]
        water = np.where(op, L["water"], 0).sum(0) * 3.6            # t/yr (kg/s x 3600 s)
        co2 = np.where(op, L["co2"], 0).sum(0) * 3.6                 # t/yr
        heat_hi = np.where(op, L["heat_hi"], 0.0)
        heat_lo = np.where(op, L["heat_lo"], 0.0)
        elec_h = np.where(op, L["elec"], 0.0)
        E_heat = (heat_hi + heat_lo).sum(0)                           # kWh_th / yr
        E_el = elec_h.sum(0)                                          # kWh_e / yr

        # process capital (per kg/s dry air), independent of the energy supply
        T0p = np.nan_to_num(np.percentile(np.where(op, T0, np.nan), 95, axis=0) - 273.15, nan=30.0)
        ntu1 = np.where(T0p > p.T_cond, np.maximum(T0p - p.T_cond, 5.0) / p.dT_warm, 0.0)
        ntu2 = np.maximum(np.minimum(T0p, p.T_cond) - (p.T_ads - 273.15), 0.0) / p.dT_cold
        area_rec = (ntu1 + ntu2) * CP_AIR / p.U_air
        area_coil = (peak(L["Q1"], op) + peak(L["Q2"], op)) / (0.05 * 1.5 * p.dT_evap)
        # sorbent inventory for the worst operating hour, cycles limited by kinetics
        sorb = peak(L["co2"] / (np.where(op, L["wcap"], np.inf) * np.maximum(L["cycles"], 1e-3)), op) * 3600
        silica = peak(L["sil"], op) * 3600 / (0.10 * 2.0)
        capex_ars = (p.c_ars_warm * peak(L["Q1"], op) + p.c_ars_cold * peak(L["Q2"], op)
                     + p.c_rej * peak(L["rejected"], op))
        if p.free_cold:                                   # LNG/LH2-air exchangers replace the ARS
            capex_ars = p.c_hx_area * (peak(L["Q1"], op) + peak(L["Q2"], op)) / (0.05 * 10.0)
        if p.drying == "freeze":
            area_rec = area_rec * 1.3                     # switching regenerators, duplicated passages
        capex_proc = (capex_ars + p.c_hx_area * (area_rec + area_coil) + p.c_fan * peak(L["elec"], op)
                      + (p.c_contactor + sorbents.get(p.sorbent).cost) * sorb + p.c_silica_sys * silica)

        for sup in supplies:
            if sup in ("local_heat", "firm"):
                # firm: heat from e-boiler/heat pump on firm clean power or PTC (cheapest levelized);
                # local_heat: geothermal / waste heat at a fixed price
                if sup == "firm":
                    heat_cost = (heat_hi.sum(0) * c_heat_hi + heat_lo.sum(0) * c_heat_lo) * 3.6e-3
                    conv = p.c_eboiler * peak(heat_hi, op) + p.c_hp * peak(heat_lo, op)
                else:
                    heat_cost = E_heat * 3.6e-3 * p.local_heat_price
                    conv = 0.0 * E_heat
                capex_sup = np.asarray(conv)[:, None]
                annual_e = (heat_cost + E_el / 1e3 * c_el)[:, None] + capex_sup * (CRF + p.om_vre)
                emis = np.zeros((n, 1))
                store = np.zeros((n, 1))
                size = np.zeros((n, 1))
                short = np.zeros((n, 1))
            else:
                if sup == "ptc":
                    unit = q_sol                                      # kW_th per m2
                    dem = heat_hi + heat_lo                           # kW_th
                    c_unit, c_store, c_conv = p.c_ptc, p.c_storage, 0.0
                    el_bought = E_el / 1e3 * c_el                     # fans etc. at firm LCOE
                else:
                    mix = {"pv": 1.0, "wind": 0.0, "hybrid": 0.5}[sup]
                    unit = mix * cf_pv + (1 - mix) * cf_w             # kW_e per kW installed
                    if unit.sum() <= 0:
                        continue
                    e_hi = heat_hi / 0.99
                    e_lo = heat_lo / cop_hp
                    dem = e_hi + e_lo + elec_h                        # kW_e equivalent
                    th_share = (e_hi + e_lo).sum(0) / np.maximum(dem.sum(0), 1e-9)
                    c_unit = mix * p.c_pv + (1 - mix) * p.c_wind
                    c_store = (th_share * p.c_storage + (1 - th_share) * p.c_batt)[:, None]
                    c_conv = p.c_eboiler * peak(heat_hi, op) + p.c_hp * peak(heat_lo, op)
                    el_bought = 0.0
                annual_dem = dem.sum(0)
                size0 = annual_dem / np.maximum(unit.sum(0), 1e-6)
                mean_dem = annual_dem / np.maximum(hours, 1)
                combos = [(fa, sh) for fa in AREA_F for sh in STORE_H]
                size = size0[:, None] * np.array([c[0] for c in combos])[None, :]
                store = mean_dem[:, None] * np.array([c[1] for c in combos])[None, :]
                short, _ = dispatch(unit, dem, size, store, p.storage_loss_h)
                if sup == "ptc":
                    short_cost = short * 3.6e-3 * p.c_backup_heat
                    emis = short * 3.6e-3 * p.co2_backup
                else:
                    short_cost = short / 1e3 * p.c_backup_el
                    emis = short / 1e3 * p.co2_backup_el
                om = p.om_frac if sup == "ptc" else p.om_vre
                capex_sup = c_unit * size + c_store * store + np.asarray(c_conv)[..., None] * (sup != "ptc")
                annual_e = capex_sup * (CRF + om) + short_cost + np.asarray(el_bought)[..., None]
            annual = capex_proc[:, None] * (CRF + p.om_frac) + annual_e
            net_co2 = co2[:, None] - emis
            lcoc = (annual - p.water_value * water[:, None]) / np.maximum(net_co2, 1e-9)
            lcoc = np.where(net_co2 > 0.05 * co2[:, None], lcoc, np.inf)
            k = np.argmin(lcoc, axis=1)
            cand = dict(
                strategy=np.full(n, STRATEGIES.index(strat)), supply=np.full(n, SUPPLIES.index(sup)),
                lcoc=lcoc[r_idx, k], lcoc_gross=annual[r_idx, k] / np.maximum(co2, 1e-9),
                lcow=(annual[r_idx, k] - p.co2_price * net_co2[r_idx, k]) / np.maximum(water, 1e-9),
                water_t=water, co2_t=co2, net_co2_t=net_co2[r_idx, k], hours=hours,
                annual=annual[r_idx, k], capex_proc=capex_proc, capex_ars=capex_ars,
                capex_supply=capex_sup[r_idx, np.minimum(k, capex_sup.shape[1] - 1)],
                supply_size=size[r_idx, np.minimum(k, size.shape[1] - 1)],
                storage_kWh=store[r_idx, np.minimum(k, store.shape[1] - 1)],
                backup_share=short[r_idx, np.minimum(k, short.shape[1] - 1)]
                / np.maximum((E_heat if sup == "ptc" else E_heat + E_el), 1e-9),
                heat_GJ_per_tco2=E_heat * 3.6e-3 / np.maximum(co2, 1e-9),
                heat_hi_GJ_per_tco2=heat_hi.sum(0) * 3.6e-3 / np.maximum(co2, 1e-9),
                exergy_GJ_per_tco2=np.where(op, L["exergy"], 0).sum(0) * 3.6e-3 / np.maximum(co2, 1e-9),
                cold_GJ_per_tco2=np.where(op, L["Q1"] + L["Q2"], 0).sum(0) * 3.6e-3 / np.maximum(co2, 1e-9),
                purity=np.nanmean(np.where(op, L["purity"], np.nan), 0),
                rinse_loss=1 - np.nanmean(np.where(op, L["rinse"], np.nan), 0),
                sorbent_kg=sorb,
                elec_GJ_per_tco2=E_el * 3.6e-3 / np.maximum(co2, 1e-9),
                heat_GJ_per_m3=E_heat * 3.6e-3 / np.maximum(water, 1e-9),
                water_per_co2=water / np.maximum(co2, 1e-9),
                chilled_share=np.where(op, L["chilled"], False).sum(0) / np.maximum(hours, 1),
                wcap_mmolg=np.nanmean(np.where(op, L["wcap"], np.nan), 0) / M_CO2 * 1e3,
                cop1=np.nanmean(np.where(op & (L["Q1"] > 0), L["cop1"], np.nan), 0),
                cop2=np.nanmean(np.where(op & L["chilled"], L["cop2"], np.nan), 0),
            )
            if best is None:
                best = cand
            else:
                better = cand["lcoc"] < best["lcoc"]
                for key in best:
                    best[key] = np.where(better, cand[key], best[key])

    if best is None:                                  # no strategy with enough operating hours
        best = {k: np.full(n, np.nan) for k in ("lcoc", "lcoc_gross", "lcow", "water_t", "co2_t",
                                                "net_co2_t", "hours", "annual", "heat_GJ_per_tco2")}
        best["strategy"] = np.zeros(n)
        best["supply"] = np.zeros(n)
        best["lcoc"] = np.full(n, np.inf)
    # climate descriptors and resources
    T_mean = T0.mean(0) - 273.15
    RH = 100 * np.clip(p_sat_ratio(w["Td"], T0), 0, 1).mean(0)
    best.update(T0_mean=T_mean, RH_mean=RH, w0_mean=L["w0"].mean(0) * 1e3,
                dni_kwh=dni.sum(0) / 1e3, cf_pv=cf_pv.mean(0), cf_wind=cf_w.mean(0),
                lcoe_firm=c_el, P_mean=w["P"].mean(0))
    # amine benchmark at the same site: same capital recovery, heat by the cheaper of
    # heat pump on firm VRE electricity or PTC heat (both levelized), electricity at c_el
    a_heat, a_el = amine_energy(T_mean, RH)
    cop_a = energy.heat_pump_cop(T0.mean(0), p.amine_T_heat)
    lcoh_hp = c_el / cop_a / 3.6                                         # $/GJ
    c_heat_a = np.minimum(lcoh_hp, lcoh_ptc)
    if p.local_heat_price is not None:
        c_heat_a = np.minimum(c_heat_a, p.local_heat_price)
    best["amine_heat_GJ"], best["amine_elec_GJ"] = a_heat, a_el
    best["amine_exergy_GJ"] = a_heat * np.clip(1 - T0.mean(0) / p.amine_T_heat, 0, 1) + a_el
    # amine range: optimistic (cheap, stable), base, pessimistic (Kim et al. reference: 50 %/yr
    # sorbent degradation, higher capital, +15 % energy)
    for tag, capex, om, fe in (("lo", p.amine_capex_lo, p.amine_om_lo, 0.85),
                               ("", p.amine_capex_tpy, p.amine_om, 1.0),
                               ("hi", p.amine_capex_hi, p.amine_om_hi, 1.15)):
        key = "lcoc_amine" + (f"_{tag}" if tag else "")
        best[key] = capex * (CRF + om) + fe * (a_heat * c_heat_a + a_el / 3.6 * c_el)
    best["ratio_vs_amine"] = best["lcoc"] / best["lcoc_amine"]
    best["ratio_vs_amine_lo"] = best["lcoc"] / best["lcoc_amine_lo"]     # vs optimistic amine
    best["ratio_vs_amine_hi"] = best["lcoc"] / best["lcoc_amine_hi"]     # vs pessimistic amine
    return best


def p_sat_ratio(Td, T0):
    from .weather import p_sat
    return p_sat(Td) / p_sat(T0)
