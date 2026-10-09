"""Parameters of the solar-ARS sub-ambient DAC + atmospheric water harvesting plant.

All plant quantities are per 1 kg/s of dry air processed at design (nominal) flow.
Values marked (a) are assumptions for screening; change them here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DacParams:
    # ---- air-side process ----------------------------------------------------
    T_ads: float = 235.0          # K contactor (adsorption) temperature
    T_cond: float = 3.0           # degC air outlet of the ARS-1 condensing coil (above 0 to avoid frost)
    dewpoint_dry: float = -40.0   # degC after silica + zeolite guard layer (below frost point at T_ads)
    dT_warm: float = 5.0          # K warm-end approach of the recuperator train (a)
    dT_cold: float = 3.0          # K cold-end approach of the recuperator train (a)
    dT_evap: float = 5.0          # K air - refrigerant approach in the ARS coils (a)
    eta_cap: float = 0.85         # CO2 capture fraction of the contactor (a)
    x_co2_ppm: float = 420.0
    dP_air: float = 1500.0        # Pa total air-side pressure drop (a)
    eta_fan: float = 0.65
    # ---- sorbents (isotherms, desorption mode, heat of adsorption in sorbents.py) ----
    sorbent: str = "13X"          # "13X" (CO2 sweep 473 K) or "CALF-20" (vacuum 0.08 bar, 293 K)
    eta_vac: float = 0.5          # vacuum pump isothermal efficiency
    cp_bed: float = 1.4           # kJ/(kg sorbent K) incl. contactor metal (a)
    cycles_per_h: float = 1.0
    h_ads_silica: float = 2800.0  # kJ/kg water
    silica_regen_factor: float = 1.25   # regen heat / adsorption heat (a)
    water_recovery_silica: float = 0.9  # fraction of silica water condensed from regen loop (a)
    # ---- absorption chillers ----------------------------------------------------
    cop_derate: float = 0.85      # realism factor on the ideal-pinch GAX COP map (a)
    fan_rej_frac: float = 0.02    # dry-cooler fan power / heat rejected
    # ---- solar field + storage -----------------------------------------------
    ptc_eta_opt: float = 0.75
    ptc_loss: float = 0.20e-3     # kW/(m2 K) at ~200 C mean HTF temperature (a)
    T_htf: float = 230.0          # degC mean HTF temperature (13X desorption at 200 C)
    storage_loss_h: float = 0.002 # fraction per hour
    # ---- economics (2030-ish, USD) ----------------------------------------------
    discount: float = 0.08
    lifetime: int = 25
    om_frac: float = 0.03         # of capex per year
    c_ptc: float = 250.0          # $/m2 aperture installed
    c_storage: float = 25.0       # $/kWh_th
    c_ars_warm: float = 700.0     # $/kW_cold (~ -2 C)
    c_ars_cold: float = 1400.0    # $/kW_cold (~ -38 C, two-stage/GAX)
    c_rej: float = 60.0           # $/kW heat rejected (dry coolers)
    c_hx_area: float = 40.0       # $/m2 air-air recuperator / coils
    U_air: float = 0.035          # kW/(m2 K) air-air
    c_fan: float = 150.0          # $/kW_e
    c_contactor: float = 25.0     # $/kg sorbent (contactor base cost, Kim et al.)
    c_sorbent: float = 2.7        # $/kg 13X
    c_silica_sys: float = 60.0    # $/kg silica installed (beds, valves)
    c_el: float = 30.0            # $/MWh electricity (legacy flat price; LCOE per cell is used now)
    c_backup_heat: float = 12.0   # $/GJ backup heat (natural gas)
    co2_backup: float = 0.056     # t CO2 / GJ backup heat (counted against capture)
    co2_price: float = 100.0      # $/t CO2 credit (used for LCOW)
    water_value: float = 0.0      # $/m3 credit for harvested water (used for LCOC)
    # ---- local renewable electricity + conversion (screening costs, 2030-ish) -----
    c_pv: float = 600.0           # $/kW_p single-axis PV
    c_wind: float = 1300.0        # $/kW onshore wind
    om_vre: float = 0.02          # of capex per year
    c_eboiler: float = 100.0      # $/kW_th electric boiler / heater
    c_hp: float = 600.0           # $/kW_th heat pump (low-grade heat)
    c_batt: float = 150.0         # $/kWh battery
    T_hp_sink: float = 363.0      # K heat-pump sink for low-grade heat (silica regen / vacuum desorption)
    firming: float = 1.3          # LCOE multiplier for firm electricity bought at cell LCOE
    c_backup_el: float = 250.0    # $/MWh diesel backup when VRE + storage fall short
    co2_backup_el: float = 0.7    # t CO2 / MWh diesel
    local_heat_price: float | None = None   # $/GJ geothermal / waste heat scenario (None = off)
    # ---- amine DAC benchmark (energy vs climate from Wenzel et al. 2026 data) -------
    amine_capex_tpy: float = 1000.0   # $ per (t CO2/yr) capacity incl. sorbent (a)
    amine_om: float = 0.05            # of capex per year incl. sorbent replacement (a)
    amine_capex_lo: float = 600.0     # optimistic amine case
    amine_om_lo: float = 0.03
    amine_capex_hi: float = 1500.0    # pessimistic case (Kim et al. reference degradation)
    amine_om_hi: float = 0.10
    amine_T_heat: float = 373.0       # K regeneration heat temperature
    # ---- product quality, drying option, exergy reference ---------------------------
    purity_min: float = 0.95          # CO2 product purity target
    drying: str = "silica"            # "silica" (condense + desiccant) or "freeze" (switching regenerators)
    freeze_loss: float = 0.2          # fraction of latent+fusion heat not recovered in "freeze" mode (a)
    T_htf_K: float = 473.0            # K delivery temperature of high-grade heat (exergy)
    free_cold: bool = False           # True: sub-ambient cold from LNG/LH2 regasification (no ARS)
    allow_chill: bool = True          # False: no ARS; adsorb only in hours when ambient <= T_ads
    energy_mode: str = "firm"         # "firm": clean firm energy at local price (same basis as amine);
                                      # "local": hourly dispatch of local PV/wind/PTC + fossil backup
    # ---- water supply benchmarks (distance to coast used as distance to source) ----
    lcow_desal: float = 1.0       # $/m3 seawater RO at the coast
    c_transport: float = 0.08     # $/m3 per 100 km, large trunk pipeline (city scale, energy+O&M share)
    pipe_capex_km_1000: float = 250e3   # $/km installed pipeline for 1000 m3/d (a)
    pipe_scale_exp: float = 0.5         # capex_km ~ Q^exp (a)
    pipe_pump_kwh_100km: float = 0.15   # kWh/m3 per 100 km (a)
    truck_base: float = 3.0             # $/m3 loading, source cost, delivery (a)
    truck_per_km: float = 0.12          # $/m3 per km one-way incl. empty return (a)
    demand_village: float = 100.0       # m3/d
    demand_town: float = 10000.0        # m3/d

    def with_(self, **kw) -> "DacParams":
        return replace(self, **kw)


def crf(p: DacParams) -> float:
    i, n = p.discount, p.lifetime
    return i * (1 + i) ** n / ((1 + i) ** n - 1)
