"""Reference plant B0, the levers of each research question, and all plant parameters.

This is the only place where the configuration is set (spec: SPESIFIKASI_SISTEM.md).
A Case is B0 with at most the lever of one research question changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

# ------------------------------------------------------------------ levers
DRYING = ("none", "cond", "cond_silica", "freeze")        # RQ1
COLD = ("ambient", "gax", "vcr", "lng")                   # RQ2
REGEN = {"vac100": (373.15, 0.01, "vacuum"),              # B0: vacuum 0.01 bar at 100 C
         "sweep200": (473.15, 1.0, "sweep")}              # sensitivity: CO2 sweep 1 bar at 200 C

# design adsorption temperatures tried for each cold source [K]; the plant adsorbs at
# min(design, ambient - 1 K), so cold hours need no chilling. GAX is limited to an evaporator
# at >= -38 C (Yamin et al. decision range) -> air at >= ~240 K with a 5 K approach.
T_ADS = {"ambient": (None,),
         "gax": (241.0, 245.0, 250.0, 255.0, 260.0, 270.0, 280.0),
         "vcr": (225.0, 230.0, 235.0, 241.0, 245.0, 250.0, 255.0, 260.0, 270.0, 280.0),
         "lng": (195.0, 205.0, 215.0, 225.0, 235.0, 245.0, 255.0)}

# drying x cold combinations that do not exist physically (no refrigeration -> no condensing
# coil, no freeze-out); reported as "n/a", never computed
NOT_APPLICABLE = {("ambient", "cond"), ("ambient", "freeze")}


@dataclass(frozen=True)
class Prm:
    """Plant parameters. (a) = assumption, varied in the Monte Carlo."""
    # ---- air side
    x_co2_ppm: float = 420.0
    eta_cap: float = 0.85             # CO2 capture fraction in the contactor (a)
    T_coil1: float = 3.0              # C air outlet of the condensing coil (above 0 C: no frost)
    dewpoint_dry: float = -40.0       # C after silica / guard bed
    dT_warm: float = 5.0              # K warm-end approach, recuperator (a)
    dT_cold: float = 3.0              # K cold-end approach, recuperator (a)
    dT_evap: float = 5.0              # K air - refrigerant approach in coils (a)
    U_air: float = 0.035              # kW/m2K air-air recuperator (Kim et al. ESI Table S5)
    U_coil: float = 0.05              # kW/m2K air coils and LNG-air exchangers (Kim et al. ESI Table S6: 44.5-95.8 W/m2K)
    reg_area_factor: float = 1.3      # area of the reversing regenerators relative to a recuperator of the same duty
    sil_dq: float = 0.10              # kg water / kg silica gel cycled
    sil_cycles: float = 2.0           # silica-gel cycles per hour
    # pressure drops [Pa]. Contactor: identical for physisorption and amine; 300 Pa gives an
    # amine fan energy (~0.8 GJ/t) consistent with the electricity in the Wenzel data (a)
    dP_contactor: float = 300.0
    dP_coil: float = 100.0            # per coil (a)
    dP_silica: float = 250.0          # (a)
    dP_recup: float = 150.0           # per pass, two passes (supply + exhaust) (a)
    dP_regen: float = 300.0           # reversing regenerator, per pass, two passes (a)
    eta_fan: float = 0.65
    # ---- sorbent handling
    regen: str = "vac100"
    # desorption pressures [bar] tried by the optimiser with regen = "vac100" (the vacuum pump cost makes deep vacuum
    # expensive); p_des fixes one pressure instead (used internally for the chosen design)
    p_des_opt: tuple = (0.01, 0.03, 0.1)
    p_des: float | None = None
    eta_vac: float = 0.5              # vacuum pump isothermal efficiency (a)
    purity_model: str = "tvsa"        # "tvsa": 5-step TVSA + IAST (tvsa.py); "henry": legacy Henry selectivity + rinse
    purity_min: float = 0.95          # CO2 product purity required (dry basis)
    water_front_factor: float = 1.0   # wet-zone length / ideal shock-front length (PrISMa SI 9: up to ~3) (a)
    wrc_pore_scaling: bool = True     # scale the WRC penalty with pore filling (our assumption); False = PrISMa WRC as is
    cp_contactor: float = 0.3         # kJ/kg K contactor structure per kg sorbent, added to the sorbent's cp (a)
    # heat-transfer-limited cycle time (PrISMa SI eq. S8; U and tube size from their Table S6)
    U_hx: float = 0.04                # kW/m2K heating/cooling medium to bed
    a_hx: float = 160.0               # m2/m3 bed (tubes 25 mm inside diameter: 4/d)
    eps_pellet: float = 0.35
    eps_bed: float = 0.37
    dT_hex_heat: float = 20.0         # K heating medium above T_des
    dT_hex_cool: float = 5.0          # K cooling medium below T_ads
    max_cycles_per_h: float = 4.0     # cap: adsorption front / valve timing (a)
    h_ads_silica: float = 2800.0      # kJ/kg water
    silica_regen_factor: float = 1.25 # (a)
    silica_regen: str = "ptc"         # "ptc": regeneration heat from the solar field; "hp": electric heat pump
    silica_hp_cop: float = 3.0        # heating COP of the regeneration heat pump (silica_regen = "hp") (a)
    # industrial heat pumps: 400 EUR per kW heat output, installed (industry average, Marina et al. 2021), in USD 2023:
    # x 1.183 USD/EUR (2021) x CEPCI 797.9/708.8 = 533 $/kW heat; used for the chiller heat pumps (per kW of heat
    # rejected, i.e. cold + compressor work) and for the silica-gel regeneration heat pump
    c_hp_heat: float = 400.0 * 1.183 * 797.9 / 708.8
    water_recovery_silica: float = 0.9
    freeze_loss: float = 0.2          # share of sublimation heat not recovered by regenerators (a)
    cycle_factor: float = 1.0         # multiplies sorbent cycles per hour (kinetics proxy) (a)
    sorbent_cost_factor: float = 1.0  # multiplies the sorbent price (a)
    # ---- cold machines
    gax_derate: float = 0.85          # realism factor on the ideal-pinch GAX COP map (a)
    gax_T_gen_max: float = 220.0      # C; 160 = strictly inside the validated range (sensitivity)
    gax_pump_frac: float = 0.005      # solution pump power / generator heat
    fan_rej_frac: float = 0.02        # dry-cooler fan power / heat rejected
    # ---- solar field, storage, PV
    ptc_eta_opt: float = 0.75
    ptc_loss: float = 0.20e-3         # kW/m2K at the oil temperature
    T_htf: float = 230.0              # C oil (as in Yamin et al.)
    storage_loss_h: float = 0.01 / 24.0   # about 1 %/day (well-insulated two-tank storage: ~1 K/day, Prieto et al. 2016)
    eta_batt: float = 0.85            # round-trip efficiency, applied on charging (Cole & Frazier 2019)
    # "offgrid": PTC + tank + PV + battery, runs only when stored energy allows (B0).
    # "grid": always-on grid electricity + gas heat at fixed prices (Kim et al. basis, V3 only)
    energy_mode: str = "offgrid"
    c_elec_grid: float = 30.0         # $/MWh (Kim et al. Table S10)
    c_heat_grid: float = 3.5          # $/GJ natural gas (Kim et al. Table S10)
    eta_thrm: float = 0.85            # heat delivered per fuel energy (Kim et al.)
    # ---- economics (USD 2024)
    discount: float = 0.08
    lifetime: int = 25
    om_proc: float = 0.03
    om_energy: float = 0.02
    c_ptc: float = 250.0              # $/m2 aperture (a)
    c_tank: float = 25.0              # $/kWh_th (a)
    c_pv: float = 600.0               # $/kWp single-axis (a)
    c_batt: float = 150.0             # $/kWh (a)
    # absorption chillers: installed 1 800 $/ton for a 1 320-ton single-stage unit, COP 0.79 (US DOE 2017, Table 3), i.e.
    # 512 $/kW cold = 226 $ per kW of heat rejected (1 + 1/COP), x CEPCI 797.9/567.5 = 317 $/kW rejected; applied per
    # kW of heat rejected by the GAX chiller, so that the cost per kW of cold rises as the COP falls at low temperature
    c_abs_rej: float = 1800.0 / 3.517 / (1.0 + 1.0 / 0.79) * 797.9 / 567.5
    # dry coolers: 49-107 EUR/kW equipment (IEA SHC Task 48, 2014), mid 78 EUR/kW x 1.329 USD/EUR x CEPCI 797.9/576.1
    # = 144 $/kW, x installation 1.20 recommended for dry cooling (8 % materials + 12 % labour; EPRI and NETL, in
    # Weiland et al. 2019) = 173 $/kW rejected
    c_rej: float = 78.0 * 1.329 * 797.9 / 576.1 * 1.20
    # recuperator / coils / LNG exchangers (air side, finned). "netl" (base): vendor-quote correlation of large direct dry
    # air coolers, C = 32.88 UA^0.75 (2017 $, UA 8.6e5-7.5e7 W/K, -25/+28 %; Weiland et al. 2019, Eq. 13), applied per
    # unit of netl_ua_unit (W/K) and installed x 1.20; the cost is per UA, so it does not depend on the heat transfer
    # coefficient. Exchangers in contact with LNG, and recuperators below 227 K, are in aluminium or stainless steel:
    # x f_mat_cryo. "unit": purchase-cost correlation used by Yamin et al. (2024), Z = 130 (A/0.093)^0.78
    # CEPCI/CEPCI_2005 per unit of hx_unit_area x installation factor hx_install (sensitivity). c_hx_area: a fixed
    # installed cost per m2 overrides both (sensitivity runs, Kim et al. basis, packed-bed regenerators).
    c_hx_area: float | None = None    # $/m2
    netl_ua_unit: float = (8.6e5 * 7.5e7) ** 0.5   # W/K per unit: geometric mean of the vendor-quote range
    netl_install: float = 1.20        # installed / equipment cost, dry coolers (Weiland et al. 2019): +8 % materials,
    netl_labour: float = 0.12         # +12 % labour
    f_hx: float = 1.0                 # multiplier on the NETL correlation (Monte Carlo: 0.75-1.28)
    # material factor for cryogenic service (LNG exchangers, recuperators below 227 K): 304/316 stainless steel, f_m = 1.3
    # relative to carbon steel (Towler & Sinnott 2022, Table 7.6; aluminium 1.07); applied, as in their Eq. 7.12, to the
    # equipment and materials but not to the labour
    f_mat_cryo: float = 1.3
    cepci_2017: float = 567.5         # CEPCI 2017 (basis of Weiland et al. 2019)
    hx_unit_area: float = 10000.0     # m2 per heat-exchanger unit ("unit" model)
    cepci_ratio: float = 797.9 / 468.2  # CEPCI 2023 (last published annual value) / CEPCI 2005
    hx_install: float = 3.5           # installed / purchased cost, heat exchangers (Hand 1958), "unit" model
    # fans and vacuum pumps: total plant cost per kW of NETL DAC case 1-NG (Rev. 1, May 2023 USD): air-handling
    # fans 70.863 M$ for 18 589 kW; vacuum compressor 15.437 M$ for 392 kW (0.04 MPa suction); NETL scales both
    # with the power, here linearly (replicated units)
    c_fan: float = 70863.0 / 18.589   # $/kW_e (3 812)
    c_vac: float = 15437.0 / 0.392    # $/kW_e (39 380)
    c_contactor: float = 25.0         # $/kg sorbent, contactor structure (Kim et al.)
    # fabrication factors of Kim et al. (ESI Table S10): installed adsorbent = 3 x price, contactor = 3 x base cost
    sorbent_fab: float = 3.0
    contactor_fab: float = 3.0
    # cost functions of Kim et al. (ESI Eqs. 30-36), used only to reproduce their result (kim_reconcile): heat
    # exchangers 2 A^0.7 x 100 $ and fans (F/1.77)^0.8 x 2000 $ for the whole plant, vacuum pump
    # (423.9 F^0.653 + 30000) x 1.07 $ (F in m3/h at atmospheric pressure), all x f_TCI 3.06, plant of 8760 t/yr
    hx_cost_model: str = "netl"       # "netl": per UA (base); "unit": hx_cost_m2 per m2; "kim": Kim et al.
    fanvac_cost_model: str = "netl"   # "netl": c_fan, c_vac per kW; "kim": Kim et al.
    kim_scale_tpy: float = 8760.0
    kim_f_tci: float = 3.06
    c_silica_sys: float = 60.0        # $/kg silica installed
    # ---- amine benchmark (same energy block, same site)
    amine_capex: tuple = (600.0, 1000.0, 1500.0)   # $ per (t/yr) at 90 % availability: lo/base/hi
    amine_om: tuple = (0.03, 0.05, 0.10)
    amine_energy_factor: tuple = (0.75, 1.0, 1.25)   # ~6 (McQueen et al.) .. 9.3 GJ/t heat (Deutz & Bardow)
    amine_T_heat: float = 373.15

    @property
    def crf(self) -> float:
        i, n = self.discount, self.lifetime
        return i * (1 + i) ** n / ((1 + i) ** n - 1)

    @property
    def hx_cost_ua(self) -> float:
        """Installed $ per W/K of the NETL correlation (Weiland et al. 2019) at the unit size netl_ua_unit, 2023 $."""
        return (self.f_hx * self.netl_install * 32.88 * self.netl_ua_unit ** (0.75 - 1.0)
                * 797.9 / self.cepci_2017)

    @property
    def f_cryo_installed(self) -> float:
        """Installed-cost factor of cryogenic service relative to carbon steel (material factor on all but labour)."""
        return ((self.netl_install - self.netl_labour) * self.f_mat_cryo + self.netl_labour) / self.netl_install

    @property
    def hx_cost_m2(self) -> float:
        """Installed $ per m2 of heat-exchanger area: fixed value; the purchase-cost correlation of Yamin et al.
        (2024) per unit times the installation factor ("unit"); or, for "netl", the equivalent cost per m2 of a coil
        (U_coil) in carbon steel."""
        if self.c_hx_area is not None:
            return self.c_hx_area
        if self.hx_cost_model == "netl":
            return self.hx_cost_ua * self.U_coil * 1e3
        a = self.hx_unit_area
        return 130.0 * (a / 0.093) ** 0.78 * self.cepci_ratio / a * self.hx_install


@dataclass(frozen=True)
class Case:
    site: str
    drying: str = "cond_silica"
    cold: str = "gax"
    sorbent: str = "13X"
    water_value: float = 0.0          # $/m3 credited to LCOC (RQ4 reports the break-even instead)
    lever: str = "B0"                 # which research question this case belongs to
    prm: Prm = field(default_factory=Prm)

    @property
    def applicable(self) -> bool:
        return (self.cold, self.drying) not in NOT_APPLICABLE

    def key(self) -> str:
        return f"{self.lever}|{self.site}|{self.drying}|{self.cold}|{self.sorbent}|{self.prm.regen}"


def b0(site: str) -> Case:
    return Case(site=site)


def with_(case: Case, **kw) -> Case:
    prm_kw = {k: kw.pop(k) for k in list(kw) if k in Prm.__dataclass_fields__}
    c = replace(case, **kw)
    return replace(c, prm=replace(c.prm, **prm_kw)) if prm_kw else c
