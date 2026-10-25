"""Input parameters (paper Table 1, 4, 8) plus modelling assumptions.

Values marked "assumed" are NOT given in the paper and were chosen so the
base case lands in the same range as the paper's Table 6/7.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Params:
    # ---- ambient / dead state ------------------------------------------------
    T0: float = 298.15            # K
    P0: float = 101.325           # kPa

    # ---- decision variables (Table 4 ranges) ---------------------------------
    Gb: float = 0.8               # kW/m2, solar direct beam irradiation
    T_gen: float = 423.2          # K, generator (weak-solution outlet) temperature (Table 6 base)
    T_eva: float = 243.9          # K, evaporator outlet temperature
    P_high: float = 5000.0        # kPa, Claude-cycle compressor outlet pressure

    # ---- solar collector (PTC) + storage tank --------------------------------
    A_col: float = 2800.0         # m2 aperture (assumed; Ex_solar in Table 6 -> ~2.8e3 m2)
    theta: float = 23.0           # deg incident angle (Table 1)
    T_sun: float = 4500.0         # K (Table 1)
    eta_opt: float = 0.75         # peak optical efficiency (assumed, LS-2 class)
    U_L: float = 4.0              # W/m2K receiver loss coefficient (assumed)
    conc_ratio: float = 25.0      # concentration ratio (assumed)
    U_T: float = 0.5              # W/m2K storage-tank loss coefficient (Table 1)
    V_ST: float = 30.0            # m3 storage tank volume (assumed)
    dT_col: float = 7.0           # K oil temperature rise in collector (Table 7)
    dT_gen_approach: float = 71.5 # K oil return temp above T_gen (Table 7: 496.2-424.7)

    # ---- branched GAX cycle ---------------------------------------------------
    T_cond: float = 313.0         # K condenser outlet (Table 1)
    eps_RHE1: float = 0.8         # (Table 1)
    x_rect: float = 0.995         # rectifier NH3 mass fraction (Table 1)
    q_eva_out: float = 0.94       # evaporator outlet vapour quality (Table 1)
    eta_pump: float = 0.50        # (Table 1)
    T_abs: float = 309.0          # K absorber outlet (Table 1)
    dT_gax_pinch: float = 5.0     # K pinch in GAX heat exchange (assumed)
    TTD_eva: float = 3.0          # K evaporator terminal temperature difference

    # ---- Claude hydrogen liquefaction cycle ----------------------------------
    T5: float = 160.15            # K outlet of RHE2 (Table 1)
    eta_com: float = 0.80         # (Table 1)
    eta_tur: float = 0.80         # (Table 1)
    x_tur: float = 0.60           # fraction of flow to the turbine (Table 7: 3.029/5.049)
    strict_pinch: bool = False    # True -> lower x_tur until RHE2-4 have no temperature cross

    # ---- economics (Table 1 + assumed prices) --------------------------------
    n_years: int = 20
    tau: float = 7000.0           # h/yr
    i_rate: float = 0.15
    phi_m: float = 1.06
    c_feed_H2: float = 9.899      # $/GJ of feed-H2 exergy (Table 7, state 1)
    c_el: float = 0.03            # $/kWh electricity (assumed)
    price_LH2: float = 1.70       # $/kg selling price for NPV (assumed)
    f_TCI: float = 4.16           # total capital investment / purchased equipment (assumed)
    # overall heat-transfer coefficients, kW/m2K (assumed)
    U_gax: float = 1.0
    U_rhe: float = 0.3

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


# NSGA-II settings and decision-variable bounds (Table 4)
NSGA = dict(pop_size=200, n_gen=200, p_crossover=0.8, p_mutation=0.01)
BOUNDS = {
    "Gb": (0.2, 1.2),
    "T_gen": (423.0, 438.0),
    "T_eva": (235.0, 244.0),
    "P_high": (3000.0, 5000.0),
}

# Table 8: monthly ambient temperature (degC) and direct beam irradiation (kW/m2)
MONTHLY = [
    ("Jan", -3.3, 0.299), ("Feb", -0.7, 0.379), ("Mar", 6.3, 0.601),
    ("Apr", 14.9, 0.656), ("May", 21.0, 0.929), ("Jun", 25.2, 0.664),
    ("Jul", 27.0, 0.406), ("Aug", 25.7, 0.431), ("Sep", 20.9, 0.552),
    ("Oct", 13.7, 0.329), ("Nov", 5.5, 0.266), ("Dec", -1.1, 0.222),
]

# Chemical Engineering Plant Cost Index
CEPCI = {1994: 368.1, 2000: 394.1, 2005: 468.2, 2021: 708.0, 2022: 816.0}
