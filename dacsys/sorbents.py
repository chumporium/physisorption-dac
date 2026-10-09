"""Physisorbent models for sub-ambient / ambient DAC.

Isotherm: multi-site Langmuir with temperature-dependent affinity (valid over pressure),
    q(T, p) = sum_i qs_i * b_i p / (1 + b_i p),   b_i = exp(a_i + E_i / (R T))   [p bar, q mmol/g]

Real sorbents are fitted to Kim et al., Energy Environ. Sci. 18, 7427 (2025), Data S1:
  * Zeolite 13X: 40 Pa points (Fig. 3a), full isotherms at 293 and 473 K (Fig. 3d) and the
    saturation plateau at 195 K (p >= 5 mbar). Desorption: CO2 sweep at 473 K, 1 bar.
  * CALF-20: 40 Pa points (Fig. 3a; column labelled "MIL-120(Al)" in the file but carrying the
    CALF-20 values quoted in the paper), full isotherms at 283/293/303 K (Fig. 3e) and the 195 K
    plateau. Desorption: vacuum 0.02 bar at 333 K (the paper's 0.08 bar / ~295 K only works after
    adsorption near 195 K). CALF-20 keeps CO2 uptake below ~40 % RH (Lin et al., Science 2021).
Low-pressure volumetric points at 195-200 K are excluded (not equilibrium).

Parametric sorbents (research question on material targets) use one Langmuir site defined by
quantities a materials scientist reports: Henry constant at 298 K, isosteric heat, saturation
capacity, plus water tolerance, kinetics, N2 affinity and price.

Run `python -m dacsys.sorbents` to refit and print checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

XLSX = Path(__file__).resolve().parent.parent / "referensi" / "d5ee01473e1_suppl (1)" / "Data S1.xlsx"
FIT = Path(__file__).with_name("sorbent_fits.json")
R = 8.314e-3
M_CO2 = 44.01
P40 = 4.0e-4


N2_QS_DEFAULT = 4.16                  # mol/kg, median N2 Langmuir saturation of 1 319 PrISMa MOFs


@dataclass(frozen=True)
class Sorbent:
    name: str
    sites: tuple                      # ((qs, a, E), ...)  qs mmol/g, a ln(1/bar), E kJ/mol
    hoa: float                        # kJ/mol heat of adsorption used in energy balances
    T_des: float                      # K
    p_des: float                      # bar CO2 partial pressure at the end of desorption
    mode: str                         # "sweep" (CO2 sweep, no vacuum) or "vacuum"
    cost: float = 2.7                 # $/kg sorbent
    hydrophobic: bool = False         # True: no drying train (condensation/frost still counted)
    rh_max: float = 0.0               # tolerated RH at T_ads if not hydrophobic (0 -> dry to dewpoint_dry)
    retention: float = 1.0            # CO2 capacity kept in humid feed (hydrophobic sorbents)
    water_ratio: float = 0.0          # co-adsorbed water per CO2 cycled (t/t), desorbed each cycle
    cycles_per_h: float = 1.0         # kinetics at 298 K
    E_diff: float = 0.0               # kJ/mol diffusion barrier (slows cycling at low T)
    S_N2_298: float = 2000.0          # CO2/N2 adsorption selectivity at 298 K, DAC conditions (assumed)
    Q_N2: float = 17.0                # kJ/mol, N2 isosteric heat
    n2_qs: float | None = None        # mol/kg N2 saturation loading (Langmuir, TVSA/IAST model); None -> N2_QS_DEFAULT
    # water (two-zone bed model, PrISMa / Webley): single-site Langmuir H2O isotherm from the
    # saturation loading, the Henry constant at 298 K and the heat of adsorption, plus the
    # water resistance coefficient WRC = CO2 capacity in wet (ternary) / dry (binary) mixture.
    # None -> legacy model (hydrophobic / retention / water_ratio).
    h2o_qsat: float | None = None     # mol/kg
    h2o_KH298: float | None = None    # mol/kg/Pa
    h2o_dH: float | None = None       # kJ/mol (positive)
    wrc: float | None = None          # -
    h2o_rh_step: float | None = None  # S-shaped (type V) isotherm: pores fill at this RH -> qsat
    cp: float = 0.85                  # kJ/kg K solid heat capacity (PrISMa median 0.85)
    rho: float = 1.17                 # g/cm3 crystal density (PrISMa median 1.17)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def twozone(self) -> bool:
        return self.h2o_KH298 is not None

    def q_h2o(self, T, p_pa, rh=None):
        """Water loading [mol/kg] at T [K], water partial pressure [Pa] and relative humidity."""
        b = (self.h2o_KH298 / self.h2o_qsat) * np.exp(self.h2o_dH / R * (1.0 / np.asarray(T, float) - 1.0 / 298.15))
        bp = b * np.asarray(p_pa, float)
        q = self.h2o_qsat * bp / (1.0 + bp)
        if self.h2o_rh_step is not None and rh is not None:
            q = np.where(np.asarray(rh) >= self.h2o_rh_step, self.h2o_qsat, q)
        return q

    def q(self, T, p_bar):
        T = np.asarray(T, float)
        p = np.asarray(p_bar, float)
        out = 0.0
        for qs, a, E in self.sites:
            bp = np.exp(a + E / (R * T)) * p
            out = out + qs * bp / (1 + bp)
        return out

    def residual(self):
        return float(self.q(self.T_des, self.p_des))

    def working_capacity(self, T_ads, p_co2_bar):
        """kg CO2 / kg sorbent (dry), before humidity retention."""
        return np.maximum(self.q(T_ads, p_co2_bar) - self.residual(), 0.0) * M_CO2 * 1e-3

    def cycles(self, T):
        """Cycles per hour at adsorption temperature T (Arrhenius slow-down below 298 K)."""
        f = np.exp(-self.E_diff / R * (1.0 / np.asarray(T, float) - 1.0 / 298.15))
        return self.cycles_per_h * np.minimum(f, 1.0)

    def selectivity_n2(self, T):
        """CO2/N2 selectivity at T; rises on cooling with the heat difference (Henry regime)."""
        dQ = max(self.hoa - self.Q_N2, 0.0)
        return self.S_N2_298 * np.exp(dQ / R * (1.0 / np.asarray(T, float) - 1.0 / 298.15))


# Water classes from the PrISMa database (Charalambous et al., Nature 2024; Zenodo 11244258):
# medians over 1 338 simulated MOFs grouped by WRC (NGCC case, the lowest CO2 pressure they
# report). (qsat mol/kg, KH298 mol/kg/Pa, dH kJ/mol, WRC)
WATER_CLASSES = {
    "phil": (16.7, 17.3, 70.7, 0.002),      # WRC < 0.01, n = 382 (13X-like)
    "mid": (19.8, 7.92e-4, 44.1, 0.131),    # 0.05 <= WRC < 0.3, n = 275
    "phob": (21.8, 1.11e-5, 20.9, 0.909),   # WRC > 0.8, n = 241
}


def parametric(name: str, KH298: float, Qst: float, qs: float, *, T_des=473.0, p_des=1.0,
               mode="sweep", hydrophobic=False, retention=1.0, water_ratio=0.0, rh_max=0.0,
               cycles_per_h=1.0, E_diff=0.0, S_N2_298=2000.0, Q_N2=17.0, cost=10.0,
               water: str | tuple | None = None, n2_qs: float | None = None) -> Sorbent:
    """One-site Langmuir sorbent from KH(298 K) [mol/kg/Pa], Qst [kJ/mol], qs [mmol/g].
    water: a WATER_CLASSES key or (qsat, KH298, dH, WRC) -> two-zone water model."""
    b298 = KH298 * 1e5 / qs                                 # 1/bar
    a = np.log(b298) - Qst / (R * 298.15)
    w = WATER_CLASSES[water] if isinstance(water, str) else water
    wkw = dict(h2o_qsat=w[0], h2o_KH298=w[1], h2o_dH=w[2], wrc=w[3]) if w is not None else {}
    return Sorbent(name, ((qs, float(a), Qst),), hoa=Qst, T_des=T_des, p_des=p_des, mode=mode,
                   cost=cost, hydrophobic=hydrophobic, rh_max=rh_max, retention=retention,
                   water_ratio=water_ratio, cycles_per_h=cycles_per_h, E_diff=E_diff,
                   S_N2_298=S_N2_298, Q_N2=Q_N2, n2_qs=n2_qs, **wkw)


# ----------------------------------------------------------------------------- fitting
def _load_data():
    x = pd.ExcelFile(XLSX)
    a = pd.read_excel(x, "Fig3a", header=None)
    e = pd.read_excel(x, "Fig3d-e", header=None)

    def pts40(col):
        T = a.iloc[2:, col].astype(float).to_numpy()
        q = a.iloc[2:, col + 1].astype(float).to_numpy()
        return [(t, P40, v, 3.0) for t, v in zip(T, q) if np.isfinite(t) and np.isfinite(v)]

    def iso(col, T, pmin=0.0, pmax=2.5, w=1.0):
        P = e.iloc[3:, col].astype(float).to_numpy()
        q = e.iloc[3:, col + 1].astype(float).to_numpy()
        ok = np.isfinite(P) & np.isfinite(q) & (P > pmin) & (P <= pmax) & (q > 0)
        return [(T, p, v, w) for p, v in zip(P[ok], q[ok])]

    return {
        "13X": pts40(4) + iso(2, 293.0) + iso(4, 473.0, w=2.0) + iso(0, 195.0, pmin=5e-3),
        "CALF-20": pts40(6) + iso(8, 283.0) + iso(10, 293.0) + iso(12, 303.0) + iso(6, 195.0, pmin=1e-3),
    }


def fit_all():
    data = _load_data()
    out = {}
    for name, pts in data.items():
        T, p, q, w = map(np.array, zip(*pts))

        def model(v):
            s = 0.0
            for i in range(2):
                qs, a, E = v[3 * i: 3 * i + 3]
                bp = np.exp(a + E / (R * T)) * p
                s = s + qs * bp / (1 + bp)
            return s

        # low-pressure points carry the DAC regime: weight them more; keep heats physisorptive
        # the 40 Pa points (DAC operating line) get x4 on top: base-case fit. Unweighted and
        # operating-line-only fits bracket the 13X uncertainty at 235-265 K (+-25-35 %).
        w = w * np.where(p < 0.01, 2.0, 1.0) * np.where(np.isclose(p, P40), 4.0, 1.0)
        res = least_squares(lambda v: w * (np.log(model(v) + 0.02) - np.log(q + 0.02)),
                            [2.0, -18.0, 45.0, 5.0, -12.0, 25.0],
                            bounds=([0.1, -60, 10, 0.1, -60, 5], [10, 20, 55, 12, 20, 55]))
        v = res.x
        rmse = float(np.sqrt(np.mean((model(v) - q) ** 2)))
        out[name] = dict(sites=[list(v[0:3]), list(v[3:6])], n=len(q), rmse=rmse)
    FIT.write_text(json.dumps(out, indent=1))
    return out, data


_CACHE: dict = {}


def load() -> dict[str, Sorbent]:
    if "all" in _CACHE:
        return _CACHE["all"]
    f = json.loads(FIT.read_text())
    # Water and N2 data: PrISMa experimental set (Zenodo 11244258, Material_Properties/
    # Water_NGCC-onshore-Experimental.csv and Isotherm-Experimental.csv). N2/CO2 Henry ratios
    # are the measured values, moved to 298 K with the heat difference where needed.
    _CACHE["all"] = {
        "13X": Sorbent("13X", tuple(tuple(s) for s in f["13X"]["sites"]), hoa=40.0,
                       T_des=473.0, p_des=1.0, mode="sweep", cost=0.85,    # Kim et al. 2025
                       S_N2_298=557.0, Q_N2=19.5, n2_qs=3.33,  # 641 at 293.15 K; n2_qs fit to PrISMa exp. N2
                       h2o_qsat=17.6, h2o_KH298=0.865502, h2o_dH=60.46, wrc=0.000281,
                       cp=0.88, rho=1.483),                    # PrISMa Zeo++-Experimental.csv
        "CALF-20": Sorbent("CALF-20", tuple(tuple(s) for s in f["CALF-20"]["sites"]), hoa=39.0,
                           T_des=333.0, p_des=0.02, mode="vacuum", cost=20.0, rh_max=0.4,
                           S_N2_298=229.0, Q_N2=15.1, n2_qs=3.05,  # 6.375e-4 / 2.78e-6; n2_qs PrISMa GCMC
                           h2o_qsat=10.5, h2o_KH298=2.25167e-4, h2o_dH=34.73, wrc=0.0800,
                           h2o_rh_step=0.4,                   # pore filling ~40 % RH (Lin et al. 2021)
                           cp=0.721, rho=1.763),
        "MIP-212": _mip212(),
    }
    return _CACHE["all"]


def _mip212() -> Sorbent:
    """MIP-212 (CuAlPyC, PrISMa experimental): one-site Langmuir through the measured CO2
    isotherm at 298 K with the measured Henry constant fixed and the measured constant heat."""
    KH, Q = 1.73984e-4, 31.5
    csv = Path(__file__).resolve().parent.parent / "data" / "prisma" / "Material_Properties" / "Isotherm-Experimental.csv"
    qs = 4.4                                                     # fallback if the data file is absent
    if csv.exists():
        import ast
        e = pd.read_csv(csv)
        r = e[(e.MOF == "CuAlPyC_exp") & (e.Molecule == "CO2")].iloc[0]
        p = np.array(ast.literal_eval(r["Pressure [bar]"])) * 1e5
        q = np.array(ast.literal_eval(r["Uptake [mol/kg]"]))
        fit = least_squares(lambda v: v[0] * KH * p / (v[0] + KH * p) - q, [4.0], bounds=([0.5], [20.0]))
        qs = float(fit.x[0])
    s = parametric("MIP-212", KH, Q, qs, S_N2_298=KH / 3.95e-6, Q_N2=16.6, n2_qs=2.70, cost=20.0,
                   water=(18.37, 5.05066e-4, 38.51, 0.5303))
    return replace(s, cp=0.842, rho=1.43211, meta={"qs_fit": qs})


def get(name: str) -> Sorbent:
    """'13X', 'CALF-20', optional suffix '_H' (hydrophobic, no drying), or a registered
    parametric sorbent (see PARAMETRIC)."""
    hydro = name.endswith("_H")
    base = name[:-2] if hydro else name
    if base in PARAMETRIC:
        s = PARAMETRIC[base]
    else:
        s = load()[base]
    return replace(s, name=name, hydrophobic=hydro or s.hydrophobic)


PARAMETRIC: dict[str, Sorbent] = {}


def register(s: Sorbent) -> Sorbent:
    PARAMETRIC[s.name] = s
    return s


if __name__ == "__main__":
    fits, data = fit_all()
    _CACHE.clear()
    S = load()
    for name, s in S.items():
        f = fits[name]
        print(f"\n{name}: {f['n']} points, RMSE {f['rmse']:.2f} mmol/g, sites " +
              "; ".join(f"qs={q:.2f} E={E:.1f}" for q, a, E in s.sites))
        for T, p, qd, _ in [pt for pt in data[name] if pt[1] == P40]:
            print(f"   40 Pa {T:.0f} K: data {qd:.2f}  fit {float(s.q(T, p)):.2f}")
        for T, p in ((293.0, 0.1), (293.0, 1.0), (473.0, 1.0), (363.0, 0.05), (333.0, 0.02)):
            print(f"   q({T:.0f} K, {p} bar) = {float(s.q(T, p)):.2f}")
        print("   q(40 Pa):", {T: round(float(s.q(T, P40)), 2) for T in (225, 235, 245, 255, 265, 288)})
        print(f"   residual at desorption ({s.mode} {s.T_des:.0f} K, {s.p_des} bar) = {s.residual():.2f}")
        print("   working capacity (mmol/g):",
              {T: round(float(s.working_capacity(T, P40)) / M_CO2 * 1e3, 2) for T in (225, 235, 245, 255)})
