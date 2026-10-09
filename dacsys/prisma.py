"""Real materials from the PrISMa database (Charalambous et al., Nature 632, 89, 2024;
Zenodo 11244258, Material_Properties) turned into Sorbent objects for the plant model.

For every MOF:
  CO2   dual-site Langmuir fitted at T_ref to the GCMC isotherm (p <= 10 bar) with the Widom
        Henry constant enforced as the p -> 0 slope; the stronger site carries the heat of
        adsorption at the lowest loading, the weaker site the heat at the highest fitted loading
        (Clausius-Clapeyron with a loading-dependent heat, as in Moubarak et al.).
  N2    CO2/N2 Henry selectivity and the average N2 heat.
  H2O   saturation loading, Henry constant, Henry heat and WRC (NGCC case: lowest CO2 pressure).
  solid heat capacity and crystal density (Zeo++ file); price 20 $/kg (assumption).

python -m dacsys.prisma   -> data/prisma/prisma_fits.json (+ fit quality printed)
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import sorbents

DIR = Path(__file__).resolve().parent.parent / "data" / "prisma" / "Material_Properties"
FITS = DIR.parent / "prisma_fits.json"
R = 8.314e-3
PRICE = 20.0                       # $/kg, all PrISMa MOFs (assumption)
PREFIX = "PR:"


def _lst(x):
    return np.array(ast.literal_eval(x), float) if isinstance(x, str) and x.startswith("[") else np.array([float(x)])


def fit_one(p_bar, q, heat, KH, T_ref):
    """Dual-site Langmuir at T_ref honouring the Henry slope KH [mol/kg/Pa]."""
    ok = (p_bar > 0) & (q > 0) & (p_bar <= 10.0)
    p_bar, q = p_bar[ok], q[ok]
    h = np.abs(heat[ok]) if len(heat) == len(ok) else np.full(ok.sum(), abs(heat[0]))
    KHb = KH * 1e5                                                     # mol/kg/bar
    if len(q) < 3:
        return None
    qmax = q.max()

    def model(v, p):
        qs1, qs2, lb1 = np.exp(v)
        b1 = lb1
        b2 = max((KHb - qs1 * b1) / qs2, 1e-12)
        return qs1 * b1 * p / (1 + b1 * p) + qs2 * b2 * p / (1 + b2 * p), b2

    def res(v):
        m, _ = model(v, p_bar)
        r = np.log(m + 1e-4) - np.log(q + 1e-4)
        p0 = 1e-9
        slope = model(v, np.array([p0]))[0][0] / p0
        return np.concatenate([r, [3.0 * (np.log(slope) - np.log(KHb))]])

    best = None
    for qs1_0, frac in ((0.3 * qmax, 0.9), (0.1 * qmax, 0.99), (0.6 * qmax, 0.5)):
        b1_0 = frac * KHb / max(qs1_0, 1e-3)
        v0 = np.log([max(qs1_0, 1e-3), max(1.2 * qmax, 1e-2), max(b1_0, 1e-6)])
        try:
            s = least_squares(res, v0, bounds=(np.log([1e-4, 1e-3, 1e-6]), np.log([50, 60, 1e14])))
        except ValueError:
            continue
        if best is None or s.cost < best.cost:
            best = s
    if best is None:
        return None
    qs1, qs2, b1 = np.exp(best.x)
    _, b2 = model(best.x, p_bar)
    # strong site = larger b, gets the low-loading heat
    E_lo, E_hi = float(h[0]), float(h[-1])
    sites = sorted([(qs1, b1), (qs2, b2)], key=lambda s: -s[1])
    out = []
    for (qs, b), E in zip(sites, (E_lo, E_hi)):
        out.append((float(qs), float(np.log(b) - E / (R * T_ref)), E))
    m, _ = model(best.x, p_bar)
    rmse_log = float(np.sqrt(np.mean((np.log(m + 1e-4) - np.log(q + 1e-4)) ** 2)))
    return dict(sites=out, rmse_log=rmse_log, n=int(len(q)), p_min=float(p_bar.min()))


def build():
    iso = pd.read_csv(DIR / "Isotherm-Simulated.csv")
    kpi = pd.read_csv(DIR / "Material_KPIs-Simulated.csv").set_index("MOF")
    wat = pd.read_csv(DIR / "Water_NGCC-onshore-Simulated.csv").set_index("MOF")
    zeo = pd.read_csv(DIR / "Zeo++-Simulated.csv").set_index("MOF")
    co2 = iso[iso.Molecule == "CO2"].set_index("MOF")
    out = {}
    for mof, r in co2.iterrows():
        if mof not in kpi.index or mof not in wat.index or mof not in zeo.index:
            continue
        f = fit_one(_lst(r["Pressure [bar]"]), _lst(r["Uptake [mol/kg]"]), _lst(r["Heat [kJ/mol]"]),
                    float(r["Henry [mol/kg/Pa]"]), float(r["T_ref [K]"]))
        if f is None:
            continue
        k, w, z = kpi.loc[mof], wat.loc[mof], zeo.loc[mof]
        vals = [w["Uptake_sat [mol/kg]"], w["Henry [mol/kg/Pa]"], w["Heat_henry [kJ/mol]"], w["WRC [-]"],
                z["Cp [J/g.K]"], z["Density [g/cm^3]"], k["Henry_Sel [-]"], k["Qavg_N2 [kJ/mol]"], k["Qavg_CO2 [kJ/mol]"]]
        if not np.all(np.isfinite(np.array(vals, float))):
            continue
        out[mof] = dict(f, KH_CO2=float(r["Henry [mol/kg/Pa]"]), T_ref=float(r["T_ref [K]"]),
                        hoa=float(-k["Qavg_CO2 [kJ/mol]"]), S_N2=float(k["Henry_Sel [-]"]),
                        Q_N2=float(-k["Qavg_N2 [kJ/mol]"]), h2o=(float(w["Uptake_sat [mol/kg]"]),
                        float(w["Henry [mol/kg/Pa]"]), float(-w["Heat_henry [kJ/mol]"]), float(w["WRC [-]"])),
                        cp=float(z["Cp [J/g.K]"]), rho=float(z["Density [g/cm^3]"]))
    FITS.write_text(json.dumps(out))
    return out


N2_FITS = DIR.parent / "prisma_n2.json"


def fit_n2_qs(p_bar, q, KH, T_ref=None) -> float | None:
    """N2 saturation loading [mol/kg] of a one-site Langmuir through the N2 isotherm with the
    Henry constant KH [mol/kg/Pa] fixed (so the CO2/N2 Henry selectivity is unchanged)."""
    ok = (p_bar > 0) & (q > 0) & (p_bar <= 10.0)
    p_bar, q = p_bar[ok], q[ok]
    if len(q) < 2:
        return None
    KHb = KH * 1e5

    def res(v):
        qs = np.exp(v[0])
        return np.log(qs * (KHb / qs) * p_bar / (1 + (KHb / qs) * p_bar) + 1e-6) - np.log(q + 1e-6)
    s = least_squares(res, [np.log(max(2 * q.max(), 1e-3))], bounds=([np.log(1e-3)], [np.log(100.0)]))
    return float(np.exp(s.x[0]))


def build_n2() -> dict:
    iso = pd.read_csv(DIR / "Isotherm-Simulated.csv")
    n2 = iso[iso.Molecule == "N2"]
    out = {}
    for r in n2.itertuples(index=False):
        qs = fit_n2_qs(_lst(r[3]), _lst(r[4]), float(r[2]))
        if qs is not None:
            out[r[0]] = qs
    N2_FITS.write_text(json.dumps(out))
    return out


def load_n2() -> dict:
    return json.loads(N2_FITS.read_text()) if N2_FITS.exists() else build_n2()


def load_fits() -> dict:
    return json.loads(FITS.read_text()) if FITS.exists() else build()


def register(mofs=None) -> list[str]:
    """Register PrISMa MOFs as sorbents named 'PR:<MOF>'. Returns the names."""
    fits = load_fits()
    n2 = load_n2()
    names = []
    for mof in (mofs or fits):
        d = fits[mof]
        s = sorbents.Sorbent(PREFIX + mof, tuple(tuple(x) for x in d["sites"]), hoa=d["hoa"], T_des=373.15,
                             p_des=0.01, mode="vacuum", cost=PRICE, S_N2_298=d["S_N2"], Q_N2=d["Q_N2"],
                             h2o_qsat=d["h2o"][0], h2o_KH298=d["h2o"][1], h2o_dH=d["h2o"][2], wrc=d["h2o"][3],
                             cp=d["cp"], rho=d["rho"], n2_qs=n2.get(mof), meta={"KH_CO2": d["KH_CO2"]})
        sorbents.register(s)
        names.append(s.name)
    return names


def origin(name: str) -> str:
    """Structure origin from the PrISMa naming (SI section 3.1.1): in-silico structures of Boyd et al.
    (str_m...), Park et al./Majumdar et al. (...+N..._charge, ddmof_...) are 'hipotetis'; CoRE-MOF 2019
    structures from the CSD (refcodes, RSM####, common names) are 'eksperimental'."""
    m = name[len(PREFIX):] if name.startswith(PREFIX) else name
    return "hipotetis" if (m.startswith(("str_m", "ddmof_")) or "+" in m) else "eksperimental"


def fit_ok(name: str, fits=None) -> bool:
    """Fit quality gate: isotherm log-RMSE <= 0.3 and the Widom Henry constant reproduced to 5 %."""
    fits = fits or load_fits()
    d, s = fits[name[len(PREFIX):]], sorbents.get(name)
    henry = float(s.q(d["T_ref"], 1e-9) / 1e-9 / 1e5 / d["KH_CO2"])
    return d["rmse_log"] <= 0.3 and abs(henry - 1) <= 0.05


def prescreen(names, T_cold=225.0, p_co2=4.2e-4, min_mmolg=0.1) -> list[str]:
    """Keep MOFs with a good fit that reach 0.1 mmol/g dry working capacity at 225 K (vacuum 100 C)."""
    fits = load_fits()
    keep = []
    for n in names:
        s = sorbents.get(n)
        if fit_ok(n, fits) and float(s.q(T_cold, p_co2) - s.q(373.15, 0.01)) >= min_mmolg:
            keep.append(n)
    return keep


if __name__ == "__main__":
    fits = build()
    rm = np.array([f["rmse_log"] for f in fits.values()])
    print(f"{len(fits)} MOFs fitted; log-RMSE median {np.median(rm):.3f}, 90th pct {np.quantile(rm, .9):.3f}, max {rm.max():.3f}")
    names = register()
    # the fit must reproduce the Widom Henry constant at T_ref
    err = []
    for n in names:
        s, d = sorbents.get(n), fits[n[len(PREFIX):]]
        err.append(float(s.q(d["T_ref"], 1e-9) / 1e-9 / 1e5 / d["KH_CO2"]))
    err = np.array(err)
    print(f"Henry reproduction ratio: median {np.median(err):.3f}, 5-95 % {np.quantile(err, .05):.3f}-{np.quantile(err, .95):.3f}")
    keep = prescreen(names)
    print(f"prescreen: {len(keep)} of {len(names)} reach >= 0.1 mmol/g at 225 K")
