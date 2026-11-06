"""One-at-a-time sensitivity of the cost to every material property in the model, for the two routes of the study:
the hydrophobic target adsorbent (ambient temperature) and Mg-MOF-74 (drying + cooling). Each property is set to a
low and a high value, the plant is re-optimised over three architectures, and the relative cost to amine DAC is
compared with the unperturbed material at six sites. Results go to a separate file (cases.csv is not touched).

python -m dacsys.material_sens [--workers 4] [--report-only]  -> results_dac/study5/material_sens.csv
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

from .results import RES

OUT = RES / "material_sens.csv"
R, T0 = 8.314e-3, 298.15
SITES = ("Singapore", "Riyadh", "London", "Kiruna", "Tibet", "Altiplano")
BASES = {"P_KH0.03_Q50_phob_S20000": (("none", "ambient"), ("cond_silica", "ambient"), ("freeze", "vcr"),
                                       ("cond_silica", "vcr")),
         "PR:MgMOF74": (("freeze", "vcr"), ("none", "vcr"), ("cond_silica", "ambient"), ("cond", "vcr"), ("cond_silica", "vcr"))}


def _crf(i, n):
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def lifetime_factor(years, i=0.08, n=25):
    """Adsorbent replaced every `years`: annualised adsorbent cost relative to one charge for the plant life."""
    return _crf(i, years) / _crf(i, n)


# property -> (description, low, high); levels are applied by perturb()
PROPS = {
    "KH": ("CO$_2$ Henry constant (factor)", 0.5, 2.0),
    "Qst": ("CO$_2$ heat of adsorption (kJ/mol, change)", -5.0, 5.0),
    "qs": ("CO$_2$ saturation capacity (factor)", 0.5, 2.0),
    "cp": ("Heat capacity (kJ/kg K)", 0.6, 1.5),
    "rho": ("Crystal density (g/cm3)", 0.8, 1.8),
    "S_N2": ("CO$_2$/N$_2$ selectivity (factor)", 0.2, 5.0),
    "Q_N2": ("N$_2$ heat of adsorption (kJ/mol)", 12.0, 22.0),
    "n2_qs": ("N$_2$ saturation capacity (mol/kg)", 2.0, 8.0),
    "wrc": ("Water resistance coefficient (change)", -0.2, 0.2),
    "h2o_KH": ("Water Henry constant (factor)", 0.1, 10.0),
    "h2o_dH": ("Water heat of adsorption (kJ/mol, change)", -10.0, 10.0),
    "h2o_qsat": ("Water saturation capacity (factor)", 0.5, 2.0),
    "h2o_step": ("S-shaped water isotherm, pore filling at RH", 0.7, 0.3),
    "E_diff": ("Diffusion barrier (kJ/mol)", 0.0, 20.0),
    "price": ("Adsorbent price (factor)", 0.5, 2.0),
    "life": ("Adsorbent lifetime (years)", 25.0, 2.0),
}


def perturb(s, prop, v):
    """Material s with one property changed; the 298 K CO2 isotherm is kept when only Q_st changes."""
    if prop == "KH":
        return replace(s, sites=tuple((q, a + math.log(v), E) for q, a, E in s.sites))
    if prop == "Qst":
        return replace(s, sites=tuple((q, a - v / (R * T0), E + v) for q, a, E in s.sites), hoa=s.hoa + v)
    if prop == "qs":          # K_H kept: b scales with 1/q_s
        return replace(s, sites=tuple((q * v, a - math.log(v), E) for q, a, E in s.sites))
    if prop in ("cp", "rho", "Q_N2", "n2_qs", "E_diff"):
        return replace(s, **{prop: v})
    if prop == "S_N2":
        return replace(s, S_N2_298=s.S_N2_298 * v)
    if prop == "wrc":
        return replace(s, wrc=float(np.clip(s.wrc + v, 0.0, 1.0)))
    if prop == "h2o_KH":
        return replace(s, h2o_KH298=s.h2o_KH298 * v)
    if prop == "h2o_dH":
        return replace(s, h2o_dH=max(s.h2o_dH + v, 5.0))
    if prop == "h2o_qsat":
        return replace(s, h2o_qsat=s.h2o_qsat * v)
    if prop == "h2o_step":
        return replace(s, h2o_rh_step=v)
    if prop == "price":
        return replace(s, cost=s.cost * v)
    if prop == "life":        # replacement every v years, as an equivalent price of one charge
        return replace(s, cost=s.cost * lifetime_factor(v))
    raise KeyError(prop)


def variants():
    out = [(b, "base", "base", 0.0) for b in BASES]
    for b in BASES:
        for p, (_, lo, hi) in PROPS.items():
            for lvl, v in (("low", lo), ("high", hi)):
                out.append((b, p, lvl, v))
    return out


def _register_base(b):
    from . import prisma, sorbents, study5
    if b.startswith("PR:"):
        prisma.register([b[3:]])
    else:
        study5.register()
    return sorbents.get(b)


_W = {}


def _weather(site):
    from . import climate_sites as cs
    from . import plant
    from .levers import Prm
    if site not in _W:
        w = cs.load(site)
        _W[site] = (w, plant.resources(w, Prm()))
    return _W[site]


def _job(args, archs=None):
    b, prop, lvl, v = args
    from . import plant, sorbents
    from .levers import Case, with_
    s = _register_base(b)
    name = b if prop == "base" else f"MS:{b}:{prop}:{lvl}"
    if prop != "base":
        sorbents.register(replace(perturb(s, prop, v), name=name))
    rows = []
    for site in SITES:
        w, res = _weather(site)
        for d, c in (archs or BASES[b]):
            case = with_(Case(site=site), sorbent=name, drying=d, cold=c, lever="MS")
            r = plant.evaluate(case, w, res)
            rows.append(dict(base=b, prop=prop, level=lvl, value=v, site=site, drying=d, cold=c,
                             status=r.get("status"), lcoc=r.get("lcoc", np.nan), heat_GJ=r.get("heat_GJ", np.nan),
                             elec_GJ=r.get("elec_GJ", np.nan), wcap=r.get("wcap_mmolg", np.nan),
                             T_design=r.get("T_design", np.nan), purity=r.get("purity", np.nan)))
    return rows


def run(workers):
    jobs = variants()
    done = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        done = set(zip(old.base, old.prop, old.level, old.drying, old.cold))
    # per architecture, so that an added architecture is computed without repeating the others
    pending = [(j, [a for a in BASES[j[0]] if (j[0], j[1], j[2]) + a not in done]) for j in jobs]
    pending = [(j, a) for j, a in pending if a]
    print(len(pending), "of", len(jobs), "variants to run", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j, a) for j, a in pending]
        for k, f in enumerate(as_completed(futs)):
            pd.DataFrame(f.result()).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(k + 1, "/", len(pending), flush=True)


def summary():
    d = pd.read_csv(OUT)
    a = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    d = d[d.status == "ok"].copy()
    d["ratio"] = d.lcoc / d.site.map(a)
    b = d.loc[d.groupby(["base", "prop", "level", "site"]).ratio.idxmin()]
    base = b[b.prop == "base"].set_index(["base", "site"]).ratio
    b = b.join(base.rename("ratio0"), on=["base", "site"])
    b["chg"] = (b.ratio / b.ratio0 - 1) * 100
    t = b.groupby(["base", "prop", "level"]).agg(chg=("chg", "median"), ratio=("ratio", "median"),
                                                 wins=("ratio", lambda v: int((v < 1).sum())),
                                                 n=("ratio", "size")).reset_index()
    return t, b


LABEL = {"KH": "$K_H$ (×0.5 / ×2)", "Qst": "$Q_{st}$ (−5 / +5 kJ mol$^{-1}$)", "qs": "$q_s$ (×0.5 / ×2)",
         "cp": "$c_p$ (0.6 / 1.5 kJ kg$^{-1}$ K$^{-1}$)", "rho": "Density (0.8 / 1.8 g cm$^{-3}$)",
         "S_N2": "CO$_2$/N$_2$ selectivity (×0.2 / ×5)", "Q_N2": "N$_2$ heat (12 / 22 kJ mol$^{-1}$)",
         "n2_qs": "N$_2$ capacity (2 / 8 mol kg$^{-1}$)", "wrc": "WRC (−0.2 / +0.2)",
         "h2o_KH": "Water Henry constant (×0.1 / ×10)", "h2o_dH": "Water heat (−10 / +10 kJ mol$^{-1}$)",
         "h2o_qsat": "Water capacity (×0.5 / ×2)", "E_diff": "Diffusion barrier (0 / 20 kJ mol$^{-1}$)",
         "price": "Price (×0.5 / ×2)", "life": "Lifetime (base 25 / 2 yr)"}
TITLE = {"P_KH0.03_Q50_phob_S20000": "Reference adsorbent, ambient temperature",
         "PR:MgMOF74": "Mg-MOF-74, drying and cooling"}


CAPTION = ("One-at-a-time sensitivity of the relative cost to the adsorbent properties{}: median change (\\%) over six "
           "sites (Singapore, Riyadh, London, Kiruna, Tibet, Altiplano) for the two values given, with the configuration "
           "re-optimised. Lifetime: replacement of the adsorbent, expressed as an equivalent price of one charge.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.workers)
    t, _ = summary()
    print(t.round(2).to_string())
