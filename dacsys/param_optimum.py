"""Where do the adsorbent and regeneration parameters stop paying off? Sweeps beyond the ranges of the main study.

A: model adsorbent (hydrophobic water class, S = 20 000, c_p 0.85, 10 USD/kg) on a grid of Q_st x K_H(298 K)
B: saturation capacity q_s of the target (Q_st 50, K_H 0.03)
C: desorption temperature x vacuum pressure for the target and for Mg-MOF-74
Each variant is re-optimised with three configurations at all 21 sites (the plant model is unchanged).

python -m dacsys.param_optimum [--workers 8] [--report-only]  -> results_dac/study5/param_optimum.csv
"""
from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .results import RES

OUT = RES / "param_optimum.csv"
QST = (40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0)
KH = (0.01, 0.03, 0.1, 0.3, 1.0)
QS = (1.0, 2.0, 4.0, 8.0, 16.0)
T_DES = (80.0, 100.0, 120.0, 150.0)          # °C
P_DES = (0.003, 0.01, 0.03, 0.1)             # bar
ARCH = {"target": (("none", "ambient"), ("cond_silica", "ambient"), ("freeze", "vcr"), ("cond_silica", "vcr")),
        "mof": (("freeze", "vcr"), ("none", "vcr"), ("cond_silica", "ambient"), ("cond", "vcr"), ("cond_silica", "vcr"))}


# D: wide one-at-a-time sweeps of every material property of the model (material_sens.perturb semantics)
SWEEP = {
    "cp": (0.4, 0.6, 1.2, 1.5, 2.0),                      # kJ/kg K (base 0.85)
    "rho": (0.5, 0.8, 1.8, 2.5),                          # g/cm3
    "S_N2": (0.05, 0.1, 0.2, 0.5, 2.0, 5.0),              # factor on the CO2/N2 selectivity
    "Q_N2": (8.0, 12.0, 22.0, 27.0),                      # kJ/mol
    "n2_qs": (1.0, 2.0, 8.0, 16.0),                       # mol/kg
    "wrc": (-0.6, -0.4, -0.2),                            # change (target 0.91; Mg-MOF-74 below 0.05)
    "h2o_KH": (0.01, 0.1, 10.0, 100.0, 1000.0),           # factor
    "h2o_dH": (-15.0, -10.0, 10.0, 20.0, 30.0),           # kJ/mol change
    "h2o_qsat": (0.25, 0.5, 2.0, 4.0),                    # factor
    "E_diff": (10.0, 20.0, 30.0, 40.0),                   # kJ/mol
    "price": (0.25, 0.5, 2.0, 4.0, 8.0),                  # factor
    "life": (1.0, 2.0, 5.0, 10.0),                        # years (base 25)
    "KH": (0.1, 0.3, 3.0, 10.0),                          # factor, Q_st kept
    "Qst": (-10.0, -5.0, 5.0, 10.0, 20.0, 30.0),          # kJ/mol change, 298 K isotherm kept
    "qs": (0.25, 0.5, 2.0, 4.0),                          # factor, K_H kept
}
MOF_PROPS = ("KH", "Qst", "qs", "cp", "S_N2", "h2o_KH", "wrc", "price")
BASES = {"target": "P_KH0.03_Q50_phob_S20000", "mof": "PR:MgMOF74"}


def regen_key(t, p):
    return f"vac{t:g}_{p:g}"


def _add_regen():
    from . import levers
    for t, p in itertools.product(T_DES, P_DES):
        levers.REGEN.setdefault(regen_key(t, p), (t + 273.15, p, "vacuum"))


def variants():
    out = []
    for q, kh in itertools.product(QST, KH):
        out.append(("A", f"PO:Q{q:g}_KH{kh:g}", dict(q=q, kh=kh, qs=4.0), "vac100", "target"))
    for qs in QS:
        out.append(("B", f"PO:Q50_KH0.03_qs{qs:g}", dict(q=50.0, kh=0.03, qs=qs), "vac100", "target"))
    for t, p in itertools.product(T_DES, P_DES):
        out.append(("C", "PO:Q50_KH0.03_qs4", dict(q=50.0, kh=0.03, qs=4.0), regen_key(t, p), "target"))
        out.append(("C", "PR:MgMOF74", None, regen_key(t, p), "mof"))
    for arch, base in BASES.items():
        for prop, vals in SWEEP.items():
            if arch == "mof" and prop not in MOF_PROPS:
                continue
            for v in vals:
                out.append(("D", f"PD:{arch}:{prop}:{v:g}", dict(base=base, prop=prop, value=v), "vac100", arch))
    return out


_W = {}


def _weather(site):
    from . import climate_sites as cs
    from . import plant
    from .levers import Prm
    if site not in _W:
        w = cs.load(site)
        _W[site] = (w, plant.resources(w, Prm()))
    return _W[site]


def _job(v, archs=None, sites=None):
    part, name, par, regen, arch = v
    from . import plant, prisma, sorbents
    from .climate_sites import SITES
    from .levers import Case, with_
    _add_regen()
    if par is None:
        prisma.register([name[3:]])
    elif "prop" in par:
        from dataclasses import replace
        from .material_sens import _register_base, perturb
        s = _register_base(par["base"])
        sorbents.register(replace(perturb(s, par["prop"], par["value"]), name=name))
    else:
        sorbents.register(sorbents.parametric(name, par["kh"], par["q"], par["qs"], S_N2_298=20000.0, cost=10.0,
                                              water="phob"))
    rows = []
    for site in (sites or SITES):
        w, res = _weather(site)
        for d, c in (archs or ARCH[arch]):
            case = with_(Case(site=site), sorbent=name, drying=d, cold=c, lever="PO", regen=regen)
            r = plant.evaluate(case, w, res)
            p_ = dict(q=np.nan, kh=np.nan, qs=np.nan, base="", prop="", value=np.nan)   # same columns in every row
            p_.update(par or {})
            rows.append(dict(part=part, sorbent=name, regen=regen, site=site, drying=d, cold=c,
                             **p_, status=r.get("status"), lcoc=r.get("lcoc", np.nan),
                             heat_GJ=r.get("heat_GJ", np.nan), elec_GJ=r.get("elec_GJ", np.nan),
                             wcap=r.get("wcap_mmolg", np.nan), purity=r.get("purity", np.nan),
                             T_design=r.get("T_design", np.nan)))
    return rows


def run(workers, only=None, sites=None):
    """sites: compute only these (added) sites for every variant; otherwise all sites per missing architecture."""
    jobs = variants()
    if only:
        jobs = jobs[:only]
    done = set()
    old = pd.read_csv(OUT) if OUT.exists() else None
    if sites:
        have = set() if old is None else set(zip(old.part, old.sorbent, old.regen, old.drying, old.cold, old.site))
        pending = [(j, [a for a in ARCH[j[4]] if not all((j[0], j[1], j[3]) + a + (s,) in have for s in sites)])
                for j in jobs]
    else:
        if old is not None:
            done = set(zip(old.part, old.sorbent, old.regen, old.drying, old.cold))
        # per architecture, so that an added architecture is computed without repeating the others
        pending = [(j, [a for a in ARCH[j[4]] if (j[0], j[1], j[3]) + a not in done]) for j in jobs]
    pending = [(j, a) for j, a in pending if a]
    print(len(pending), "of", len(jobs), "variants to run", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j, a, sites) for j, a in pending]
        for k, f in enumerate(as_completed(futs)):
            pd.DataFrame(f.result()).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(k + 1, "/", len(pending), flush=True)


def summary():
    d = pd.read_csv(OUT)
    a = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    d = d[d.status == "ok"].copy()
    d["ratio"] = d.lcoc / d.site.map(a)
    b = d.loc[d.groupby(["part", "sorbent", "regen", "site"]).ratio.idxmin()]
    return b.groupby(["part", "sorbent", "regen"]).agg(
        ratio=("ratio", "median"), lo=("ratio", "min"), hi=("ratio", "max"), wins=("ratio", lambda v: int((v < 1).sum())),
        n=("ratio", "size"), heat=("heat_GJ", "median"), elec=("elec_GJ", "median"), wc=("wcap", "median")).reset_index()


BASE_VAL = {"Qst": 50.0, "KH": 0.03, "qs": 4.0, "cp": 0.85, "rho": None, "S_N2": 20000.0, "h2o_KH": 1.11e-5,
            "E_diff": 0.0, "price": 10.0, "life": 25.0}
ABS = {"cp", "rho", "E_diff", "life", "Q_N2", "n2_qs"}                     # perturb sets absolute values
CHG = {"Qst", "h2o_dH", "wrc"}                                               # perturb adds a change
LAB = {"Qst": "$Q_{st}$ (kJ mol$^{-1}$)", "KH": "$K_H$ (mol kg$^{-1}$ Pa$^{-1}$)", "qs": "$q_s$ (mmol g$^{-1}$)",
       "cp": "$c_p$ (kJ kg$^{-1}$ K$^{-1}$)", "S_N2": "CO$_2$/N$_2$ selectivity", "h2o_KH": "water $K_H$ (mol kg$^{-1}$ Pa$^{-1}$)",
       "rho": "density (g cm$^{-3}$)", "E_diff": "diffusion barrier (kJ mol$^{-1}$)", "life": "lifetime (yr)",
       "price": "price (USD kg$^{-1}$)"}


def sweeps():
    """Median relative cost of the target along each property (absolute property values), base included."""
    t = summary()
    base = t[(t.part == "A") & (t.sorbent == "PO:Q50_KH0.03")].ratio.iloc[0]
    d = t[(t.part == "D") & t.sorbent.str.startswith("PD:target:")].copy()
    d[["prop", "v"]] = d.sorbent.str.split(":", expand=True).iloc[:, 2:4]
    d["v"] = d.v.astype(float)
    from . import sorbents, study5
    study5.register()
    s = sorbents.get("P_KH0.03_Q50_phob_S20000")
    bv = dict(BASE_VAL, rho=s.rho)
    out = {}
    for prop, g in d.groupby("prop"):
        if prop not in LAB:
            continue
        b = bv[prop]
        x = g.v if prop in ABS else (b + g.v if prop in CHG else b * g.v)
        out[prop] = pd.concat([pd.Series(g.ratio.to_numpy(), index=x.to_numpy()), pd.Series([base], index=[b])]).sort_index()
    return out, base


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--sites", default=None, help="comma-separated added sites (only these are computed)")
    a = ap.parse_args()
    if not a.report_only:
        run(a.workers, a.only, a.sites.split(",") if a.sites else None)
    print(summary().round(3).to_string())
