"""RQ5 property decomposition: which property separates the best real MOF from the material target?

Six property groups are switched between the real MOF (PrISMa str_m4_o14_o24_acs_sym.190) and the
RQ5 target (P_KH0.03_Q50_phob_S20000). Every one of the 2^6 = 64 hybrids is evaluated on the same
plant with the architecture re-optimised (drying x cold without LNG, FREE_ARCH), and the Shapley value
of each group is its average marginal effect on ln(LCOC/amine) over all orders -- so the attribution
does not depend on the order in which properties are switched.

Groups
  price   : $/kg sorbent
  N2      : CO2/N2 selectivity at 298 K and N2 heat
  water   : H2O isotherm (qsat, KH, heat, S-step) and WRC
  solid   : heat capacity and crystal density
  heat    : CO2 heat of adsorption (temperature dependence; the 298 K isotherm is kept)
  iso298  : CO2 isotherm at 298 K (Henry constant, saturation capacity, shape)

python -m dacsys.decomp [--workers 12] [--sites all] [--real RSM2114,NEYZAU_clean]
    -> results_dac/study5/decomp_cases.csv (resumable per real MOF and site), decomp_best.csv, decomp_shapley.csv
Architectures: ARCH4, those that are the optimum of any real MOF or the reference adsorbent without LNG (GAX never is);
resumable per (MOF, site, architecture).
"""
from __future__ import annotations

import argparse
import itertools
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import climate_sites as cs
from . import sorbents
from .levers import Case, with_

OUT = Path(__file__).resolve().parent.parent / "results_dac" / "study5"
REAL = "PR:str_m4_o14_o24_acs_sym.190"
REALS = ("PR:str_m4_o14_o24_acs_sym.190", "PR:RSM2114", "PR:NEYZAU_clean", "PR:MgMOF74")
ARCH4 = (("none", "ambient"), ("cond_silica", "ambient"), ("none", "vcr"), ("freeze", "vcr"), ("cond", "vcr"), ("cond_silica", "vcr"))
TARGET = "P_KH0.03_Q50_phob_S20000"
GROUPS = ("price", "N2", "water", "solid", "heat", "iso298")
SITES5 = ("Singapore", "Riyadh", "London", "Kiruna", "Tibet")
R = 8.314e-3
T0 = 298.15


def _b298(site):
    qs, a, E = site
    return qs, a + E / (R * T0)                       # (qs, ln b at 298 K)


def hname(real_name: str, mask) -> str:
    return "DX:" + "".join("1" if m else "0" for m in mask) + ("" if real_name == REAL else ":" + real_name[3:])


def hybrid(real: sorbents.Sorbent, tgt: sorbents.Sorbent, mask: tuple[bool, ...]) -> sorbents.Sorbent:
    """Real MOF with the groups flagged in mask taken from the target."""
    use = dict(zip(GROUPS, mask))
    src = tgt if use["iso298"] else real
    heat_src = tgt if use["heat"] else real
    sites = []
    for i, s in enumerate(src.sites):
        qs, lnb = _b298(s)
        # per-site heat only when isotherm and heat come from the same material; otherwise its mean heat
        E = s[2] if src is heat_src else heat_src.hoa
        sites.append((qs, lnb - E / (R * T0), E))
    kw = dict(sites=tuple(sites), hoa=heat_src.hoa)
    kw["cost"] = (tgt if use["price"] else real).cost
    n2 = tgt if use["N2"] else real
    kw.update(S_N2_298=n2.S_N2_298, Q_N2=n2.Q_N2, n2_qs=n2.n2_qs)
    w = tgt if use["water"] else real
    kw.update(h2o_qsat=w.h2o_qsat, h2o_KH298=w.h2o_KH298, h2o_dH=w.h2o_dH, wrc=w.wrc, h2o_rh_step=w.h2o_rh_step)
    so = tgt if use["solid"] else real
    kw.update(cp=so.cp, rho=so.rho)
    name = hname(real.name, mask)
    return replace(real, name=name, meta={}, **kw)


def _register(real_name: str = REAL):
    from . import prisma, study5
    study5.register()
    prisma.register([real_name[len(prisma.PREFIX):]])
    real, tgt = sorbents.get(real_name), sorbents.get(TARGET)
    names = []
    for mask in itertools.product((False, True), repeat=len(GROUPS)):
        names.append(sorbents.register(hybrid(real, tgt, mask)).name)
    return names


def _job(real_name, site, names, archs=None):
    from . import plant
    _register(real_name)
    w = cs.load(site)
    base = Case(site=site)
    res = plant.resources(w, base.prm)
    rows = []
    for n in names:
        for d, c in (archs or ARCH4):
            case = with_(base, sorbent=n, drying=d, cold=c, lever="DX")
            r = plant.evaluate(case, w, res)
            rows.append(dict(real=real_name, site=site, sorbent=n, mask=n[3:9], drying=d, cold=c, status=r.get("status"),
                             lcoc=r.get("lcoc", np.inf), purity=r.get("purity"), wcap=r.get("wcap_mmolg"),
                             water_per_co2=r.get("water_per_co2"), heat_GJ=r.get("heat_GJ"),
                             elec_GJ=r.get("elec_GJ"), T_design=r.get("T_design")))
    return rows


def shapley(best: pd.DataFrame) -> pd.DataFrame:
    """Shapley value of each group on ln(ratio), per site. Infeasible hybrids get a penalty ratio."""
    out = []
    k = len(GROUPS)
    for (real, site), g in best.groupby(["real", "site"]):
        v = {tuple(c == "1" for c in m): math.log(r) for m, r in zip(g["mask"], g.ratio_pen)}
        for i, grp in enumerate(GROUPS):
            phi = 0.0
            for mask, val in v.items():
                if mask[i]:
                    continue
                on = mask[:i] + (True,) + mask[i + 1:]
                s = sum(mask)
                phi += math.factorial(s) * math.factorial(k - s - 1) / math.factorial(k) * (v[on] - val)
            out.append(dict(real=real, site=site, group=grp, phi_ln=phi))
    return pd.DataFrame(out)


def _load():
    d = pd.read_csv(OUT / "decomp_cases.csv", dtype={"mask": str})
    if "real" not in d:                                   # first 5-site run (single MOF, 8 architectures)
        d["real"], d["mask"] = REAL, d.sorbent.str[3:9]
    d["real"] = d.real.fillna(REAL)
    d["mask"] = d["mask"].fillna(d.sorbent.str[3:9])
    arch = set(ARCH4)
    return d[[(a, b) in arch for a, b in zip(d.drying, d.cold)]]


def run(workers, sites, reals):
    part = OUT / "decomp_cases.csv"
    # per architecture, so that an added architecture is computed without repeating the others
    done = set(zip(_load().real, _load().site, _load().drying, _load().cold)) if part.exists() else set()
    jobs = []
    for r in reals:                                        # MOF by MOF, so each finishes early
        names = _register(r)
        for s in sites:
            archs = [a for a in ARCH4 if (r, s) + a not in done]
            if archs:
                jobs += [(r, s, names[i:i + 8], archs) for i in range(0, len(names), 8)]
    print(f"{len(jobs)} jobs", flush=True)
    buf = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_job, *j): (j[0], j[1]) for j in jobs}
        need = pd.Series([k for k in futs.values()]).value_counts().to_dict()
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            buf.setdefault(key, []).extend(f.result())
            need[key] -= 1
            if need[key] == 0:                             # a (MOF, site) is complete: write it
                pd.DataFrame(buf.pop(key)).to_csv(part, mode="a", header=not part.exists(), index=False)
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} jobs", flush=True)
    return summarise()


def summarise():
    d = _load()
    a = pd.read_csv(OUT / "amine.csv").set_index("site")["amine_lcoc_base"]
    d["ratio"] = d.lcoc / d.site.map(a)
    ok = d[(d.status == "ok") & np.isfinite(d.ratio)]
    best = ok.loc[ok.groupby(["real", "site", "mask"]).ratio.idxmin()]
    # a (MOF, site) is only used once all 64 hybrids are in; infeasible hybrids get 2 x the worst feasible ratio
    n = d.groupby(["real", "site"])["mask"].nunique()
    full = n[n == 2 ** len(GROUPS)].index
    allm = pd.MultiIndex.from_tuples([(r, s, "".join(m)) for r, s in full
                                      for m in itertools.product("01", repeat=len(GROUPS))],
                                     names=["real", "site", "mask"])
    best = best.set_index(["real", "site", "mask"]).reindex(allm).reset_index()
    pen = best.groupby(["real", "site"]).ratio.transform("max") * 2.0
    best["infeasible"] = best.ratio.isna()
    best["ratio_pen"] = best.ratio.fillna(pen)
    best.to_csv(OUT / "decomp_best.csv", index=False)
    sh = shapley(best)
    sh.to_csv(OUT / "decomp_shapley.csv", index=False)
    fac = sh.assign(f=np.exp(sh.phi_ln)).pivot_table(index="real", columns="group", values="f",
                                                      aggfunc=["min", "median", "max"])
    print(fac.round(2).to_string())
    one = best[best["mask"].isin(["".join("1" if j == i else "0" for j in range(len(GROUPS)))
                                  for i in range(len(GROUPS))] + ["0" * len(GROUPS), "1" * len(GROUPS)])]
    print(one.pivot_table(index="real", columns="mask", values="ratio", aggfunc="median").round(2).to_string())
    print("infeasible hybrids:", int(best.infeasible.sum()), "of", len(best))
    return sh


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sites", default="5", help="'5' (one per Koppen group) or 'all'")
    ap.add_argument("--real", default=None, help="comma-separated PrISMa MOFs (without PR:); default: REALS")
    a = ap.parse_args()
    reals = ["PR:" + x for x in a.real.split(",")] if a.real else list(REALS)
    run(a.workers, list(cs.SITES) if a.sites == "all" else list(SITES5), reals)
