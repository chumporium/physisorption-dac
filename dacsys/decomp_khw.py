"""Threshold scan: the real MOF of dacsys.decomp with its water Henry constant lowered step by step.

python -m dacsys.decomp_khw [--workers 5] -> results_dac/study5/decomp_khw.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from . import climate_sites as cs
from . import decomp, sorbents
from .levers import Case, with_

KHW = (27.8144, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 1e-3, 1e-4, 1.11e-5)     # mol/kg/Pa at 298 K


def _register():
    decomp._register()
    real = sorbents.get(decomp.REAL)
    for k in KHW:
        sorbents.register(replace(real, name=f"DK:{k:g}", meta={}, h2o_KH298=k))


def _job(site):
    from . import plant, study5
    _register()
    w = cs.load(site)
    base = Case(site=site)
    res = plant.resources(w, base.prm)
    rows = []
    for k in KHW:
        for d, c in study5.FREE_ARCH:
            case = with_(base, sorbent=f"DK:{k:g}", drying=d, cold=c, lever="DK")
            if case.applicable:
                r = plant.evaluate(case, w, res)
                rows.append(dict(site=site, KHw=k, drying=d, cold=c, status=r.get("status"),
                                 lcoc=r.get("lcoc", np.inf), wcap=r.get("wcap_mmolg"),
                                 water_per_co2=r.get("water_per_co2")))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(_job, decomp.SITES5) for r in rs]
    d = pd.DataFrame(rows)
    am = pd.read_csv(decomp.OUT / "amine.csv").set_index("site")["amine_lcoc_base"]
    d["ratio"] = d.lcoc / d.site.map(am)
    d.to_csv(decomp.OUT / "decomp_khw.csv", index=False)
    ok = d[d.status == "ok"]
    b = ok.loc[ok.groupby(["site", "KHw"]).ratio.idxmin()]
    print(b.pivot(index="KHw", columns="site", values="ratio").sort_index(ascending=False).round(2).to_string())
    b["arch"] = b.drying + "/" + b.cold
    print(b.pivot(index="KHw", columns="site", values="arch").sort_index(ascending=False).to_string())
