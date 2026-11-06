"""LNG cold: cost of cold recovery versus the LNG needed per tonne of CO2 (and hence the global potential), and the
break-even price of LNG cold. The plant is evaluated with the recuperator approach fixed (3, 8, 20 K or none) for 13X
and Mg-MOF-74 with LNG cold at every site; everything else is optimised as usual.

Global potential on the resource basis of Kim et al. (2025): 327.9 t LNG per t CO2 without cold recovery corresponds
to 4.2 Mt CO2/yr with the current regasification capacity (6.6 Mt/yr in 2050), i.e. potential = 4.2 * 327.9 / (t LNG
per t CO2); cold per tonne of LNG = 290.4 GJ / 327.9 t = 0.886 GJ.

python -m dacsys.lng_scale [--workers 4] -> results_dac/study5/lng_scale.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .results import RES

OUT = RES / "lng_scale.csv"
SORBENTS = ("13X", "PR:MgMOF74")
DRYERS = ("freeze", "cond")
DT = {"3 K": 3.0, "8 K": 8.0, "20 K": 20.0, "none": None}
GJ_PER_T_LNG = 290.4 / 327.9                     # Kim et al. 2025
KIM_T_LNG, KIM_MT_NOW, KIM_MT_2050 = 327.9, 4.2, 6.6


def _job(site):
    from . import climate_sites as cs
    from . import plant, prisma
    from .levers import Case, Prm, with_
    prisma.register(["MgMOF74"])
    w = cs.load(site)
    res = plant.resources(w, Prm())
    rows = []
    for sb in SORBENTS:
        for dr in DRYERS:
            for lab, dt in DT.items():
                case = with_(Case(site=site), sorbent=sb, drying=dr, cold="lng", lever="LNGS")
                r = plant.evaluate(case, w, res, dT_list=[dt])
                rows.append(dict(site=site, sorbent=sb, drying=dr, recup=lab, status=r.get("status"),
                                 lcoc=r.get("lcoc", np.nan), cold_GJ=r.get("cold_GJ", np.nan),
                                 heat_GJ=r.get("heat_GJ", np.nan), elec_GJ=r.get("elec_GJ", np.nan),
                                 T_design=r.get("T_design", np.nan), wcap=r.get("wcap_mmolg", np.nan),
                                 capex_process=r.get("capex_process", np.nan),
                                 co2_t_per_kgs=r.get("co2_t_per_kgs", np.nan)))
    return rows


def run(workers):
    from .climate_sites import SITES
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_job, list(SITES)) for r in rs]
    pd.DataFrame(rows).to_csv(OUT, index=False)


def summary(drying=None):
    """Best dryer per site, adsorbent and approach, or only the given dryer."""
    d = pd.read_csv(OUT)
    if drying:
        d = d[d.drying == drying]
    a = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    d = d[d.status == "ok"].copy()
    d["amine"] = d.site.map(a)
    d["ratio"] = d.lcoc / d.amine
    b = d.loc[d.groupby(["site", "sorbent", "recup"]).lcoc.idxmin()].copy()      # best dryer per option
    b["t_lng"] = b.cold_GJ / GJ_PER_T_LNG
    b["potential_now"] = KIM_MT_NOW * KIM_T_LNG / b.t_lng
    b["potential_2050"] = KIM_MT_2050 * KIM_T_LNG / b.t_lng
    b["cold_price"] = (b.amine - b.lcoc) / b.cold_GJ                              # USD per GJ cold at parity
    return b


LEVELS = ["none", "20 K", "8 K", "3 K"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.workers)
    b = summary()
    cols = ["ratio", "lcoc", "cold_GJ", "t_lng", "potential_now", "potential_2050", "cold_price", "T_design"]
    print(b.groupby(["sorbent", "recup"])[cols].median().round(2).to_string())
