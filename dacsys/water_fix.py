"""Recompute the recovered water of freeze-out cases after the water-accounting fix in plant.air_train
(ice in the reversing regenerators is sublimed into the exhaust and is not recovered). The LCOC does not
depend on the recovered water (water_value = 0), so only the water columns change; the LCOC is re-checked.

python -m dacsys.water_fix [--workers 4]  -> results_dac/study5/water_fix.csv, patches cases.csv (backup kept)
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .fig_paper1 import RES

LEVERS = ("PR2", "PRL", "RQ1x2", "B0")
OUT = RES / "water_fix.csv"
AFTER_FIX = ("RQ5cp", "DB3")


_W = {}


def _weather(site):
    from . import climate_sites as cs
    from . import plant
    from .levers import Prm
    if site not in _W:
        w = cs.load(site)
        _W[site] = (w, plant.resources(w, Prm()))
    return _W[site]


def _job(args):
    """One sorbent (so that its TVSA table is built once per worker), all its freeze-out cases."""
    sorbent, rows = args
    from . import plant, prisma
    from .levers import Case, with_
    if sorbent.startswith("PR:"):
        prisma.register([sorbent[3:]])
    out = []
    for r in rows:
        w, res = _weather(r["site"])
        case = with_(Case(site=r["site"]), sorbent=sorbent, drying=r["drying"], cold=r["cold"], lever=r["lever"])
        x = plant.evaluate(case, w, res, T_list=[float(r["T_design"])])
        out.append(dict(key=r["key"], water_t_per_kgs=x.get("water_t_per_kgs", np.nan),
                        water_per_co2=x.get("water_per_co2", np.nan), lcoc_new=x.get("lcoc", np.nan)))
    return out


def run(workers):
    c = pd.read_csv(RES / "cases.csv", low_memory=False)
    x = c[c.lever.isin(LEVERS) & (c.drying == "freeze") & (c.status == "ok")]
    jobs = [(s, g[["key", "lever", "site", "drying", "cold", "T_design"]].to_dict("records"))
            for s, g in x.groupby("sorbent")]
    jobs.sort(key=lambda j: -len(j[1]))
    print(len(x), "cases in", len(jobs), "jobs", flush=True)
    out = []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for k, f in enumerate(as_completed(futs)):
            out += f.result()
            print(k + 1, "/", len(jobs), flush=True)
    f = pd.DataFrame(out)
    f.to_csv(OUT, index=False)
    return f


def apply():
    c = pd.read_csv(RES / "cases.csv", low_memory=False)
    f = pd.read_csv(OUT).set_index("key")
    old = c.set_index("key").loc[f.index]
    dl = (f.lcoc_new / old.lcoc - 1).abs()
    print("LCOC check: max rel. difference", float(dl.max()))
    assert dl.max() < 1e-6
    bak = RES / "cases_before_waterfix.csv"
    if not bak.exists():
        shutil.copy(RES / "cases.csv", bak)
    c = c.set_index("key")
    c.loc[f.index, "water_t_per_kgs"] = f.water_t_per_kgs
    c.loc[f.index, "water_per_co2"] = f.water_per_co2
    # levers first run after the fix in plant.air_train are correct as computed
    c["water_fixed"] = (c.drying != "freeze") | c.index.isin(f.index) | c.lever.isin(AFTER_FIX)
    c.reset_index().to_csv(RES / "cases.csv", index=False)
    print("patched", len(f), "rows; freeze rows not recomputed:", int((~c.water_fixed).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--apply-only", action="store_true")
    a = ap.parse_args()
    if not a.apply_only:
        run(a.workers)
    apply()
