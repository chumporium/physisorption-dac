"""Paths and helpers for the stored results of the case study (results_dac/study5)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .climate_sites import SITES

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_dac" / "study5"
TARGET = "P_KH0.03_Q50_phob_S20000"          # hydrophobic reference adsorbent
ORDER = list(SITES)


def load():
    """All cases of the main runs with the base amine LCOC and the relative cost (ratio)."""
    d = pd.read_csv(RES / "cases.csv")
    a = pd.read_csv(RES / "amine.csv").set_index("site")
    d = d.join(a[["amine_lcoc_lo", "amine_lcoc_base", "amine_lcoc_hi"]], on="site")
    d["ratio"] = d.lcoc / d.amine_lcoc_base
    d = d[~d.lever.str.startswith(("S_", "W_", "PR1", "DB1", "DX", "DW", "DK"))]
    return d, a


def best(x, by):
    """Lowest-cost feasible case of each group."""
    x = x[x.status == "ok"]
    return x.loc[x.groupby(by).lcoc.idxmin()]


def m13(d):
    """Zeolite 13X cases."""
    return d[(d.sorbent == "13X") & d.lever.isin(["B0", "RQ1x2"])]


def tgt(d, name=TARGET):
    """Cases of a model adsorbent without LNG."""
    return d[(d.sorbent == name) & d.lever.isin(["RQ5", "RQ5free"]) & (d.cold != "lng")]


def db(d):
    """Stage-2 PrISMa cases: the best MOFs of the stage-1 screen x every dryer x cold source x site."""
    return d[d.lever.isin(["PR2", "PRL"])]


def db_best(d, lng=False):
    """Best stage-2 MOF and configuration at each site, without or with LNG cold."""
    p = db(d)
    p = p[(p.cold == "lng") if lng else (p.cold != "lng")]
    return best(p, ["site"]).set_index("site").reindex(ORDER)


def iast_flagged(threshold=0.1):
    """PrISMa MOFs whose binary IAST prediction (CO2 or N2) deviates from GCMC (log ratio > threshold)."""
    f = pd.read_csv(ROOT / "data" / "prisma" / "Flags" / "IAST-Flag_NGCC-onshore.csv").set_index("MOF")
    return {"PR:" + m for m in f.index[(f.CO2 > threshold) | (f.N2 > threshold)]}
