"""V9: our 5-step TVSA + IAST cycle (dacsys/tvsa.py) against the PrISMa process layer itself.

PrISMa reports, per MOF, the purity, recovery and working capacity of its equilibrium 5-step TVSA cycle for
flue-gas case studies (Zenodo 11244258, KPIs_Simulated). We run our cycle on the same MOFs under the same
conditions for the dry NGCC case (no water, so only the cycle and the mixture model are tested):
  feed 4.32 % CO2 / 95.68 % N2 (dry basis), 1.01 bar, adsorption 37 C, intermediate heating 47 C,
  heating 120 C, vacuum 0.2 bar (TVSA02) or 0.6 bar (TVSA06)                      (PrISMa SI, Table S2, 3.2.1)

python -m dacsys.validate_prisma [--workers 12] -> results_dac/study5/validation_prisma.csv (+ summary printed)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import prisma, sorbents, tvsa
from .levers import Prm

KPI = Path(__file__).resolve().parent.parent / "data" / "prisma" / "KPIs_Simulated"
OUT = Path(__file__).resolve().parent.parent / "results_dac" / "study5" / "validation_prisma.csv"
CASES = {"TVSA02": 0.2, "TVSA06": 0.6}
T_LOW, T_MED, T_HIGH = 310.15, 320.15, 393.15
P_FEED, Y_CO2 = 1.01, 0.0432


def _job(mofs):
    prisma.register(mofs)
    p = Prm()
    rows = []
    for m in mofs:
        base = sorbents.get(prisma.PREFIX + m)
        for case, pvac in CASES.items():
            sb = replace(base, T_des=T_HIGH, p_des=pvac, mode="vacuum")
            with np.errstate(all="ignore"):
                r = tvsa.cycle(sb, np.array([T_LOW]), P_FEED, Y_CO2, np.array([T_MED - T_LOW]), T_HIGH, pvac,
                               tvsa.void_volume(sb, p), p.eta_vac, P_amb=1.01325, P_vs=pvac)
            rows.append(dict(MOF=m, case=case, purity_ours=100 * float(r["purity"][0]),
                             recovery_ours=100 * float(r["recovery"][0]), wc_ours=float(r["captured"][0])))
    return rows


def run(workers):
    kpi = {c: pd.read_csv(KPI / f"NGCC-onshore_Storage_UK_{c}-dry.csv").set_index("MOF") for c in CASES}
    fits = prisma.load_fits()
    mofs = sorted(set(kpi["TVSA02"].index) & set(fits))
    chunks = [mofs[i:i + 25] for i in range(0, len(mofs), 25)]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rs in ex.map(_job, chunks) for r in rs]
    d = pd.DataFrame(rows)
    for c in CASES:
        k = kpi[c][["purity", "recovery", "working_capacity"]]
        k.columns = ["purity_prisma", "recovery_prisma", "wc_prisma"]
        d.loc[d.case == c, ["purity_prisma", "recovery_prisma", "wc_prisma"]] = k.reindex(d[d.case == c].MOF).to_numpy()
    prisma.register(mofs)
    d["fit_ok"] = [prisma.fit_ok(prisma.PREFIX + m, fits) for m in d.MOF]
    d.to_csv(OUT, index=False)
    return summary(d)


def summary(d=None):
    d = pd.read_csv(OUT) if d is None else d
    d = d[d.fit_ok & d.purity_prisma.notna()]
    out = []
    for c, g in d.groupby("case"):
        for q in ("purity", "recovery", "wc"):
            a, b = g[f"{q}_prisma"], g[f"{q}_ours"]
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            rho = a.rank().corr(b.rank())
            err = (b - a) if q != "wc" else (b / a.clip(lower=1e-6) - 1) * 100
            out.append(dict(case=c, kpi=q, n=len(a), spearman=round(rho, 3), pearson=round(a.corr(b), 3),
                            median_diff=round(float(err.median()), 2), mad=round(float(err.abs().median()), 2),
                            within5=round(float((err.abs() <= 5).mean()), 3)))
    s = pd.DataFrame(out)
    print(s.to_string(index=False))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    run(ap.parse_args().workers)
