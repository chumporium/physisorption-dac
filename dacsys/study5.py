"""Run every case of the five research questions on one plant architecture.

python -m dacsys.study5 [--workers 8] [--only matrix,rq5,sens]
    -> results_dac/study5/cases.jsonl   (one line per case, appended; finished cases are skipped)
    -> results_dac/study5/amine.csv     (amine benchmark per site, same energy block)
    -> results_dac/study5/cases.csv     (flat table of everything, rebuilt at the end)

Case groups
  matrix : 13X, every drying x cold combination, every site   -> RQ1, RQ2, RQ1x2, RQ3 (+ RQ4)
  rq5    : parametric + screening sorbents, B0 architecture and free architecture, every site
  sens   : B0 sensitivities (sweep regeneration, strict GAX generator limit, CALF-20, cycle time)
  prisma1: PrISMa real MOFs (prescreened) x 3 architectures x 5 climate sites
  prisma2: the 40 best PrISMa MOFs of prisma1 x every architecture x every site
  prismaL: the same 40 MOFs with LNG cold (upper bound: is any real MOF viable even with free cold?)
  db1    : stage-1 screen extended to the other cold sources: every prescreened PrISMa MOF with freeze-out +
           solar GAX and freeze-out + LNG at the five climate sites (PR1 covered ambient and heat pump)
  db2    : stage 2 over the full lever matrix: the best DB_TOP MOFs of each cold source (union) x every
           drying x cold combination x every site (levers PR2 / PRL, so earlier stage-2 cases are reused)
  db3    : every prescreened PrISMa MOF x every site x (silica gel, no chiller) and (freeze-out, heat pump)
  rq5cp  : hydrophobic target (Q_st 50, S 20000) with solid heat capacity x saturation capacity x K_H, 3 architectures
  water  : water-model robustness -- wet front 2x / 3x longer than ideal, and PrISMa WRC without pore-filling
           scaling -- for 13X (every drying x cold), the target, the mid-water target and the 10 best PrISMa MOFs
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

from . import climate_sites as cs
from . import prisma, sorbents
from .levers import COLD, DRYING, Case, b0, with_

OUT = Path(__file__).resolve().parent.parent / "results_dac" / "study5"
JSONL = OUT / "cases.jsonl"

# parametric material grid for RQ5 (one Langmuir site, qs 4 mmol/g, $10/kg)
KH = (0.003, 0.01, 0.03, 0.1)          # mol/kg/Pa at 298 K
QST = (30.0, 40.0, 50.0)               # kJ/mol
WATER = ("phil", "mid", "phob")        # PrISMa water classes (sorbents.WATER_CLASSES)
S_N2 = (2000.0, 20000.0)               # CO2/N2 selectivity at 298 K
# screening materials (user's CoRE MOF 2025 GCMC): KH, qs, hydrophobic, retention, water t/t, S_N2, E_diff, $/kg
SCREEN = {
    "KENVOR": (0.005947, 2.0, True, 0.76, 0.43, 4602.0, 18.4, 50.0),
    "KENVOR02": (0.002700, 2.0, True, 0.94, 0.21, 2217.0, 17.8, 50.0),
    "KENVIL": (0.002967, 2.0, True, 0.77, 0.58, 2729.0, 12.3, 50.0),
    "SIFSIX-3-Ni": (0.034005, 2.5, False, 1.0, 0.0, 6691.0, 0.0, 30.0),
    "jp9b03221": (0.146761, 2.5, False, 1.0, 0.0, 28630.0, 0.0, 30.0),
}
PR_SITES = ("Singapore", "Riyadh", "London", "Kiruna", "Tibet")   # one per main Koppen group
PR_ARCH = (("none", "ambient"), ("none", "vcr"), ("freeze", "vcr"))
PR_TOP = 40
# stage 1 also covers the dryer that is cheapest for most stage-2 pairs with each cold source after the freeze-out
# correction (freeze-out needs a costed regenerator): silica gel + heat pump, no dryer + GAX, condensation + LNG
DB1_ARCH = (("freeze", "gax"), ("freeze", "lng"), ("cond_silica", "vcr"), ("none", "gax"), ("cond", "lng"))
DB_TOP = 30
COLD_CLASS = {"ambient": "ambient", "gax": "gax", "vcr": "vcr", "lng": "lng"}
WATER_VARIANTS = {"W_f2": dict(water_front_factor=2.0), "W_f3": dict(water_front_factor=3.0),
                  "W_noscale": dict(wrc_pore_scaling=False)}
WATER_SORBENTS = ("P_KH0.03_Q50_phob_S20000", "P_KH0.03_Q50_mid_S20000")
FREE_ARCH = [(d, c) for d in ("none", "cond", "cond_silica", "freeze") for c in ("ambient", "gax", "vcr")
             if (c, d) not in (("ambient", "freeze"), ("ambient", "cond"))]
# RQ5 extension: solid heat capacity and saturation capacity of the hydrophobic target (Q_st 50, S 20000)
CP_GRID = (0.6, 0.85, 1.2, 1.5)        # kJ/kg K (PrISMa median 0.85)
QS_GRID = (2.0, 4.0, 8.0)              # mmol/g
KH_CP = (0.01, 0.03, 0.1)              # mol/kg/Pa at 298 K
ARCH_CP = (("none", "ambient"), ("cond_silica", "ambient"), ("freeze", "vcr"), ("cond_silica", "vcr"))
# every prescreened MOF at every site with the two architectures that win without LNG (no two-stage preselection)
DB3_ARCH = (("cond_silica", "ambient"), ("freeze", "vcr"), ("cond_silica", "vcr"))


def cp_name(kh, cp, qs):
    return f"P_KH{kh:g}_Q50_phob_S20000_cp{cp:g}_qs{qs:g}"


def register() -> list[str]:
    """Parametric grid (water class from PrISMa medians), screening materials, and the real
    PrISMa experimental materials. Every sorbent uses the two-zone water model."""
    names = []
    for kh, q, wat, s in itertools.product(KH, QST, WATER, S_N2):
        n = f"P_KH{kh:g}_Q{q:g}_{wat}_S{s:g}"
        sorbents.register(sorbents.parametric(n, kh, q, 4.0, S_N2_298=s, cost=10.0, water=wat))
        names.append(n)
    for m, (kh, qs, hyd, ret, wr, s, ed, cost) in SCREEN.items():
        # water isotherm of the matching PrISMa class; WRC = the user's GCMC retention
        qsat, khw, dhw, _ = sorbents.WATER_CLASSES["phob" if hyd else "phil"]
        for q in (30.0, 40.0):
            n = f"M_{m}_Q{q:g}"
            sorbents.register(sorbents.parametric(n, kh, q, qs, S_N2_298=s, E_diff=ed, cost=cost,
                                                  water=(qsat, khw, dhw, ret if hyd else 0.002)))
            names.append(n)
    for kh, cp, qs in itertools.product(KH_CP, CP_GRID, QS_GRID):
        s = sorbents.parametric(cp_name(kh, cp, qs), kh, 50.0, qs, S_N2_298=20000.0, cost=10.0, water="phob")
        sorbents.register(replace(s, cp=cp))
    return names + ["CALF-20", "MIP-212"]


def _prisma_top(n=PR_TOP) -> list[str]:
    """Best PrISMa MOFs of stage 1 by their lowest cost ratio to amine over the five sites."""
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    d = d[(d.lever == "PR1") & (d.status == "ok")]
    a = pd.read_csv(OUT / "amine.csv").set_index("site")["amine_lcoc_base"]
    d["ratio"] = d.lcoc / d.site.map(a)
    return list(d.groupby("sorbent").ratio.min().sort_values().index[:n])


def _db_top(k=DB_TOP) -> list[str]:
    """Union of the k best PrISMa MOFs for each cold source in stage 1 (PR1 + DB1, five sites), by their
    lowest cost ratio to amine; so a MOF that is best with GAX or with LNG is not missed."""
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    d = d[d.lever.isin(["PR1", "DB1"]) & (d.status == "ok")]
    a = pd.read_csv(OUT / "amine.csv").set_index("site")["amine_lcoc_base"]
    d["ratio"] = d.lcoc / d.site.map(a)
    top = []
    for _, g in d.groupby("cold"):
        for n in g.groupby("sorbent").ratio.min().sort_values().index[:k]:
            if n not in top:
                top.append(n)
    return top


def _have(levers) -> set:
    """(sorbent, site, drying, cold) already evaluated under the given levers."""
    out = set()
    for l in JSONL.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["lever"] in levers:
                out.add((r["sorbent"], r["site"], r["drying"], r["cold"]))
    return out


def cases(groups) -> list[Case]:
    names = register()
    pr = prisma.prescreen(prisma.register()) if {"prisma1", "prisma2", "prismaL", "water", "db1", "db2", "db3"} & set(groups) else []
    top = _prisma_top() if {"prisma2", "prismaL", "water"} & set(groups) else []
    dbtop = sorted(set(_db_top()) | set(_prisma_top())) if "db2" in groups else []
    have = _have(("PR1", "PR2", "DB1")) if "db3" in groups else set()
    out = []
    for site in cs.SITES:
        base = b0(site)
        if "matrix" in groups:
            for d, c in itertools.product(DRYING, COLD):
                lever = "B0" if (d, c) == ("cond_silica", "gax") else "RQ1x2"
                out.append(with_(base, drying=d, cold=c, lever=lever))
        if "rq5" in groups:
            for n in names:
                out.append(with_(base, sorbent=n, lever="RQ5"))
                for d, c in FREE_ARCH:
                    if (d, c) != ("cond_silica", "gax"):
                        out.append(with_(base, sorbent=n, drying=d, cold=c, lever="RQ5free"))
        if "sens" in groups:
            out.append(with_(base, regen="sweep200", lever="S_sweep200"))
            out.append(with_(base, gax_T_gen_max=160.0, lever="S_Tgen160"))
            out.append(with_(base, sorbent="CALF-20", lever="S_CALF20"))
            # cycle-time sensitivity: the old fixed 1 cycle/h (Kim et al.) for 13X and the target
            out.append(with_(base, drying="freeze", cold="vcr", max_cycles_per_h=1.0, lever="S_cyc1_13X"))
            out.append(with_(base, sorbent="P_KH0.03_Q50_phob_S20000", drying="none", cold="ambient",
                             max_cycles_per_h=1.0, lever="S_cyc1_target"))
            # purity model: legacy Henry-selectivity model instead of the TVSA/IAST cycle (step 4)
            out.append(with_(base, drying="freeze", cold="vcr", purity_model="henry", lever="S_henry_13X"))
            out.append(with_(base, sorbent="P_KH0.03_Q50_phob_S20000", drying="none", cold="ambient",
                             purity_model="henry", lever="S_henry_target"))
        if "prisma1" in groups and site in PR_SITES:
            for n in pr:
                for d, c in PR_ARCH:
                    out.append(with_(base, sorbent=n, drying=d, cold=c, lever="PR1"))
        if "db1" in groups and site in PR_SITES:
            for n in pr:
                for d, c in DB1_ARCH:
                    out.append(with_(base, sorbent=n, drying=d, cold=c, lever="DB1"))
        if "db2" in groups:
            for n in dbtop:
                for d, c in itertools.product(DRYING, COLD):
                    out.append(with_(base, sorbent=n, drying=d, cold=c, lever="PRL" if c == "lng" else "PR2"))
        if "prisma2" in groups:
            for n in top:
                for d, c in FREE_ARCH:
                    out.append(with_(base, sorbent=n, drying=d, cold=c, lever="PR2"))
        if "water" in groups:
            for lv, kw in WATER_VARIANTS.items():
                for d, c in itertools.product(DRYING, COLD):
                    out.append(with_(base, drying=d, cold=c, lever=lv, **kw))
                for n in list(WATER_SORBENTS) + top[:10]:
                    for d, c in FREE_ARCH:
                        out.append(with_(base, sorbent=n, drying=d, cold=c, lever=lv, **kw))
        if "db3" in groups:
            for n in pr:
                for d, c in DB3_ARCH:
                    if (n, site, d, c) not in have:
                        out.append(with_(base, sorbent=n, drying=d, cold=c, lever="DB3"))
        if "rq5cp" in groups:
            for kh, cp, qs in itertools.product(KH_CP, CP_GRID, QS_GRID):
                for d, c in ARCH_CP:
                    out.append(with_(base, sorbent=cp_name(kh, cp, qs), drying=d, cold=c, lever="RQ5cp"))
        if "prismaL" in groups:
            for n in top:
                for d in DRYING:
                    out.append(with_(base, sorbent=n, drying=d, cold="lng", lever="PRL"))
    return [c for c in out if c.applicable]


def _row(case: Case, r: dict) -> dict:
    return dict(key=case.key(), lever=case.lever, site=case.site, drying=case.drying, cold=case.cold,
                sorbent=case.sorbent, regen=case.prm.regen, gax_T_gen_max=case.prm.gax_T_gen_max, **r)


_SITE_CACHE: dict = {}
_REGISTERED = False


def _job(site: str, batch: list[Case]) -> list[dict]:
    """Evaluate a batch; weather and energy resources are cached per worker (the batch may span several sites)."""
    global _REGISTERED
    from . import plant
    if not _REGISTERED:
        register()
        prisma.register()
        _REGISTERED = True
    out = []
    for c in batch:
        key = (c.site, c.prm)
        if key not in _SITE_CACHE:
            w = cs.load(c.site)
            _SITE_CACHE[key] = (w, plant.resources(w, c.prm))
        w, res = _SITE_CACHE[key]
        out.append(_row(c, plant.evaluate(c, w, res)))
    return out


def _amine(site: str) -> dict:
    from . import plant
    w = cs.load(site)
    return dict(site=site, koppen=cs.SITES[site][2], **plant.amine(w))


def run(workers: int, groups, part: str | None = None):
    """part = "k/n": run only the sorbents with index k (0-based) modulo n into cases_part<k>.jsonl, so that several
    processes can share the work; merge() appends the part files to cases.jsonl and rebuilds cases.csv."""
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in [JSONL] + sorted(OUT.glob("cases_part*.jsonl")):
        if f.exists():
            done |= {json.loads(l)["key"] for l in f.read_text().splitlines() if l.strip()}
    todo = [c for c in cases(groups) if c.key() not in done]
    target = JSONL
    if part:
        k, n = (int(x) for x in part.split("/"))
        names = sorted({c.sorbent for c in todo})
        mine = set(names[k::n])
        todo = [c for c in todo if c.sorbent in mine]
        target = OUT / f"cases_part{k}.jsonl"
    print(f"{len(todo)} cases to run ({len(done)} already done)", flush=True)
    # batches of ~20 cases per job keep weather loading cheap and checkpoints frequent
    # batches of one sorbent (so that its TVSA table is built once per worker) and at most 40 cases
    batches = []
    by_sorbent: dict = {}
    for c in todo:
        by_sorbent.setdefault(c.sorbent, []).append(c)
    for n, cs_ in by_sorbent.items():
        batches += [(cs_[0].site, cs_[i:i + 40]) for i in range(0, len(cs_), 40)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        amine_f = [ex.submit(_amine, s) for s in cs.SITES] if not (OUT / "amine.csv").exists() else []
        futs = [ex.submit(_job, s, b) for s, b in batches]
        n = 0
        with target.open("a") as fh:
            for f in as_completed(futs):
                for row in f.result():
                    fh.write(json.dumps(row) + "\n")
                    n += 1
                fh.flush()
                print(f"  {n}/{len(todo)} cases, {(time.time() - t0) / 60:.1f} min", flush=True)
        if amine_f:
            pd.DataFrame([f.result() for f in amine_f]).to_csv(OUT / "amine.csv", index=False)
    if part:
        print("part done; run merge() after all parts", flush=True)
        return
    merge()


def merge():
    """Append the part files to cases.jsonl (without duplicate keys), rebuild cases.csv, re-apply the water fix."""
    keys = {json.loads(l)["key"] for l in JSONL.read_text().splitlines() if l.strip()}
    for f in sorted(OUT.glob("cases_part*.jsonl")):
        with JSONL.open("a") as fh:
            for l in f.read_text().splitlines():
                if l.strip() and json.loads(l)["key"] not in keys:
                    fh.write(l + "\n")
                    keys.add(json.loads(l)["key"])
        f.rename(f.with_suffix(".merged"))
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    pd.DataFrame(rows).to_csv(OUT / "cases.csv", index=False)
    print("written", OUT / "cases.csv")
    # cases.jsonl holds the freeze-out water of the old accounting; re-apply the recomputed values
    from . import water_fix
    if water_fix.OUT.exists():
        water_fix.apply()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="matrix,rq5,sens")
    ap.add_argument("--part", default=None, help="k/n: share the work between n processes")
    ap.add_argument("--merge", action="store_true", help="merge part files and rebuild cases.csv")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run(a.workers, set(a.only.split(",")), a.part)
