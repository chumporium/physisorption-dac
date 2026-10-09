"""Robustness and extension runs for the merged paper (24 September 2026).

  A  hx     heat-exchanger cost and recuperator heat transfer (incl. a cheap packed-bed regenerator scenario):
            LNG cold-recovery curve (13X, Mg-MOF-74) and the best configuration without LNG at every site
  B  pore   pore filling by water (S-shaped water isotherm) of hydrophobic adsorbents at all 21 sites
  C  hpeff  isentropic efficiency of the heat pump (COP maps rebuilt for 0.6 and 0.8)
  D  silica silica-gel regeneration by an electric heat pump (and with heat recovery) for the 75 stage-2 MOFs
  E  lngt   LNG cold at twelve large LNG import terminals (ERA5 2018 at the terminal)
  F  years  the best configurations and amine DAC with the weather of 2015, 2019, 2020 and 2021
  G  silica silica-gel regeneration with heat recovery only (solar heat)
  H  oat    one-at-a-time variation of the design assumptions not covered elsewhere, for four configurations

python -m dacsys.extra_runs [--parts A,B,C,D,E,F] [--workers 4]  -> results_dac/study5/extra_<part>.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .fig_paper1 import RES


def out_path(part):
    return RES / f"extra_{part}.csv"


TARGET = "P_KH0.03_Q50_phob_S20000"
# heat-exchanger scenarios: NETL correlation (base) at the smallest and largest unit of the vendor range and without the
# cryogenic material factor; the purchase-cost correlation of Yamin et al. per 10 000 m2 unit, installed (previous
# basis) and purchase cost only; fixed installed costs of 100 $/m2 (Kim et al.) and 30 $/m2; packed-bed regenerators
# (assumed 10 $/m2 and doubled U)
HX = {"base": {}, "ua_min": dict(netl_ua_unit=8.6e5), "ua_max": dict(netl_ua_unit=7.5e7),
      "cryo_cs": dict(f_mat_cryo=1.0), "cryo_hi": dict(f_mat_cryo=2.93), "yamin": dict(hx_cost_model="unit"),
      "purchase": dict(hx_cost_model="unit", hx_install=1.0), "hx100": dict(c_hx_area=100.0),
      "hx30": dict(c_hx_area=30.0), "regen": dict(c_hx_area=10.0, U_air=0.07)}
DT = {"3 K": 3.0, "8 K": 8.0, "20 K": 20.0, "none": None}
PORE = {TARGET: (None, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95),
        "P_KH0.01_Q50_phob_S20000": (None, 0.8, 0.9, 0.95), "P_KH0.1_Q50_phob_S20000": (None, 0.8, 0.9, 0.95)}
AMB3 = (("none", "ambient"), ("cond_silica", "ambient"), ("freeze", "vcr"), ("cond_silica", "vcr"))
ETAS = (0.6, 0.7, 0.8)
SILICA = {"base": {}, "hp3": dict(silica_regen="hp"), "hp3_f1": dict(silica_regen="hp", silica_regen_factor=1.0)}
SILICA2 = {"f1": dict(silica_regen_factor=1.0)}              # part G: heat recovery only, solar heat
YEARS = (2015, 2019, 2020, 2021)
# part H: low / high values of the design assumptions that are not varied in the Monte Carlo analysis or in parts A-G
OAT = {"dewpoint_dry": (-50.0, -30.0), "T_coil1": (1.0, 6.0), "dT_evap": (3.0, 8.0), "U_coil": (0.0445, 0.0958),
       "reg_area_factor": (1.0, 2.0), "dP_coil": (50.0, 200.0), "dP_silica": (125.0, 500.0), "dP_recup": (75.0, 300.0),
       "dP_regen": (150.0, 600.0), "cp_contactor": (0.15, 1.0), "dT_hex_heat": (10.0, 30.0), "dT_hex_cool": (2.5, 10.0),
       "sil_dq": (0.05, 0.15), "sil_cycles": (1.0, 4.0), "gax_pump_frac": (0.0025, 0.01),
       "fan_rej_frac": (0.01, 0.04), "storage_loss_h": (0.0, 0.004), "water_recovery_silica": (0.7, 1.0)}


def vcr_map_path(eta):
    """COP map of the heat pump for an isentropic efficiency (0.70 = the base map)."""
    base = Path(__file__).with_name("vcr_map.npz")
    return base if abs(eta - 0.70) < 1e-9 else base.with_name(f"vcr_map_eta{eta:.2f}.npz")


def build_vcr_maps():
    from . import vcr
    orig_eta, orig_out = vcr.ETA_IS, vcr.OUT
    for eta in ETAS:
        path = vcr_map_path(eta)
        if path.exists():
            continue
        vcr.ETA_IS, vcr.OUT = eta, path
        t = time.time()
        vcr.build()
        print(f"COP map eta={eta} built in {time.time() - t:.0f} s", flush=True)
    vcr.ETA_IS, vcr.OUT = orig_eta, orig_out
    vcr._load.cache_clear()


# ------------------------------------------------------------------ task lists
def _best_configs():
    from . import fig_paper1 as F
    d, a = F.load()
    nl, ln = F.db_best(d), F.db_best(d, lng=True)
    m = F.m13(d)
    x13 = F.best(m[m.cold != "lng"], ["site"]).set_index("site")
    tg = F.best(F.tgt(d), ["site"]).set_index("site")
    return nl, ln, x13, tg


def tasks(part):
    from .climate_sites import SITES, TERMINALS
    nl, ln, x13, tg = _best_configs()
    out = []

    def t(**k):
        base = dict(part=part, scenario="base", place=None, year=2018, sorbent=None, drying=None, cold=None,
                    recup="opt", prm={}, dT=None, eta=0.7, kind="site")
        base.update(k)
        out.append(base)

    if part == "A":
        for sc, kw in HX.items():
            for site in SITES:
                for sb in ("13X", "PR:MgMOF74"):
                    for dr in ("freeze", "cond"):
                        for lab, dt in DT.items():
                            t(scenario=sc, place=site, sorbent=sb, drying=dr, cold="lng", recup=lab, prm=kw, dT=[dt])
                for dr, co in (("freeze", "vcr"), ("cond_silica", "ambient"), ("none", "vcr"), ("cond", "vcr"),
                               ("cond_silica", "vcr")):
                    t(scenario=sc, place=site, sorbent=nl.at[site, "sorbent"], drying=dr, cold=co, prm=kw)
                t(scenario=sc, place=site, sorbent="13X", drying="freeze", cold="vcr", prm=kw)
    elif part == "B":
        for sb, steps in PORE.items():
            for st in steps:
                name = sb if st is None else f"PF:{sb}:{st:g}"
                for site in SITES:
                    for dr, co in AMB3:
                        t(scenario="none" if st is None else f"{st:g}", place=site, sorbent=name, drying=dr, cold=co)
    elif part == "C":
        for eta in ETAS:
            for site in SITES:
                for dr, co in (("freeze", "vcr"), ("none", "vcr"), ("cond", "vcr"), ("cond_silica", "vcr")):
                    t(scenario=f"eta{eta:.1f}", place=site, sorbent=nl.at[site, "sorbent"], drying=dr, cold=co, eta=eta)
                t(scenario=f"eta{eta:.1f}", place=site, sorbent="13X", drying="freeze", cold="vcr", eta=eta)
    elif part == "D":
        from . import fig_paper1 as F
        d, _ = F.load()
        mofs = sorted(F.db(d).sorbent.unique())
        for sc, kw in SILICA.items():
            if sc == "base":
                continue
            for site in SITES:
                for sb in mofs:
                    t(scenario=sc, place=site, sorbent=sb, drying="cond_silica", cold="ambient", prm=kw)
                t(scenario=sc, place=site, sorbent=nl.at[site, "sorbent"], drying="cond_silica", cold="vcr", prm=kw)
    elif part == "G":
        from . import fig_paper1 as F
        d, _ = F.load()
        for sc, kw in SILICA2.items():
            for site in SITES:
                for sb in sorted(F.db(d).sorbent.unique()):
                    t(scenario=sc, place=site, sorbent=sb, drying="cond_silica", cold="ambient", prm=kw)
    elif part == "H":
        cfgs = {}
        for site in SITES:
            cfgs[site] = [("best_noLNG", nl.at[site, "sorbent"], nl.at[site, "drying"], nl.at[site, "cold"]),
                          ("best_LNG", ln.at[site, "sorbent"], ln.at[site, "drying"], ln.at[site, "cold"]),
                          ("target", tg.at[site, "sorbent"], tg.at[site, "drying"], tg.at[site, "cold"]),
                          ("13X_B0", "13X", "cond_silica", "gax")]
        for site in SITES:
            for tag, sb, dr, co in cfgs[site]:
                t(scenario="base", place=site, sorbent=sb, drying=dr, cold=co, config=tag)
                for par, vals in OAT.items():
                    for lvl, v in zip(("low", "high"), vals):
                        t(scenario=f"{par}:{lvl}", place=site, sorbent=sb, drying=dr, cold=co, config=tag, prm={par: v})
    elif part == "E":
        for term in TERMINALS:
            for sb in ("13X", "PR:MgMOF74"):
                for dr in ("freeze", "cond"):
                    for lab, dt in DT.items():
                        t(kind="term", place=term, sorbent=sb, drying=dr, cold="lng", recup=lab, dT=[dt])
            t(kind="term", place=term, sorbent="amine")
    elif part == "F":
        for yr in YEARS:
            for site in SITES:
                for tag, row in (("best_noLNG", nl.loc[site]), ("best_LNG", ln.loc[site]), ("13X", x13.loc[site]),
                                 ("target", tg.loc[site])):
                    t(kind="year", scenario=tag, place=site, year=yr, sorbent=row.sorbent, drying=row.drying,
                      cold=row.cold)
                t(kind="year", scenario="amine", place=site, year=yr, sorbent="amine")
    return out


# ------------------------------------------------------------------ worker
_CACHE: dict = {}
_REG = False


def _weather(kind, place, year):
    from . import climate_sites as cs
    from . import plant
    from .levers import Prm
    key = (kind, place, year)
    if key not in _CACHE:
        if kind == "site" and year == 2018:
            w = cs.load(place)
        elif kind == "term":
            lat, lon = cs.TERMINALS[place]
            w = cs.load_point(place, lat, lon, year)
        else:
            lat, lon = cs.SITES[place][:2]
            w = cs.load_point(place, lat, lon, year)
        _CACHE[key] = (w, plant.resources(w, Prm()))
    return _CACHE[key]


def _sorbent(name):
    from . import material_sens as M
    from . import sorbents
    from dataclasses import replace
    if name.startswith("PF:"):
        _, base, st = name.split(":")
        try:
            sorbents.get(name)
        except KeyError:
            sorbents.register(replace(M.perturb(sorbents.get(base), "h2o_step", float(st)), name=name))
    return name


def _set_eta(eta):
    from . import vcr
    path = vcr_map_path(eta)
    if vcr.OUT != path:
        vcr.OUT = path
        vcr._load.cache_clear()


def _job(batch):
    global _REG
    from . import plant, prisma, study5
    from .levers import Case, with_
    if not _REG:
        study5.register()
        prisma.register()
        _REG = True
    rows = []
    for tk in batch:
        w, res = _weather(tk["kind"], tk["place"], tk["year"])
        rec = {k: v for k, v in tk.items() if k not in ("prm", "dT")}
        if tk["sorbent"] == "amine":
            r = plant.amine(w, res)
            rec.update(status="ok", **{k: r[k] for k in r if k.startswith("amine")})
            rows.append(rec)
            continue
        _set_eta(tk["eta"])
        name = _sorbent(tk["sorbent"])
        case = with_(Case(site=tk["place"]), sorbent=name, drying=tk["drying"], cold=tk["cold"],
                     lever="X" + tk["part"], **tk["prm"])
        r = plant.evaluate(case, w, res, dT_list=tk["dT"])
        for k in ("status", "lcoc", "heat_GJ", "heat_sil_GJ", "elec_GJ", "elec_cold_GJ", "cold_GJ", "T_design",
                  "wcap_mmolg", "water_per_co2", "dT_rec", "capex_process", "co2_t_per_kgs"):
            rec[k] = r.get(k, np.nan)
        rows.append(rec)
    _set_eta(0.7)
    return rows


def run(parts, workers, places=None):
    """places: compute only the tasks of these (added) sites and replace their rows in the existing part files."""
    from . import climate_sites as cs
    if "C" in parts:
        build_vcr_maps()
    if "E" in parts and not places:
        for term, (lat, lon) in cs.TERMINALS.items():
            cs.fetch_point(term, lat, lon, 2018)
        print("terminal weather ready", flush=True)
    if "F" in parts:
        for yr in YEARS:
            for site, (lat, lon, _, _) in cs.SITES.items():
                if not places or site in places:
                    cs.fetch_point(site, lat, lon, yr)
        print("multi-year weather ready", flush=True)
    for part in parts:
        old = None
        if places:
            if part == "E":                                   # LNG terminals: not a site
                continue
            if out_path(part).exists():
                old = pd.read_csv(out_path(part))
                old = old[~old.place.isin(places)]
        elif out_path(part).exists():
            print("part", part, "already done", flush=True)
            continue
        tk = tasks(part)
        if places:
            tk = [x for x in tk if x["place"] in places]
        tk.sort(key=lambda x: (str(x["sorbent"]), x["eta"], x["kind"], str(x["place"]), x["year"]))
        batches = [tk[i:i + 30] for i in range(0, len(tk), 30)]
        t0, rows = time.time(), []
        print(f"part {part}: {len(tk)} cases in {len(batches)} batches", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(_job, b) for b in batches]
            for k, f in enumerate(as_completed(futs)):
                rows += f.result()
                if (k + 1) % 10 == 0 or k + 1 == len(batches):
                    print(f"  part {part}: {k + 1}/{len(batches)} batches, {(time.time() - t0) / 60:.1f} min", flush=True)
        new = pd.DataFrame(rows)
        (new if old is None else pd.concat([old, new], ignore_index=True)).to_csv(out_path(part), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,C,D,E,F,G,H")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--places", default=None, help="comma-separated added sites (only these are computed)")
    a = ap.parse_args()
    run(a.parts.split(","), a.workers, a.places.split(",") if a.places else None)
