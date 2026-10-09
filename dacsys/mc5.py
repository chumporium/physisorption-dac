"""Monte Carlo robustness of the physisorption-vs-amine boundary (spec section 9).

For every site, the architecture chosen in the base run is kept and the uncertain assumptions
are sampled jointly; operating variables (T_ads, recuperator, sizes) are re-optimised per sample.
Three physisorption configurations per site, each against the amine range of the same sample:
  13X_best   : 13X, best architecture without LNG (RQ1 x RQ2)
  13X_lng    : 13X with free LNG cold (Kim et al. reference)
  target     : material target P_KH0.03_Q50_phob_S20000, best architecture without LNG (RQ5)
  prisma     : best real PrISMa MOF of the site, best architecture without LNG (stage 1 + 2)

python -m dacsys.mc5 --only prisma   adds one configuration to an existing run (resumable per config)

python -m dacsys.mc5 [--n 200] [--workers 26] -> results_dac/study5/mc.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import climate_sites as cs
from .levers import Case, Prm, with_

OUT = Path(__file__).resolve().parent.parent / "results_dac" / "study5"
TARGET = "P_KH0.03_Q50_phob_S20000"
_P0 = Prm()


def sample(rng, n):
    """Joint samples of the uncertain assumptions (independent unless stated)."""
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))   # noqa: E731
    u = lambda lo, hi: rng.uniform(lo, hi, n)                           # noqa: E731
    cap_amine = u(600.0, 1500.0)
    return pd.DataFrame(dict(
        dP_contactor=lu(100.0, 1000.0),
        f_gax=u(0.7, 1.4), f_vcr=lu(0.75, 2.5),                          # heat pumps: 300-1000 EUR/kW (Marina et al.)
        sorbent_cost_factor=lu(0.5, 2.0),
        c_ptc=u(150.0, 350.0), c_pv=u(400.0, 900.0), c_batt=u(100.0, 250.0), c_tank=u(15.0, 40.0),
        discount=u(0.05, 0.10),
        cycle_factor=lu(0.5, 2.0),
        gax_derate=u(0.75, 0.95), freeze_loss=u(0.1, 0.4), eta_cap=u(0.7, 0.9),
        amine_capex=cap_amine,
        amine_om=0.03 + (cap_amine - 600.0) / 900.0 * 0.07,         # degradation rises with cost
        amine_energy=u(0.75, 1.25),                                      # ~6 GJ/t (McQueen) .. 9.3 GJ/t (Deutz & Bardow)
        netl_ua_unit=lu(8.6e5, 7.5e7),                                  # heat-exchanger unit, NETL vendor range (91-30 USD/m2)
        f_aux=lu(0.5, 2.0),                                              # fans, dry coolers, contactor, silica gel
        f_vac=lu(0.07, 1.0),                                             # vacuum pumps: Kim et al. (~2 700 $/kW) .. NETL
        f_hx=u(0.75, 1.28),                                              # NETL heat-exchanger correlation, -25/+28 %
    ))


def _prm_kw(s: pd.Series) -> dict:
    return dict(dP_contactor=s.dP_contactor, c_abs_rej=_P0.c_abs_rej * s.f_gax, c_hp_heat=_P0.c_hp_heat * s.f_vcr,
                sorbent_cost_factor=s.sorbent_cost_factor, c_ptc=s.c_ptc, c_pv=s.c_pv,
                c_batt=s.c_batt, c_tank=s.c_tank, discount=s.discount, cycle_factor=s.cycle_factor,
                gax_derate=s.gax_derate, freeze_loss=s.freeze_loss, eta_cap=s.eta_cap,
                amine_capex=(s.amine_capex,) * 3, amine_om=(s.amine_om,) * 3,
                amine_energy_factor=(s.amine_energy,) * 3, netl_ua_unit=s.netl_ua_unit, f_hx=s.f_hx,
                c_fan=_P0.c_fan * s.f_aux, c_vac=_P0.c_vac * s.f_vac, c_rej=_P0.c_rej * s.f_aux,
                c_contactor=_P0.c_contactor * s.f_aux, c_silica_sys=_P0.c_silica_sys * s.f_aux)


def _job(site, configs, samples):
    from . import plant, prisma, study5
    study5.register()
    prisma.register()
    w = cs.load(site)
    rows = []
    for i, s in samples.iterrows():
        kw = _prm_kw(s)
        res = plant.resources(w, with_(Case(site=site), **kw).prm)
        am = plant.amine(w, res, with_(Case(site=site), **kw).prm)["amine_lcoc_base"]
        for name, (sorb, dry, cold) in configs.items():
            r = plant.evaluate(with_(Case(site=site), sorbent=sorb, drying=dry, cold=cold, **kw), w, res)
            rows.append(dict(site=site, sample=i, config=name, lcoc=r.get("lcoc", np.inf), amine=am))
    return rows


def run(n, workers, only=None):
    d = pd.read_csv(OUT / "cases.csv")
    ok = d[d.status == "ok"]
    m = ok[(ok.sorbent == "13X") & ok.lever.isin(["B0", "RQ1x2"])]
    t = ok[(ok.sorbent == TARGET) & ok.lever.isin(["RQ5", "RQ5free"]) & (ok.cold != "lng")]
    pr = ok[ok.lever.isin(["PR1", "PR2"])]
    prl = ok[ok.lever == "PRL"]                                    # stage-2 MOFs with LNG cold
    samples = sample(np.random.default_rng(42), n)
    samples.to_csv(OUT / "mc_samples.csv", index_label="sample")
    jobs = []
    for site in cs.SITES:
        cfg = {}
        for name, x in (("13X_best", m[(m.site == site) & (m.cold != "lng")]),
                        ("13X_lng", m[(m.site == site) & (m.cold == "lng")]),
                        ("target", t[t.site == site]),
                        ("prisma", pr[pr.site == site]),
                        ("prisma_lng", prl[prl.site == site])):
            if len(x) and (only is None or name in only):
                b = x.loc[x.lcoc.idxmin()]
                cfg[name] = (b.sorbent, b.drying, b.cold)
        for k in range(0, n, 10):
            jobs.append((site, cfg, samples.iloc[k:k + 10]))
    part = OUT / "mc_part.csv"
    done = set()
    if part.exists():
        prev = pd.read_csv(part)
        done = set(zip(prev.site, prev["sample"], prev.config))
    jobs = [(st, {k: v for k, v in cfg.items() if not all((st, i, k) in done for i in smp.index)}, smp)
            for st, cfg, smp in jobs]
    jobs = [j for j in jobs if j[1]]
    print(f"{len(jobs)} jobs to run", flush=True)
    from concurrent.futures import as_completed
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            pd.DataFrame(f.result()).to_csv(part, mode="a", header=not part.exists(), index=False)
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)} jobs", flush=True)
    rows = pd.read_csv(part).to_dict("records")
    df = pd.DataFrame(rows)
    df["ratio"] = df.lcoc / df.amine
    df.to_csv(OUT / "mc.csv", index=False)
    summ = df.groupby(["site", "config"])["ratio"].agg(
        p10=lambda x: x.quantile(0.1), median="median", p90=lambda x: x.quantile(0.9),
        p_win=lambda x: (x < 1).mean()).reset_index()
    summ.to_csv(OUT / "mc_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--only", default=None, help="comma-separated configurations")
    a = ap.parse_args()
    run(a.n, a.workers, a.only.split(",") if a.only else None)
