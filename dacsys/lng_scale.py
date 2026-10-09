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

from .fig_paper1 import RES

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


def figure():
    import matplotlib.pyplot as plt
    from . import fig_paper1 as F
    F.rc()
    b = summary()                                # best dryer (freeze-out needs a recuperator approach)
    fig, axes = plt.subplots(1, 2, figsize=(F.DW, 3.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    F.style(ax, grid="y")
    for sb, col, mk, lab in (("13X", F.C_BLUE, "o", "13X"), ("PR:MgMOF74", F.C_ORANGE, "s", "Mg-MOF-74")):
        g = b[b.sorbent == sb].groupby("recup")
        med, lo, hi = g.ratio.median(), g.ratio.quantile(0.1), g.ratio.quantile(0.9)
        for col_pot, filled in (("potential_now", True), ("potential_2050", False)):
            x = g[col_pot].median().reindex(LEVELS)
            ax.errorbar(x, med.reindex(LEVELS), yerr=[med.reindex(LEVELS) - lo.reindex(LEVELS),
                                                       hi.reindex(LEVELS) - med.reindex(LEVELS)],
                        fmt=mk + "-", ms=4, lw=1.0, color=col, mfc=col if filled else F.SURF, capsize=2,
                        label=f"{lab}, {'current' if filled else '2050'} capacity")
    # heat-exchanger cost scenarios (dacsys.extra_runs part A), Mg-MOF-74, current capacity
    xa = RES / "extra_A.csv"
    if xa.exists():
        from .extra_analysis import lng_curve
        e = pd.read_csv(xa)
        e["amine"] = e.place.map(pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base)
        eb = lng_curve(e, ["scenario"])
        for sc, ls, lab in (("yamin", "-.", "Mg-MOF-74, current, correlation of Yamin et al. (652 USD m$^{-2}$)"),
                            ("regen", ":", "Mg-MOF-74, current, regenerators (10 USD m$^{-2}$, doubled $U$)")):
            g = eb[(eb.scenario == sc) & (eb.sorbent == "PR:MgMOF74")].groupby("recup")
            ax.plot(g.pot_now.median().reindex(LEVELS), g.ratio.median().reindex(LEVELS), ls=ls, marker="s", ms=2.5,
                    lw=1.0, color=F.C_ORANGE, alpha=0.8, label=lab)
    ax.axhline(1, color=F.INK, lw=0.8)
    ax.axvspan(103, 142, color=F.NEUTRAL, alpha=0.35, lw=0)
    ax.text(121, 0.55, "Kim et al.\n2050", ha="center", va="bottom", fontsize=5.8, color=F.INK2)
    ax.set_xscale("log")
    ax.set_xlim(3, 200)
    ax.set_xticks([3, 5, 10, 20, 50, 100, 200], ["3", "5", "10", "20", "50", "100", "200"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale("log")
    ax.set_ylim(0.5, 20)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Global potential with LNG cold (Mt CO$_2$ yr$^{-1}$)")
    ax.set_ylabel("Relative cost")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0)
    F.panel(ax, "a", x=-0.12)
    # (b) installed heat-exchanger cost at which Mg-MOF-74 with LNG cold reaches the amine cost, per site
    ax = axes[1]
    F.style(ax, grid="x")
    hp = pd.read_csv(RES / "hx_parity.csv")
    hp = hp[hp.case == "lng"].set_index("place").reindex(F.ORDER)
    y = np.arange(len(hp))[::-1]
    below = hp.flag == "<"
    ax.scatter(hp.parity_cost[~below], y[~below.to_numpy()], s=12, color=F.C_ORANGE, zorder=3,
               label="Mg-MOF-74 with LNG, parity cost")
    if below.any():
        ax.scatter(hp.parity_cost[below], y[below.to_numpy()], s=14, marker="<", color=F.C_ORANGE, zorder=3,
                   label="below 30 USD m$^{-2}$")
    from .extra_analysis import hx_cost
    for v, ls, t in ((hx_cost("base", lng=True), "-", "this study (stainless steel)"), (hx_cost("yamin"), "--", "Yamin et al., installed"),
                     (100.0, ":", "100 USD m$^{-2}$")):
        ax.axvline(v, color=F.INK2, lw=0.8, ls=ls, label=t)
    ax.set_xscale("log")
    ax.set_xlim(20, 1200)
    ax.set_xticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, [F.lab(s_) for s_ in hp.index], fontsize=5.8)
    ax.set_ylim(-0.7, len(hp) - 0.3)
    ax.set_xlabel("Installed heat-exchanger cost at parity (USD m$^{-2}$)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
    F.panel(ax, "b", x=-0.33)
    fig.tight_layout(w_pad=1.0)
    F.save(fig, "fig15_lng")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.workers)
    figure()
    b = summary()
    cols = ["ratio", "lcoc", "cold_GJ", "t_lng", "potential_now", "potential_2050", "cold_price", "T_design"]
    print(b.groupby(["sorbent", "recup"])[cols].median().round(2).to_string())
