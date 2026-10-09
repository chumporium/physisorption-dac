"""Discussion analyses for Paper 1: energy breakdown, LCOC breakdown, adsorption-temperature study and
Monte Carlo sensitivity. All from results_dac/study5 except the temperature study (new plant evaluations).

python -m dacsys.analysis_paper1 [--tads]   -> docs/paper1/figs/fig8..10, docs/paper1/tables/energy.tex, compare.tex
"""
from __future__ import annotations

import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fig_paper1 as F
from .levers import Prm

P = Prm()
FE, FP = P.crf + P.om_energy, P.crf + P.om_proc
TADS_CSV = F.RES / "tads_study.csv"
COMP = ["Solar heat (PTC, tank)", "PV and battery", "Chiller", "Adsorbent and contactor", "Other process equipment"]
CCOL = [F.C_ORANGE, "#c9a227", F.C_BLUE, F.C_GREEN, F.NEUTRAL]


def configs(d):
    """Best case per site for each configuration (rows indexed by site)."""
    m, p = F.m13(d), F.db(d)
    ok = lambda x: x[x.status == "ok"]                                         # noqa: E731
    return {
        "B0 (13X)": ok(m[m.lever == "B0"]).set_index("site"),
        "13X, no LNG": F.best(m[m.cold != "lng"], ["site"]).set_index("site"),
        "Best MOF, GAX": F.best(p[p.cold == "gax"], ["site"]).set_index("site"),
        "Best MOF, no LNG": F.best(p[p.cold != "lng"], ["site"]).set_index("site"),
        "Best MOF, LNG": F.best(p[p.cold == "lng"], ["site"]).set_index("site"),
        "Reference adsorbent": F.best(F.tgt(d), ["site"]).set_index("site"),
    }


def breakdown(r):
    """LCOC components (USD/t) of one case row; they sum to the LCOC."""
    solar = (P.c_ptc * r.A_ptc_m2 + P.c_tank * r.tank_kWh) * FE / r.co2_t_per_kgs
    pvb = (r.capex_energy * FE / r.co2_t_per_kgs) - solar
    chill = r.capex_cold * FP / r.co2_t_per_kgs
    sorb = r.capex_sorbent * FP / r.co2_t_per_kgs
    other = (r.capex_process - r.capex_cold - r.capex_sorbent) * FP / r.co2_t_per_kgs
    return [solar, pvb, chill, sorb, other]


def amine_split(site):
    """Amine LCOC split into capture plant and energy supply (base case)."""
    from . import climate_sites as cs
    from . import plant
    from .process import amine_energy
    w = cs.load(site)
    res = plant.resources(w, P)
    T_mean = float(w["T0"].mean() - 273.15)
    RH = float(100 * np.clip(plant.p_sat(w["Td"]) / plant.p_sat(w["T0"]), 0, 1).mean())
    heat, elec = (float(x) for x in amine_energy(T_mean, RH))
    H = len(w["T0"])
    ran, eb = plant.energy_block(np.full(H, heat / 3.6e-3), np.full(H, elec / 3.6e-3), np.ones(H, bool), res, P)
    hours = ran.sum(0)
    plant_ann = P.amine_capex[1] * 8760 * 0.9 * (P.crf + P.amine_om[1])
    lcoc = (plant_ann + eb["capex"] * FE) / np.maximum(hours, 1)
    k = int(np.argmin(np.where(hours >= plant.MIN_HOURS, lcoc, np.inf)))
    return plant_ann / hours[k], eb["capex"][k] * FE / hours[k]


# ------------------------------------------------------------------ energy table
def energy_table(d, a):
    rows = []
    for name, x in configs(d).items():
        med = x.median(numeric_only=True)
        rows.append([name, f"{med.heat_GJ:.1f}", f"{med.elec_GJ:.1f}", f"{med.elec_fan_GJ:.1f}",
                     f"{med.elec_vac_GJ:.1f}", f"{med.elec_cold_GJ:.1f}", f"{med.cold_GJ:.0f}",
                     f"{med.wcap_mmolg:.2f}", f"{med.lcoc:,.0f}".replace(",", "{,}")])
    rows.append(["Amine DAC", f"{a.amine_heat_GJ.median():.1f}", f"{a.amine_elec_GJ.median():.1f}", "--", "--", "--",
                 "--", "--", f"{a.amine_lcoc_base.median():,.0f}".replace(",", "{,}")])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    (F.TAB / "energy.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{Energy demand, working capacity and LCOC of the main "
        "configurations (median over 21 sites). Electricity for cooling includes the compressor or solution pump and "
        "the dry-cooler fans. At each site the components add up to the total; their medians do not.}\n\\label{tab:energy}\n"
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        " & Heat & \\multicolumn{4}{c}{Electricity (GJ t$^{-1}$)} & Cold & $WC$ & LCOC \\\\\n\\cmidrule(lr){3-6}\n"
        "Configuration & (GJ t$^{-1}$) & total & fan & vacuum & cooling & (GJ t$^{-1}$) & (mmol g$^{-1}$) & (USD t$^{-1}$) \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n", encoding="utf-8")
    return rows


# ------------------------------------------------------------------ Fig 8 LCOC breakdown
SITES3 = ["Singapore", "London", "Altiplano"]


def fig_breakdown(d, a):
    cf = configs(d)
    names = list(cf) + ["Amine DAC"]
    fig, axes = plt.subplots(1, 3, figsize=(F.DW, 2.9), sharey=True)
    out = {}
    for ax, site in zip(axes, SITES3):
        F.style(ax, grid="x")
        y = np.arange(len(names))[::-1]
        for yi, n in zip(y, names):
            if n == "Amine DAC":
                plant_c, energy_c = amine_split(site)
                parts = [plant_c, energy_c]
                left = 0.0
                for v, col in zip(parts, (F.MUTED, F.C_PURPLE)):
                    ax.barh(yi, v, left=left, color=col, height=0.62, edgecolor=F.SURF, lw=0.5)
                    left += v
                out[(site, n)] = parts
            else:
                r = cf[n].loc[site]
                parts = breakdown(r)
                left = 0.0
                for v, col in zip(parts, CCOL):
                    ax.barh(yi, v, left=left, color=col, height=0.62, edgecolor=F.SURF, lw=0.5)
                    left += v
                out[(site, n)] = parts
            ax.text(left + 80, yi, f"{left:,.0f}", va="center", fontsize=6, color=F.INK)
        ax.set_yticks(y, names)
        ax.grid(axis="y", visible=False)
        ax.set_title(F.lab(site), loc="left", fontsize=7, color=F.INK)
        ax.set_xlabel("LCOC (USD t$^{-1}$)")
    xmax = 1.2 * max(sum(v) for v in out.values())
    for ax in axes:
        ax.set_xlim(0, xmax)
    from matplotlib.patches import Patch
    h = [Patch(color=c, label=l) for c, l in zip(CCOL, COMP)] + \
        [Patch(color=F.MUTED, label="Amine capture plant"), Patch(color=F.C_PURPLE, label="Amine energy supply")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=6.3, bbox_to_anchor=(0.55, -0.1))
    fig.tight_layout(w_pad=0.6)
    F.save(fig, "fig8_cost_breakdown")
    return out


# ------------------------------------------------------------------ Fig 9 adsorption temperature study
def tads_study(site="London"):
    from . import climate_sites as cs
    from . import plant, prisma
    from .levers import Case, with_
    prisma.register(["MgMOF74"])
    w = cs.load(site)
    res = plant.resources(w, P)
    rows = []
    for sorb in ("13X", "PR:MgMOF74"):
        for cold, Ts in (("vcr", (225.0, 230.0, 235.0, 241.0, 245.0, 250.0, 255.0, 260.0, 270.0, 280.0)),
                         ("gax", (241.0, 245.0, 250.0, 255.0, 260.0, 270.0, 280.0))):
            for T in Ts:
                case = with_(Case(site=site), sorbent=sorb, drying="cond_silica", cold=cold)
                r = plant.evaluate(case, w, res, T_list=[T])
                if r.get("status") != "ok":
                    continue
                s = pd.Series(r)
                rows.append(dict(sorbent=sorb, cold=cold, T=T, lcoc=r["lcoc"], cold_GJ=r["cold_GJ"],
                                 wcap=r["wcap_mmolg"], elec_cold=r["elec_cold_GJ"],
                                 **dict(zip(COMP, breakdown(s)))))
    t = pd.DataFrame(rows)
    t.to_csv(TADS_CSV, index=False)
    return t


def fig_tads(a, site="London"):
    t = pd.read_csv(TADS_CSV)
    am = a.at[site, "amine_lcoc_base"]
    fig, axes = plt.subplots(1, 2, figsize=(F.DW, 2.6))
    ax = axes[0]
    F.style(ax, grid="y")
    for (sorb, cold), g in t.groupby(["sorbent", "cold"]):
        col = F.C_BLUE if cold == "vcr" else F.C_ORANGE
        ls = "-" if sorb == "13X" else "--"
        mk = "o" if sorb == "13X" else "s"
        ax.plot(g["T"], g.lcoc, ls=ls, marker=mk, ms=3, color=col, lw=1.2,
                label=f"{'13X' if sorb == '13X' else 'Mg-MOF-74'}, {'heat pump' if cold == 'vcr' else 'GAX'}")
    ax.axhline(am, color=F.INK, lw=0.8)
    ax.text(279, am * 1.05, "amine", ha="right", fontsize=6.3, color=F.INK2)
    ax.set_yscale("log")
    ax.set_yticks([300, 500, 1000, 2000, 5000], ["300", "500", "1000", "2000", "5000"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Design adsorption temperature (K)")
    ax.set_ylabel("LCOC (USD t$^{-1}$)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    F.panel(ax, "a", x=-0.12)
    ax = axes[1]
    F.style(ax, grid="y")
    g = t[(t.sorbent == "PR:MgMOF74") & (t.cold == "vcr")].sort_values("T")
    ax.stackplot(g["T"], [g[c] for c in COMP], colors=CCOL, labels=COMP, alpha=0.9)
    ax.set_xlabel("Design adsorption temperature (K)")
    ax.set_ylabel("LCOC components (USD t$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("Mg-MOF-74, silica gel and heat pump", loc="left", fontsize=7, color=F.INK)
    F.panel(ax, "b", x=-0.12)
    fig.tight_layout(w_pad=1.2)
    F.save(fig, "fig9_tads")
    return t


# ------------------------------------------------------------------ Fig 10 Monte Carlo sensitivity
LABELS = {"dP_contactor": "Contactor pressure drop", "f_gax": "GAX chiller cost", "f_vcr": "Heat pump cost",
          "sorbent_cost_factor": "Adsorbent price", "c_ptc": "PTC cost", "c_pv": "PV cost", "c_batt": "Battery cost",
          "c_tank": "Storage tank cost", "discount": "Discount rate", "cycle_factor": "Cycle rate",
          "gax_derate": "GAX COP factor", "freeze_loss": "Freeze-out heat loss", "eta_cap": "Capture fraction",
          "amine_capex": "Amine capital and O&M cost", "amine_energy": "Amine energy demand",
          "netl_ua_unit": "Heat-exchanger unit size", "f_hx": "Heat-exchanger cost", "f_aux": "Fan and other equipment cost", "f_vac": "Vacuum pump cost"}


def fig_sens():
    mc = pd.read_csv(F.RES / "mc.csv")
    s = pd.read_csv(F.RES / "mc_samples.csv").set_index("sample")
    mc = mc[np.isfinite(mc.ratio)]
    cfgs = [("prisma", "Best MOF, no LNG", F.C_BLUE), ("prisma_lng", "Best MOF, LNG", F.C_PURPLE),
            ("13X_lng", "13X, LNG", F.C_ORANGE), ("target", "Reference adsorbent", F.C_GREEN)]
    cfgs = [c for c in cfgs if c[0] in set(mc.config)]
    res = {}
    for c, _, _ in cfgs:
        x = mc[mc.config == c].copy()
        x["z"] = np.log(x.ratio) - x.groupby("site").ratio.transform(lambda v: np.log(v).mean())
        x = x.join(s, on="sample")
        res[c] = {k: x[k].rank().corr(x.z.rank()) for k in LABELS}
    df = pd.DataFrame(res)
    order = df.abs().max(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(F.SW * 1.45, 3.7))
    F.style(ax, grid="x")
    y = np.arange(len(order))
    for i, (c, name, col) in enumerate(cfgs):
        w = 0.8 / len(cfgs)
        ax.barh(y + (i - (len(cfgs) - 1) / 2) * w, df.loc[order, c], height=w * 0.95, color=col, label=name)
    ax.axvline(0, color=F.INK, lw=0.7)
    ax.set_yticks(y, [LABELS[k] for k in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Spearman correlation with relative cost")
    ax.legend(frameon=False, loc="lower right", fontsize=6.3)
    F.save(fig, "fig10_sensitivity")
    df.to_csv(F.RES / "mc_sensitivity.csv")
    return df


def main(run_tads):
    F.rc()
    d, a = F.load()
    e = energy_table(d, a)
    for r in e:
        print(r)
    out = fig_breakdown(d, a)
    for k, v in out.items():
        tot = sum(v)
        print(k, round(tot), [f"{x / tot:.0%}" for x in v])
    if run_tads or not TADS_CSV.exists():
        tads_study()
    t = fig_tads(a)
    print(t.groupby(["sorbent", "cold"]).apply(lambda g: g.loc[g.lcoc.idxmin(), ["T", "lcoc", "cold_GJ", "wcap"]]))
    print(fig_sens().round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tads", action="store_true")
    main(ap.parse_args().tads)
