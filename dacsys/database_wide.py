"""Database-wide answers to the research questions for Paper 1: distributions over all screened MOFs instead of the
best MOF only. Stage 1: 1,115 MOFs x 8 configurations x 5 sites; stage 2: 79 MOFs x 14 configurations x 21 sites.
All from results_dac/study5/cases.csv (no new plant runs).

python -m dacsys.database_wide  -> docs/paper1/tables/database.tex, docs/paper1/figs/fig12_database.pdf|png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fig_paper1 as F
from .analysis_paper1 import breakdown

TIE = 1.005                      # cost ratios within 0.5 % are treated as equal (chiller hardly used)


def load():
    c = pd.read_csv(F.RES / "cases.csv", low_memory=False)
    a = pd.read_csv(F.RES / "amine.csv").set_index("site").amine_lcoc_base
    c["ratio"] = c.lcoc / c.site.map(a)
    c["amine"] = c.site.map(a)
    ok = c[c.status == "ok"]
    s1 = ok[ok.lever.isin(["PR1", "DB1"])].copy()
    s1["cfg"] = s1.drying + "/" + s1.cold
    s2 = ok[ok.lever.isin(["PR2", "PRL"])]
    k = pd.read_csv(F.RES / "sites_koppen.csv").set_index("site")
    n_s1 = c[c.lever.isin(["PR1", "DB1"])].sorbent.nunique()
    return s1, s2, k, n_s1


def stats(s1, s2, k, n_s1):
    """Everything the text and the table need: stage 1, stage 2 (per MOF) and the best MOF at each site."""
    r = stage1(s1, n_s1)
    r.update(stage2(s2, k))
    r["best"] = stage2(s2.assign(sorbent="best"), k)
    r["full"] = full(k)
    return r


def full(k):
    """All prescreened MOFs at all 21 sites with the two no-LNG architectures of study5 group db3."""
    from .db3_check import full_pairs
    b = full_pairs()
    if not (b.lever == "DB3").any():
        return None
    nl = b.set_index(["sorbent", "site"]).ratio
    per = nl.unstack("site")
    per = per[per.notna().all(axis=1)]
    td, dni = k.Td_mean_2018_C, k.DNI_kWh_m2_yr
    return dict(nl=nl, below1=int((nl < 1).sum()), mofs=b.sorbent.nunique(), n_all20=len(per),
                rho_td=per.apply(lambda x: x.rank().corr(td[x.index].rank()), axis=1),
                rho_dni=per.apply(lambda x: x.rank().corr(dni[x.index].rank()), axis=1))


from .study5 import DB1_ARCH, PR_ARCH  # noqa: E402
N_S1 = len(PR_ARCH) + len(DB1_ARCH)


def stage1(s1, n_s1):
    """1,115 MOFs x stage-1 configurations x 5 sites (columns 'drying/cold')."""
    r = {}
    p = s1.pivot_table(index=["sorbent", "site"], columns="cfg", values="ratio", aggfunc="min")
    cold = {c: c.split("/")[1] for c in p.columns}
    by = lambda co: [c for c in p.columns if cold[c] == co]                   # noqa: E731
    r["s1_pairs"] = len(p)
    r["s1_mofs"], r["s1_feasible"] = n_s1, s1.sorbent.nunique()
    # drying with the heat pump: best dryer of stage 1 (freeze-out, silica gel) against no dryer
    dried = [c for c in by("vcr") if not c.startswith("none/")]
    fz = pd.DataFrame({"dry": p[dried].min(axis=1), "none": p["none/vcr"]}).dropna()
    r["s1_dry_n"] = len(fz)
    r["s1_dry_share"] = float((fz["dry"] < fz["none"]).mean())
    r["s1_dry_med"] = float((fz["dry"] / fz["none"] - 1).median() * 100)
    # GAX against heat pump with the same dryers (stage 1 has no silica gel with GAX)
    same = [d for d in ("none", "freeze") if f"{d}/gax" in p and f"{d}/vcr" in p]
    gax = p[[f"{d}/gax" for d in same]].min(axis=1)
    hp = p[[f"{d}/vcr" for d in same]].min(axis=1)
    gh = pd.DataFrame({"gax": gax, "hp": hp}).dropna()
    r["s1_gh"] = gh["gax"] / gh["hp"]
    r["s1_gh_n"] = len(gh)
    r["s1_gax_cheaper"] = int((r["s1_gh"] * TIE < 1).sum())
    nl = p[[c for c in p.columns if cold[c] != "lng"]].min(axis=1).dropna()
    ln = p[by("lng")].min(axis=1).dropna()
    r["s1_nl"], r["s1_ln"] = nl, ln
    r["s1_nl_min"], r["s1_nl_below1"] = float(nl.min()), int((nl < 1).sum())
    r["s1_ln_share"] = float((ln < 1).mean())
    lw = ln.unstack("site")
    r["s1_ln_all5"], r["s1_ln_mofs"] = int((lw.lt(1).sum(axis=1) == 5).sum()), len(lw)
    return r


def stage2(s2, k):
    """Per MOF (or per 'best' when all sorbents are merged) and site, with all 14 configurations."""
    r = {}
    r["s2_mofs"] = s2.sorbent.nunique()
    cell = s2.groupby(["sorbent", "site", "cold", "drying"]).ratio.min().unstack("drying")
    r["dry"] = {}
    for co in ("vcr", "gax", "lng"):
        x = cell.xs(co, level="cold")
        ch = (x.drop(columns="none").min(axis=1) / x["none"] - 1).dropna() * 100
        which = x.drop(columns="none").loc[ch.index].idxmin(axis=1)
        r["dry"][co] = dict(ch=ch, med=float(ch.median()), p10=float(ch.quantile(.1)), p90=float(ch.quantile(.9)),
                            share=float((ch < 0).mean()), freeze=float((which == "freeze").mean()))
    bc = s2[s2.cold != "lng"].groupby(["sorbent", "site", "cold"]).ratio.min().unstack("cold")
    nl2 = bc.min(axis=1)
    near = bc.le(nl2 * TIE, axis=0)                                   # options within 0.5 % of the best
    r["s2_pairs"] = len(nl2)
    r["cold_vcr"] = float((near.vcr & ~near.gax).mean() + (near.vcr & near.gax & ~near.ambient).mean())
    r["cold_amb"] = float((near.ambient & ~near.vcr).mean())
    r["cold_gax_strict"] = int((near.gax & ~near.vcr & ~near.ambient).sum())
    r["cold_gax_tie"] = int((near.gax & near.vcr).sum())
    gh2 = (bc.gax / bc.vcr).dropna()
    r["s2_gh"] = gh2
    r["gh_med"], r["gh_p10"], r["gh_p90"] = (float(gh2.median()), float(gh2.quantile(.1)), float(gh2.quantile(.9)))
    ln2 = s2[s2.cold == "lng"].groupby(["sorbent", "site"]).ratio.min()
    r["s2_nl"], r["s2_ln"] = nl2, ln2
    r["s2_nl_min"], r["s2_nl_med"], r["s2_nl_below1"] = float(nl2.min()), float(nl2.median()), int((nl2 < 1).sum())
    r["s2_ln_med"], r["s2_ln_max"], r["s2_ln_share"] = float(ln2.median()), float(ln2.max()), float((ln2 < 1).mean())
    for co in ("vcr", "gax", "lng"):
        b = F.best(s2[s2.cold == co], ["sorbent", "site"])
        r[f"T_{co}"] = b.T_design.quantile([.25, .5, .75]).tolist()
    fv = F.best(s2[(s2.cold == "vcr") & (s2.drying == "freeze")], ["sorbent", "site"])
    r["fv"] = fv
    r["cold_cv"] = float(fv.groupby("site").cold_GJ.agg(lambda v: v.std() / v.mean()).median())
    ld = fv[fv.site == "London"]
    r["ld_wc"], r["ld_cold"] = (ld.wcap_mmolg.min(), ld.wcap_mmolg.max()), (ld.cold_GJ.min(), ld.cold_GJ.max())
    bn = F.best(s2[s2.cold != "lng"], ["sorbent", "site"])
    sh = np.array([breakdown(x)[3] / x.lcoc for _, x in bn.iterrows()])
    r["sorb_med"], r["sorb_p10"], r["sorb_p90"] = (float(np.median(sh)), float(np.quantile(sh, .1)),
                                                   float(np.quantile(sh, .9)))
    r["sorb_min"], r["sorb_max"] = float(sh.min()), float(sh.max())
    td, dni = k.Td_mean_2018_C, k.DNI_kWh_m2_yr
    rr = nl2.unstack("site")
    r["rho_td"] = rr.apply(lambda x: x.rank().corr(td[x.index].rank()), axis=1)
    r["rho_dni"] = rr.apply(lambda x: x.rank().corr(dni[x.index].rank()), axis=1)
    lc = bn.set_index(["sorbent", "site"]).lcoc.unstack("site")
    r["rho_lc_dni"] = lc.apply(lambda x: x.rank().corr(dni[x.index].rank()), axis=1)
    r["best_site"] = rr.idxmin(axis=1).value_counts().to_dict()
    need = (bn.lcoc - bn.amine) / bn.water_per_co2
    r["need"], r["water"] = need, bn.water_per_co2
    return r


# ------------------------------------------------------------------ table
def table(r):
    b = r["best"]

    def m(x, f):
        """Number with a typeset minus sign."""
        s = f.format(x)
        return s.replace("-", "$-$", 1) if s.startswith("-") else s

    def rng(lo, hi, f):
        sep = " to " if lo < 0 else "--"
        return f"{m(lo, f)}{sep}{m(hi, f)}"

    def q(v, f="{:.2f}"):
        """Median (10th--90th percentile) over MOF--site pairs."""
        v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
        return f"{m(v.median(), f)} ({rng(v.quantile(.1), v.quantile(.9), f)})"

    def qb(v, f="{:.2f}"):
        """Best MOF: median (range) over the 21 sites, or the single value of a site-level statistic."""
        v = pd.Series(v).replace([np.inf, -np.inf], np.nan).dropna()
        return m(v.iloc[0], f) if len(v) == 1 else f"{m(v.median(), f)} ({rng(v.min(), v.max(), f)})"

    def dry(x):
        return f"{m(x['med'], '{:+.0f}')} ({rng(x['p10'], x['p90'], '{:+.0f}')})"

    def n(x):
        return f"{x:,}".replace(",", "{,}")

    d, db = r["dry"], b["dry"]
    fu = r.get("full")

    def fq(key, f="{:.2f}"):
        return q(fu[key], f) if fu else "--"

    rows = [
        ("Drying", "Cost change by the best dryer, heat pump (\\%)",
         m(r['s1_dry_med'], '{:+.0f}') + '$^a$', dry(d["vcr"]), m(db['vcr']['med'], '{:+.0f}')),
        ("", "Cost change by the best dryer, GAX (\\%)", "--", dry(d["gax"]), m(db['gax']['med'], '{:+.0f}')),
        ("", "Cost change by the best dryer, LNG (\\%)", "--", dry(d["lng"]), m(db['lng']['med'], '{:+.0f}')),
        ("", "Pairs where drying lowers the cost, heat pump (\\%)", f"{r['s1_dry_share'] * 100:.1f}$^a$",
         f"{d['vcr']['share'] * 100:.1f}", f"{db['vcr']['share'] * 100:.0f}"),
        ("", "Freeze-out is the best dryer, heat pump / GAX (\\%)", "--",
         f"{d['vcr']['freeze'] * 100:.0f} / {d['gax']['freeze'] * 100:.0f}",
         f"{db['vcr']['freeze'] * 100:.0f} / {db['gax']['freeze'] * 100:.0f}"),
        ("Cold source", "Cost with GAX / cost with heat pump", q(r["s1_gh"]), q(r["s2_gh"]), qb(b["s2_gh"])),
        ("", "Pairs with GAX cheaper than heat pump", f"{r['s1_gax_cheaper']} of {n(r['s1_gh_n'])}",
         f"{r['cold_gax_strict']} of {n(r['s2_pairs'])}", f"{b['cold_gax_strict']} of {b['s2_pairs']}"),
        ("", "Relative cost without LNG", q(r["s1_nl"]), q(r["s2_nl"]), fq("nl"), qb(b["s2_nl"])),
        ("", "Pairs below amine cost without LNG", n(r["s1_nl_below1"]), n(r["s2_nl_below1"]),
         n(fu["below1"]) if fu else "--", f"{b['s2_nl_below1']}"),
        ("", "Relative cost with LNG", q(r["s1_ln"]), q(r["s2_ln"]), qb(b["s2_ln"])),
        ("", "Pairs below amine cost with LNG (\\%)", f"{r['s1_ln_share'] * 100:.0f}", f"{r['s2_ln_share'] * 100:.0f}",
         f"{b['s2_ln_share'] * 100:.0f}"),
        ("", "Optimal $T_{ads}$, heat pump / GAX / LNG (K)$^b$", "--",
         f"{r['T_vcr'][1]:.0f} / {r['T_gax'][1]:.0f} / {r['T_lng'][1]:.0f}",
         f"{b['T_vcr'][1]:.0f} / {b['T_gax'][1]:.0f} / {b['T_lng'][1]:.0f}"),
        ("Location", "Spearman $\\rho$ over 21 sites: relative cost and dew point$^c$", "--", q(r["rho_td"]), fq("rho_td"), qb(b["rho_td"])),
        ("", "Spearman $\\rho$ over 21 sites: relative cost and DNI$^c$", "--", q(r["rho_dni"]), fq("rho_dni"), qb(b["rho_dni"])),
        ("", "Spearman $\\rho$ over 21 sites: LCOC and DNI$^c$", "--", q(r["rho_lc_dni"]), qb(b["rho_lc_dni"])),
        ("Water", "Water recovered (t per t CO$_2$)", "--", q(r["water"], "{:.1f}"), qb(b["water"], "{:.1f}")),
        ("", "Water price for parity (USD m$^{-3}$)", "--", q(r["need"], "{:.0f}"), qb(b["need"], "{:.0f}")),
        ("Adsorbent", "Adsorbent share of LCOC without LNG (\\%)", "--",
         f"{r['sorb_med'] * 100:.1f} ({r['sorb_p10'] * 100:.1f}--{r['sorb_p90'] * 100:.0f})",
         f"{b['sorb_med'] * 100:.1f} ({b['sorb_min'] * 100:.1f}--{b['sorb_max'] * 100:.0f})"),
    ]
    rows = [x if len(x) == 6 else x[:4] + ("--",) + x[4:] for x in rows]
    body = "\n".join(" & ".join(x) + r" \\" for x in rows)
    for tag in ("Cold source", "Location", "Water", "Adsorbent"):
        body = body.replace(f"\n{tag} &", f"\n\\midrule\n{tag} &")
    txt = ("\\begin{table*}[t]\n\\centering\\footnotesize\n"
           "\\caption{Results over all evaluated MOFs, each MOF with its lowest-cost configuration at each site. Stage 1: "
           f"{n(r['s1_mofs'])} MOFs ({n(r['s1_feasible'])} feasible in at least one case) at 5 sites with {N_S1} "
           f"configurations; stage 2: {r['s2_mofs']} MOFs at 21 sites with all 14 configurations; values: median "
           "(10th--90th percentile) over MOF--site pairs. Best MOF: lowest-cost MOF at each site; values: median "
           "(range) over 21 sites; with LNG, Tibet (0.995) is counted as below the amine cost. $^a$Best of freeze-out and silica gel with heat pump compared with no drying with heat pump. $^b$Median. "
           "$^c$Stage 2 and all MOFs: one value per MOF. All MOFs: " +
           (f"{n(fu['mofs'])} MOFs feasible at one or more of 21 sites with silica gel without chiller, silica gel with "
            f"heat pump or freeze-out with heat pump; $\\rho$ for the {n(fu['n_all20'])} MOFs feasible at all sites." if fu else "not available.")
           + "}\n\\label{tab:database}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{llllll}\n\\toprule\n"
           "Topic & Quantity & Stage 1 & Stage 2 & All MOFs & Best MOF \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    (F.TAB / "database.tex").write_text(txt, encoding="utf-8")


# ------------------------------------------------------------------ figure
def ecdf(ax, v, **kw):
    v = np.sort(np.asarray(v))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kw)


def figure(r):
    fig, axes = plt.subplots(2, 3, figsize=(F.DW, 4.9))
    # (a) drying
    ax = axes[0, 0]
    F.style(ax, grid="x")
    for co, col, name in (("vcr", F.C_BLUE, "heat pump"), ("gax", F.C_ORANGE, "GAX"), ("lng", F.C_GREEN, "LNG")):
        ecdf(ax, r["dry"][co]["ch"], color=col, lw=1.3, label=name)
    ax.axvline(0, color=F.INK, lw=0.7)
    ax.set_xlim(-70, 40)
    ax.set_xlabel("Cost change by the best dryer (%)")
    ax.set_ylabel("Fraction of MOF–site pairs")
    ax.legend(frameon=False, fontsize=6, loc="lower right", title="cold source", title_fontsize=6)
    F.panel(ax, "a", x=-0.18)
    # (b) RQ2: relative cost with and without LNG
    ax = axes[0, 1]
    F.style(ax, grid="x")
    ecdf(ax, r["s1_nl"], color=F.C_BLUE, lw=1.0, ls="--", label="no LNG, stage 1")
    ecdf(ax, r["s2_nl"], color=F.C_BLUE, lw=1.4, label="no LNG, stage 2")
    if r.get("full"):
        ecdf(ax, r["full"]["nl"], color=F.C_PURPLE, lw=1.0, ls=":", label="no LNG, all MOFs")
    ecdf(ax, r["s1_ln"], color=F.C_GREEN, lw=1.0, ls="--", label="LNG cold, stage 1")
    ecdf(ax, r["s2_ln"], color=F.C_GREEN, lw=1.4, label="LNG cold, stage 2")
    ax.axvline(1, color=F.INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 400)
    F.logx(ax, [0.5, 1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xlabel("Relative cost")
    ax.legend(frameon=False, fontsize=5.0, loc="lower right", bbox_to_anchor=(1.04, 0.0), handlelength=1.2, borderaxespad=0.1)
    F.panel(ax, "b", x=-0.12)
    # (c) RQ2: GAX vs heat pump
    ax = axes[0, 2]
    F.style(ax, grid="x")
    ecdf(ax, r["s1_gh"], color=F.C_ORANGE, lw=1.0, ls="--", label="stage 1 (same dryers)")
    ecdf(ax, r["s2_gh"], color=F.C_ORANGE, lw=1.4, label="stage 2 (best dryer)")
    ax.axvline(1, color=F.INK, lw=0.8)
    ax.set_xlim(0.9, 3.0)
    ax.set_xlabel("Cost with GAX chiller / cost with heat pump")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    F.panel(ax, "c", x=-0.12)
    # (d) cold demand vs working capacity (stage-2 MOFs, freeze-out + heat pump: deep cooling)
    ax = axes[1, 0]
    F.style(ax, grid="y")
    fv = r["fv"]
    for site, col in (("Singapore", F.C_ORANGE), ("London", F.C_BLUE), ("Kiruna", F.C_PURPLE)):
        g = fv[fv.site == site]
        ax.scatter(g.wcap_mmolg, g.cold_GJ, s=7, color=col, lw=0, alpha=0.85, label=site)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3], ["0.1", "0.3", "1", "3"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 150)
    ax.set_xlabel("Working capacity (mmol g$^{-1}$)")
    ax.set_ylabel("Cold demand (GJ t$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="center right", markerscale=1.8, ncol=3, columnspacing=0.6,
              handletextpad=0.1)
    F.panel(ax, "d", x=-0.18)
    # (e) RQ3: per-MOF rank correlations over 21 sites
    ax = axes[1, 1]
    F.style(ax, grid="x")
    data = [r["rho_td"], r["rho_dni"], r["rho_lc_dni"]]
    labels = ["relative cost\nand dew point", "relative cost\nand DNI", "LCOC\nand DNI"]
    rng = np.random.default_rng(1)
    fu = r.get("full")
    for i, v in enumerate(data):
        if fu and i < 2:                                   # all MOFs feasible at all 21 sites, below stage 2
            w = fu[("rho_td", "rho_dni")[i]]
            ax.scatter(w, i + 0.25 + rng.uniform(-0.07, 0.07, len(w)), s=3, color=F.C_PURPLE, alpha=0.35, lw=0,
                       label=f"all MOFs ({len(w)})" if i == 0 else None)
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=6, color=F.C_BLUE, alpha=0.6, lw=0,
                   label=f"stage 2 ({len(v)})" if i == 0 else None)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=F.INK, lw=1.2)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", markerscale=2, handletextpad=0.1)
    ax.axvline(0, color=F.NEUTRAL, lw=0.7)
    ax.set_yticks(range(3), labels, fontsize=6.3)
    ax.set_ylim(-0.6, 2.6)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman $\\rho$ over 21 sites (one point per MOF)")
    F.panel(ax, "e", x=-0.3)
    # (f) RQ4: water price for parity
    ax = axes[1, 2]
    F.style(ax, grid="x")
    ax.axvspan(0.38, 2.97, color=F.C_BLUE, alpha=0.12, lw=0)
    ax.text(1.05, 0.93, "desalination", ha="center", fontsize=6, color=F.INK2, transform=ax.get_xaxis_transform())
    need = r["need"].replace([np.inf, -np.inf], np.nan).dropna()
    ecdf(ax, need, color=F.C_ORANGE, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 3000)
    F.logx(ax, [1, 10, 100, 1000])
    ax.set_xlabel("Water price for parity (USD m$^{-3}$)")
    ax.set_ylabel("Fraction of MOF–site pairs")
    F.panel(ax, "f", x=-0.12)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    F.save(fig, "fig12_database")


def main():
    F.rc()
    s1, s2, k, n_s1 = load()
    r = stats(s1, s2, k, n_s1)
    table(r)
    figure(r)
    skip = {"dry", "s1_gh", "s2_gh", "s1_nl", "s1_ln", "s2_nl", "s2_ln", "fv", "rho_td", "rho_dni", "rho_lc_dni",
            "need", "water"}
    for key, v in r.items():
        if key not in skip:
            print(key, np.round(v, 3) if isinstance(v, float) else v)
    for co, x in r["dry"].items():
        print("dry", co, {k_: round(v, 3) for k_, v in x.items() if k_ != "ch"})
    for n in ("rho_td", "rho_dni", "rho_lc_dni", "need", "s1_gh", "s2_gh", "s1_nl", "s2_nl", "s1_ln", "s2_ln"):
        v = r[n].replace([np.inf, -np.inf], np.nan).dropna()
        print(n, "min", round(v.min(), 2), "P10", round(v.quantile(.1), 2), "med", round(v.median(), 2),
              "P90", round(v.quantile(.9), 2), "max", round(v.max(), 2), "n", len(v))


if __name__ == "__main__":
    main()
