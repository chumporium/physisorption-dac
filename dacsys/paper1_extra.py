"""Additional tables and figures for Paper 1 and its Supplementary Information (all from existing results).

python -m dacsys.paper1_extra
  main text: tables/screening.tex, drying.tex, cop.tex; figs/fig11_climate, fig16_map
  SI:        tables/si_params.tex, si_mc.tex, si_yamin.tex, si_water.tex, si_selectivity.tex, cpqs.tex
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import ars_map, prisma, sorbents, vcr
from . import fig_paper1 as F
from .levers import COLD, DRYING, NOT_APPLICABLE, Prm

P = Prm()
DRY_T = {"none": "None", "cond": "Condensation", "cond_silica": "Cond.\\ and silica gel", "freeze": "Freeze-out"}
COLD_T = {"ambient": "None", "gax": "GAX chiller", "vcr": "Heat pump", "lng": "LNG cold"}


P2TAB = F.TAB                                             # merged paper: material tables go to the SI


def _write(name, text, where=None):
    where = where or F.TAB
    where.mkdir(parents=True, exist_ok=True)
    (where / name).write_text(text, encoding="utf-8")


def _signed(x):
    """Signed integer percentage with a typeset minus; -0 and +0 are printed as 0."""
    s = f"{x:+.0f}"
    return "0" if s in ("+0", "-0") else s.replace("-", "$-$")


def _num(x, fmt):
    return "--" if x is None or not np.isfinite(x) else format(x, fmt).replace(",", "{,}")


# ------------------------------------------------------------------ screening funnel
def screening_table():
    from .study5 import DB1_ARCH, PR_ARCH
    fits = prisma.load_fits()
    full = pd.read_csv(F.RES / "cases.csv", low_memory=False)
    a = pd.read_csv(F.RES / "amine.csv").set_index("site").amine_lcoc_base
    names = prisma.register()
    ok = [n for n in names if prisma.fit_ok(n, fits)]
    assert len(prisma.prescreen(names)) == full[full.lever.isin(["PR1", "DB1"])].sorbent.nunique()
    s1 = full[full.lever.isin(["PR1", "DB1"])]
    s2 = full[full.lever.isin(["PR2", "PRL"])].copy()
    s2["ratio"] = s2.lcoc / s2.site.map(a)
    s1ok = s1[s1.status == "ok"]

    def split(ns):
        o = pd.Series([prisma.origin(n) for n in ns])
        return f"{len(ns):,} ({(o == 'eksperimental').sum():,}/{(o == 'hipotetis').sum():,})".replace(",", "{,}")

    feas_n = s1ok[s1ok.cold != "lng"].sorbent.unique()
    feas_l = s1ok[s1ok.cold == "lng"].sorbent.unique()
    b2n = F.best(s2[s2.cold != "lng"], ["site", "sorbent"])
    b2l = F.best(s2[s2.cold == "lng"], ["site", "sorbent"])
    win_l = (b2l.groupby("sorbent").ratio.apply(lambda v: (v < 1).sum()) == 20).sum()
    rows = [
        ("PrISMa database", split(names), "--", "--", "--"),
        ("Isotherm fit accepted", split(ok), "--", "--", "log.\\ RMSE $\\le$ 0.3, $K_H$ within 5\\,\\%"),
        ("Pre-screen", split(s1.sorbent.unique()), "--", "--", "$WC_{dry}\\ge0.1$ mmol g$^{-1}$ at 225 K"),
        ("Stage 1", split(s1.sorbent.unique()), f"{len(PR_ARCH) + len(DB1_ARCH)} $\\times$ 5", f"{len(s1):,}".replace(",", "{,}"),
         f"feasible: {split(feas_n)} without LNG, {len(feas_l):,} with LNG".replace(",", "{,}")),
        ("Stage 2", split(s2.sorbent.unique()), f"14 $\\times$ {len(F.ORDER)}", f"{len(s2):,}".replace(",", "{,}"),
         f"below amine cost: {int((b2n.ratio < 1).sum())} MOF--site pairs without LNG; "
         f"{win_l} MOFs at all {len(F.ORDER)} sites with LNG"),
    ]
    s3 = full[full.lever == "DB3"]
    if len(s3):
        from .db3_check import full_pairs
        fp = full_pairs()
        from .study5 import DB3_ARCH
        arch = full.lever.isin(["PR1", "PR2", "DB1", "DB3"]) & pd.Series(
            [(a, b) in DB3_ARCH for a, b in zip(full.drying, full.cold)], index=full.index)
        n3 = len(full[arch].drop_duplicates(["sorbent", "site", "drying", "cold"]))
        rows.append(("All MOFs", split(s1.sorbent.unique()), f"{len(DB3_ARCH)} $\\times$ {len(F.ORDER)}", f"{n3:,}".replace(",", "{,}"),
                     f"feasible: {fp.sorbent.nunique():,} MOFs; below amine cost: {int((fp.ratio < 1).sum())} "
                     f"MOF--site pairs; lowest {fp.ratio.min():.2f}".replace(",", "{,}")))
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write("screening.tex",
           "\\begin{table*}[t]\n\\centering\\footnotesize\n"
           "\\caption{Screening of the PrISMa database. Number of MOFs with experimental/hypothetical structures in "
           "brackets. Configurations: dryer and cold-source combinations.}\n\\label{tab:screening}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllll}\n\\toprule\n"
           "Step & MOFs & Configurations $\\times$ sites & Cases & Criterion or result \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    return rows


# ------------------------------------------------------------------ drying x cold detail (RQ1/RQ2)
def drying_table(d):
    p = F.db(d)
    pok = p[p.status == "ok"]
    b = F.best(pok, ["site", "drying", "cold"])
    rows, out = [], []
    for co in COLD:
        ref = b[(b.cold == co) & (b.drying == "none")].set_index("site").ratio
        for dr in DRYING:
            if (co, dr) in NOT_APPLICABLE:
                continue
            x = b[(b.cold == co) & (b.drying == dr)].set_index("site")
            if not len(x):
                continue
            med = x.median(numeric_only=True)
            chg = (x.ratio / ref.reindex(x.index) - 1).median() * 100 if dr != "none" else np.nan
            Td = x.T_design.where(x.cold != "ambient").median()
            out.append(dict(cold=co, drying=dr, n=len(x), T=Td, wc=med.wcap_mmolg, heat=med.heat_GJ,
                            heat_sil=med.heat_sil_GJ, elec=med.elec_GJ, elec_cold=med.elec_cold_GJ,
                            cold_GJ=med.cold_GJ, water=med.water_per_co2, lcoc=med.lcoc, ratio=med.ratio, chg=chg))
            dlab = "Silica gel" if (co == "ambient" and dr == "cond_silica") else DRY_T[dr]   # no coil without cold
            rows.append([COLD_T[co], dlab, f"{len(x)}", _num(Td, ".0f"), _num(med.wcap_mmolg, ".2f"),
                         _num(med.heat_GJ, ".1f"), _num(med.heat_sil_GJ, ".1f"), _num(med.elec_GJ, ".1f"),
                         _num(med.elec_cold_GJ, ".1f"), _num(med.cold_GJ, ".0f"), _num(med.water_per_co2, ".1f"),
                         _num(med.lcoc, ",.0f"), _num(med.ratio, ".2f"),
                         "--" if not np.isfinite(chg) else _signed(chg)])
    body = []
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            body.append("\\midrule")
        body.append(" & ".join(r if r[0] != prev else [""] + r[1:]) + r" \\")
        prev = r[0]
    _write("drying.tex",
           "\\begin{table*}[t]\n\\centering\\footnotesize\n"
           "\\caption{Effect of the dryer for each cold source: best MOF at each site, median over the sites where the "
           "combination is feasible ($n$). $T_{ads}$: design adsorption temperature (--: ambient); heat, silica: "
           "regeneration of silica gel, including the guard bed that dries the air in the freeze-out option in hours "
           "above 0\\,°C; electricity, total: fans, vacuum pumps and cooling; water: water recovered from the air; "
           "change: median over the sites of the change in relative cost compared with no drying at the same site (the "
           "relative cost is a median over the $n$ sites, so the two columns need not agree).}\n"
           "\\label{tab:drying}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrrrrrrr}\n\\toprule\n"
           "Cold source & Dryer & $n$ & $T_{ads}$ & $WC$ & \\multicolumn{2}{c}{Heat (GJ t$^{-1}$)} & "
           "\\multicolumn{2}{c}{Electricity (GJ t$^{-1}$)} & Cold & Water & LCOC & Relative & Change \\\\\n"
           "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
           " & & & (K) & (mmol g$^{-1}$) & total & silica & total & cooling & (GJ t$^{-1}$) & (t t$^{-1}$) & "
           "(USD t$^{-1}$) & cost & (\\%) \\\\\n\\midrule\n" + "\n".join(body) +
           "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")
    return pd.DataFrame(out)


# ------------------------------------------------------------------ COP of the two chillers (RQ2)
def cop_table():
    T_eva = (-10.0, -20.0, -30.0, -38.0, -45.0)
    T_amb = (0.0, 15.0, 30.0, 40.0)
    rows, out = [], []
    for te in T_eva:
        g = [P.gax_derate * ars_map.cop_env(np.array([ta]), np.array([te]), P.gax_T_gen_max)[0][0] for ta in T_amb]
        h = [float(vcr.cop(np.array([ta]), np.array([te]))[0]) for ta in T_amb]
        out.append(dict(T_eva=te, gax=g, hp=h))
        rows.append([f"{te:.0f}".replace("-", "$-$"), f"{te + 273.15 + P.dT_evap:.0f}"] + [_num(v, ".2f") for v in g] +
                    [_num(v, ".2f") for v in h])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    hdr = " & ".join(f"{t:.0f}\\,°C" for t in T_amb)
    _write("cop.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n"
           "\\caption{COP of the solar GAX chiller (heat input, correction factor 0.85 included) and of the heat pump "
           "(electricity input) as a function of the evaporator and ambient temperature. $T_{air}$: lowest air "
           "temperature with a 5\\,K approach. --: outside the operating range of the GAX chiller.}\n"
           "\\label{tab:cop}\n\\resizebox{\\columnwidth}{!}{\\begin{tabular}{rrrrrrrrrr}\n\\toprule\n"
           "$T_{eva}$ & $T_{air}$ & \\multicolumn{4}{c}{GAX chiller, ambient} & \\multicolumn{4}{c}{Heat pump, ambient} \\\\\n"
           "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
           f"(°C) & (K) & {hdr} & {hdr} \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return out


# ------------------------------------------------------------------ Fig 11 climate drivers (RQ3)
SHORT = {"Atacama coast": "Atacama", "Mexico City": "Mexico C."}


def fig_climate(d, a):
    k = pd.read_csv(F.RES / "sites_koppen.csv").set_index("site").reindex(F.ORDER)
    b = F.db_best(d)
    m = F.m13(d)
    x13 = F.best(m[m.cold != "lng"], ["site"]).set_index("site").reindex(F.ORDER)
    am = a.amine_lcoc_base.reindex(F.ORDER)
    dry = b.cold == "ambient"
    rho = lambda x, y: pd.Series(x).rank().corr(pd.Series(y).rank())      # noqa: E731
    stats = dict(phys_dni=rho(k.DNI_kWh_m2_yr, b.lcoc), amine_dni=rho(k.DNI_kWh_m2_yr, am),
                 ratio_td=rho(k.Td_mean_2018_C, b.ratio), ratio_dni=rho(k.DNI_kWh_m2_yr, b.ratio),
                 cold_td=rho(k.Td_mean_2018_C[~dry], b.cold_GJ[~dry]), r13_td=rho(k.Td_mean_2018_C, x13.ratio))
    fig, axes = plt.subplots(1, 3, figsize=(F.DW, 3.0))
    # (a) both costs fall with irradiance
    ax = axes[0]
    F.style(ax, grid="y")
    ax.scatter(k.DNI_kWh_m2_yr, b.lcoc, s=14, color=F.C_BLUE, edgecolor=F.SURF, lw=0.4, zorder=3,
               label="best MOF, without LNG")
    ax.scatter(k.DNI_kWh_m2_yr, am, s=14, color=F.INK2, marker="s", edgecolor=F.SURF, lw=0.4, zorder=3,
               label="amine DAC")
    off_a = {"Reykjavik": (4, 4, "left")}
    for s_ in ("Altiplano", "Reykjavik", "Riyadh"):
        dx, dy, ha = off_a.get(s_, (3, 3, "left"))
        ax.annotate(SHORT.get(s_, s_), (k.at[s_, "DNI_kWh_m2_yr"], b.at[s_, "lcoc"]), fontsize=5.6, color=F.INK2,
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.set_yscale("log")
    ax.set_ylim(150, 5000)
    ax.set_yticks([200, 500, 1000, 2000, 5000], ["200", "500", "1000", "2000", "5000"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("DNI (kWh m$^{-2}$ yr$^{-1}$)")
    ax.set_ylabel("LCOC (USD t$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.text(0.98, 0.97, f"$\\rho$ = {stats['phys_dni']:.2f} (MOF)\n$\\rho$ = {stats['amine_dni']:.2f} (amine)".replace("-", "\u2212"),
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=F.INK2)
    F.panel(ax, "a", x=-0.16)
    # (b) cooling demand and water removed rise with dew point
    ax = axes[1]
    F.style(ax, grid="y")
    ax.scatter(k.Td_mean_2018_C[~dry], b.cold_GJ[~dry], s=14, color=F.C_BLUE, edgecolor=F.SURF, lw=0.4, zorder=3,
               label="cold demand (GJ t$^{-1}$)")
    ax.scatter(k.Td_mean_2018_C, b.water_per_co2, s=14, color=F.C_GREEN, marker="^", edgecolor=F.SURF, lw=0.4,
               zorder=3, label="water recovered (t t$^{-1}$)")
    ax.set_xlabel("Mean dew point (°C)")
    ax.set_ylabel("Cold (GJ t$^{-1}$) or water (t t$^{-1}$)")
    ax.text(0.03, 0.97, f"$\\rho$ = {stats['cold_td']:.2f} (cold)\n$\\rho$ = {rho(k.Td_mean_2018_C, b.water_per_co2):.2f}"
            " (water)", transform=ax.transAxes, va="top", fontsize=6, color=F.INK2)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    F.panel(ax, "b", x=-0.16)
    # (c) relative cost follows the dew point
    ax = axes[2]
    F.style(ax, grid="y")
    ax.scatter(k.Td_mean_2018_C[~dry], b.ratio[~dry], s=16, color=F.C_BLUE, edgecolor=F.SURF, lw=0.4, zorder=3,
               label="best MOF, heat pump")
    ax.scatter(k.Td_mean_2018_C[dry], b.ratio[dry], s=18, color=F.C_ORANGE, marker="D", edgecolor=F.SURF, lw=0.4,
               zorder=3, label="best MOF, silica gel, no chiller")
    ax.scatter(k.Td_mean_2018_C, x13.ratio, s=16, facecolor="none", edgecolor=F.INK, lw=0.6, zorder=2,
               label="13X, best without LNG")
    off = {"Dubai": (0, 7, "center"), "Riyadh": (6, -9, "left"), "Singapore": (2, -9, "left"),
           "Altiplano": (0, -10, "center"), "Gobi": (3, -9, "left")}
    for s_ in ("Altiplano", "Dubai", "Reykjavik", "Singapore", "Riyadh", "Gobi"):
        dx, dy, ha = off.get(s_, (3, -7, "left"))
        ax.annotate(SHORT.get(s_, s_), (k.at[s_, "Td_mean_2018_C"], b.at[s_, "ratio"]), fontsize=5.6, color=F.INK2,
                    xytext=(dx, dy), textcoords="offset points", ha=ha)
    ax.axhline(1, color=F.INK, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 25)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Mean dew point (°C)")
    ax.set_ylabel("Relative cost")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    ax.text(0.03, 0.97, f"$\\rho$ = {stats['ratio_td']:.2f} (best MOF)\n$\\rho$ = {stats['r13_td']:.2f} (13X)",
            transform=ax.transAxes, va="top", fontsize=6, color=F.INK2)
    F.panel(ax, "c", x=-0.16)
    fig.tight_layout(w_pad=1.0)
    F.save(fig, "fig11_climate")
    return stats


LAND = F.ROOT / "data" / "naturalearth" / "ne_110m_land.geojson"    # Natural Earth 1:110m land, public domain
# label offsets (points) and alignment where neighbouring sites would overlap
MAP_OFF = {"London": (-4, 3, "right"), "Madrid": (-4, -5, "right"), "Rome": (4, -5, "left"),
           "Kiruna": (4, 2, "left"), "Reykjavik": (-4, 2, "right"), "Atacama coast": (-4, -3, "right"),
           "Altiplano": (4, 2, "left"), "Shanghai": (4, -5, "left"), "Seoul": (4, 3, "left"),
           "Singapore": (-4, 3, "right"), "Jakarta": (-4, -4, "right"), "Darwin": (4, -3, "left"), "Tibet": (4, -5, "left"),
           "Gobi": (-4, 4, "right"), "Astana": (4, 3, "left"), "Riyadh": (-4, -4, "right")}


def _land(ax):
    import json
    from matplotlib.patches import Polygon
    for f in json.loads(LAND.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for rings in polys:
            for k, ring in enumerate(rings):                  # exterior ring, then holes (lakes) in white
                ax.add_patch(Polygon(ring, closed=True, lw=0, zorder=1, facecolor=F.GRID if k == 0 else F.SURF))


def fig_map(d):
    """Relative cost of the best MOF at the sites on a world map, without and with LNG cold."""
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=np.log10(0.25), vcenter=0.0, vmax=np.log10(30))      # as the heat maps (Fig. 3)
    pos = pd.DataFrame({s: v[:2] for s, v in F.SITES.items()}, index=["lat", "lon"]).T.reindex(F.ORDER)
    fig, axes = plt.subplots(2, 1, figsize=(F.DW, 6.0))
    out = {}
    for ax, lng, letter, title in ((axes[0], False, "a", "Best MOF without LNG cold"),
                                   (axes[1], True, "b", "Best MOF with LNG cold")):
        b = F.db_best(d, lng=lng)
        out[title] = b.ratio
        _land(ax)
        sc = ax.scatter(pos.lon, pos.lat, c=np.log10(b.ratio), cmap=F.DIV, norm=norm, s=34, edgecolor=F.INK,
                        lw=0.5, zorder=3)
        for s_ in F.ORDER:
            dx, dy, ha = MAP_OFF.get(s_, (4, 2, "left"))
            ax.annotate(f"{SHORT.get(s_, s_)} {b.at[s_, 'ratio']:.1f}", (pos.at[s_, "lon"], pos.at[s_, "lat"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=5.8, color=F.INK,
                        zorder=4)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 80)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(F.NEUTRAL)
            s.set_linewidth(0.5)
        ax.set_title(title, loc="left", fontsize=8, color=F.INK)
        F.panel(ax, letter, x=-0.01, y=1.0)
    ticks = [0.25, 0.5, 1, 2, 5, 10, 20]
    fig.tight_layout(h_pad=1.2)
    fig.subplots_adjust(bottom=0.13)
    x0, x1 = axes[1].get_position().x0, axes[1].get_position().x1
    cb = fig.colorbar(sc, cax=fig.add_axes([x0, 0.065, x1 - x0, 0.016]), orientation="horizontal")
    cb.set_ticks(np.log10(ticks), labels=[f"{t:g}" for t in ticks])
    cb.outline.set_linewidth(0.4)
    cb.set_label("Relative cost (LCOC / amine LCOC at the same site); below 1: physisorption cheaper", fontsize=7)
    F.save(fig, "fig16_map")
    return pd.DataFrame(out)


# ------------------------------------------------------------------ SI tables
def si_params():
    p = P
    O = "assumed (Table~\\ref{tab:si_oat})"               # design assumption, tested one at a time
    D = "design value"
    rows = [
        ("\\textit{Air side}", ""), ("CO$_2$ in air", f"{p.x_co2_ppm:.0f} ppm", D),
        ("CO$_2$ capture fraction", f"{p.eta_cap}", "\\cite{keith2018,netl2025,kim2025}"),
        ("Condensing coil outlet", f"{p.T_coil1:.0f} °C", D + "; Table~\\ref{tab:si_oat}"),
        ("Dew point after silica gel", f"{p.dewpoint_dry:.0f} °C".replace("-", "$-$"), O),
        ("Recuperator approach (warm / cold end)", f"{p.dT_warm:.0f} K / 3, 8 or 20 K or no recuperator", "cold end optimised"),
        ("Air--refrigerant approach in coils", f"{p.dT_evap:.0f} K", O),
        ("Heat transfer coefficient: recuperator / coils", f"{p.U_air * 1e3:.0f} / {p.U_coil * 1e3:.0f} W m$^{{-2}}$ K$^{{-1}}$",
         "\\cite{kim2025}"),
        ("Regenerator area relative to a recuperator", f"{p.reg_area_factor}", O),
        ("Pressure drop: contactor", f"{p.dP_contactor:.0f} Pa", "\\cite{netl2025}"),
        ("Pressure drop: coil / silica gel", f"{p.dP_coil:.0f} / {p.dP_silica:.0f} Pa", O),
        ("Pressure drop: recuperator / regenerator (per pass)", f"{p.dP_recup:.0f} / {p.dP_regen:.0f} Pa",
         "\\cite{ashrae2010}"),
        ("Fan efficiency", f"{p.eta_fan}", "\\cite{keith2018,kim2025}"),
        ("\\textit{Adsorption cycle}", ""),
        ("Regeneration", "100 °C; " + " / ".join(f"{x:g}" for x in p.p_des_opt) + " bar", "pressure optimised"),
        ("Minimum CO$_2$ purity", f"{p.purity_min}", D),
        ("Vacuum pump isothermal efficiency", f"{p.eta_vac}", "\\cite{kim2025}"),
        ("Contactor heat capacity", f"{p.cp_contactor} kJ kg$^{{-1}}$ K$^{{-1}}$ per kg adsorbent (75 wt\\% adsorbent)",
         "\\cite{stampi2024}; Table~\\ref{tab:si_oat}"),
        ("Bed heat transfer: $U$ / specific area", f"{p.U_hx * 1e3:.0f} W m$^{{-2}}$ K$^{{-1}}$ / {p.a_hx:.0f} m$^{{2}}$ m$^{{-3}}$",
         "\\cite{charalambous2024}"),
        ("Bed / pellet porosity", f"{p.eps_bed} / {p.eps_pellet}", "\\cite{charalambous2024}"),
        ("Heating / cooling medium approach", f"{p.dT_hex_heat:.0f} / {p.dT_hex_cool:.0f} K", O),
        ("Maximum cycles per hour", f"{p.max_cycles_per_h:.0f}", "\\cite{stampi2024}; Table~\\ref{tab:si_variants}"),
        ("\\textit{Dryers}", ""),
        ("Silica gel adsorption heat", f"{p.h_ads_silica:.0f} kJ kg$^{{-1}}$ water", "\\cite{abdelgaied2023}"),
        ("Silica gel working capacity / cycles", f"{p.sil_dq} kg kg$^{{-1}}$ / {p.sil_cycles:.0f} h$^{{-1}}$", O),
        ("Silica gel regeneration margin", f"{p.silica_regen_factor}", "assumed (Table~\\ref{tab:si_silica})"),
        ("Water recovery from silica gel", f"{p.water_recovery_silica}", O),
        ("Freeze-out: unrecovered sublimation heat", f"{p.freeze_loss}", "assumed (Table~\\ref{tab:si_mc})"),
        ("\\textit{Cold supply}", ""),
        ("GAX COP correction factor", f"{p.gax_derate}", "\\cite{aprile2016,herold2016}"),
        ("GAX evaporator / generator limit", f"$\\ge-38$ °C / $\\le${p.gax_T_gen_max:.0f} °C", "\\cite{yamin2024}"),
        ("GAX solution pump", f"{p.gax_pump_frac * 100:.1f} \\% of generator heat", "\\cite{herold2016}"),
        ("Dry cooler fans", f"{p.fan_rej_frac * 100:.0f} \\% of heat rejected", "\\cite{fedrizzi2014}"),
        ("Heat pump isentropic efficiency / condensing approach", f"{vcr.ETA_IS} / {vcr.DT_COND:.0f} K",
         "assumed (Table~\\ref{tab:si_hpeff})"),
        ("\\textit{Energy supply}", ""),
        ("PTC optical efficiency / heat loss", f"{p.ptc_eta_opt} / {p.ptc_loss * 1e3:.2f} W m$^{{-2}}$ K$^{{-1}}$",
         "\\cite{kalogirou2024,burkholder2009}"),
        ("Incidence angle modifier", "LS-2 collector", "\\cite{dudley1994}"),
        ("Thermal oil temperature", f"{p.T_htf:.0f} °C", "\\cite{yamin2024}"),
        ("Storage loss", f"{p.storage_loss_h * 100:.3f} \\% h$^{{-1}}$ (1 \\% d$^{{-1}}$)", "\\cite{prieto2016}"),
        ("Battery round-trip efficiency", f"{p.eta_batt}", "\\cite{cole2019}"),
        ("\\textit{Economics (USD 2024, installed)}", ""),
        ("Discount rate / lifetime", f"{p.discount * 100:.0f} \\% / {p.lifetime} yr", "\\cite{fasihi2019,ieaghg2021}"),
        ("O\\&M: process / energy supply", f"{p.om_proc * 100:.0f} / {p.om_energy * 100:.0f} \\% of capital yr$^{{-1}}$",
         "\\cite{keith2018,cole2019}"),
        ("PTC / storage tank", f"{p.c_ptc:.0f} USD m$^{{-2}}$ / {p.c_tank:.0f} USD kWh$^{{-1}}$", "\\cite{yamin2024,akar2024,irena2020tes}"),
        ("PV / battery", f"{p.c_pv:.0f} USD kW$_p^{{-1}}$ / {p.c_batt:.0f} USD kWh$^{{-1}}$", "\\cite{irena2025}"),
        ("GAX chiller / heat pump", f"{p.c_abs_rej:.0f} / {p.c_hp_heat:.0f} USD per kW heat rejected",
         "\\cite{doe2017abs} / \\cite{marina2021}"),
        ("Dry coolers", f"{p.c_rej:.0f} USD per kW heat rejected (installation factor {p.netl_install:.2f})",
         "\\cite{fedrizzi2014,weiland2019}"),
        ("Heat exchangers (recuperator, coils, LNG)",
         f"Eq.~(\\ref{{eq:si_hx}}): {p.hx_cost_ua:.2f} USD per W K$^{{-1}}$ installed "
         f"({p.hx_cost_m2:.0f} USD m$^{{-2}}$ at 50 W m$^{{-2}}$ K$^{{-1}}$)",
         "\\cite{weiland2019}; unit size: geometric mean of the vendor range (Table~\\ref{tab:si_hx})"),
        ("Material factor, LNG exchangers", f"{p.f_mat_cryo:g} (304/316 stainless steel; not on labour)", "\\cite{towler2022}; 1--2.93 in Table~\\ref{tab:si_hx}"),
        ("Fans / vacuum pumps", f"{p.c_fan:,.0f} / {p.c_vac:,.0f} USD kW$^{{-1}}$".replace(",", "{,}"), "\\cite{netl2025}"),
        ("Adsorbent price: 13X / MOFs / model", f"{sorbents.get('13X').cost} / {prisma.PRICE:.0f} / 10 USD kg$^{{-1}}$, "
         f"$\\times${p.sorbent_fab:.0f} for fabrication", "\\cite{kim2025} / \\cite{desantis2017} / assumed"),
        ("Contactor structure", f"{p.c_contactor:.0f} USD kg$^{{-1}}$ adsorbent $\\times${p.contactor_fab:.0f}", "\\cite{kim2025}"),
        ("Silica gel system", f"{p.c_silica_sys:.0f} USD kg$^{{-1}}$", "assumed (Table~\\ref{tab:si_mc})"),
        ("Amine plant capital (low / base / high)", " / ".join(f"{c:.0f}" for c in p.amine_capex) + " USD per t yr$^{-1}$",
         "\\cite{fasihi2019,ieaghg2021}"),
        ("Amine O\\&M (low / base / high)", " / ".join(f"{c * 100:.0f}" for c in p.amine_om) + " \\% of capital yr$^{-1}$",
         "\\cite{fasihi2019}"),
        ("Amine energy factor (low / base / high)", " / ".join(f"{c}" for c in p.amine_energy_factor),
         "\\cite{mcqueen2021,deutz2021}"),
    ]
    rows = [r if len(r) == 3 else (r[0], r[1], "") for r in rows]
    body = "\n".join((f"\\multicolumn{{3}}{{l}}{{{a_}}} \\\\" if not b_ else f"{a_} & {b_} & {c_} \\\\")
                     for a_, b_, c_ in rows)
    _write("si_params.tex",
           "\\begin{table}[p]\n\\centering\\footnotesize\\renewcommand{\\arraystretch}{0.9}\n"
           "\\caption{Model parameters of the base case and their sources. "
           "Costs are installed costs in USD 2023 (CEPCI); the heat-exchanger correlation gives equipment costs and is multiplied by "
           "the installation factor of dry cooling \\cite{weiland2019}. Values marked `assumed' are assumptions of this "
           "study; the table or section in brackets shows where their effect is tested.}\n"
           "\\label{tab:si_params}\n\\begin{tabular}{p{0.37\\textwidth}p{0.36\\textwidth}p{0.21\\textwidth}}\n\\toprule\n"
           "Parameter & Value & Source \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def si_mc():
    rows = [("Contactor pressure drop", "100--1000 Pa", "log-uniform"),
            ("GAX chiller cost", "0.7--1.4 $\\times$ base", "uniform"),
            ("Heat pump cost", "0.75--2.5 $\\times$ base (300--1000 EUR kW$^{-1}$ heat \\cite{marina2021})", "log-uniform"),
            ("Adsorbent price", "0.5--2 $\\times$ base", "log-uniform"),
            ("PTC cost", "150--350 USD m$^{-2}$", "uniform"), ("PV cost", "400--900 USD kW$_p^{-1}$", "uniform"),
            ("Battery cost", "100--250 USD kWh$^{-1}$", "uniform"), ("Storage tank cost", "15--40 USD kWh$^{-1}$", "uniform"),
            ("Discount rate", "5--10 \\%", "uniform"), ("Cycle rate", "0.5--2 $\\times$ base", "log-uniform"),
            ("GAX COP correction factor", "0.75--0.95", "uniform"), ("Freeze-out heat loss", "0.1--0.4", "uniform"),
            ("Capture fraction", "0.7--0.9", "uniform"),
            ("Amine capital cost", "600--1500 USD per t yr$^{-1}$", "uniform"),
            ("Amine O\\&M", "3--10 \\% yr$^{-1}$, linear in capital cost", "coupled"),
            ("Amine energy demand", "0.75--1.25 $\\times$ base \\cite{mcqueen2021,deutz2021}", "uniform"),
            ("Heat-exchanger unit size", "$UA$ = 8.6$\\times$10$^5$--7.5$\\times$10$^7$ W K$^{-1}$ ("
             + f"{Prm(netl_ua_unit=8.6e5).hx_cost_m2:.0f}--{Prm(netl_ua_unit=7.5e7).hx_cost_m2:.0f}"
             + " USD m$^{-2}$ installed) \\cite{weiland2019}", "log-uniform"),
            ("Heat-exchanger cost correlation", "0.75--1.28 $\\times$ base \\cite{weiland2019}", "uniform"),
            ("Fans, dry coolers, contactor, silica gel", "0.5--2 $\\times$ base (together)", "log-uniform"),
            ("Vacuum pumps", "0.07--1 $\\times$ base (Kim et al.\\ \\cite{kim2025} to NETL \\cite{netl2025})", "log-uniform")]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write("si_mc.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Parameters of the Monte Carlo analysis (200 samples, "
           "the same samples at every site).}\n\\label{tab:si_mc}\n\\begin{tabular}{lll}\n\\toprule\n"
           "Parameter & Range & Distribution \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def si_yamin():
    t = pd.read_csv(F.ROOT / "results" / "paper" / "table6_base_vs_optimum.csv", index_col=0)
    names = {"LHPR": ("LH$_2$ production rate", "m$^3$ h$^{-1}$", ".1f"), "COP_sys": ("System COP", "--", ".3f"),
             "eta_II": ("Exergy efficiency", "\\%", ".1f"), "SI": ("Sustainability index", "--", ".2f"),
             "Zdot": ("Capital cost rate", "USD h$^{-1}$", ".1f"), "c_LH2": ("LH$_2$ cost", "USD kg$^{-1}$", ".3f")}
    rows = []
    for k, (n, u, f) in names.items():
        r = t.loc[k].astype(float)
        dev = [(r["Base"] / r["Base (paper)"] - 1) * 100, (r["Optimum (paper DV)"] / r["Optimum (paper)"] - 1) * 100]
        rows.append([n, u, format(r["Base (paper)"], f), format(r["Base"], f), _signed(dev[0]),
                     format(r["Optimum (paper)"], f), format(r["Optimum (paper DV)"], f), _signed(dev[1])])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write("si_yamin.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Reproduction of the solar GAX--liquefaction system "
           "of Yamin et al.\\ (Table 6 of the original paper) with the model used for the energy supply.}\n"
           "\\label{tab:si_yamin}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrr}\n\\toprule\n"
           " & & \\multicolumn{3}{c}{Base case} & \\multicolumn{3}{c}{Optimum} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
           "Quantity & Unit & Yamin et al. & This model & Dev.\\ (\\%) & Yamin et al. & This model & Dev.\\ (\\%) \\\\\n"
           "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return rows


def si_water(a):
    w = pd.read_csv(F.RES / "cases.csv", low_memory=False)
    w["ratio"] = w.lcoc / w.site.map(a.amine_lcoc_base)
    wv = w[w.lever.str.startswith("W_")]
    sorbs = list(wv.sorbent.unique())
    base = w[w.sorbent.isin(sorbs) & ~w.lever.str.startswith(("S_", "W_", "DX", "DW", "DK"))]
    variants = [("Base", base), ("Water front $\\times$2", wv[wv.lever == "W_f2"]),
                ("Water front $\\times$3", wv[wv.lever == "W_f3"]), ("WRC not scaled", wv[wv.lever == "W_noscale"])]
    res = {}
    for nm, x in variants:
        b = F.best(x[x.cold != "lng"], ["site", "sorbent"])
        g = b.groupby("sorbent").ratio
        res[nm] = pd.DataFrame({"lo": g.min(), "hi": g.max(), "win": g.apply(lambda v: (v < 1).sum())})
    order = ["13X"] + sorted(s for s in sorbs if s.startswith("PR:")) + \
            ["P_KH0.03_Q50_mid_S20000", "P_KH0.03_Q50_phob_S20000"]
    lab = {"P_KH0.03_Q50_mid_S20000": "Model adsorbent, intermediate (WRC 0.13)",
           "P_KH0.03_Q50_phob_S20000": "Model adsorbent, hydrophobic (WRC 0.91)"}
    rows = []
    for s in order:
        cells = [f"{r.at[s, 'lo']:.2f}--{r.at[s, 'hi']:.2f} ({int(r.at[s, 'win'])})" for r in res.values()]
        rows.append(" & ".join([lab.get(s, F.short(s, 20))] + cells) + r" \\")
    _write("si_water.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Sensitivity to the water co-adsorption model: "
           "relative cost without LNG, range over 21 sites (number of sites below amine cost), each adsorbent with its "
           "lowest-cost configuration. PrISMa MOFs: the ten lowest-cost MOFs of the first screening stage. Water front: "
           "length of the water-loaded zone relative to the ideal shock front. WRC not scaled: the WRC penalty is "
           "applied as given in PrISMa, independent of the water loading (Eq.~1 of the main text).}\n\\label{tab:si_water}\n"
           "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lllll}\n\\toprule\n"
           "Adsorbent & " + " & ".join(n for n, _ in variants) + " \\\\\n\\midrule\n" + "\n".join(rows).replace("_", "\\_") +
           "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return res


def si_selectivity(d):
    h = d[d.lever.isin(["RQ5", "RQ5free"]) & d.sorbent.str.startswith("P_") & (d.cold != "lng")].copy()
    ex = h.sorbent.str.extract(r"P_KH(?P<KH>[\d.]+)_Q(?P<Q>[\d.]+)_(?P<water>\w+?)_S(?P<S>[\d.]+)$")
    h = pd.concat([h, ex], axis=1)
    h["KH"], h["Q"], h["S"] = h.KH.astype(float), h.Q.astype(float), h.S.astype(float)
    bs = F.best(h, ["site", "sorbent"])
    t = bs.groupby(["water", "Q", "KH", "S"]).ratio.agg(med="median", win=lambda v: int((v < 1).sum())).unstack("S")
    rows = []
    wl = {"phob": "Hydrophobic", "mid": "Intermediate"}
    for wat in ("phob", "mid"):
        for q in (40.0, 50.0):
            for kh in sorted(bs.KH.unique()):
                r = t.loc[(wat, q, kh)]
                rows.append([wl[wat] if (q, kh) == (40.0, min(bs.KH)) else "", f"{q:.0f}", f"{kh:g}",
                             f"{r[('med', 2000.0)]:.2f} ({int(r[('win', 2000.0)])})",
                             f"{r[('med', 20000.0)]:.2f} ({int(r[('win', 20000.0)])})"])
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write("si_selectivity.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Effect of the CO$_2$/N$_2$ selectivity of the "
           "model adsorbents: median relative cost without LNG over 21 sites (number of sites below amine "
           "cost). $K_H$ in mol kg$^{-1}$ Pa$^{-1}$, $Q_{st}$ in kJ mol$^{-1}$. Not shown: $Q_{st}$ = 30 and the "
           "hydrophilic class, for which no case is below amine cost.}\n\\label{tab:si_sel}\n"
           "\\begin{tabular}{lrrll}\n\\toprule\n"
           "Water class & $Q_{st}$ & $K_H$ & $S$ = 2\\,000 & $S$ = 20\\,000 \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", P2TAB)
    return t


def cpqs_table(d):
    """RQ5 extension: hydrophobic target (Q_st 50, S 20000) with solid heat capacity x saturation capacity x K_H."""
    from .study5 import CP_GRID, KH_CP, QS_GRID, cp_name
    x = d[d.lever == "RQ5cp"]
    if not len(x):
        return None
    b = F.best(x, ["site", "sorbent"])
    rows, out = [], []
    for kh in KH_CP:
        for qs in QS_GRID:
            cells = []
            for cp in CP_GRID:
                g = b[b.sorbent == cp_name(kh, cp, qs)]
                med, win = g.ratio.median(), int((g.ratio < 1).sum())
                out.append(dict(KH=kh, qs=qs, cp=cp, ratio=med, wins=win, n=len(g), heat=g.heat_GJ.median(),
                                wc=g.wcap_mmolg.median(), amb=(g.cold == "ambient").mean()))
                cells.append(f"{med:.2f} ({win})" if len(g) else "--")
            rows.append([f"{kh:g}" if qs == QS_GRID[0] else "", f"{qs:g}"] + cells)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    for kh in KH_CP[1:]:
        body = body.replace(f"\n{kh:g} &", f"\n\\midrule\n{kh:g} &")
    hdr = " & ".join(f"{cp:g}" for cp in CP_GRID)
    _write("cpqs.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n"
           "\\caption{Effect of the heat capacity $c_p$ and the saturation capacity $q_s$ of a hydrophobic adsorbent "
           "($Q_{st}$ = 50\\,kJ\\,mol$^{-1}$, $S$ = 20\\,000): median relative cost without LNG over 21 sites (number of "
           "sites below amine cost), each with its lowest-cost configuration. A contactor heat capacity of "
           "0.3\\,kJ\\,kg$^{-1}$\\,K$^{-1}$ per kg adsorbent is added to $c_p$.}\n\\label{tab:cpqs}\n"
           "\\begin{tabular}{rrrrrr}\n\\toprule\n"
           "$K_H$ & $q_s$ & \\multicolumn{4}{c}{$c_p$ (kJ\\,kg$^{-1}$\\,K$^{-1}$)} \\\\\n\\cmidrule(lr){3-6}\n"
           f"(mol\\,kg$^{{-1}}$\\,Pa$^{{-1}}$) & (mmol\\,g$^{{-1}}$) & {hdr} \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", P2TAB)
    return pd.DataFrame(out)


def si_variants(a):
    """Model-assumption variants: cycle rate, purity model, and exclusion of IAST-flagged MOFs."""
    c = pd.read_csv(F.RES / "cases.csv", low_memory=False)
    c["ratio"] = c.lcoc / c.site.map(a.amine_lcoc_base)
    ok = c[c.status == "ok"]

    def cell(x):
        v = F.best(x, ["site"]).ratio
        return [f"{v.median():.2f} ({v.min():.2f}--{v.max():.2f})", f"{int((v < 1).sum())}"]

    base13 = ok[(ok.sorbent == "13X") & ok.lever.isin(["RQ1x2", "B0"]) & (ok.drying == "freeze") & (ok.cold == "vcr")]
    baset = ok[(ok.sorbent == F.TARGET) & ok.lever.isin(["RQ5", "RQ5free"]) & (ok.drying == "none") & (ok.cold == "ambient")]
    s2 = ok[ok.lever.isin(["PR2", "PRL"])]
    flag = F.iast_flagged()
    rows = [["13X, freeze-out, heat pump", "Base"] + cell(base13),
            ["", "One cycle per hour at most"] + cell(ok[ok.lever == "S_cyc1_13X"]),
            ["", "Purity from Henry selectivity"] + cell(ok[ok.lever == "S_henry_13X"]),
            ["Reference adsorbent, no drying, ambient", "Base"] + cell(baset),
            ["", "One cycle per hour at most"] + cell(ok[ok.lever == "S_cyc1_target"]),
            ["", "Purity from Henry selectivity"] + cell(ok[ok.lever == "S_henry_target"]),
            ["Best MOF without LNG", "All stage-2 MOFs"] + cell(s2[s2.cold != "lng"]),
            ["", "IAST-flagged MOFs excluded"] + cell(s2[(s2.cold != "lng") & ~s2.sorbent.isin(flag)]),
            ["Best MOF with LNG", "All stage-2 MOFs"] + cell(s2[s2.cold == "lng"]),
            ["", "IAST-flagged MOFs excluded"] + cell(s2[(s2.cold == "lng") & ~s2.sorbent.isin(flag)])]
    n_flag = len(set(s2.sorbent) & flag)
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    _write("si_variants.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Variants of the model assumptions: relative cost, "
           "median (range) over 21 sites, and number of sites below amine cost. IAST-flagged: MOFs for which the "
           "binary IAST prediction of CO$_2$ or N$_2$ loading deviates from GCMC by more than 0.1 (logarithmic ratio, "
           f"PrISMa flag); {n_flag} of the {s2.sorbent.nunique()} stage-2 MOFs, mostly because of the N$_2$ loading. "
           "With LNG, Tibet (0.995) is counted as below the amine cost; the main text treats it as parity.}}\n"
           "\\label{tab:si_variants}\n\\begin{tabular}{llll}\n\\toprule\n"
           "Configuration & Variant & Relative cost & Sites \\\\\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return rows


def si_sites():
    """Coordinates and elevation of the ERA5 grid points used for the sites."""
    k = pd.read_csv(F.RES / "sites_koppen.csv").set_index("site").reindex(F.ORDER)

    def deg(v, pos, neg):
        return f"{abs(v):.2f}\\,°{pos if v >= 0 else neg}"

    rows = [f"{s} & {F.KOP[s]} & {deg(r.grid_lat, 'N', 'S')} & {deg(r.grid_lon, 'E', 'W')} & "
            + f"{r.elev_m:,.0f} & {r.MAP_mm:,.0f}".replace(",", "{,}") + " \\\\" for s, r in k.iterrows()]
    _write("si_sites.tex",
           "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{ERA5 grid points of the 21 sites: coordinates, "
           "elevation of the grid cell and mean annual precipitation (1991--2020).}\n\\label{tab:si_sites}\n"
           "\\begin{tabular}{llrrrr}\n\\toprule\n"
           "Site & Köppen & Latitude & Longitude & Elevation (m) & Precipitation (mm yr$^{-1}$) \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    F.rc()
    d, a = F.load()
    t = cpqs_table(d)
    if t is not None:
        print(t.round(3).to_string())
    print("screening", screening_table())
    dt = drying_table(d)
    print(dt.round(2).to_string())
    for c in cop_table():
        print(c)
    print("climate", {k: round(v, 2) for k, v in fig_climate(d, a).items()})
    print("map", fig_map(d).round(2).to_string())
    si_params()
    si_mc()
    print(si_yamin())
    si_water(a)
    si_selectivity(d)
    si_variants(a)
    si_sites()


if __name__ == "__main__":
    main()
