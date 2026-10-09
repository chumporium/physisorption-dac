"""Journal figures and tables for Paper 1 (English), from results_dac/study5.

python -m dacsys.fig_paper1 -> docs/paper1/figs/*.pdf|png, docs/paper1/tables/*.tex
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .climate_sites import SITES  # noqa: E402
from .levers import COLD, DRYING, NOT_APPLICABLE, Prm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results_dac" / "study5"
FIG = ROOT / "docs" / "paper1" / "figs"
TAB = ROOT / "docs" / "paper1" / "tables"

# categorical order (validated: dataviz validate_palette, light surface)
C_BLUE, C_ORANGE, C_GREEN, C_PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a63d2"
INK, INK2, MUTED, GRID, NEUTRAL, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#c3c2b7", "#ffffff"
# diverging, log10(ratio) centred on 1: green = physisorption cheaper, orange = amine cheaper
DIV = LinearSegmentedColormap.from_list("div", ["#12875f", "#7fd0ae", "#efeeea", "#f3a47f", "#b8481c"])
DW, SW = 7.48, 3.54                                   # double / single column width [in]

TARGET = "P_KH0.03_Q50_phob_S20000"
DRY_L = {"none": "No drying", "cond": "Condensation", "cond_silica": "Condensation\nand silica gel",
         "freeze": "Freeze-out\n(regenerators)"}
COLD_L = {"ambient": "No chiller", "gax": "Solar GAX\nabsorption", "vcr": "Heat pump\nR290/R744",
          "lng": "Free LNG\ncold"}
KOP = {s: v[2] for s, v in SITES.items()}
ORDER = list(SITES)


def rc():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.5,
        "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": NEUTRAL, "axes.linewidth": 0.6, "xtick.color": INK2,
        "ytick.color": INK2, "axes.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.dpi": 400, "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42})


def style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=500)
    plt.close(fig)


def logx(ax, ticks):
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, weight="bold", color=INK, ha="right", va="bottom")


def lab(s):
    return f"{s} ({KOP[s]})"


# ------------------------------------------------------------------ data
def load():
    d = pd.read_csv(RES / "cases.csv")
    a = pd.read_csv(RES / "amine.csv").set_index("site")
    d = d.join(a[["amine_lcoc_lo", "amine_lcoc_base", "amine_lcoc_hi"]], on="site")
    d["ratio"] = d.lcoc / d.amine_lcoc_base
    d = d[~d.lever.str.startswith(("S_", "W_", "PR1", "DB1", "DX", "DW", "DK"))]
    return d, a


def best(x, by):
    x = x[x.status == "ok"]
    return x.loc[x.groupby(by).lcoc.idxmin()]


def m13(d):
    return d[(d.sorbent == "13X") & d.lever.isin(["B0", "RQ1x2"])]


def iast_flagged(threshold=0.1):
    """PrISMa MOFs whose binary IAST prediction (CO2 or N2) deviates from GCMC (log ratio > threshold)."""
    f = pd.read_csv(ROOT / "data" / "prisma" / "Flags" / "IAST-Flag_NGCC-onshore.csv").set_index("MOF")
    return {"PR:" + m for m in f.index[(f.CO2 > threshold) | (f.N2 > threshold)]}


def tgt(d, name=TARGET):
    return d[(d.sorbent == name) & d.lever.isin(["RQ5", "RQ5free"]) & (d.cold != "lng")]


# ------------------------------------------------------------------ Fig 1 schematic
def fig1():
    """Plant schematic: energy supply (top), air train (middle), products (bottom); orthogonal energy buses."""
    AIR, HEAT, ELEC, COLDC, WATER, PROD, TAG = "#6f7f95", C_ORANGE, "#b58f10", C_BLUE, C_GREEN, INK2, C_PURPLE
    fig, ax = plt.subplots(figsize=(DW, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x0, x1, y0, y1, text, fc="#ffffff", bold=False):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8", fc=fc,
                                    ec=INK2, lw=0.7, zorder=3))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=6.6, color=INK, zorder=4,
                weight="bold" if bold else "normal", linespacing=1.15)

    def line(pts, c, lw=1.1, arrow=True):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys, color=c, lw=lw, zorder=2, solid_capstyle="butt")
        if arrow:
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=c, lw=lw,
                                         shrinkA=0, shrinkB=0, zorder=2))

    def tag(x, y, text):
        ax.text(x, y, text, ha="center", va="center", fontsize=5.8, color="#ffffff", weight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.22", fc=TAG, ec="none"))

    # background bands
    ax.add_patch(FancyBboxPatch((0.5, 35.5), 90, 13.2, boxstyle="round,pad=0,rounding_size=1", fc="#f4f3ee",
                                ec="none", zorder=0))
    ax.text(1.5, 47.4, "Energy and cold supply (off-grid; LNG cold as external option)", fontsize=6.6, color=INK2,
            va="center")
    ax.text(1.5, 27.8, "Air train", fontsize=6.6, color=INK2, va="center")

    # top row: energy supply
    ty0, ty1 = 38, 45
    box(2, 16, ty0, ty1, "Solar field\n(PTC, 230 °C)")
    box(20, 32, ty0, ty1, "Thermal\nstorage")
    box(43, 63, ty0, ty1, "Cold source\nGAX, heat pump or LNG", bold=True)
    box(74, 88, ty0, ty1, "PV and battery")
    # middle row: air train
    my0, my1 = 17, 25
    mid = [(2, 11, "Ambient\nair"), (14, 24, "Coil 1\n(+3 °C)\noptional"), (27, 37, "Dryer\noptional"),
           (40, 51, "Recuperator"),
           (54, 64, "Coil 2\n($T_{ads}$)"), (67, 79, "Adsorbent\ncontactor")]
    for x0, x1, t in mid:
        box(x0, x1, my0, my1, t, fc="#eef3fa" if t in ("Ambient\nair", "Exhaust") else "#ffffff")
    for (a0, a1, _), (b0, b1, _) in zip(mid[:-1], mid[1:]):
        line([(a1, (my0 + my1) / 2), (b0, (my0 + my1) / 2)], AIR, lw=1.3)
    # bottom row: products
    by0, by1 = 3, 9.5
    box(14, 37, by0, by1, "Product water", fc="#eef8f3")
    box(67, 79, by0, by1, "CO$_2$ (≥ 95 %)", fc="#eef8f3")
    box(40, 51, by0, by1, "Exhaust", fc="#eef3fa")
    # return air: contactor -> recuperator (cold recovery) -> exhaust
    line([(69.5, my0), (69.5, 13), (48.5, 13), (48.5, my0)], AIR, lw=1.3)
    line([(43, my0), (43, by1)], AIR, lw=1.3)
    ax.text(59, 13.5, "CO$_2$-depleted air", fontsize=5.8, color=AIR, ha="center", va="bottom")

    # heat: collectors -> storage -> GAX generator; heat bus to dryer (silica gel) and contactor (desorption)
    line([(16, 41.5), (20, 41.5)], HEAT)
    line([(32, 41.5), (43, 41.5)], HEAT)
    yh = 32.5
    line([(26, ty0), (26, yh), (73, yh), (73, my1)], HEAT)
    line([(32, yh), (32, my1)], HEAT)
    ax.text(40, yh + 0.9, "heat", fontsize=5.8, color=HEAT, ha="center", va="bottom")
    # cold: cold source -> coils
    yc = 29
    line([(53, ty0), (53, yc), (19, yc), (19, my1)], COLDC)
    line([(59, yc), (59, my1)], COLDC, arrow=False)
    line([(53, yc), (59, yc), (59, my1)], COLDC)
    ax.text(44, yc - 0.6, "cold", fontsize=5.8, color=COLDC, ha="center", va="top")
    # electricity: heat pump, fans and vacuum pumps
    line([(74, 40), (63, 40)], ELEC)
    line([(77, ty0), (77, my1)], ELEC)
    ax.text(68.5, 40.8, "electricity", fontsize=5.8, color=ELEC, ha="center", va="bottom")
    # products
    line([(19, my0), (19, by1)], WATER)
    line([(32, my0), (32, by1)], WATER)
    line([(75.5, my0), (75.5, by1)], PROD, lw=1.3)

    # legend
    lx, ly = 83, 13
    for k, (c, t) in enumerate(((AIR, "air"), (HEAT, "heat"), (COLDC, "cold"), (ELEC, "electricity"),
                                (WATER, "water"), (PROD, "CO$_2$"))):
        yy = ly - k * 2.0
        if t == "questions 1 to 5":
            ax.add_patch(FancyBboxPatch((lx, yy - 0.6), 3, 1.2, boxstyle="round,pad=0,rounding_size=0.4", fc=c,
                                        ec="none"))
        else:
            ax.plot([lx, lx + 3], [yy, yy], color=c, lw=1.3)
        ax.text(lx + 4, yy, t, fontsize=5.8, color=INK2, va="center")
    save(fig, "fig1_system")


# ------------------------------------------------------------------ Fig 2 Kim reconciliation
KIM_EN = [("K0", "Assumptions of Kim et al. (grid power and gas)"), ("plant life", "Plant life 25 instead of 15 yr"),
          ("capture fraction", "Capture fraction 0.85 instead of 0.60; pressure drop and fan of this model"),
          ("cycle rate", "Cycle rate limited by heat transfer instead of 1 h$^{-1}$"),
          ("heat-exchanger cost", "Heat exchangers: correlation of Weiland et al., installed"),
          ("O&M", "Process O&M 3 % instead of 0"),
          ("adsorption temperature", "Adsorption temperature optimised"),
          ("regeneration", "Vacuum at 100 °C instead of sweep at 200 °C"), ("NETL", "Fans and vacuum pumps: NETL costs"),
          ("grid power and gas ->", "Off-grid solar instead of grid and gas"),
          ("best dryer", "No LNG: best dryer and heat pump"), ("B0", "No LNG: reference plant B0")]


def fig2():
    k = pd.read_csv(RES / "kim_waterfall.csv")
    names = []
    for s in k.step:
        names.append(next((en for key, en in KIM_EN if key in s), s).replace("\\$", "$"))
    fig, ax = plt.subplots(figsize=(SW * 1.45, 2.6))
    style(ax)
    y = np.arange(len(k))[::-1]
    grid_mode = (k.energy_mode == "grid").to_numpy()
    ax.barh(y, k.lcoc, color=np.where(grid_mode, C_BLUE, C_ORANGE), height=0.62)
    ax.scatter(k.amine_base, y, marker="|", s=60, color=INK, lw=1.2, zorder=3, label="amine, same energy basis")
    for yi, v, am in zip(y, k.lcoc, k.amine_base):
        ax.text(max(v, am if am < v * 1.3 else 0) * 1.08, yi, f"{v:,.0f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(68.2, color=INK2, lw=0.8, ls="--")
    ax.text(70, y[0] + 0.55, "Kim et al.: 68 USD/t", fontsize=6.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(40, 5000)
    logx(ax, [50, 100, 200, 500, 1000, 2000])
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Levelised cost of capture, 13X, Madrid (USD per t CO$_2$)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_BLUE, label="grid power and gas (Kim et al. basis)"),
                       Patch(color=C_ORANGE, label="off-grid solar (this work)"),
                       plt.Line2D([], [], marker="|", ls="", color=INK, ms=8, label="amine, same basis")],
              loc="upper center", bbox_to_anchor=(0.3, -0.16), ncol=2, frameon=False)
    save(fig, "fig2_kim")


# ------------------------------------------------------------------ database helpers
def db(d):
    """Stage-2 PrISMa cases: the best MOFs of the stage-1 screen x every dryer x cold source x site."""
    return d[d.lever.isin(["PR2", "PRL"])]


def short(n, k=12):
    m = n[3:] if n.startswith("PR:") else n
    m = m.replace("_clean", "").replace("MgMOF74", "Mg-MOF-74")
    return m if len(m) <= k else m[:k - 1] + "…"


def heat(ax, fig, med, txt, cb_label, letter, x_panel):
    norm = TwoSlopeNorm(vmin=np.log10(0.25), vcenter=0.0, vmax=np.log10(30))
    im = ax.imshow(np.log10(med), cmap=DIV, norm=norm, aspect="auto")
    for (i, j), t in txt.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=6.4, color=MUTED if t == "n/a" else INK)
    return im


# ------------------------------------------------------------------ Fig 3 drying x cooling
def fig3(d):
    m, p = m13(d), db(d)
    pok = p[p.status == "ok"]
    env = pok.groupby(["site", "drying", "cold"]).ratio.min()          # best MOF per site and lever cell
    fig, axes = plt.subplots(1, 2, figsize=(DW, 3.6), gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    med = np.full((len(DRYING), len(COLD)), np.nan)
    txt = {}
    for i, dr in enumerate(DRYING):
        for j, co in enumerate(COLD):
            if (co, dr) in NOT_APPLICABLE:
                txt[(i, j)] = "n/a"
                continue
            e = env.xs((dr, co), level=("drying", "cold")) if (dr, co) in env.droplevel(0).index else pd.Series(dtype=float)
            x13 = m[(m.drying == dr) & (m.cold == co) & (m.status == "ok")].ratio
            if len(e):
                med[i, j] = e.median()
            txt[(i, j)] = (f"{e.median():.2f}" if len(e) else "–") + (f"\n13X: {x13.median():.2f}" if len(x13) else "\n13X –")
    im = heat(ax, fig, med, txt, None, "a", -0.28)
    b0 = (DRYING.index("cond_silica"), COLD.index("gax"))
    ax.add_patch(plt.Rectangle((b0[1] - 0.5, b0[0] - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ax.text(b0[1] + 0.45, b0[0] - 0.4, "B0", ha="right", va="top", fontsize=6.5, weight="bold", color=INK)
    ax.set_xticks(range(len(COLD)), [COLD_L[c] for c in COLD])
    ax.set_yticks(range(len(DRYING)), [DRY_L[x] for x in DRYING])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    ticks = [0.25, 0.5, 1, 2, 5, 10, 30]
    cb.set_ticks(np.log10(ticks), labels=[f"{t:g}" for t in ticks])
    cb.set_label("relative cost, best MOF (median)")
    cb.outline.set_visible(False)
    panel(ax, "a", x=-0.28)
    ax = axes[1]
    style(ax)
    b = pok.groupby(["site", "cold"]).ratio.min()
    y = np.arange(len(ORDER))[::-1]
    for co, col, mk in (("ambient", NEUTRAL, "o"), ("gax", C_ORANGE, "s"), ("vcr", C_BLUE, "D"), ("lng", C_GREEN, "^")):
        v = [b.get((s_, co), np.nan) for s_ in ORDER]
        # GAX drawn larger and underneath: where it ties with the heat pump (chiller hardly used) it stays visible
        ax.scatter(v, y, s=30 if co == "gax" else 16, color=col, marker=mk, edgecolor=SURF, lw=0.6,
                   zorder=2 if co == "gax" else 3,
                   label="best MOF, " + {"ambient": "no chiller", "gax": "solar GAX", "vcr": "heat pump",
                                         "lng": "free LNG cold"}[co])
    x13 = m[(m.status == "ok") & (m.cold != "lng")].groupby("site").ratio.min()
    ax.scatter([x13.get(s_, np.nan) for s_ in ORDER], y, s=22, facecolor="none", edgecolor=INK, lw=0.7, zorder=4,
               label="13X, best without LNG")
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.25, 40)
    logx(ax, [0.3, 0.5, 1, 2, 5, 10, 20])
    ax.set_yticks(y, [lab(s_) for s_ in ORDER])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Relative cost, best MOF and dryer per cold source")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.13), ncol=2, frameon=False, handletextpad=0.2)
    panel(ax, "b", x=-0.33)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_drying_cooling")


# ------------------------------------------------------------------ Fig 4 location + Monte Carlo
def fig4():
    mc = pd.read_csv(RES / "mc.csv")
    mc = mc[np.isfinite(mc.ratio)]
    cfg = [("prisma", C_BLUE, "o", "best PrISMa MOF, without LNG"), ("13X_best", NEUTRAL, "D", "13X, without LNG"),
           ("prisma_lng", C_PURPLE, "v", "best PrISMa MOF, with LNG cold"), ("13X_lng", C_ORANGE, "^", "13X, with LNG cold"),
           ("target", C_GREEN, "s", "reference adsorbent, ambient")]
    cfg = [c for c in cfg if c[0] in set(mc.config)]
    fig, ax = plt.subplots(figsize=(SW * 1.45, 4.6))
    style(ax)
    y = np.arange(len(ORDER))[::-1]
    off = {"prisma": 0.32, "13X_best": 0.16, "prisma_lng": 0.0, "13X_lng": -0.16, "target": -0.32}
    for c, col, mk, name in cfg:
        g = mc[mc.config == c].groupby("site").ratio
        q10, q50, q90 = g.quantile(0.1), g.median(), g.quantile(0.9)
        for yi, s_ in zip(y, ORDER):
            if s_ in q50.index:
                ax.plot([q10[s_], q90[s_]], [yi + off[c]] * 2, color=col, lw=1.4, alpha=0.5, solid_capstyle="round")
        ax.scatter([q50.get(s_, np.nan) for s_ in ORDER], y + off[c], s=12, marker=mk, color=col, edgecolor=SURF,
                   lw=0.5, zorder=3, label=name)
    ax.axvline(1, color=INK, lw=0.8)
    ax.axvspan(0.1, 1, color=C_GREEN, alpha=0.05, lw=0)
    ax.text(0.17, -1.2, "physisorption cheaper", ha="left", va="bottom", fontsize=6.3, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 25)
    logx(ax, [0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_ylim(-1.3, len(ORDER) - 0.3)
    ax.set_yticks(y, [lab(s_) for s_ in ORDER])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Relative cost (Monte Carlo median; bar: 10th to 90th percentile)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.08), ncol=2, frameon=False)
    save(fig, "fig4_location_mc")


# ------------------------------------------------------------------ Fig 5 water
def db_best(d, lng=False):
    p = db(d)
    p = p[(p.cold == "lng") if lng else (p.cold != "lng")]
    return best(p, ["site"]).set_index("site").reindex(ORDER)


def fig5(d):
    b = db_best(d)
    need = (b.lcoc - b.amine_lcoc_base) / b.water_per_co2
    fig, ax = plt.subplots(figsize=(SW * 1.45, 3.4))
    style(ax)
    y = np.arange(len(ORDER))[::-1]
    ax.axvspan(0.38, 2.97, color=C_BLUE, alpha=0.12, lw=0)          # 0.35-2.7 EUR/m3 (desal2024), 1.1 USD/EUR
    ax.text(1.05, -1.0, "seawater desalination", ha="center", va="center", fontsize=6.0, color=INK2)
    ax.set_ylim(-1.5, len(ORDER) - 0.4)
    ax.barh(y, need, color=C_ORANGE, height=0.62)
    for yi, s_ in zip(y, ORDER):
        ax.text(need[s_] * 1.08, yi, f"{need[s_]:.0f}; {b.at[s_, 'water_per_co2']:.1f}", va="center", fontsize=6.2, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2000)
    logx(ax, [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    ax.set_yticks(y, [lab(s_) for s_ in ORDER])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Water price for cost parity with amine DAC (USD m$^{-3}$)\n"
                  "bar labels: price (USD m$^{-3}$); water recovered (t per t CO$_2$)")
    save(fig, "fig5_water")
    return need, b.water_per_co2


# ------------------------------------------------------------------ Fig 6 material envelope + database
def fig6(d):
    from . import prisma
    p = d[d.lever.isin(["RQ5", "RQ5free"]) & d.sorbent.str.startswith("P_") & (d.cold != "lng")].copy()
    ex = p.sorbent.str.extract(r"P_KH(?P<KH>[\d.]+)_Q(?P<Q>[\d.]+)_(?P<water>\w+?)_S(?P<S>[\d.]+)$")
    p = pd.concat([p, ex], axis=1)
    p["KH"], p["Q"], p["S"] = p.KH.astype(float), p.Q.astype(float), p.S.astype(float)
    bs = best(p, ["site", "sorbent"])
    KHs, Qs = sorted(p.KH.unique()), sorted(p.Q.unique())
    norm = TwoSlopeNorm(vmin=np.log10(0.25), vcenter=0.0, vmax=np.log10(30))
    fig, axes = plt.subplots(1, 4, figsize=(DW, 3.1), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35]),
                             layout="constrained")
    titles = {"phob": "Hydrophobic (WRC 0.91)", "mid": "Intermediate (0.13)*", "phil": "Hydrophilic (0.002)"}
    for k, (ax, wat, letter) in enumerate(zip(axes[:3], ("phob", "mid", "phil"), "abc")):
        x = bs[(bs.water == wat) & (bs.S == 20000.0)]
        mn = np.full((len(KHs), len(Qs)), np.nan)
        for i, kh in enumerate(KHs):
            for j, q in enumerate(Qs):
                c = x[(x.KH == kh) & (x.Q == q)]
                if len(c):
                    mn[i, j] = c.ratio.median()
                    ax.text(j, i, f"{mn[i, j]:.2f}\n{(c.ratio < 1).sum()}/{len(c)}", ha="center", va="center", fontsize=5.8, color=INK)
        im = ax.imshow(np.log10(mn), cmap=DIV, norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(Qs)), [f"{q:.0f}" for q in Qs])
        ax.set_yticks(range(len(KHs)), [f"{kk:g}" for kk in KHs] if k == 0 else [])
        ax.set_xlabel("$Q_{st}$ (kJ mol$^{-1}$)")
        ax.set_title(titles[wat], loc="left", color=INK, fontsize=7)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        panel(ax, letter, x=-0.05)
    axes[0].set_ylabel("$K_{H}$ CO$_2$, 298 K (mol kg$^{-1}$ Pa$^{-1}$)")
    cb = fig.colorbar(im, ax=axes[:3], shrink=0.7, pad=0.01, location="bottom", aspect=35)
    ticks = [0.25, 0.5, 1, 2, 5, 10, 30]
    cb.set_ticks(np.log10(ticks), labels=[f"{t:g}" for t in ticks])
    cb.set_label("median relative cost (a–c)")
    cb.outline.set_visible(False)
    # (d) where the real materials sit: best cost ratio of every screened PrISMa MOF vs its CO2 affinity
    ax = axes[3]
    style(ax, grid="y")
    fits = prisma.load_fits()
    full = pd.read_csv(RES / "cases.csv")
    a = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    full = full[full.lever.isin(["PR1", "DB1", "PR2", "DB3"]) & (full.status == "ok") & (full.cold != "lng")]
    r = (full.lcoc / full.site.map(a)).groupby(full.sorbent).min()
    mofs = pd.DataFrame({"r": r})
    mofs["KH"] = [fits[n[3:]]["KH_CO2"] for n in mofs.index]
    mofs["WRC"] = [fits[n[3:]]["h2o"][3] for n in mofs.index]
    mofs["exp"] = [prisma.origin(n) == "eksperimental" for n in mofs.index]
    for sel, col, name in ((mofs.WRC < 0.05, C_ORANGE, "WRC < 0.05"), ((mofs.WRC >= 0.05) & (mofs.WRC < 0.5), C_BLUE, "0.05–0.5"),
                           (mofs.WRC >= 0.5, C_GREEN, "WRC ≥ 0.5")):
        x = mofs[sel]
        ax.scatter(x.KH, x.r, s=np.where(x.exp, 7, 5), color=col, alpha=0.75, lw=0, label=name,
                   marker="o")
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 30)
    ax.set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("$K_H$ CO$_2$, 298 K (mol kg$^{-1}$ Pa$^{-1}$)")
    ax.set_ylabel("Lowest relative cost, without LNG")
    ax.set_title(f"{len(mofs)} PrISMa MOFs", loc="left", color=INK, fontsize=7)
    ax.legend(loc="upper right", frameon=False, markerscale=2, handletextpad=0.1, fontsize=6)
    panel(ax, "d", x=-0.2)
    save(fig, "fig6_materials")
    return mofs


# ------------------------------------------------------------------ Fig 7 unifying waterfall
REP = ["Singapore", "Riyadh", "London", "Kiruna", "Tibet"]


def fig7(d, a):
    m, p = m13(d), db(d)
    fig, axes = plt.subplots(1, len(REP), figsize=(DW, 2.6), sharey=True)
    steps = ["B0", "MOF,\nGAX", "No\nLNG", "Ref.\nads.", "LNG"]
    cols = [NEUTRAL, C_ORANGE, C_BLUE, C_GREEN, "#ffffff"]
    vals = {}
    for ax, site in zip(axes, REP):
        style(ax, grid="y")
        x, q = m[m.site == site], p[p.site == site]
        v = [x[(x.lever == "B0") & (x.status == "ok")].lcoc.min(),
             best(q[q.cold == "gax"], ["site"]).lcoc.min(),
             best(q[q.cold != "lng"], ["site"]).lcoc.min(),
             best(tgt(d[d.site == site]), ["site"]).lcoc.min(),
             best(q[q.cold == "lng"], ["site"]).lcoc.min()]
        vals[site] = v
        bars = ax.bar(range(5), v, color=cols, width=0.68, edgecolor=[SURF] * 4 + [C_GREEN], lw=[0.8] * 4 + [1.0],
                      hatch=[None] * 4 + ["////"])
        bars[4].set_edgecolor(C_GREEN)
        base = a.at[site, "amine_lcoc_base"]
        for i, vi in enumerate(v):
            yl = vi * 1.12
            if 0.86 < yl / base < 1.22:               # the label would be crossed by the amine line: place it above
                yl = base * 1.1
            ax.text(i, yl, f"{vi:,.0f}", ha="center", fontsize=5.8, color=INK)
        ax.axhspan(a.at[site, "amine_lcoc_lo"], a.at[site, "amine_lcoc_hi"], color=INK, alpha=0.07, lw=0)
        ax.axhline(a.at[site, "amine_lcoc_base"], color=INK, lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(100, 7000)
        ax.set_yticks([100, 200, 500, 1000, 2000, 5000], ["100", "200", "500", "1000", "2000", "5000"])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(range(5), steps, fontsize=6.5)
        ax.set_title(lab(site), loc="left", fontsize=7, color=INK)
    axes[0].set_ylabel("LCOC (USD t$^{-1}$)")
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig7_waterfall")
    return vals


# ------------------------------------------------------------------ tables
def tables(d, a):
    TAB.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv(RES / "sites_koppen.csv").set_index("site")
    m = m13(d)
    nl = best(m[m.cold != "lng"], ["site"]).set_index("site")
    dbn, dbl = db_best(d), db_best(d, lng=True)
    tg = best(tgt(d), ["site"]).set_index("site")
    x13l = best(m[m.cold == "lng"], ["site"]).set_index("site")
    flag = iast_flagged()
    rows = []
    for s_ in ORDER:
        r = k.loc[s_]
        t_, td_ = (f"{v:.1f}".replace("-", "$-$") for v in (r.T_mean_2018_C, r.Td_mean_2018_C))
        mof = short(dbn.at[s_, "sorbent"], 30) + ("$^a$" if dbn.at[s_, "sorbent"] in flag else "")
        rows.append(f"{s_} & {KOP[s_]} & {t_} & {td_} & {r.DNI_kWh_m2_yr:,.0f} & "
                    f"{a.at[s_, 'amine_lcoc_base']:,.0f} & {nl.at[s_, 'ratio']:.2f} & {dbn.at[s_, 'ratio']:.2f} & "
                    f"{mof} & {x13l.at[s_, 'ratio']:.2f} & {dbl.at[s_, 'ratio']:.2f} & {tg.at[s_, 'ratio']:.2f} \\\\")
    body = "\n".join(rows).replace(",", "{,}").replace("_", "\\_")
    (TAB / "sites.tex").write_text(
        "\\begin{table*}[t]\n\\centering\\footnotesize\n"
        "\\caption{Sites, climate data (2018), amine LCOC (base case) and relative cost of physisorption (LCOC divided "
        "by the amine LCOC at the same site), each with the lowest-cost dryer and cold source. Best MOF: lowest-cost "
        "MOF of the second screening stage. Reference: hydrophobic reference adsorbent with $K_H$ = "
        "0.03\\,mol\\,kg$^{-1}$\\,Pa$^{-1}$ and $Q_{st}$ = 50\\,kJ\\,mol$^{-1}$ at ambient temperature. $^a$IAST "
        "prediction for N$_2$ flagged as uncertain in PrISMa.}\n"
        "\\label{tab:sites}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{llrrrrrrlrrr}\n\\toprule\n"
        " & & & & & & \\multicolumn{3}{c}{Without LNG} & \\multicolumn{2}{c}{With LNG} & Ambient \\\\\n"
        "\\cmidrule(lr){7-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-12}\n"
        "Site & Köppen & $\\bar T$ & $\\bar T_d$ & DNI & Amine & 13X & Best MOF & MOF & 13X & Best MOF & Reference \\\\\n"
        " & & (°C) & (°C) & (kWh m$^{-2}$ yr$^{-1}$) & (USD t$^{-1}$) & & & & & & \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n",
        encoding="utf-8")
    p = Prm()
    par = [("CO$_2$ in air", f"{p.x_co2_ppm:.0f} ppm"), ("Capture fraction", f"{p.eta_cap}"),
           ("Product purity", f"$\\geq$ {p.purity_min}"), ("Regeneration", "vacuum at 100 °C, 0.01 / 0.03 / 0.1 bar (optimised)"),
           ("Dryer dew point (silica)", f"${p.dewpoint_dry:.0f}$ °C"), ("Contactor pressure drop", f"{p.dP_contactor:.0f} Pa"),
           ("Vacuum pump efficiency", f"{p.eta_vac}"), ("Max. cycles per hour", f"{p.max_cycles_per_h:.0f}"),
           ("GAX evaporator limit / COP correction factor", "$\\geq -38$ °C / 0.85"),
           ("Discount rate / lifetime", f"{p.discount:.0%} / {p.lifetime} yr".replace("%", "\\%")),
           ("PTC / tank / PV / battery", f"{p.c_ptc:.0f} USD m$^{{-2}}$ / {p.c_tank:.0f} USD kWh$^{{-1}}$ / "
                                         f"{p.c_pv:.0f} USD kW$_p^{{-1}}$ / {p.c_batt:.0f} USD kWh$^{{-1}}$"),
           ("GAX chiller / heat pump", f"{p.c_abs_rej:.0f} / {p.c_hp_heat:.0f} USD per kW heat rejected"),
           ("Dry coolers", f"{p.c_rej:.0f} USD per kW heat rejected"),
           ("Heat exchangers (installed)", f"{p.hx_cost_ua:.2f} USD per W K$^{{-1}}$ ({p.hx_cost_m2:.0f} USD m$^{{-2}}$ "
            f"at 50 W m$^{{-2}}$ K$^{{-1}}$; material factor {p.f_mat_cryo:g} in cryogenic service)"),
           ("Fans / vacuum pumps", f"{p.c_fan:,.0f} / {p.c_vac:,.0f} USD kW$^{{-1}}$".replace(",", "{,}")),
           ("Adsorbent price (13X / MOFs), fabrication factor", f"0.85 / 20 USD kg$^{{-1}}$, $\\times${p.sorbent_fab:.0f}"),
           ("Contactor structure", f"{p.c_contactor:.0f} USD per kg adsorbent $\\times${p.contactor_fab:.0f}"),
           ("Amine capital cost (low / base / high)", " / ".join(f"{c:.0f}" for c in p.amine_capex) + " USD per t yr$^{-1}$")]
    (TAB / "params.tex").write_text(
        "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Key parameters (full list in Table~S1).}\n"
        "\\label{tab:params}\n\\begin{tabular}{p{0.42\\columnwidth}p{0.5\\columnwidth}}\n\\toprule\n"
        "Parameter & Value \\\\\n\\midrule\n"
        + "\n".join(f"{a_} & {b_} \\\\" for a_, b_ in par) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


def main():
    rc()
    d, a = load()
    fig1()
    fig2()
    fig3(d)
    fig4()
    need, wpc = fig5(d)
    mofs = fig6(d)
    vals = fig7(d, a)
    tables(d, a)
    print("water breakeven", need.min().round(1), need.max().round(1), "water t/t", wpc.min().round(1), wpc.max().round(1))
    print("fig6d MOFs", len(mofs), "min ratio", mofs.r.min().round(2), "n<1", int((mofs.r < 1).sum()))
    print("waterfall", {k: [round(x) for x in v] for k, v in vals.items()})
    print("written", FIG, TAB)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ SI: validation against PrISMa (V9)
def fig_validation():
    v = pd.read_csv(RES / "validation_prisma.csv")
    v = v[v.fit_ok & v.purity_prisma.notna() & (v.case == "TVSA02")]
    c = pd.read_csv(RES / "cases.csv")
    s2 = set(c[c.lever.isin(["PR2", "PRL"])].sorbent.str[3:])
    fig, axes = plt.subplots(1, 3, figsize=(DW, 2.5))
    for ax, q, name, log in zip(axes, ("purity", "recovery", "wc"),
                                ("CO$_2$ purity (%)", "CO$_2$ recovery (%)", "working capacity (mol kg$^{-1}$)"),
                                (False, False, True)):
        style(ax, grid=None)
        x, y = v[f"{q}_prisma"], v[f"{q}_ours"]
        ins = v.MOF.isin(s2)
        ax.scatter(x[~ins], y[~ins], s=4, color=NEUTRAL, lw=0, alpha=0.8, label="all MOFs")
        ax.scatter(x[ins], y[ins], s=9, color=C_BLUE, lw=0, label="MOFs used in stage 2")
        lo, hi = (1e-3, 10) if log else (0, 101)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        rho = x.rank().corr(y.rank())
        ax.text(0.04, 0.80 if log else 0.96, f"Spearman {rho:.2f}\nn = {len(v)}", transform=ax.transAxes, va="top",
                fontsize=6.5, color=INK2)
        ax.set_xlabel(f"PrISMa, {name}")
        ax.set_ylabel(f"this work, {name}")
    axes[2].legend(loc="lower right", frameon=False, markerscale=1.8, fontsize=6)
    fig.tight_layout(w_pad=1.0)
    save(fig, "figS1_validation_prisma")
