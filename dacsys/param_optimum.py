"""Where do the adsorbent and regeneration parameters stop paying off? Sweeps beyond the ranges of the main study.

A: model adsorbent (hydrophobic water class, S = 20 000, c_p 0.85, 10 USD/kg) on a grid of Q_st x K_H(298 K)
B: saturation capacity q_s of the target (Q_st 50, K_H 0.03)
C: desorption temperature x vacuum pressure for the target and for Mg-MOF-74
Each variant is re-optimised with three configurations at all 21 sites (the plant model is unchanged).

python -m dacsys.param_optimum [--workers 8] [--report-only]  -> results_dac/study5/param_optimum.csv
"""
from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .fig_paper1 import RES

OUT = RES / "param_optimum.csv"
QST = (40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0)
KH = (0.01, 0.03, 0.1, 0.3, 1.0)
QS = (1.0, 2.0, 4.0, 8.0, 16.0)
T_DES = (80.0, 100.0, 120.0, 150.0)          # °C
P_DES = (0.003, 0.01, 0.03, 0.1)             # bar
ARCH = {"target": (("none", "ambient"), ("cond_silica", "ambient"), ("freeze", "vcr"), ("cond_silica", "vcr")),
        "mof": (("freeze", "vcr"), ("none", "vcr"), ("cond_silica", "ambient"), ("cond", "vcr"), ("cond_silica", "vcr"))}


# D: wide one-at-a-time sweeps of every material property of the model (material_sens.perturb semantics)
SWEEP = {
    "cp": (0.4, 0.6, 1.2, 1.5, 2.0),                      # kJ/kg K (base 0.85)
    "rho": (0.5, 0.8, 1.8, 2.5),                          # g/cm3
    "S_N2": (0.05, 0.1, 0.2, 0.5, 2.0, 5.0),              # factor on the CO2/N2 selectivity
    "Q_N2": (8.0, 12.0, 22.0, 27.0),                      # kJ/mol
    "n2_qs": (1.0, 2.0, 8.0, 16.0),                       # mol/kg
    "wrc": (-0.6, -0.4, -0.2),                            # change (target 0.91; Mg-MOF-74 below 0.05)
    "h2o_KH": (0.01, 0.1, 10.0, 100.0, 1000.0),           # factor
    "h2o_dH": (-15.0, -10.0, 10.0, 20.0, 30.0),           # kJ/mol change
    "h2o_qsat": (0.25, 0.5, 2.0, 4.0),                    # factor
    "E_diff": (10.0, 20.0, 30.0, 40.0),                   # kJ/mol
    "price": (0.25, 0.5, 2.0, 4.0, 8.0),                  # factor
    "life": (1.0, 2.0, 5.0, 10.0),                        # years (base 25)
    "KH": (0.1, 0.3, 3.0, 10.0),                          # factor, Q_st kept
    "Qst": (-10.0, -5.0, 5.0, 10.0, 20.0, 30.0),          # kJ/mol change, 298 K isotherm kept
    "qs": (0.25, 0.5, 2.0, 4.0),                          # factor, K_H kept
}
MOF_PROPS = ("KH", "Qst", "qs", "cp", "S_N2", "h2o_KH", "wrc", "price")
BASES = {"target": "P_KH0.03_Q50_phob_S20000", "mof": "PR:MgMOF74"}


def regen_key(t, p):
    return f"vac{t:g}_{p:g}"


def _add_regen():
    from . import levers
    for t, p in itertools.product(T_DES, P_DES):
        levers.REGEN.setdefault(regen_key(t, p), (t + 273.15, p, "vacuum"))


def variants():
    out = []
    for q, kh in itertools.product(QST, KH):
        out.append(("A", f"PO:Q{q:g}_KH{kh:g}", dict(q=q, kh=kh, qs=4.0), "vac100", "target"))
    for qs in QS:
        out.append(("B", f"PO:Q50_KH0.03_qs{qs:g}", dict(q=50.0, kh=0.03, qs=qs), "vac100", "target"))
    for t, p in itertools.product(T_DES, P_DES):
        out.append(("C", "PO:Q50_KH0.03_qs4", dict(q=50.0, kh=0.03, qs=4.0), regen_key(t, p), "target"))
        out.append(("C", "PR:MgMOF74", None, regen_key(t, p), "mof"))
    for arch, base in BASES.items():
        for prop, vals in SWEEP.items():
            if arch == "mof" and prop not in MOF_PROPS:
                continue
            for v in vals:
                out.append(("D", f"PD:{arch}:{prop}:{v:g}", dict(base=base, prop=prop, value=v), "vac100", arch))
    return out


_W = {}


def _weather(site):
    from . import climate_sites as cs
    from . import plant
    from .levers import Prm
    if site not in _W:
        w = cs.load(site)
        _W[site] = (w, plant.resources(w, Prm()))
    return _W[site]


def _job(v, archs=None, sites=None):
    part, name, par, regen, arch = v
    from . import plant, prisma, sorbents
    from .climate_sites import SITES
    from .levers import Case, with_
    _add_regen()
    if par is None:
        prisma.register([name[3:]])
    elif "prop" in par:
        from dataclasses import replace
        from .material_sens import _register_base, perturb
        s = _register_base(par["base"])
        sorbents.register(replace(perturb(s, par["prop"], par["value"]), name=name))
    else:
        sorbents.register(sorbents.parametric(name, par["kh"], par["q"], par["qs"], S_N2_298=20000.0, cost=10.0,
                                              water="phob"))
    rows = []
    for site in (sites or SITES):
        w, res = _weather(site)
        for d, c in (archs or ARCH[arch]):
            case = with_(Case(site=site), sorbent=name, drying=d, cold=c, lever="PO", regen=regen)
            r = plant.evaluate(case, w, res)
            p_ = dict(q=np.nan, kh=np.nan, qs=np.nan, base="", prop="", value=np.nan)   # same columns in every row
            p_.update(par or {})
            rows.append(dict(part=part, sorbent=name, regen=regen, site=site, drying=d, cold=c,
                             **p_, status=r.get("status"), lcoc=r.get("lcoc", np.nan),
                             heat_GJ=r.get("heat_GJ", np.nan), elec_GJ=r.get("elec_GJ", np.nan),
                             wcap=r.get("wcap_mmolg", np.nan), purity=r.get("purity", np.nan),
                             T_design=r.get("T_design", np.nan)))
    return rows


def run(workers, only=None, sites=None):
    """sites: compute only these (added) sites for every variant; otherwise all sites per missing architecture."""
    jobs = variants()
    if only:
        jobs = jobs[:only]
    done = set()
    old = pd.read_csv(OUT) if OUT.exists() else None
    if sites:
        have = set() if old is None else set(zip(old.part, old.sorbent, old.regen, old.drying, old.cold, old.site))
        todo = [(j, [a for a in ARCH[j[4]] if not all((j[0], j[1], j[3]) + a + (s,) in have for s in sites)])
                for j in jobs]
    else:
        if old is not None:
            done = set(zip(old.part, old.sorbent, old.regen, old.drying, old.cold))
        # per architecture, so that an added architecture is computed without repeating the others
        todo = [(j, [a for a in ARCH[j[4]] if (j[0], j[1], j[3]) + a not in done]) for j in jobs]
    todo = [(j, a) for j, a in todo if a]
    print(len(todo), "of", len(jobs), "variants to run", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, j, a, sites) for j, a in todo]
        for k, f in enumerate(as_completed(futs)):
            pd.DataFrame(f.result()).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(k + 1, "/", len(todo), flush=True)


def summary():
    d = pd.read_csv(OUT)
    a = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    d = d[d.status == "ok"].copy()
    d["ratio"] = d.lcoc / d.site.map(a)
    b = d.loc[d.groupby(["part", "sorbent", "regen", "site"]).ratio.idxmin()]
    return b.groupby(["part", "sorbent", "regen"]).agg(
        ratio=("ratio", "median"), lo=("ratio", "min"), hi=("ratio", "max"), wins=("ratio", lambda v: int((v < 1).sum())),
        n=("ratio", "size"), heat=("heat_GJ", "median"), elec=("elec_GJ", "median"), wc=("wcap", "median")).reset_index()


BASE_VAL = {"Qst": 50.0, "KH": 0.03, "qs": 4.0, "cp": 0.85, "rho": None, "S_N2": 20000.0, "h2o_KH": 1.11e-5,
            "E_diff": 0.0, "price": 10.0, "life": 25.0}
ABS = {"cp", "rho", "E_diff", "life", "Q_N2", "n2_qs"}                     # perturb sets absolute values
CHG = {"Qst", "h2o_dH", "wrc"}                                               # perturb adds a change
LAB = {"Qst": "$Q_{st}$ (kJ mol$^{-1}$)", "KH": "$K_H$ (mol kg$^{-1}$ Pa$^{-1}$)", "qs": "$q_s$ (mmol g$^{-1}$)",
       "cp": "$c_p$ (kJ kg$^{-1}$ K$^{-1}$)", "S_N2": "CO$_2$/N$_2$ selectivity", "h2o_KH": "water $K_H$ (mol kg$^{-1}$ Pa$^{-1}$)",
       "rho": "density (g cm$^{-3}$)", "E_diff": "diffusion barrier (kJ mol$^{-1}$)", "life": "lifetime (yr)",
       "price": "price (USD kg$^{-1}$)"}


def sweeps():
    """Median relative cost of the target along each property (absolute property values), base included."""
    t = summary()
    base = t[(t.part == "A") & (t.sorbent == "PO:Q50_KH0.03")].ratio.iloc[0]
    d = t[(t.part == "D") & t.sorbent.str.startswith("PD:target:")].copy()
    d[["prop", "v"]] = d.sorbent.str.split(":", expand=True).iloc[:, 2:4]
    d["v"] = d.v.astype(float)
    from . import sorbents, study5
    study5.register()
    s = sorbents.get("P_KH0.03_Q50_phob_S20000")
    bv = dict(BASE_VAL, rho=s.rho)
    out = {}
    for prop, g in d.groupby("prop"):
        if prop not in LAB:
            continue
        b = bv[prop]
        x = g.v if prop in ABS else (b + g.v if prop in CHG else b * g.v)
        out[prop] = pd.concat([pd.Series(g.ratio.to_numpy(), index=x.to_numpy()), pd.Series([base], index=[b])]).sort_index()
    return out, base


TAB_ROWS = [  # property, label, unit/kind, base-level key
    ("Qst", "$Q_{st}$ (change, kJ mol$^{-1}$)", "chg"), ("KH", "$K_H$ (factor)", "fac"), ("qs", "$q_s$ (factor)", "fac"),
    ("cp", "$c_p$ (kJ kg$^{-1}$ K$^{-1}$)", "abs"), ("rho", "Density (g cm$^{-3}$)", "abs"),
    ("S_N2", "CO$_2$/N$_2$ selectivity (factor)", "fac"), ("Q_N2", "N$_2$ heat (kJ mol$^{-1}$)", "abs"),
    ("n2_qs", "N$_2$ capacity (mol kg$^{-1}$)", "abs"), ("wrc", "WRC (change)", "chg"),
    ("h2o_KH", "Water Henry constant (factor)", "fac"), ("h2o_dH", "Water heat (change, kJ mol$^{-1}$)", "chg"),
    ("h2o_qsat", "Water capacity (factor)", "fac"), ("E_diff", "Diffusion barrier (kJ mol$^{-1}$)", "abs"),
    ("price", "Price (factor)", "fac"), ("life", "Lifetime (yr)", "abs")]
BASE_ABS = {"cp": {"target": 0.85}, "Q_N2": {"target": 17.0}, "E_diff": {"target": 0.0, "mof": 0.0},
            "life": {"target": 25.0, "mof": 25.0}}


def table():
    """SI table: median relative cost (sites below amine cost) along every property, target and Mg-MOF-74."""
    from . import fig_paper1 as F
    from . import prisma, sorbents, study5
    t = summary()
    # Mg-MOF-74 base: desorption at 100 C with the pressure optimised at each site, as in the part-D variants
    raw = pd.read_csv(OUT)
    raw = raw[(raw.part == "C") & (raw.sorbent == "PR:MgMOF74") & raw.regen.str.startswith("vac100_") & (raw.status == "ok")]
    am = pd.read_csv(RES / "amine.csv").set_index("site").amine_lcoc_base
    rs = (raw.lcoc / raw.site.map(am)).groupby([raw.site, raw.drying, raw.cold]).min().groupby(level=0).min()
    base = {"target": t[(t.part == "A") & (t.sorbent == "PO:Q50_KH0.03")].iloc[0],
            "mof": pd.Series(dict(ratio=rs.median(), wins=int((rs < 1).sum())))}
    study5.register()
    prisma.register(["MgMOF74"])
    sb = {"target": sorbents.get("P_KH0.03_Q50_phob_S20000"), "mof": sorbents.get("PR:MgMOF74")}
    d = t[t.part == "D"].copy()
    d[["arch", "prop", "v"]] = d.sorbent.str.split(":", expand=True).iloc[:, 1:4]
    d["v"] = d.v.astype(float)

    def base_level(prop, kind, arch):
        if kind == "fac":
            return 1.0
        if kind == "chg":
            return 0.0
        return BASE_ABS.get(prop, {}).get(arch, getattr(sb[arch], {"n2_qs": "n2_qs", "rho": "rho", "cp": "cp"}.get(prop, prop), None))

    rows = []
    for prop, lab, kind in TAB_ROWS:
        for arch, name in (("target", "Reference"), ("mof", "Mg-MOF-74")):
            g = d[(d.arch == arch) & (d.prop == prop)]
            if g.empty:
                continue
            bl = base_level(prop, kind, arch)
            pts = [(v, r, w, False) for v, r, w in zip(g.v, g.ratio, g.wins)]
            if bl is not None:
                pts.append((bl, base[arch].ratio, base[arch].wins, True))
            pts.sort()
            rr = [p[1] for p in pts]
            j = int(np.argmin(rr))
            real_min = 0 < j < len(rr) - 1 and rr[j] < 0.99 * min(rr[0], rr[-1])   # a minimum, not a plateau
            cells = []
            for k, (v, r, w, isb) in enumerate(pts):
                if kind == "fac":
                    vs = f"$\\times${v:g}"
                elif kind == "chg":
                    vs = "0" if v == 0 else f"{v:+g}"
                else:
                    vs = f"{v:.3g}"
                vs = vs.replace("-", "$-$")
                rs = f"{r:.2f}"
                if isb:
                    rs = f"\\underline{{{rs}}}"
                if real_min and k == j:
                    rs = f"\\textbf{{{rs}}}"
                cells.append(f"{vs}: {rs}" + (f" ({w})" if arch == "target" and w < 21 else ""))
            rows.append(f"{lab if arch == 'target' or g.empty else ''} & {name} & " + "; ".join(cells) + " \\\\")
    body = "\n".join(rows)
    (F.TAB / "si_optimum.tex").write_text(
        "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{Median relative cost over 21 sites when each adsorbent "
        "property is varied over a wide range, for the reference adsorbent (ambient temperature) and for Mg-MOF-74 (drying "
        "and cooling); base value underlined, minimum inside the range in bold, number of sites below amine cost in "
        "brackets where fewer than 21. Change and factor refer to the base value of each property.}\n\\label{tab:si_optimum}\n"
        "\\begin{tabular}{p{0.24\\textwidth}p{0.1\\textwidth}p{0.6\\textwidth}}\n\\toprule\n"
        "Property & Adsorbent & Value: relative cost \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")
    return rows


def figure():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from . import fig_paper1 as F
    F.rc()
    t = summary()
    fig = plt.figure(figsize=(F.DW, 6.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 1, 1], hspace=0.75, wspace=0.55)
    norm = TwoSlopeNorm(vmin=np.log10(0.2), vcenter=0.0, vmax=np.log10(4))
    # (a) Q_st x K_H
    ax = fig.add_subplot(gs[0, :2])
    a = t[t.part == "A"].copy()
    a["q"] = a.sorbent.str.extract(r"Q([\d.]+)_")[0].astype(float)
    a["kh"] = a.sorbent.str.extract(r"KH([\d.]+)")[0].astype(float)
    m = a.pivot(index="kh", columns="q", values="ratio")
    ax.imshow(np.log10(m.to_numpy()), cmap=F.DIV, norm=norm, aspect="auto", origin="lower")
    for i, kh in enumerate(m.index):
        for j, q in enumerate(m.columns):
            best = m.loc[kh].idxmin() == q
            ax.text(j, i, f"{m.loc[kh, q]:.2f}", ha="center", va="center", fontsize=5.8, color=F.INK,
                    weight="bold" if best else "normal")
    ax.add_patch(plt.Rectangle((list(m.columns).index(50.0) - 0.5, list(m.index).index(0.03) - 0.5), 1, 1, fill=False,
                               ec=F.INK, lw=1.2))
    ax.set_xticks(range(len(m.columns)), [f"{q:g}" for q in m.columns])
    ax.set_yticks(range(len(m.index)), [f"{k:g}" for k in m.index])
    ax.set_xlabel("$Q_{st}$ (kJ mol$^{-1}$)")
    ax.set_ylabel("$K_H$ (mol kg$^{-1}$ Pa$^{-1}$)")
    ax.set_title("Reference adsorbent (bold: best $Q_{st}$ for each $K_H$)", loc="left", fontsize=6.8)
    F.panel(ax, "a", x=-0.1)
    # (b) regeneration conditions, target
    ax = fig.add_subplot(gs[0, 2:])
    c = t[(t.part == "C") & (t.sorbent != "PR:MgMOF74")].copy()
    c["T"] = c.regen.str.extract(r"vac([\d.]+)_")[0].astype(float)
    c["p"] = c.regen.str.extract(r"_([\d.]+)$")[0].astype(float)
    r = c.pivot(index="T", columns="p", values="ratio")
    ax.imshow(np.log10(r.to_numpy()), cmap=F.DIV, norm=norm, aspect="auto", origin="lower")
    for i, T in enumerate(r.index):
        for j, p in enumerate(r.columns):
            ax.text(j, i, f"{r.loc[T, p]:.2f}", ha="center", va="center", fontsize=6, color=F.INK,
                    weight="bold" if r.loc[T, p] == r.to_numpy().min() else "normal")
    ax.add_patch(plt.Rectangle((list(r.columns).index(0.01) - 0.5, list(r.index).index(100.0) - 0.5), 1, 1,
                               fill=False, ec=F.INK, lw=1.2))
    ax.set_xticks(range(len(r.columns)), [f"{p:g}" for p in r.columns])
    ax.set_yticks(range(len(r.index)), [f"{T:g}" for T in r.index])
    ax.set_xlabel("Desorption pressure (bar)")
    ax.set_ylabel("Desorption temperature (°C)")
    ax.set_title("Reference adsorbent: regeneration conditions", loc="left", fontsize=6.8)
    F.panel(ax, "b", x=-0.2)
    # (c) one property at a time
    sw, base = sweeps()
    order = ["Qst", "KH", "qs", "cp", "S_N2", "h2o_KH", "rho", "E_diff"]
    for k, prop in enumerate(order):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        F.style(ax, grid="y")
        s = sw[prop]
        ax.plot(s.index, s.to_numpy(), marker="o", ms=2.5, lw=1.0, color=F.C_BLUE)
        b = BASE_VAL[prop] if BASE_VAL[prop] is not None else s.index[np.argmin(np.abs(s.to_numpy() - base))]
        ax.plot([b], [base], marker="s", ms=4, color=F.INK, zorder=3)
        j = int(np.argmin(s.to_numpy()))
        if 0 < j < len(s) - 1 and s.iloc[j] < 0.99 * min(s.iloc[0], s.iloc[-1]):   # a real minimum, not a plateau
            ax.plot([s.index[j]], [s.iloc[j]], marker="*", ms=7, color=F.C_ORANGE, zorder=4)
        if prop in ("KH", "S_N2", "h2o_KH", "qs"):
            ax.set_xscale("log")
        ax.set_xlabel(LAB[prop], fontsize=6.5)
        if k % 4 == 0:
            ax.set_ylabel("Relative cost")
        ax.tick_params(labelsize=6)
        if k == 0:
            F.panel(ax, "c", x=-0.3)
    F.save(fig, "figS6_optimum")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--sites", default=None, help="comma-separated added sites (only these are computed)")
    a = ap.parse_args()
    if not a.report_only:
        run(a.workers, a.only, a.sites.split(",") if a.sites else None)
    print(summary().round(3).to_string())
