"""Summaries of the robustness and extension runs (dacsys.extra_runs), parts A-F.

python -m dacsys.extra_analysis [--parts A,B,C,D,E,F]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import fig_paper1 as F
from .extra_runs import out_path
from .lng_scale import GJ_PER_T_LNG, KIM_MT_2050, KIM_MT_NOW, KIM_T_LNG

AMINE = pd.read_csv(F.RES / "amine.csv").set_index("site").amine_lcoc_base


def _load(part):
    d = pd.read_csv(out_path(part))
    if "place" in d and "amine_lcoc_base" not in d:
        d["amine"] = d.place.map(AMINE)
    return d


def lng_curve(d, group, drying=None):
    """Best dryer (or the given dryer) per (group, place, sorbent, recup); median ratio, potential, cold price."""
    x = d[(d.cold == "lng") & (d.status == "ok")].copy()
    if drying:
        x = x[x.drying == drying]
    x["ratio"] = x.lcoc / x.amine
    b = x.loc[x.groupby(group + ["place", "sorbent", "recup"]).lcoc.idxmin()].copy()
    b["t_lng"] = b.cold_GJ / GJ_PER_T_LNG
    b["pot_now"] = KIM_MT_NOW * KIM_T_LNG / b.t_lng
    b["pot_2050"] = KIM_MT_2050 * KIM_T_LNG / b.t_lng
    b["cold_price"] = (b.amine - b.lcoc) / b.cold_GJ
    return b


HX_SC = ("base", "ua_min", "ua_max", "cryo_cs", "cryo_hi", "yamin", "purchase", "hx100", "hx30")   # change only the cost per m2


def hx_cost(sc, lng=False):
    """Installed heat-exchanger cost per m2 of a coil (U_coil) in a part-A scenario, with the cryogenic material factor for
    the LNG exchangers (lng=True); None if the scenario changes more than the cost."""
    from .extra_runs import HX
    from .levers import Prm
    kw = HX[sc]
    if "U_air" in kw:
        return None
    p = Prm(**kw)
    c = p.hx_cost_m2
    if lng and p.hx_cost_model == "netl" and p.c_hx_area is None:
        c *= p.f_cryo_installed
    return c


def hx_labels():
    from .levers import Prm
    c = lambda **kw: f"{Prm(**kw).hx_cost_m2:.0f}"                    # noqa: E731
    fm = Prm().f_mat_cryo
    return {"base": f"Base: correlation of Weiland et al., installed ({c()} USD m$^{{-2}}$; material factor {fm:g} in cryogenic service)",
            "ua_min": f"Weiland et al., smallest unit of the vendor range ({c(netl_ua_unit=8.6e5)} USD m$^{{-2}}$)",
            "ua_max": f"Weiland et al., largest unit of the vendor range ({c(netl_ua_unit=7.5e7)} USD m$^{{-2}}$)",
            "cryo_cs": f"Weiland et al., LNG exchangers in carbon steel ({c()} USD m$^{{-2}}$)",
            "cryo_hi": f"Weiland et al., LNG exchangers with material factor 2.93 (stainless-steel air cooler, Turton et al.)",
            "yamin": f"Correlation of Yamin et al., 10\\,000 m$^2$ units, installed ({c(hx_cost_model='unit')} USD m$^{{-2}}$)",
            "purchase": f"Correlation of Yamin et al., purchase cost only ({c(hx_cost_model='unit', hx_install=1.0)} USD m$^{{-2}}$)",
            "hx100": "Fixed 100 USD m$^{-2}$", "hx30": "Fixed 30 USD m$^{-2}$",
            "regen": "Regenerators (10 USD m$^{-2}$, doubled $U$)"}


def hx_parity():
    """Installed heat-exchanger cost at which the relative cost reaches 1, per site: interpolated in log(cost) between the
    part-A scenarios that change only the cost per m2. Mg-MOF-74 with LNG (best dryer and recuperator approach) and
    the best MOF without LNG. Returns the per-site table and a summary; '<' / '>' when parity lies outside the range."""
    d = _load("A")
    scs = [sc for sc in d.scenario.unique() if sc in HX_SC]
    b = lng_curve(d, ["scenario"])                     # best dryer (freeze-out or condensation) and approach
    lng = b[b.sorbent == "PR:MgMOF74"].groupby(["scenario", "place"]).ratio.min()
    x = d[(d.cold != "lng") & (d.status == "ok") & (d.sorbent != "13X")].copy()
    x["ratio"] = x.lcoc / x.amine
    nolng = x.groupby(["scenario", "place"]).ratio.min()
    rows = []
    for name, s in (("lng", lng), ("no_lng", nolng)):
        # without LNG the material factor does not apply, so "cryo_cs" and "cryo_hi" duplicate the base case
        costs = {sc: hx_cost(sc, lng=(name == "lng")) for sc in scs if name == "lng" or sc not in ("cryo_cs", "cryo_hi")}
        for place in sorted({p for _, p in s.index}):
            pts = sorted((costs[sc], s.get((sc, place), np.nan)) for sc in costs)
            c = np.array([p[0] for p in pts])
            r = np.array([p[1] for p in pts])
            ok = np.isfinite(r)
            c, r = c[ok], r[ok]
            if len(c) < 2:
                val, flag = np.nan, ""
            elif r[0] >= 1:
                val, flag = c[0], "<"
            elif r[-1] < 1:
                val, flag = c[-1], ">"
            else:
                i = int(np.argmax(r >= 1))
                f = (1 - r[i - 1]) / (r[i] - r[i - 1])
                val, flag = float(np.exp(np.log(c[i - 1]) + f * (np.log(c[i]) - np.log(c[i - 1])))), ""
            rows.append(dict(case=name, place=place, parity_cost=val, flag=flag))
    t = pd.DataFrame(rows)
    t.to_csv(F.RES / "hx_parity.csv", index=False)
    return t


def part_A():
    d = _load("A")
    b = lng_curve(d, ["scenario"])                     # best dryer (freeze-out or condensation) per approach
    t = b.groupby(["scenario", "sorbent", "recup"]).agg(ratio=("ratio", "median"), lo=("ratio", "min"),
                                                         hi=("ratio", "max"), wins=("ratio", lambda v: (v < 1).sum()),
                                                         pot_now=("pot_now", "median"), pot_2050=("pot_2050", "median"),
                                                         price=("cold_price", "median"))
    print("A: LNG curve\n", t.round(2).to_string())
    x = d[(d.cold != "lng") & (d.status == "ok") & (d.sorbent != "13X")].copy()
    x["ratio"] = x.lcoc / x.amine
    best = x.loc[x.groupby(["scenario", "place"]).lcoc.idxmin()]
    s = best.groupby("scenario").ratio.agg(["median", "min", "max", lambda v: (v < 1).sum()])
    print("A: best MOF without LNG\n", s.round(2).to_string())
    print(best.groupby("scenario").cold.value_counts().unstack().fillna(0).astype(int).to_string())
    y = d[(d.sorbent == "13X") & (d.cold == "vcr") & (d.status == "ok")].copy()
    y["ratio"] = y.lcoc / y.amine
    print("A: 13X freeze+HP\n", y.groupby("scenario").ratio.agg(["median", "min", "max"]).round(2).to_string())
    return t, best


def part_B():
    d = _load("B")
    x = d[d.status == "ok"].copy()
    x["ratio"] = x.lcoc / x.amine
    b = x.loc[x.groupby(["sorbent", "scenario", "place"]).lcoc.idxmin()]
    b["base"] = b.sorbent.str.replace(r"^PF:", "", regex=True).str.replace(r":[\d.]+$", "", regex=True)
    t = b.groupby(["base", "scenario"]).agg(ratio=("ratio", "median"), lo=("ratio", "min"), hi=("ratio", "max"),
                                           wins=("ratio", lambda v: (v < 1).sum()), n=("ratio", "size"),
                                           cooled=("cold", lambda v: (v != "ambient").sum()))
    print("B: pore filling\n", t.round(2).to_string())
    return t, b


def part_C():
    d = _load("C")
    x = d[d.status == "ok"].copy()
    x["ratio"] = x.lcoc / x.amine
    mof = x[x.sorbent != "13X"]
    b = mof.loc[mof.groupby(["scenario", "place"]).lcoc.idxmin()].set_index(["scenario", "place"]).ratio
    dd, _ = F.load()
    amb = F.db(dd)
    amb = amb[(amb.cold == "ambient") & (amb.status == "ok")].groupby("site").ratio.min()
    rows = []
    for sc in sorted(x.scenario.unique()):
        hp = b.loc[sc]
        tot = pd.concat([hp, amb.reindex(hp.index)], axis=1).min(axis=1)
        rows.append(dict(scenario=sc, hp_med=hp.median(), hp_lo=hp.min(), hp_hi=hp.max(), best_med=tot.median(),
                         best_lo=tot.min(), best_hi=tot.max(), wins=int((tot < 1).sum()),
                         x13=x[(x.sorbent == "13X") & (x.scenario == sc)].ratio.median()))
    t = pd.DataFrame(rows)
    print("C: heat pump efficiency\n", t.round(2).to_string())
    return t


def part_D():
    d = _load("D")
    x = d[d.status == "ok"].copy()
    x["ratio"] = x.lcoc / x.amine
    dd, _ = F.load()
    base_best = F.db_best(dd)
    rows = []
    for sc, g in x.groupby("scenario"):
        amb = g[g.cold == "ambient"]
        ba = amb.loc[amb.groupby("place").lcoc.idxmin()].set_index("place")
        hpsil = g[g.cold == "vcr"].set_index("place").ratio
        new = pd.concat([ba.ratio, hpsil, base_best.ratio], axis=1).min(axis=1)
        for site in base_best.index:
            rows.append(dict(scenario=sc, site=site, base=base_best.at[site, "ratio"],
                             silica_amb=ba.ratio.get(site, np.nan), silica_hp=hpsil.get(site, np.nan), new=new[site],
                             mof=ba.sorbent.get(site, None)))
    t = pd.DataFrame(rows)
    s = t.groupby("scenario").agg(base=("base", "median"), new=("new", "median"), lo=("new", "min"),
                                  wins=("new", lambda v: (v < 1).sum()),
                                  improved=("new", lambda v: 0))
    for sc in t.scenario.unique():
        g = t[t.scenario == sc]
        s.loc[sc, "improved"] = int((g.new < g.base - 1e-9).sum())
    print("D: heat-pump silica regeneration\n", s.round(2).to_string())
    print(t[t.site.isin(["Altiplano", "Tibet", "Gobi", "Yakutsk", "Reykjavik", "Kiruna"])].round(2).to_string())
    return t


def part_E():
    d = _load("E")
    am = d[d.sorbent == "amine"].set_index("place")
    d = d[d.sorbent != "amine"].copy()
    d["amine"] = d.place.map(am.amine_lcoc_base)
    b = lng_curve(d, [])
    t = b.groupby(["sorbent", "recup"]).agg(ratio=("ratio", "median"), lo=("ratio", "min"), hi=("ratio", "max"),
                                           wins=("ratio", lambda v: (v < 1).sum()), pot_now=("pot_now", "median"),
                                           price=("cold_price", "median"), cold=("cold_GJ", "median"))
    print("E: LNG terminals\n", t.round(2).to_string())
    per = b[b.recup == "none"].pivot_table(index="place", columns="sorbent", values="ratio")
    per["amine"] = am.amine_lcoc_base
    from . import climate_sites as cs
    for term in per.index:
        lat, lon = cs.TERMINALS[term]
        w = cs.load_point(term, lat, lon, 2018)
        per.loc[term, "T_mean"] = w["T0"].mean() - 273.15
        per.loc[term, "Td_mean"] = w["Td"].mean() - 273.15
        per.loc[term, "DNI"] = w["dni"].sum() / 1e3
    print(per.round(2).to_string())
    return t, b


def part_F():
    d = _load("F")
    am = d[d.scenario == "amine"].set_index(["place", "year"]).amine_lcoc_base
    x = d[(d.scenario != "amine") & (d.status == "ok")].copy()
    x["amine_y"] = [am.get((p, y), np.nan) for p, y in zip(x.place, x.year)]
    x["ratio"] = x.lcoc / x.amine_y
    dd, _ = F.load()
    nl, ln, _, _ = (F.db_best(dd), F.db_best(dd, lng=True), None, None)
    m = F.m13(dd)
    x13 = F.best(m[m.cold != "lng"], ["site"]).set_index("site").ratio
    tg = F.best(F.tgt(dd), ["site"]).set_index("site").ratio
    ref = {"best_noLNG": nl.ratio, "best_LNG": ln.ratio, "13X": x13, "target": tg}
    x["ratio2018"] = [ref[s].get(p, np.nan) for s, p in zip(x.scenario, x.place)]
    x["dev"] = (x.ratio / x.ratio2018 - 1) * 100
    t = x.groupby(["scenario", "year"]).agg(ratio=("ratio", "median"), lo=("ratio", "min"), hi=("ratio", "max"),
                                           wins=("ratio", lambda v: (v < 1).sum()), n=("ratio", "size"),
                                           dev_med=("dev", "median"), dev_max=("dev", lambda v: v.abs().max()))
    print("F: other weather years\n", t.round(2).to_string())
    amd = pd.DataFrame({"amine": am}).reset_index()
    amd["a2018"] = amd.place.map(AMINE)
    print("amine dev % per year:", amd.assign(dev=(amd.amine / amd.a2018 - 1) * 100).groupby("year").dev
          .agg(["median", "min", "max"]).round(1).to_dict())
    return t, x


# ------------------------------------------------------------------ SI tables
def _m(x, f="{:.2f}"):
    s = f.format(x)
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def _write(name, caption, label, colspec, header, rows, wide=True):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    wrap = ("\\resizebox{\\textwidth}{!}{", "}") if wide else ("", "")
    (F.TAB / name).write_text(
        "\\begin{table}[t]\n\\centering\\footnotesize\n\\caption{" + caption + "}\n\\label{" + label + "}\n" + wrap[0] +
        "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}" + wrap[1] + "\n\\end{table}\n", encoding="utf-8")


def si_tables():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        tA, bestA = part_A()
        tC = part_C()
        tE, bE = part_E()
        tF, _ = part_F()
    # heat exchangers
    lab = hx_labels()
    d = _load("A")
    y = d[(d.sorbent == "13X") & (d.cold == "vcr") & (d.status == "ok")].copy()
    y["ratio"] = y.lcoc / y.amine
    rows = []
    for sc in lab:
        g = bestA[bestA.scenario == sc].ratio
        cells = [lab[sc], f"{g.median():.2f} ({g.min():.2f}--{g.max():.2f})", f"{int((g < 1).sum())}",
                 f"{y[y.scenario == sc].ratio.median():.2f}"]
        for rc in ("none", "20 K", "8 K", "3 K"):
            r = tA.loc[(sc, "PR:MgMOF74", rc)]
            cells.append(f"{r.ratio:.2f} ({int(r.wins)})")
        cells.append(f"{tA.loc[(sc, 'PR:MgMOF74', '3 K')].pot_now:.0f} / {tA.loc[(sc, 'PR:MgMOF74', '3 K')].pot_2050:.0f}")
        rows.append(cells)
    _write("si_hx.tex", "Sensitivity to the heat-exchanger cost. Cost per m$^2$ of a coil ($U$ = 50\\,W\\,m$^{-2}$\\,K$^{-1}$) in "
           "carbon steel; the correlation of Weiland et al. is applied per UA, so that its cost per m$^2$ of a recuperator is lower in "
           "proportion to $U$. "
           "Without LNG: best MOF and configuration at each site, median (range) over 21 sites and number of sites below "
           "amine cost; 13X with freeze-out and heat pump, median. With LNG: Mg-MOF-74 with the best dryer (condensation without recuperator) and a fixed "
           "recuperator approach, "
           "median relative cost (sites below amine cost) and global potential with a 3\\,K approach (current / 2050 "
           "capacity, Mt\\,yr$^{-1}$).", "tab:si_hx", "lllllllll",
           " & \\multicolumn{3}{c}{Without LNG} & \\multicolumn{5}{c}{With LNG, Mg-MOF-74} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-9}\nScenario & Best MOF & Sites & 13X & No recup. & 20\\,K & 8\\,K & 3\\,K & "
           "Potential", rows)
    # installed heat-exchanger cost at parity with amine DAC
    hp = hx_parity()
    rows = []
    for case, name in (("lng", "Mg-MOF-74 with LNG (best dryer)"), ("no_lng", "Best MOF without LNG")):
        g = hp[hp.case == case]
        v = g.parity_cost
        fmt = lambda r: f"{r.flag}{r.parity_cost:.0f}"                  # noqa: E731
        lo, hi = g.loc[v.idxmin()], g.loc[v.idxmax()]
        med = f"<{v.median():.0f}" if (g.flag == "<").mean() > 0.5 else f"{v.median():.0f}"
        rows.append([name, med, f"{fmt(lo)}--{fmt(hi)}",
                     f"{int(((v > hx_cost('base', lng=(case == 'lng'))) & (g.flag != '<')).sum())}", f"{int((g.flag == '<').sum())}"])
    _write("si_hxparity.tex", "Installed heat-exchanger cost (USD\\,m$^{-2}$) at which the relative cost reaches 1, "
           "interpolated between the scenarios of Table~\\ref{tab:si_hx} that change only the cost per m$^2$ "
           f"({min(c for c in map(hx_cost, HX_SC) if c):.0f}--{max(c for c in map(hx_cost, HX_SC) if c):.0f}\\,USD\\,m$^{{-2}}$). "
           "Median and range over 21 sites; `$<$': no parity within the range.", "tab:si_hxparity", "lllll",
           "Case & Median & Range & Sites above base cost & Sites below range", rows, wide=False)
    # heat pump efficiency
    rows = [[f"{float(r.scenario[3:]):.1f}", f"{r.hp_med:.2f} ({r.hp_lo:.2f}--{r.hp_hi:.2f})",
             f"{r.best_med:.2f} ({r.best_lo:.2f}--{r.best_hi:.2f})", f"{int(r.wins)}", f"{r.x13:.2f}"] for r in tC.itertuples()]
    _write("si_hpeff.tex", "Effect of the isentropic efficiency of the heat pump on the relative cost without LNG, median "
           "(range) over 21 sites. Heat pump: lowest-cost MOF of each site in the base case, operated with the heat pump "
           "(no drying, condensation or freeze-out). Best configuration: including operation without chiller. Sites: "
           "number of sites below amine cost. 13X: freeze-out with heat pump (median).", "tab:si_hpeff", "lllll",
           "$\\eta_{is}$ & Heat pump & Best configuration & Sites & 13X", rows, wide=False)
    # weather years
    names = {"best_noLNG": "Best MOF, without LNG", "best_LNG": "Best MOF, with LNG", "13X": "13X, without LNG",
             "target": "Reference adsorbent"}
    rows = []
    for sc, lab_ in names.items():
        cells = [lab_]
        for yr in (2015, 2019, 2020, 2021):
            r = tF.loc[(sc, yr)]
            cells.append(f"{r.ratio:.2f} ({r.lo:.2f}--{r.hi:.2f}; {int(r.wins)})")
        rows.append(cells)
    _write("si_years.tex", "Relative cost with the weather of other years. At each site the adsorbent, dryer and cold "
           "source selected for 2018 were kept, and the design temperature, recuperator approach, operating hours and "
           "energy supply were re-optimised; amine DAC was recalculated for the same year: median (range; number of sites below amine cost) "
           "over 21 sites. 2018: " + _base2018() + ". With LNG, a relative cost of 0.995 or higher (Tibet in 2018) is "
           "counted as below the amine cost if it is below 1; the main text treats 0.995 as parity.", "tab:si_years", "lllll",
           "Configuration & 2015 & 2019 & 2020 & 2021", rows)
    # LNG terminals
    from . import climate_sites as cs
    per = bE[bE.recup == "none"].pivot_table(index="place", columns="sorbent", values="ratio")
    cold = bE[bE.recup == "none"].pivot_table(index="place", columns="sorbent", values="cold_GJ")
    r3 = bE[bE.recup == "3 K"].pivot_table(index="place", columns="sorbent", values="ratio")
    am = _load("E")
    am = am[am.sorbent == "amine"].set_index("place").amine_lcoc_base
    rows = []
    for term in cs.TERMINALS:
        lat, lon = cs.TERMINALS[term]
        w = cs.load_point(term, lat, lon, 2018)
        rows.append([term, f"{w['T0'].mean() - 273.15:.1f}", f"{w['Td'].mean() - 273.15:.1f}", f"{am[term]:.0f}",
                     f"{per.at[term, '13X']:.2f}", f"{per.at[term, 'PR:MgMOF74']:.2f}",
                     f"{cold.at[term, 'PR:MgMOF74']:.0f}", f"{r3.at[term, 'PR:MgMOF74']:.2f}"])
    _write("si_terminals.tex", "LNG cold at twelve large LNG import terminals (ERA5 2018 at the approximate terminal "
           "location). Relative cost without recuperator (condensation drying; freeze-out requires the regenerators), cold demand of "
           "Mg-MOF-74, and relative cost of Mg-MOF-74 with a 3\\,K recuperator approach.", "tab:si_terminals",
           "lrrrrrrr", "Terminal & $\\bar T$ (°C) & $\\bar T_d$ (°C) & Amine (USD\\,t$^{-1}$) & 13X & Mg-MOF-74 & "
           "Cold (GJ\\,t$^{-1}$) & Mg-MOF-74, 3\\,K", rows)
    # silica regeneration (parts D and G)
    dd, _ = F.load()
    base_best = F.db_best(dd)
    p = F.db(dd)
    p = p[(p.status == "ok") & (p.drying == "cond_silica") & (p.cold == "ambient")]
    s0 = p.loc[p.groupby("site").lcoc.idxmin()].set_index("site").ratio

    def best_amb(part, sc):
        x = _load(part)
        x = x[(x.status == "ok") & (x.scenario == sc) & (x.cold == "ambient")].copy()
        x["ratio"] = x.lcoc / x.amine
        return x.groupby("place").ratio.min()
    cols = {"Solar heat (base)": s0, "Solar heat, heat recovery": best_amb("G", "f1"),
            "Heat pump (COP 3)": best_amb("D", "hp3"), "Heat pump, heat recovery": best_amb("D", "hp3_f1")}
    rows = []
    for site in ("Altiplano", "Gobi", "Tibet", "Yakutsk", "Reykjavik", "Kiruna", "Astana"):
        rows.append([site, f"{base_best.at[site, 'ratio']:.2f}"] + [f"{v.get(site, np.nan):.2f}" for v in cols.values()])
    _write("si_silica.tex", "Silica-gel drying without chiller with different regeneration: lowest relative cost over the "
           f"{F.db(dd).sorbent.nunique()} second-stage MOFs at the cold sites. Heat recovery: regeneration heat 2.8 instead of 3.5\\,MJ per kg water. "
           "Best: best configuration of the base case (any dryer and cold source).", "tab:si_silica", "lrrrrr",
           "Site & Best (base) & " + " & ".join(cols), rows)
    si_oat()


def _base2018():
    """Base-year (2018) values quoted in the caption of the weather-year table."""
    from .analysis_paper1 import configs
    dd, a = F.load()
    cf = configs(dd)
    out = []
    for key, lab_ in (("Best MOF, no LNG", "best MOF without LNG"), ("Best MOF, LNG", "with LNG"),
                      ("13X, no LNG", "13X"), ("Reference adsorbent", "reference adsorbent")):
        r = cf[key].ratio
        out.append(f"{lab_} {r.median():.2f} ({r.min():.2f}--{r.max():.2f}; {int((r < 1).sum())})")
    return ", ".join(out)


OAT_LABELS = {"dewpoint_dry": ("Dew point after silica gel", "°C", 1), "T_coil1": ("Condensing coil outlet", "°C", 1),
              "dT_evap": ("Air--refrigerant approach", "K", 1), "U_coil": ("Coil heat transfer coefficient", "W m$^{-2}$ K$^{-1}$", 1e3),
              "reg_area_factor": ("Regenerator area factor", "--", 1), "dP_coil": ("Pressure drop, coil", "Pa", 1),
              "dP_silica": ("Pressure drop, silica gel", "Pa", 1), "dP_recup": ("Pressure drop, recuperator", "Pa", 1),
              "dP_regen": ("Pressure drop, regenerator", "Pa", 1), "cp_contactor": ("Contactor heat capacity", "kJ kg$^{-1}$ K$^{-1}$", 1),
              "dT_hex_heat": ("Heating medium approach", "K", 1), "dT_hex_cool": ("Cooling medium approach", "K", 1),
              "sil_dq": ("Silica gel working capacity", "kg kg$^{-1}$", 1), "sil_cycles": ("Silica gel cycles", "h$^{-1}$", 1),
              "gax_pump_frac": ("GAX solution pump", "\\% of heat", 100), "fan_rej_frac": ("Dry cooler fans", "\\% of heat", 100),
              "storage_loss_h": ("Storage loss", "\\% h$^{-1}$", 100),
              "water_recovery_silica": ("Water recovery, silica gel$^a$", "--", 1)}
OAT_CFG = (("best_noLNG", "Best MOF, no LNG"), ("best_LNG", "Best MOF, LNG"), ("target", "Reference"),
           ("13X_B0", "13X, B0"))


def part_H():
    """Median change over the 21 sites of the LCOC (water per CO2 for the silica-gel water recovery) when one
    assumption is set to its low or high value; per configuration."""
    d = _load("H")
    d = d[d.status == "ok"].copy()
    base = d[d.scenario == "base"].set_index(["config", "place"])
    x = d[d.scenario != "base"].copy()
    x[["par", "lvl"]] = x.scenario.str.split(":", expand=True)
    key = list(zip(x.config, x.place))
    col = np.where(x.par == "water_recovery_silica", "water_per_co2", "lcoc")
    ref = np.array([base.at[k, c] if k in base.index else np.nan for k, c in zip(key, col)], float)
    val = np.where(col == "lcoc", x.lcoc, x.water_per_co2)
    x["change"] = 100.0 * (val / ref - 1.0)
    t = x.groupby(["par", "lvl", "config"]).change.median().unstack("config")
    print("H: one-at-a-time (median % change over sites)\n", t.round(1).to_string())
    return t


def si_oat():
    from .extra_runs import OAT
    from .levers import Prm
    t = part_H()
    p = Prm()
    rows = []
    for par, (lo, hi) in OAT.items():
        name, unit, k = OAT_LABELS[par]
        fmt = lambda v: f"{v * k:.3g}".replace("-", "$-$")                   # noqa: E731
        cells = [name, f"{fmt(getattr(p, par))} ({fmt(lo)}--{fmt(hi)}) {unit}"]
        for cfg, _ in OAT_CFG:
            a_, b_ = t.at[(par, "low"), cfg], t.at[(par, "high"), cfg]
            cells.append(_signed(a_) + " / " + _signed(b_))
        rows.append(cells)
    _write("si_oat.tex", "Effect of the design assumptions that are not varied in the Monte Carlo analysis or in the other "
           "runs: median change of the LCOC over the 21 sites (\\%) with the low / high value, the plant re-optimised in "
           "each case. Best MOF: best configuration without LNG at each site; reference: reference adsorbent, best configuration; "
           "B0: 13X with condensation, silica gel and GAX. $^a$Change of the water recovered per tonne of CO$_2$ "
           "(the LCOC does not depend on it).", "tab:si_oat", "llllll",
           "Parameter & Base (range) & " + " & ".join(n for _, n in OAT_CFG), rows)


def _signed(v):
    if not np.isfinite(v):
        return "--"
    return ("$+$" if v > 0.05 else ("$-$" if v < -0.05 else "")) + f"{abs(v):.1f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,C,D,E,F")
    ap.add_argument("--tables", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    if a.tables:
        si_tables()
    else:
        for p in a.parts.split(","):
            globals()[f"part_{p}"]()
