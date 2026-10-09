"""Matplotlib figures mirroring the paper's Figs. 2-10."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .system import HEADLINE, UNITS  # noqa: E402

LABEL = {"Gb": "G$_b$ (kW/m$^2$)", "T_gen": "T$_{gen}$ (K)", "T_eva": "T$_{eva}$ (K)",
         "P_high": "P$_{high}$ (kPa)"}
NAME = {"LHPR": "LHPR", "COP_sys": "COP$_{sys}$", "eta_II": r"$\eta_{II,sys}$", "SI": "SI$_{sys}$",
        "Zdot": r"$\dot Z_{sys}$", "c_LH2": "c$_{L,H2}$", "PP": "PP$_{sys}$", "NPV": "NPV$_{sys}$"}


def _ylab(m):
    unit = UNITS[m].replace("$", r"\$")
    return f"{NAME[m]} ({unit})"


COLORS = ["#2a7f62", "#1f9e3a", "#7b1fa2", "#ef6c00", "#8e2c2c", "#d81b9a", "#9e9d24", "#1a3fd6"]


def sweep_figure(var, df, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, m, c in zip(axes.flat, HEADLINE, COLORS):
        if m in df:
            ax.plot(df[var], df[m], "o-", color=c, ms=4)
        ax.set_xlabel(LABEL[var])
        ax.set_ylabel(_ylab(m))
        ax.grid(alpha=0.3)
    fig.suptitle(f"Influence of {LABEL[var]} on the system indicators")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_figure(si, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    left = None
    cols = ["#56708c", "#f2b705", "#a9c4e0", "#d9d9d9"]
    rows = si.index[::-1]
    import numpy as np
    left = np.zeros(len(rows))
    for var, c in zip(si.columns, cols):
        vals = si.loc[rows, var].to_numpy()
        ax.barh(range(len(rows)), vals, left=left, color=c, edgecolor="k", label=var)
        for i, (l, v) in enumerate(zip(left, vals)):
            if v > 0.04:
                ax.text(l + v / 2, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
        left += vals
    ax.set_yticks(range(len(rows)), [NAME.get(r, r) for r in rows])
    ax.set_xlim(0, 1)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Sensitivity index")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def monthly_figure(dfm, path: Path):
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    for ax, m, c in zip(axes.flat, HEADLINE, COLORS):
        if m in dfm:
            vals = dfm[m]
            ax.bar(dfm.index, vals, color=c)
            for i, v in enumerate(vals):
                if v == v:
                    txt = f"{v/1e6:.1f}M" if m == "NPV" else f"{v:.3g}"
                    ax.text(i, v, txt, ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(_ylab(m))
    fig.suptitle("System performance metrics per month")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pareto_figure(front, best, hist, path: Path):
    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(front["eta_II"], front["c_LH2"], s=14, c="#5b6ee1", edgecolor="k", lw=0.3)
    ax.scatter([best["eta_II"]], [best["c_LH2"]], c="red", s=40, zorder=3,
               label="Best optimal solution by TOPSIS")
    ax.set_xlabel(r"$\eta_{II,sys}$ (%)")
    ax.set_ylabel(_ylab("c_LH2"))
    ax.legend()
    ax.grid(alpha=0.3)
    for i, v in enumerate(["T_gen", "T_eva", "P_high", "Gb"]):
        a = fig.add_subplot(3, 2, 3 + i)
        if len(hist):
            a.scatter(hist["gen"], hist[v], s=12, edgecolor="k", lw=0.3)
        a.set_xlabel("Number of generations")
        a.set_ylabel(LABEL[v])
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def npv_figure(curves: dict, path: Path):
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(next(iter(curves.values()))[0])
    x = np.arange(1, n + 1)
    w = 0.4
    for i, (lab, (curve, pp)) in enumerate(curves.items()):
        ax.bar(x + (i - 0.5) * w, curve, w, label=f"{lab}, PP={pp:.2f} yr")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Lifetime of system (year)")
    ax.set_ylabel(r"NPV (\$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def exergy_figure(r: dict, path: Path):
    items = sorted(r["ExD"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#c0392b")
    ax.set_xlabel("Exergy destruction rate (kW)")
    ax.set_title(f"Total {r['ExD_total']:.0f} kW  |  fuel {r['Ex_F']:.0f} kW  |  product {r['Ex_P']:.0f} kW")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
