"""Why the PrISMa MOFs fail without cooling: Shapley attribution of the cost gap to property groups, the water
affinity threshold, and the co-variation of CO2 and water affinity over the database.

python -m dacsys.materials_why -> docs/paper1/figs/fig14_why.pdf|png (numbers printed)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fig_paper1 as F
from . import prisma

GROUPS = {"water": "Water isotherm and WRC", "heat": "CO$_2$ heat of adsorption", "iso298": "CO$_2$ isotherm at 298 K",
          "N2": "N$_2$ co-adsorption", "price": "Price", "solid": "Heat capacity, density"}
MOFS = {"PR:str_m4_o14_o24_acs_sym.190": "hypothetical MOF (str_m4_o14_o24_acs_sym.190)", "PR:MgMOF74": "Mg-MOF-74", "PR:NEYZAU_clean": "NEYZAU",
        "PR:RSM2114": "RSM2114"}


def stats():
    s = pd.read_csv(F.RES / "decomp_shapley.csv")
    sh = s.groupby(["real", "group"]).phi_ln.median().unstack("group")
    k = pd.read_csv(F.RES / "decomp_khw.csv")
    kh = k[k.status == "ok"].groupby(["KHw", "site"]).ratio.min().unstack("site")
    fits = prisma.load_fits()
    kc = np.array([v["KH_CO2"] for v in fits.values()])
    kw = np.array([v["h2o"][1] for v in fits.values()])
    wrc = np.array([v["h2o"][3] for v in fits.values()])
    r = np.corrcoef(np.log10(kc), np.log10(kw))[0, 1]
    return sh, kh, (kc, kw, wrc, r)


def figure():
    F.rc()
    sh, kh, (kc, kw, wrc, r) = stats()
    fig, axes = plt.subplots(1, 3, figsize=(F.DW, 2.9), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    # (a) Shapley: factor by which each property group of the target lowers the cost of the real MOF
    ax = axes[0]
    F.style(ax, grid="x")
    order = list(GROUPS)
    y = np.arange(len(order))[::-1]
    cols = [F.C_BLUE, F.C_ORANGE, F.C_GREEN, F.C_PURPLE]
    for i, (m, lab) in enumerate(MOFS.items()):
        v = np.exp(-sh.loc[m, order].to_numpy())
        ax.barh(y + (i - 1.5) * 0.19, v - 1, left=1, height=0.18, color=cols[i], label=lab.replace(" (", "\n("))
    ax.axvline(1, color=F.INK, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([0.7, 1, 2, 3, 5], ["0.7", "1", "2", "3", "5"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, [GROUPS[g] for g in order], fontsize=6.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cost reduction factor (Shapley)")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    F.panel(ax, "a", x=-0.02)
    # (b) water affinity threshold
    ax = axes[1]
    F.style(ax, grid="y")
    pal = [F.C_BLUE, F.C_ORANGE, F.C_GREEN, F.C_PURPLE, F.INK2]
    for c, site in zip(pal, kh.columns):
        ax.plot(kh.index, kh[site], marker="o", ms=2.5, lw=1.0, color=c, label=site)
    ax.axhline(1, color=F.INK, lw=0.8)
    ax.axvspan(kh.index.min() * 0.7, 1e-4, color=F.C_GREEN, alpha=0.07, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Water Henry constant of the\nhypothetical MOF (mol kg$^{-1}$ Pa$^{-1}$)")
    ax.set_ylabel("Relative cost")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 25)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    F.panel(ax, "b", x=-0.2)
    # (c) CO2 and water affinity over the database
    ax = axes[2]
    F.style(ax, grid=None)
    for sel, col, lab in ((wrc < 0.05, F.C_ORANGE, "WRC < 0.05"), ((wrc >= 0.05) & (wrc < 0.5), F.C_BLUE, "0.05–0.5"),
                          (wrc >= 0.5, F.C_GREEN, "WRC ≥ 0.5")):
        ax.scatter(kc[sel], kw[sel], s=3, color=col, alpha=0.6, lw=0, label=lab)
    ax.add_patch(plt.Rectangle((0.01, 1e-8), 10, 1e-4 - 1e-8, fill=False, ec=F.INK, lw=0.9, ls="--"))
    ax.text(0.012, 2e-8, "reference", fontsize=6, color=F.INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 3)
    ax.set_ylim(1e-8, 1e4)
    ax.set_xlabel("CO$_2$ Henry constant, 298 K\n(mol kg$^{-1}$ Pa$^{-1}$)")
    ax.set_ylabel("Water Henry constant (mol kg$^{-1}$ Pa$^{-1}$)")
    ax.text(0.03, 0.97, f"r = {r:.2f} (log)\nn = {len(kc)}", transform=ax.transAxes, va="top", fontsize=6, color=F.INK2)
    ax.legend(frameon=False, fontsize=5.6, loc="lower left", bbox_to_anchor=(-0.02, 0.99), ncol=3, markerscale=2.5,
              handletextpad=0.05, columnspacing=0.6, borderaxespad=0)
    F.panel(ax, "c", x=-0.22)
    fig.tight_layout(w_pad=0.8)
    F.save(fig, "fig14_why")


if __name__ == "__main__":
    sh, kh, (kc, kw, wrc, r) = stats()
    print(np.exp(-sh).round(2).to_string())
    print(kh.round(2).to_string())
    print("r", round(r, 3), "KH>=0.01:", int((kc >= 0.01).sum()), "of which KHw<=1e-4:", int(((kc >= 0.01) & (kw <= 1e-4)).sum()),
          "WRC>0.5:", int(((kc >= 0.01) & (wrc > 0.5)).sum()))
    figure()
