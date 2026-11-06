"""Air-cooled NH3-H2O (GAX) absorption chiller performance map.

Uses the Patek-Klomfar based GAX model of lh2sys. Heat is rejected to ambient air:
condenser outlet = T_amb + DT_COND, absorber outlet = T_amb + DT_ABS.
For each (T_amb, T_eva) the generator temperature giving the highest COP
(<= T_GEN_MAX) is selected.

Run once:  python -m dacsys.ars_map   -> dacsys/ars_map.npz
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("ars_map.npz")
DT_COND, DT_ABS = 10.0, 8.0
T_AMB = np.arange(-10.0, 52.5, 2.5)           # degC
T_EVA = np.arange(-55.0, 5.0, 2.5)            # degC (evaporator outlet)
T_GEN = np.arange(100.0, 230.0, 10.0)         # degC candidates
T_GEN_MAX = 220.0


def _cop(args):
    ta, te, tg = args
    from lh2sys.params import Params
    from lh2sys.gax import solve_gax
    try:
        g = solve_gax(Params(T_eva=te + 273.15, T_gen=tg + 273.15,
                             T_cond=ta + DT_COND + 273.15, T_abs=ta + DT_ABS + 273.15), 1.0)
        c = float(g["COP"])
        return c if 0.0 < c < 3.0 else np.nan
    except Exception:
        return np.nan


def build():
    jobs = [(ta, te, tg) for ta in T_AMB for te in T_EVA for tg in T_GEN]
    with ProcessPoolExecutor() as ex:
        cops = np.array(list(ex.map(_cop, jobs, chunksize=32))).reshape(len(T_AMB), len(T_EVA), len(T_GEN))
    ok = T_GEN <= T_GEN_MAX
    c = np.where(ok[None, None, :], cops, np.nan)
    best = np.nanmax(np.where(np.isnan(c), -1, c), axis=2)
    best[best <= 0] = np.nan
    tg_best = T_GEN[np.argmax(np.where(np.isnan(c), -1, c), axis=2)]
    tg_best = np.where(np.isnan(best), np.nan, tg_best)
    np.savez(OUT, T_amb=T_AMB, T_eva=T_EVA, T_gen=T_GEN, cop_all=cops, cop=best, T_gen_best=tg_best)
    return best, tg_best


@lru_cache(maxsize=1)
def _load():
    d = np.load(OUT)
    return d["T_amb"], d["T_eva"], d["cop"], d["T_gen_best"]


def cop(T_amb_C, T_eva_C):
    """Bilinear interpolation of best COP; NaN where the chiller cannot operate."""
    ta, te, c, _ = _load()
    x = np.clip((np.asarray(T_amb_C, float) - ta[0]) / (ta[1] - ta[0]), 0, len(ta) - 1.000001)
    y = np.clip((np.asarray(T_eva_C, float) - te[0]) / (te[1] - te[0]), 0, len(te) - 1.000001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    return ((1 - fx) * (1 - fy) * c[i, j] + fx * (1 - fy) * c[i + 1, j]
            + (1 - fx) * fy * c[i, j + 1] + fx * fy * c[i + 1, j + 1])


# Operating envelope (spec V6). T_eva below -38 C is outside the decision range of Yamin et al.
# (235-244 K) and not used. T_gen up to 220 C stays inside the Patek-Klomfar validity range
# (<= 500 K) but above the 165 C validated in the paper; the plant reports that share.
T_EVA_MIN = -38.0
T_GEN_VALIDATED = 165.0


@lru_cache(maxsize=4)
def _load_limited(T_gen_max: float):
    d = np.load(OUT)
    c = np.where((d["T_gen"] <= T_gen_max)[None, None, :], d["cop_all"], np.nan)
    ok = np.isfinite(c).any(axis=2)
    best = np.where(ok, np.nanmax(np.where(np.isfinite(c), c, -1), axis=2), np.nan)
    tg = np.where(ok, d["T_gen"][np.argmax(np.where(np.isfinite(c), c, -1), axis=2)], np.nan)
    return d["T_amb"], d["T_eva"], best, tg


def cop_env(T_amb_C, T_eva_C, T_gen_max=T_GEN_MAX):
    """Best COP inside the envelope (T_eva >= T_EVA_MIN, T_gen <= T_gen_max); NaN outside.
    Also returns the generator temperature used [C]."""
    ta, te, c, tg = _load_limited(float(T_gen_max))
    T_amb_C, T_eva_C = np.asarray(T_amb_C, float), np.asarray(T_eva_C, float)
    x = np.clip((T_amb_C - ta[0]) / (ta[1] - ta[0]), 0, len(ta) - 1.000001)
    y = np.clip((T_eva_C - te[0]) / (te[1] - te[0]), 0, len(te) - 1.000001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    v = ((1 - fx) * (1 - fy) * c[i, j] + fx * (1 - fy) * c[i + 1, j]
         + (1 - fx) * fy * c[i, j + 1] + fx * fy * c[i + 1, j + 1])
    g = np.fmax(np.fmax(tg[i, j], tg[i + 1, j]), np.fmax(tg[i, j + 1], tg[i + 1, j + 1]))
    v = np.where(T_eva_C >= T_EVA_MIN - 1e-9, v, np.nan)
    return v, g


def T_gen_needed(T_amb_C, T_eva_C):
    ta, te, _, tg = _load()
    i = np.clip(np.round((np.asarray(T_amb_C) - ta[0]) / (ta[1] - ta[0])).astype(int), 0, len(ta) - 1)
    j = np.clip(np.round((np.asarray(T_eva_C) - te[0]) / (te[1] - te[0])).astype(int), 0, len(te) - 1)
    return tg[i, j]


if __name__ == "__main__":
    import time
    t = time.time()
    best, tg = build()
    print(f"built {best.size} points in {time.time() - t:.0f} s -> {OUT}")
    for te in (-2.5, -35.0, -40.0, -45.0):
        j = int(np.argmin(abs(T_EVA - te)))
        print(f"T_eva {T_EVA[j]:6.1f} C: " + "  ".join(
            f"{ta:4.0f}C:{best[i, j]:.2f}@{tg[i, j]:.0f}" for i, ta in enumerate(T_AMB) if ta % 10 == 0))
