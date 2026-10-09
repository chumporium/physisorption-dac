"""Electric vapour-compression refrigeration with natural refrigerants (RQ2 option 2).

Single stage R290 (propane) or R744/R290 cascade, air-cooled condenser, real-fluid
properties from CoolProp. The better feasible option is used at every (T_amb, T_eva).

  single stage : evaporator outlet saturated vapour + 5 K superheat, condenser outlet
                 saturated liquid - 2 K subcooling, isentropic efficiency ETA_IS,
                 pressure ratio <= PR_MAX, discharge temperature <= T_DIS_MAX
  cascade      : R744 low stage condensing at T_mid + DT_CASC into the R290 high stage
                 evaporating at T_mid; T_mid chosen for the highest overall COP

Run once:  python -m dacsys.vcr   -> dacsys/vcr_map.npz
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from CoolProp.CoolProp import PropsSI

OUT = Path(__file__).with_name("vcr_map.npz")
DT_COND = 10.0                 # K condensing above ambient (same as the absorption chiller)
ETA_IS = 0.70                  # compressor isentropic efficiency (electrical, incl. motor)
SUPERHEAT, SUBCOOL = 5.0, 2.0  # K
PR_MAX = 10.0                  # single-stage pressure ratio limit
T_DIS_MAX = 130.0 + 273.15     # K discharge temperature limit
DT_CASC = 5.0                  # K cascade heat-exchanger approach
COP_MAX = 8.0                  # cap for small lifts (fans, part load, approach losses) (a)
T_AMB = np.arange(-10.0, 52.5, 2.5)
T_EVA = np.arange(-55.0, 5.0, 2.5)


def _stage(fluid: str, Te: float, Tc: float):
    """COP of one vapour-compression stage; NaN if outside limits. Temperatures in K."""
    try:
        pe = PropsSI("P", "T", Te, "Q", 1, fluid)
        pc = PropsSI("P", "T", Tc, "Q", 0, fluid)
        if pc / pe > PR_MAX:
            return np.nan
        h1 = PropsSI("H", "T", Te + SUPERHEAT, "P", pe, fluid)
        s1 = PropsSI("S", "T", Te + SUPERHEAT, "P", pe, fluid)
        h2s = PropsSI("H", "P", pc, "S", s1, fluid)
        h2 = h1 + (h2s - h1) / ETA_IS
        if PropsSI("T", "P", pc, "H", h2, fluid) > T_DIS_MAX:
            return np.nan
        h3 = PropsSI("H", "T", Tc - SUBCOOL, "P", pc, fluid)
        return (h1 - h3) / (h2 - h1)
    except ValueError:
        return np.nan


def cop_point(T_amb_C: float, T_eva_C: float) -> tuple[float, str]:
    Tc, Te = T_amb_C + DT_COND + 273.15, T_eva_C + 273.15
    best, kind = _stage("R290", Te, Tc), "R290"
    t_crit_744 = PropsSI("Tcrit", "R744")
    for Tm in np.arange(Te + 10.0, min(Tc - 10.0, t_crit_744 - DT_CASC - 3.0), 2.5):
        c_lo = _stage("R744", Te, Tm + DT_CASC)
        c_hi = _stage("R290", Tm, Tc)
        if not (np.isfinite(c_lo) and np.isfinite(c_hi)):
            continue
        # Q_e = 1: W_lo = 1/c_lo, high stage lifts 1 + W_lo
        w = 1.0 / c_lo + (1.0 + 1.0 / c_lo) / c_hi
        c = 1.0 / w
        if not np.isfinite(best) or c > best:
            best, kind = c, "R744/R290"
    return best, kind


def build():
    cop = np.full((len(T_AMB), len(T_EVA)), np.nan)
    casc = np.zeros_like(cop, bool)
    for i, ta in enumerate(T_AMB):
        for j, te in enumerate(T_EVA):
            if te >= ta:                           # no lift needed: free cooling, not a chiller
                continue
            c, k = cop_point(ta, te)
            cop[i, j], casc[i, j] = c, k != "R290"
    np.savez(OUT, T_amb=T_AMB, T_eva=T_EVA, cop=cop, cascade=casc)
    return cop, casc


@lru_cache(maxsize=1)
def _load():
    d = np.load(OUT)
    return d["T_amb"], d["T_eva"], d["cop"], d["cascade"]


def cop(T_amb_C, T_eva_C):
    """Bilinear interpolation of the COP map; NaN where no option is feasible."""
    ta, te, c, _ = _load()
    x = np.clip((np.asarray(T_amb_C, float) - ta[0]) / (ta[1] - ta[0]), 0, len(ta) - 1.000001)
    y = np.clip((np.asarray(T_eva_C, float) - te[0]) / (te[1] - te[0]), 0, len(te) - 1.000001)
    i, j = x.astype(int), y.astype(int)
    fx, fy = x - i, y - j
    v = ((1 - fx) * (1 - fy) * c[i, j] + fx * (1 - fy) * c[i + 1, j]
         + (1 - fx) * fy * c[i, j + 1] + fx * fy * c[i + 1, j + 1])
    return np.minimum(v, COP_MAX)


def is_cascade(T_amb_C, T_eva_C):
    ta, te, _, k = _load()
    i = np.clip(np.round((np.asarray(T_amb_C) - ta[0]) / (ta[1] - ta[0])).astype(int), 0, len(ta) - 1)
    j = np.clip(np.round((np.asarray(T_eva_C) - te[0]) / (te[1] - te[0])).astype(int), 0, len(te) - 1)
    return k[i, j]


if __name__ == "__main__":
    c, k = build()
    for ta in (0.0, 20.0, 40.0):
        i = int(np.argmin(abs(T_AMB - ta)))
        row = "  ".join(f"{te:.0f}:{c[i, j]:.2f}{'c' if k[i, j] else ''}" for j, te in enumerate(T_EVA)
                        if te in (-50, -40, -30, -20, -10, -2.5))
        print(f"T_amb {ta:4.0f} C | {row}")
