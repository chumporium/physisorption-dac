"""Parametric study, sensitivity index, monthly case study, NSGA-II + TOPSIS."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gax import Infeasible
from .params import BOUNDS, MONTHLY, NSGA, Params
from .system import HEADLINE, evaluate

# Table 6 base mode and optimum decision variables
BASE_DV = dict(T_gen=423.2, T_eva=243.9, P_high=5000.0, Gb=0.8)
PAPER_OPT_DV = dict(T_gen=424.7, T_eva=241.5, P_high=3000.0, Gb=1.13)

SWEEPS = {
    "Gb": np.linspace(0.2, 1.2, 11),
    "T_gen": np.linspace(423.0, 438.0, 11),
    "T_eva": np.linspace(235.0, 244.0, 10),
    "P_high": np.linspace(3000.0, 5000.0, 11),
}


def _safe(p: Params):
    try:
        return evaluate(p)
    except (Infeasible, ValueError):  # ValueError: CoolProp / root-finding failure
        return None


def row(r: dict) -> dict:
    return {k: float(r[k]) for k in ["Gb", "T_gen", "T_eva", "P_high", "T0"] + HEADLINE
            + ["m_H2", "COP_GAX", "W_com", "W_tur", "W_pump", "Ex_F", "Ex_P", "ExD_total",
               "yL", "x_tur", "min_approach"]}


def parametric(p: Params) -> dict[str, pd.DataFrame]:
    """Figs. 2-5: vary one decision variable, others at p's values."""
    out = {}
    for var, values in SWEEPS.items():
        rows = []
        for v in values:
            r = _safe(p.with_(**{var: float(v)}))
            rows.append(row(r) if r else {var: float(v)})
        out[var] = pd.DataFrame(rows)
    return out


def sensitivity_index(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fig. 6: range of each output per variable, normalised over variables."""
    finite = {var: df.replace([np.inf, -np.inf], np.nan) for var, df in sweeps.items()}
    rng = pd.DataFrame({var: {m: df[m].max() - df[m].min() for m in HEADLINE if m in df}
                        for var, df in finite.items()})
    si = rng.div(rng.sum(axis=1), axis=0)
    si.loc["Mean"] = si.mean()
    return si


def monthly(p: Params) -> pd.DataFrame:
    """Table 8 / Fig. 10: real ambient temperature and irradiation per month."""
    rows = []
    for month, t_amb, gb in MONTHLY:
        r = _safe(p.with_(T0=t_amb + 273.15, Gb=gb))
        d = row(r) if r else {"Gb": gb, "T0": t_amb + 273.15}
        d["Month"] = month
        rows.append(d)
    return pd.DataFrame(rows).set_index("Month")


def topsis(F: np.ndarray, benefit: list[bool], w=None) -> int:
    """Index of the TOPSIS best compromise among rows of F."""
    w = np.ones(F.shape[1]) / F.shape[1] if w is None else np.asarray(w)
    R = F / np.linalg.norm(F, axis=0)
    V = R * w
    best = np.where(benefit, V.max(axis=0), V.min(axis=0))
    worst = np.where(benefit, V.min(axis=0), V.max(axis=0))
    d_best = np.linalg.norm(V - best, axis=1)
    d_worst = np.linalg.norm(V - worst, axis=1)
    return int(np.argmax(d_worst / (d_best + d_worst)))


# ------------------------------- NSGA-II --------------------------------------
_VARS = list(BOUNDS)


def _objectives(x, base: Params):
    """Top-level (picklable) evaluation used by the worker pool."""
    r = _safe(base.with_(**dict(zip(_VARS, map(float, x)))))
    if r is None:
        return [1e3, 1e3], [1.0]
    return [-r["eta_II"], r["c_LH2"]], [-1.0]


def optimize(p: Params, pop_size=NSGA["pop_size"], n_gen=NSGA["n_gen"], workers=1, seed=1,
             verbose=True):
    """Bi-objective NSGA-II: max exergy efficiency, min LH2 cost (Table 4)."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize

    xl = np.array([BOUNDS[v][0] for v in _VARS])
    xu = np.array([BOUNDS[v][1] for v in _VARS])

    pool = None
    if workers > 1:
        from multiprocessing import Pool
        pool = Pool(workers)

    class LH2Problem(Problem):
        def __init__(self):
            super().__init__(n_var=len(_VARS), n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kw):
            jobs = [(x, p) for x in X]
            res = pool.starmap(_objectives, jobs) if pool else [_objectives(*j) for j in jobs]
            out["F"] = np.array([r[0] for r in res])
            out["G"] = np.array([r[1] for r in res])

    algo = NSGA2(pop_size=pop_size,
                 crossover=SBX(prob=NSGA["p_crossover"], eta=15),
                 mutation=PM(prob=1.0, prob_var=NSGA["p_mutation"], eta=20),
                 eliminate_duplicates=True)
    try:
        res = minimize(LH2Problem(), algo, ("n_gen", n_gen), seed=seed, verbose=verbose,
                       save_history=True)
    finally:
        if pool is not None:
            pool.close()

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    front = pd.DataFrame(X, columns=_VARS)
    front["eta_II"] = -F[:, 0]
    front["c_LH2"] = F[:, 1]
    front = front.sort_values("eta_II").reset_index(drop=True)
    k = topsis(front[["eta_II", "c_LH2"]].to_numpy(), benefit=[True, False])
    best = front.iloc[k].to_dict()

    # best individual of every generation (for the scatter plots of Fig. 7)
    hist = []
    for g, algo_g in enumerate(res.history, start=1):
        Fg = algo_g.opt.get("F")
        Xg = algo_g.opt.get("X")
        if len(Fg) == 0:
            continue
        kk = topsis(np.column_stack([-Fg[:, 0], Fg[:, 1]]), benefit=[True, False])
        hist.append(dict(gen=g, **dict(zip(_VARS, Xg[kk])), eta_II=-Fg[kk, 0], c_LH2=Fg[kk, 1]))
    return front, best, pd.DataFrame(hist)
