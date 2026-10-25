"""Product purity and CO2 recovery from an equilibrium 5-step TVSA cycle with IAST (step 4).

Follows the PrISMa process layer (Charalambous et al., Nature 2024, SI section 3.2.1; Santori et al.),
per kg sorbent, dry CO2/N2 feed (O2 and Ar are lumped with N2):
  1 AS   adsorption at T_ads and the feed pressure P; bed and pellet voids hold feed gas.
  2 VS   vacuum at T_ads: the gas phase is removed down to P_vac with the adsorbed phase frozen,
         then the bed re-equilibrates; repeated on a falling pressure ladder -> waste (N2-rich).
  3 IHS  intermediate heating to T_med under P_vac, vented -> waste (desorbs the weak N2 first).
  4 HS   heating to T_des under P_vac -> CO2 product.
  5 CS   repressurisation and cooling with feed (closes the cycle; no gas leaves).
Mixture equilibrium: IAST with the multi-site Langmuir CO2 isotherm and a one-site Langmuir N2
isotherm (Henry constant = K_H,CO2 / S_N2 at 298 K, heat Q_N2, saturation n2_qs).

Outputs per T_ads (all per kg sorbent and cycle):
  product CO2 and N2, CO2 vented, purity, recovery = product / captured, and the isothermal work
  to pump the vented gas from its pressure to the ambient pressure.
The plant model interpolates these on a T_ads grid (tvsa.table), computed once per sorbent and site.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import sorbents

R = 8.314                      # J/mol/K
RK = 8.314e-3                  # kJ/mol/K
T_PUMP = 300.0                 # K, gas temperature at the vacuum pump inlet (after recuperation)
N_VS = 24                      # pressure ladder of the vacuum step
N_IHS = 6
N_HS = 8
T_GRID = np.arange(170.0, 330.1, 4.0)
T_MED_OFFSETS = (0.0, 10.0, 25.0, 45.0)   # K above T_ads; 0 = no intermediate heating


# ------------------------------------------------------------------ pure isotherms
def co2_sites(sb):
    return [(qs, a, E) for qs, a, E in sb.sites]


def n2_sites(sb):
    KH298 = sum(qs * np.exp(a + E / (RK * 298.15)) for qs, a, E in sb.sites)      # mol/kg/bar
    qs = sb.n2_qs or sorbents.N2_QS_DEFAULT
    b298 = KH298 / sb.S_N2_298 / qs
    return [(qs, float(np.log(b298) - sb.Q_N2 / (RK * 298.15)), sb.Q_N2)]


def _b(sites, T):
    return [(qs, np.exp(a + E / (RK * T))) for qs, a, E in sites]


def q_pure(bs, p):
    return sum(qs * b * p / (1.0 + b * p) for qs, b in bs)


def pi_pure(bs, p):
    return sum(qs * np.log1p(b * p) for qs, b in bs)


def p_inv(bs, pi, p0=None):
    """Pressure whose reduced spreading pressure equals pi (Newton in ln p)."""
    Q = sum(qs for qs, _ in bs)
    KH = sum(qs * b for qs, b in bs)
    p = np.expm1(np.minimum(pi / Q, 50.0)) / (KH / Q) if p0 is None else p0
    p = np.maximum(p, 1e-30)
    for _ in range(30):
        f = pi_pure(bs, p) - pi
        du = np.clip(-f / np.maximum(q_pure(bs, p), 1e-300), -3.0, 3.0)
        p = p * np.exp(du)
        if np.all(np.abs(du) < 1e-10):
            break
    return p


def iast2(bs1, bs2, p1, p2):
    """Binary IAST: loadings [mol/kg] at partial pressures p1, p2 [bar]."""
    p1 = np.maximum(p1, 1e-30)
    p2 = np.maximum(p2, 1e-30)
    pi = np.maximum(pi_pure(bs1, p1), pi_pure(bs2, p2))        # lower bound: g(pi) >= 0
    P1 = P2 = None
    for _ in range(40):
        P1 = p_inv(bs1, pi, P1)
        P2 = p_inv(bs2, pi, P2)
        x1, x2 = p1 / P1, p2 / P2
        g = x1 + x2 - 1.0
        gp = -(x1 / q_pure(bs1, P1) + x2 / q_pure(bs2, P2))
        step = -g / gp
        pi = np.maximum(pi + step, 1e-300)
        if np.all(np.abs(g) < 1e-10):
            break
    x1, x2 = p1 / P1, p2 / P2
    s = x1 + x2
    x1, x2 = x1 / s, x2 / s
    qt = 1.0 / (x1 / q_pure(bs1, P1) + x2 / q_pure(bs2, P2))
    return x1 * qt, x2 * qt


# ------------------------------------------------------------------ bed equilibrium
def flash(bs1, bs2, c, N1, N2, p1, p2):
    """Partial pressures [bar] so that adsorbed + gas = N_i (mol/kg); c = gas mol/kg per bar."""
    u = np.log(np.maximum(np.stack([p1, p2]), 1e-30))
    N = np.stack([N1, N2])
    for _ in range(40):
        p = np.exp(u)
        q = np.stack(iast2(bs1, bs2, p[0], p[1]))
        F = q + c * p - N
        h = 1e-6
        J = np.empty((2, 2) + u.shape[1:])
        for j in range(2):
            uj = u.copy()
            uj[j] += h
            pj = np.exp(uj)
            qj = np.stack(iast2(bs1, bs2, pj[0], pj[1]))
            J[:, j] = ((qj + c * pj - N) - F) / h
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        d0 = -(J[1, 1] * F[0] - J[0, 1] * F[1]) / det
        d1 = -(-J[1, 0] * F[0] + J[0, 0] * F[1]) / det
        du = np.clip(np.stack([d0, d1]), -2.0, 2.0)
        du = np.where(np.isfinite(du), du, 0.0)
        u = u + du
        if np.all(np.abs(du) < 1e-9):
            break
    p = np.exp(u)
    q = iast2(bs1, bs2, p[0], p[1])
    return p[0], p[1], q[0], q[1]


def release(bs1, bs2, c, N1, N2, P, p1, p2):
    """Open-end desorption at total pressure P (one implicit batch-distillation step): the bed
    releases n mol/kg of gas with the composition of its own gas phase until the pressure is P.
    Returns (removed CO2, removed N2, p1, p2). If the bed is already below P nothing leaves."""
    e1, e2, _, _ = flash(bs1, bs2, c, N1, N2, p1, p2)
    below = e1 + e2 <= P
    y = np.clip(np.where(below, 0.5, e1 / (e1 + e2)), 1e-15, 1 - 1e-15)
    z = np.log(y / (1 - y))
    n = np.where(below, 0.0, c * (e1 + e2 - P))                 # first guess: the excess gas

    def F(z, n):
        y1 = 1.0 / (1.0 + np.exp(-z))
        q1, q2 = iast2(bs1, bs2, y1 * P, (1 - y1) * P)
        return np.stack([q1 + (c * P + n) * y1 - N1, q2 + (c * P + n) * (1 - y1) - N2])

    for _ in range(60):
        f0 = F(z, n)
        hz, hn = 1e-6, 1e-9 + 1e-6 * np.abs(n)
        fz = (F(z + hz, n) - f0) / hz
        fn = (F(z, n + hn) - f0) / hn
        det = fz[0] * fn[1] - fn[0] * fz[1]
        dz = -(fn[1] * f0[0] - fn[0] * f0[1]) / det
        dn = -(-fz[1] * f0[0] + fz[0] * f0[1]) / det
        dz = np.where(np.isfinite(dz), np.clip(dz, -3, 3), 0.0)
        dn = np.where(np.isfinite(dn), dn, 0.0)
        z = z + dz
        n = np.maximum(n + dn, 0.0)
        if np.all((np.abs(dz) < 1e-10) & (np.abs(dn) < 1e-12 + 1e-9 * n)):
            break
    y1 = 1.0 / (1.0 + np.exp(-z))
    r1 = np.where(below, 0.0, n * y1)
    r2 = np.where(below, 0.0, n * (1 - y1))
    return r1, r2, np.where(below, e1, y1 * P), np.where(below, e2, (1 - y1) * P)


def void_volume(sb, p) -> float:
    """m3 of gas-filled void (bed + pellet) per kg sorbent (PrISMa: eps_bed 0.37, eps_pellet 0.35)."""
    rho_bulk = sb.rho * 1e3 * (1 - p.eps_pellet) * (1 - p.eps_bed)
    return (p.eps_bed + (1 - p.eps_bed) * p.eps_pellet) / rho_bulk


def cycle(sb, T_ads, P_feed, y_co2, T_med_off, T_des, P_vac, V, eta_vac, P_amb=1.01325, P_vs=None):
    """Run the 5-step cycle for arrays of T_ads (and T_med offsets). Returns a dict of arrays."""
    T_ads = np.asarray(T_ads, float)
    P_vs = P_vac if P_vs is None else P_vs
    c1, c2 = co2_sites(sb), n2_sites(sb)
    cf = lambda T: 1e5 * V / (R * T)                           # mol/kg per bar
    # 1 adsorption
    T = T_ads
    p1 = np.full_like(T, y_co2 * P_feed)
    p2 = np.full_like(T, (1 - y_co2) * P_feed)
    q1, q2 = iast2(_b(c1, T), _b(c2, T), p1, p2)
    N1 = q1 + cf(T) * p1
    N2 = q2 + cf(T) * p2
    N1_ads = N1.copy()
    vent1 = np.zeros_like(T)
    vent2 = np.zeros_like(T)
    work = np.zeros_like(T)                                    # kJ/kg sorbent

    def out(T, P, p1, p2, vent):
        """Release gas at total pressure P; vented gas is pumped to ambient (work), product is not
        (product pumping is counted in plant.py)."""
        nonlocal N1, N2, work
        r1, r2, p1, p2 = release(_b(c1, T), _b(c2, T), cf(T), N1, N2, P, p1, p2)
        N1, N2 = N1 - r1, N2 - r2
        if vent:
            work = work + (r1 + r2) * R * T_PUMP * np.log(P_amb / P) / eta_vac / 1e3
        return r1, r2, p1, p2

    # 2 vacuum step at T_ads down to P_vs (falling pressure ladder) -> waste
    for P in np.geomspace(P_feed, P_vs, N_VS + 1)[1:]:
        r1, r2, p1, p2 = out(T, P, p1, p2, True)
        vent1, vent2 = vent1 + r1, vent2 + r2
    # 3 intermediate heating to T_med under P_vac (vented), 4 heating to T_des under P_vac (product)
    prod1 = np.zeros_like(T)
    prod2 = np.zeros_like(T)
    T_med = np.minimum(T_ads + T_med_off, T_des)
    for n, T_from, T_to, to_product in ((N_IHS, T_ads, T_med, False),
                                        (N_HS, T_med, np.full_like(T, T_des), True)):
        for f in np.linspace(0, 1, n + 1)[1:]:
            T = T_from + f * (T_to - T_from)
            r1, r2, p1, p2 = out(T, P_vac, p1, p2, not to_product)
            if to_product:
                prod1, prod2 = prod1 + r1, prod2 + r2
            else:
                vent1, vent2 = vent1 + r1, vent2 + r2
    captured = prod1 + vent1
    purity = prod1 / np.maximum(prod1 + prod2, 1e-30)
    return dict(prod_co2=prod1, prod_n2=prod2, vent_co2=vent1, vent_n2=vent2, captured=captured,
                ads_co2=N1_ads, purity=purity, recovery=prod1 / np.maximum(captured, 1e-30), work=work)


@lru_cache(maxsize=256)
def table(sb, P_feed: float, y_co2: float, V: float, eta_vac: float, purity_min: float):
    """Cycle results on T_GRID, best T_med offset per T_ads: the largest CO2 product among the
    offsets that reach purity_min (otherwise the purest). Cached per sorbent and site pressure."""
    T = np.repeat(T_GRID, len(T_MED_OFFSETS))
    off = np.tile(np.array(T_MED_OFFSETS), len(T_GRID))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        r = cycle(sb, T, P_feed, y_co2, off, sb.T_des, sb.p_des, V, eta_vac)
    shape = (len(T_GRID), len(T_MED_OFFSETS))
    prod, pur = r["prod_co2"].reshape(shape), r["purity"].reshape(shape)
    ok = pur >= purity_min
    score = np.where(ok, prod, -1.0 + pur)                     # feasible first, then purest
    k = np.argmax(score, axis=1)
    pick = lambda a: a.reshape(shape)[np.arange(len(T_GRID)), k]
    return dict(T=T_GRID, prod_co2=pick(r["prod_co2"]), purity=pick(r["purity"]), recovery=pick(r["recovery"]),
                work=pick(r["work"]), T_med_off=np.array(T_MED_OFFSETS)[k], ok=ok[np.arange(len(T_GRID)), k])


def at(sb, T_ads, P_feed, y_co2, V, eta_vac, purity_min):
    """Interpolate the cycle table at hourly T_ads. 'ok' is False if either neighbour fails."""
    t = table(sb, round(float(P_feed), 2), float(y_co2), round(float(V), 7), float(eta_vac), float(purity_min))
    x = np.clip(np.asarray(T_ads, float), T_GRID[0], T_GRID[-1])
    out = {k: np.interp(x, t["T"], t[k]) for k in ("prod_co2", "purity", "recovery", "work")}
    i = np.clip(np.searchsorted(t["T"], x), 1, len(T_GRID) - 1)
    out["ok"] = t["ok"][i] & t["ok"][i - 1]
    return out
