"""Validation tests for the integrated plant (spec V3, V5, V6)."""
import numpy as np
import pytest

from dacsys import ars_map, climate_sites, plant, vcr
from dacsys.levers import COLD, DRYING, Case, Prm, with_


@pytest.fixture(scope="module")
def madrid():
    return climate_sites.load("Madrid")


# ---- V5: balances ---------------------------------------------------------------------------
@pytest.mark.parametrize("drying", DRYING)
@pytest.mark.parametrize("cold", ["gax", "vcr", "lng"])
def test_water_balance_closes_every_hour(madrid, drying, cold):
    case = with_(Case(site="Madrid"), drying=drying, cold=cold)
    T = {"gax": 245.0, "vcr": 230.0, "lng": 200.0}[cold]
    L = plant.air_train(madrid, case, T, 3.0)
    removed = L["cond"] + L["sil"] + L["frz"] + L["frost"]
    assert np.allclose(L["w0"] - removed, L["w_feed"], atol=1e-12)
    assert (L["water"] <= removed + L["m_w"] + 1e-12).all()   # + water desorbed from the bed
    # ice in the reversing regenerators leaves with the exhaust and is not recovered
    assert (L["water"] <= removed - L["frz"] + L["m_w"] + 1e-12).all()


def test_chiller_first_law(madrid):
    L = plant.air_train(madrid, Case(site="Madrid"), 245.0, 3.0)
    assert np.allclose(L["rejected"], L["Q1"] + L["Q2"] + L["q_gen"])


def test_dispatch_creates_no_energy():
    rng = np.random.default_rng(0)
    H = 500
    q_sol, pv = rng.uniform(0, 1, H), rng.uniform(0, 1, H)
    dem_th, dem_el = rng.uniform(0, 2, H), rng.uniform(0, 1, H)
    want = np.ones(H, bool)
    A, S_th, P_pv, S_el = (np.array([1.5]), np.array([5.0]), np.array([1.2]), np.array([3.0]))
    ran = plant.dispatch(want, dem_th, dem_el, q_sol, pv, A, S_th, P_pv, S_el, Prm())[:, 0]
    used_th, used_el = (dem_th * ran).sum(), (dem_el * ran).sum()
    assert used_th <= (q_sol * A[0]).sum() + 0.5 * S_th[0] + 1e-9
    assert used_el <= (pv * P_pv[0]).sum() + 0.5 * S_el[0] + 1e-9


# ---- V6: cold-machine envelopes -------------------------------------------------------------
def test_gax_envelope():
    c, _ = ars_map.cop_env([20.0, 20.0], [-35.0, -45.0])
    assert np.isfinite(c[0]) and 0.1 < c[0] < 1.5
    assert np.isnan(c[1])                                  # below -38 C: outside the envelope
    strict, g = ars_map.cop_env([30.0], [-35.0], 160.0)
    ext, _ = ars_map.cop_env([30.0], [-35.0], 220.0)
    assert g[0] <= 160.0 and strict[0] <= ext[0] + 1e-12


@pytest.mark.parametrize("ta,te", [(0, -40), (20, -30), (35, -45), (40, -2)])
def test_vcr_below_carnot(ta, te):
    carnot = (te + 273.15) / ((ta + 10.0) - te)
    c = float(vcr.cop(ta, te))
    assert 0.3 * carnot < c < 0.8 * carnot or c == vcr.COP_MAX


# ---- V3: reproduction of Kim et al. (EES 2025), 68.2 $/t --------------------------------------
def test_kim_reproduction(madrid):
    from dataclasses import replace
    from dacsys import sorbents
    sorbents.register(replace(sorbents.get("13X"), name="13X_kim", cost=0.85))
    case = with_(Case(site="Madrid"), drying="none", cold="lng", sorbent="13X_kim", regen="sweep200",
                 energy_mode="grid", lifetime=25, c_hx_area=40.0)
    r = plant.evaluate(case, madrid, T_list=[195.0])
    assert 50.0 < r["lcoc"] < 100.0
    a = plant.amine(madrid, p=with_(case, lifetime=15).prm)
    assert abs(a["amine_lcoc_base"] - 171.3) / 171.3 < 0.2  # their amine case, same energy prices


def test_dark_winter_site_feasible():
    """Two-stage search must not drop designs whose cheapest proxies cannot run (Yakutsk)."""
    w = climate_sites.load("Yakutsk")
    case = with_(Case(site="Yakutsk"), drying="freeze", cold="vcr")
    r = plant.evaluate(case, w)
    assert r["status"] == "ok" and r["lcoc"] < 2500.0


# ---- two-zone water model (PrISMa / Webley) ----------------------------------------------------
def test_two_zone_limits(madrid):
    from dataclasses import replace
    from dacsys import sorbents
    sb = plant.sorbent_for(Case(site="Madrid"))
    T = np.full(3, 241.0); P = np.full(3, 1.0e5)
    wdry = sb.working_capacity(T, 4.2e-4 * P / 1e5)
    # no water in the feed -> dry capacity, alpha 0
    wc, a, r = plant.two_zone(sb, wdry, np.zeros(3), P, T, Prm())
    assert np.allclose(wc, wdry) and np.allclose(a, 0) and np.allclose(r, 0)
    # water-blind sorbent (WRC = 1) keeps its dry capacity at any humidity
    wc, a, r = plant.two_zone(replace(sb, wrc=1.0), wdry, np.full(3, 3e-4), P, T, Prm())
    assert np.allclose(wc, wdry)
    # 13X fed air saturated at 241 K: small wet zone, ~0.9 mol water per mol CO2
    wc, a, r = plant.two_zone(sb, wdry, np.full(3, 1.9e-4), P, T, Prm())
    assert 0.02 < a[0] < 0.2 and 0.5 < r[0] < 1.2 and wc[0] < wdry[0]


def test_13x_without_silica_now_feasible(madrid):
    r = plant.evaluate(with_(Case(site="Madrid"), drying="cond", cold="gax"), madrid)
    assert r["status"] == "ok"


# ---- step 3: heat-transfer cycle time and material heat capacity -------------------------------
def test_cycle_time_physical():
    sb = plant.sorbent_for(Case(site="Madrid"))
    T = np.array([241.0, 290.0])
    wc = sb.working_capacity(T, np.full(2, 4.2e-4))
    n = plant.cycles_per_h(sb, Prm(), np.maximum(wc, 1e-3), np.zeros(2), T, sb.cp + Prm().cp_contactor)
    assert np.all(n > 0.5) and np.all(n <= Prm().max_cycles_per_h + 1e-9)
    # a slower heat exchanger gives fewer cycles
    n2 = plant.cycles_per_h(sb, with_(Case(site="Madrid"), U_hx=0.004).prm, np.maximum(wc, 1e-3),
                            np.zeros(2), T, sb.cp + Prm().cp_contactor)
    assert np.all(n2 < n)


# ---- step 2: PrISMa materials --------------------------------------------------------------------
def test_prisma_fits_reproduce_henry():
    from dacsys import prisma, sorbents
    fits = prisma.load_fits()
    names = prisma.register()
    good = [n for n in names if prisma.fit_ok(n, fits)]
    assert len(good) >= 0.98 * len(names)                 # >= 98 % of the 1 338 fits pass the gate
    for n in prisma.prescreen(names[:200]):              # everything that enters the study passes it
        d = fits[n[len(prisma.PREFIX):]]
        assert abs(sorbents.get(n).q(d["T_ref"], 1e-9) / 1e-9 / 1e5 / d["KH_CO2"] - 1) <= 0.05


# ---- RQ5 property decomposition --------------------------------------------------------------------
def test_decomp_hybrid_endpoints():
    from dacsys import decomp, sorbents
    names = decomp._register()
    real, tgt = sorbents.get(decomp.REAL), sorbents.get(decomp.TARGET)
    T = np.array([200.0, 241.0, 298.15, 373.15])
    p = np.array([4.2e-4, 4.2e-4, 1e-2, 1e-2])
    for n, ref in ((names[0], real), (names[-1], tgt)):
        h = sorbents.get(n)
        assert np.allclose(h.q(T, p), ref.q(T, p), rtol=1e-10)
        for f in ("hoa", "cost", "S_N2_298", "Q_N2", "h2o_KH298", "wrc", "cp", "rho"):
            assert getattr(h, f) == getattr(ref, f)
    # switching only the heat keeps the 298 K isotherm of the real MOF
    heat_only = sorbents.get("DX:" + "".join("1" if g == "heat" else "0" for g in decomp.GROUPS))
    assert np.allclose(heat_only.q(298.15, p), real.q(298.15, p), rtol=1e-10)
    assert not np.allclose(heat_only.q(241.0, 4.2e-4), real.q(241.0, 4.2e-4))


# ---- step 4: TVSA purity with IAST ------------------------------------------------------------------
def test_iast_reduces_to_extended_langmuir():
    from dacsys import tvsa
    T = np.array([250.0])
    s1 = [(3.0, -10.0, 40.0)]
    s2 = [(3.0, -9.0, 18.0)]                              # same saturation -> IAST = extended Langmuir
    b1, b2 = tvsa._b(s1, T), tvsa._b(s2, T)
    p1, p2 = np.array([4e-4]), np.array([0.8])
    q1, q2 = tvsa.iast2(b1, b2, p1, p2)
    d = 1 + b1[0][1] * p1 + b2[0][1] * p2
    assert np.allclose(q1, 3.0 * b1[0][1] * p1 / d, rtol=1e-6)
    assert np.allclose(q2, 3.0 * b2[0][1] * p2 / d, rtol=1e-6)


def test_tvsa_cycle_mass_balance_and_limits():
    from dataclasses import replace as rep
    from dacsys import tvsa
    sb = plant.sorbent_for(Case(site="London"))
    V = tvsa.void_volume(sb, Prm())
    T = np.array([200.0, 225.0, 250.0])
    r = tvsa.cycle(sb, T, 1.0, 420e-6, np.zeros(3), sb.T_des, sb.p_des, V, 0.5)
    # everything captured leaves as product or vent; what stays is the regeneration residual
    resid = r["ads_co2"] - r["captured"]
    assert np.all(resid > 0) and np.allclose(resid, sb.q(sb.T_des, sb.p_des), rtol=0.1)
    assert np.all((r["purity"] > 0) & (r["purity"] <= 1)) and np.all(r["recovery"] > 0.9)
    # a far more selective sorbent gives a purer product
    s2 = rep(sb, S_N2_298=sb.S_N2_298 * 100)
    r2 = tvsa.cycle(s2, T, 1.0, 420e-6, np.zeros(3), sb.T_des, sb.p_des, V, 0.5)
    assert np.all(r2["purity"] > r["purity"]) and np.all(r2["purity"] > 0.999)
    # the vacuum step raises purity above the Henry-regime estimate without it (13X, 225-250 K)
    old = 1 / (1 + (0.7808 / 420e-6) / sb.selectivity_n2(T))
    assert np.all(r["purity"][1:] > old[1:])


# ---- water-model robustness -------------------------------------------------------------------------
def test_water_front_factor_and_wrc_scaling():
    from dataclasses import replace
    sb = plant.sorbent_for(Case(site="Madrid"))
    T = np.full(2, 241.0); P = np.full(2, 1.0e5)
    w = np.array([2e-4, 5e-4])
    wc0 = sb.working_capacity(T, np.full(2, 4.2e-4))
    base = plant.two_zone(sb, wc0, w, P, T, Prm())
    slow = plant.two_zone(sb, wc0, w, P, T, replace(Prm(), water_front_factor=3.0))
    assert np.all(slow[0] < base[0]) and np.all(slow[1] >= base[1])     # longer wet zone -> less CO2
    mid = replace(sb, h2o_qsat=19.8, h2o_KH298=7.92e-4, h2o_dH=44.1, wrc=0.131)
    a = plant.two_zone(mid, wc0, w, P, T, Prm())
    b = plant.two_zone(mid, wc0, w, P, T, replace(Prm(), wrc_pore_scaling=False))
    assert np.all(b[0] <= a[0] + 1e-15)                                   # unscaled WRC is the harsher case
