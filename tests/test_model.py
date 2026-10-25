"""Consistency checks: run with  python -m pytest tests"""
import pytest

from lh2sys import nh3h2o as aw
from lh2sys.analysis import BASE_DV, PAPER_OPT_DV
from lh2sys.params import Params
from lh2sys.system import evaluate


def test_ammonia_water_pure_limits():
    assert aw.T_bubble(101.325, 0.0) == pytest.approx(373.1, abs=0.5)      # water
    assert aw.P_bubble(313.0, 1.0) == pytest.approx(1555, rel=0.01)        # NH3 at 40 C
    assert aw.h_vapour(373.15, 0.0) - aw.h_liquid(373.15, 0.0) == pytest.approx(2257, rel=0.01)


@pytest.mark.parametrize("dv", [BASE_DV, PAPER_OPT_DV])
@pytest.mark.parametrize("strict", [False, True])
def test_balances_close(dv, strict):
    r = evaluate(Params(strict_pinch=strict, **dv))
    assert abs(r["gax"]["balance"]) < 1e-6                              # GAX energy balance
    assert r["ExD_total"] == pytest.approx(r["Ex_F"] - r["Ex_P"], rel=1e-9)
    assert r["states"][10]["m"] == pytest.approx(r["states"][1]["m"])   # m10 == m1
    m = {k: v["m"] for k, v in r["states"].items()}
    h = {k: v["h"] for k, v in r["states"].items()}
    residuals = [
        m[2] * h[2] + m[17] * h[17] - m[3] * h[3],                     # mixer 3
        m[4] * (h[4] - h[5]) - m[14] * (h[17] - h[14]),                # RHE2
        m[6] * (h[6] - h[7]) - m[13] * (h[14] - h[13]),                # RHE3
        m[7] * (h[7] - h[8]) - m[11] * (h[12] - h[11]),                # RHE4
        m[9] * h[9] - m[10] * h[10] - m[11] * h[11],                   # separator
        m[12] * h[12] + m[16] * h[16] - m[13] * h[13],                 # mixer 13
        m[1] * (h[1] - h[2]) - r["Q_eva"],                             # GAX evaporator
    ]
    assert max(map(abs, residuals)) < 1e-6
    if strict:
        assert r["min_approach"] >= 1.99
        assert all(v > -1e-6 for v in r["ExD"].values())                # 2nd law per component


def test_paper_mode_close_to_table6():
    r = evaluate(Params(**BASE_DV))
    assert r["c_LH2"] == pytest.approx(1.403, rel=0.03)
    assert r["eta_II"] == pytest.approx(45.69, rel=0.15)
