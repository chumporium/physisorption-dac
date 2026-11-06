"""Checks for the solar-ARS DAC screening model: python -m pytest tests"""
import numpy as np
import pytest
from CoolProp.HumidAirProp import HAPropsSI

from dacsys import ars_map
from dacsys.params import DacParams
from dacsys.process import hourly_loads, working_capacity_13x
from dacsys.weather import humidity_ratio, sun


@pytest.mark.parametrize("Td", [233.15, 268.15, 278.15, 298.15])
def test_humidity_ratio_vs_coolprop(Td):
    w = humidity_ratio(np.array(Td), np.array(101325.0))
    ref = HAPropsSI("W", "T", Td + 1.0, "P", 101325, "D", Td)
    assert float(w) == pytest.approx(ref, rel=0.02)


def test_solar_noon_zenith():
    # 2018-06-21, solar noon at lon 0 ~ 12:02 UTC; stamp 13 -> mid-hour 12:30
    cz, _ = sun(np.array([24 * 171 + 13]), "2018-01-01T00:00", np.array([23.44]), np.array([0.0]))
    assert float(cz[0, 0]) > 0.99


def test_working_capacity_monotonic():
    wc = [working_capacity_13x(T) for T in (225, 235, 245)]
    assert wc[0] > wc[1] > wc[2] > 0


def test_cop_map_trends():
    assert ars_map.cop(10.0, -40.0) > ars_map.cop(40.0, -40.0)     # hotter ambient -> worse
    assert ars_map.cop(30.0, -2.5) > ars_map.cop(30.0, -40.0)      # colder evaporator -> worse


def test_loads_match_hand_estimate():
    # Jakarta-like air 30 C / dew 25 C: cond+silica water ~0.020 kg/kg, heat ~ 4-7 GJ/m3
    p = DacParams()
    shape = (1, 1)
    w = dict(T0=np.full(shape, 303.15), Td=np.full(shape, 298.15), P=np.full(shape, 101325.0))
    L = hourly_loads(w, p)
    assert L["water"][0, 0] == pytest.approx(0.0195, rel=0.1)
    heat_per_m3 = L["heat"][0, 0] / L["water"][0, 0] / 1e3      # kJ/kg -> GJ/m3
    assert 3.0 < heat_per_m3 < 8.0
