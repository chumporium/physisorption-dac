"""Climate-representative sites: one per major Koppen-Geiger class.

Hourly 2018 weather and 1991-2020 daily normals come from the Open-Meteo archive
restricted to the ERA5 reanalysis (models=era5, 0.25 deg), so every site shares one
data pipeline. The intended Koppen class of each site is verified from the normals.

python -m dacsys.climate_sites            -> download (cached) + Koppen check table
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "openmeteo_era5"
API = "https://archive-api.open-meteo.com/v1/archive"
YEAR = 2018
HOURLY = ["temperature_2m", "dew_point_2m", "surface_pressure", "shortwave_radiation",
          "direct_radiation", "direct_normal_irradiance", "wind_speed_100m", "precipitation"]

# name -> (lat, lon, intended Koppen class, description)
SITES = {
    # A - tropical
    "Singapore": (1.35, 103.82, "Af", "tropical rainforest"),
    "Jakarta": (-6.20, 106.85, "Am", "tropical monsoon"),
    "Darwin": (-12.46, 130.84, "Aw", "tropical savanna"),
    # B - arid
    "Riyadh": (24.70, 46.70, "BWh", "hot desert, inland"),
    "Dubai": (25.20, 55.30, "BWh", "hot desert, humid coast (Persian Gulf)"),
    "Atacama coast": (-23.60, -70.20, "BWk", "cool coastal desert (MAT < 18 C)"),
    "Kano": (12.00, 8.52, "BSh", "hot steppe (Sahel)"),
    "Gobi": (43.60, 104.40, "BWk", "cold desert"),
    "Madrid": (40.42, -3.70, "BSk", "cold steppe"),
    # C - temperate
    "Shanghai": (31.23, 121.47, "Cfa", "humid subtropical"),
    "London": (51.51, -0.13, "Cfb", "oceanic"),
    "Rome": (41.90, 12.50, "Csa", "Mediterranean"),
    "Mexico City": (19.43, -99.13, "Cwb", "subtropical highland"),
    # D - continental
    "Chicago": (41.88, -87.63, "Dfa", "humid continental, hot summer"),
    "Seoul": (37.57, 126.98, "Dwa", "monsoon continental"),
    "Astana": (51.20, 71.40, "Dfb", "continental, warm summer"),
    "Kiruna": (67.80, 20.20, "Dfc", "subarctic"),
    "Yakutsk": (62.00, 129.70, "Dwc", "extreme subarctic, dry winter"),
    # E / highland
    "Reykjavik": (64.10, -21.80, "Dfc", "maritime subarctic (Cfc with the -3 C boundary)"),
    "Tibet": (33.00, 90.00, "ET", "tundra highland (about 4 700 m)"),
    "Altiplano": (-22.00, -68.00, "ET", "dry highland (about 4 000 m)"),
}


def _get(params: dict, tries: int = 5) -> dict:
    url = API + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def fetch(name: str) -> Path:
    """Download (once) hourly 2018 + daily 1991-2020 series for a site."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{name.replace(' ', '_')}.npz"
    if out.exists():
        return out
    lat, lon, _, _ = SITES[name]
    base = dict(latitude=lat, longitude=lon, models="era5", timezone="GMT", wind_speed_unit="ms")
    h = _get(base | dict(start_date=f"{YEAR}-01-01", end_date=f"{YEAR}-12-31", hourly=",".join(HOURLY)))
    d = _get(base | dict(start_date="1991-01-01", end_date="2020-12-31",
                         daily="temperature_2m_mean,precipitation_sum"))
    arr = {v: np.array(h["hourly"][v], dtype=float) for v in HOURLY}
    np.savez(out, **arr, d_time=np.array(d["daily"]["time"]),
             d_T=np.array(d["daily"]["temperature_2m_mean"], dtype=float),
             d_P=np.array(d["daily"]["precipitation_sum"], dtype=float),
             grid_lat=h["latitude"], grid_lon=h["longitude"], elevation=h["elevation"])
    return out


def load(name: str) -> dict:
    """Hourly weather in the units the plant model uses (K, Pa, W/m2, m/s)."""
    z = np.load(fetch(name))
    return dict(T0=z["temperature_2m"] + 273.15, Td=z["dew_point_2m"] + 273.15,
                P=z["surface_pressure"] * 100.0, ghi=z["shortwave_radiation"],
                fdir=z["direct_radiation"], dni=z["direct_normal_irradiance"],
                wind100=z["wind_speed_100m"], precip=z["precipitation"],
                lat=float(z["grid_lat"]), lon=float(z["grid_lon"]), elevation=float(z["elevation"]))


# ------------------------------------------------------------------ other years and other points (robustness runs)
EXTRA = DATA.parent / "openmeteo_era5_extra"

# large LNG import terminals (approximate coordinates of the terminal area; ERA5 cell of 0.25 deg)
TERMINALS = {
    "Futtsu (JP)": (35.33, 139.83), "Incheon (KR)": (37.40, 126.62), "Dapeng (CN)": (22.60, 114.50),
    "Yung-An (TW)": (22.83, 120.20), "Dahej (IN)": (21.70, 72.55), "Map Ta Phut (TH)": (12.68, 101.15),
    "Rotterdam (NL)": (51.97, 4.05), "Isle of Grain (UK)": (51.45, 0.71), "Montoir (FR)": (47.30, -2.15),
    "Barcelona (ES)": (41.35, 2.16), "Swinoujscie (PL)": (53.92, 14.28), "Quintero (CL)": (-32.78, -71.53),
}


def fetch_point(label: str, lat: float, lon: float, year: int) -> Path:
    """Download (once) hourly ERA5 for any point and year (same variables and source as fetch)."""
    EXTRA.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in label)
    out = EXTRA / f"{safe}_{year}.npz"
    if out.exists():
        return out
    base = dict(latitude=lat, longitude=lon, models="era5", timezone="GMT", wind_speed_unit="ms")
    h = _get(base | dict(start_date=f"{year}-01-01", end_date=f"{year}-12-31", hourly=",".join(HOURLY)))
    arr = {v: np.array(h["hourly"][v], dtype=float) for v in HOURLY}
    np.savez(out, **arr, grid_lat=h["latitude"], grid_lon=h["longitude"], elevation=h["elevation"])
    time.sleep(1.0)
    return out


def load_point(label: str, lat: float, lon: float, year: int) -> dict:
    """Hourly weather of any point and year, as load(); 29 February is dropped so that every year has 8 760 h."""
    z = np.load(fetch_point(label, lat, lon, year))
    keep = slice(None)
    n = len(z["temperature_2m"])
    if n == 8784:                                         # leap year: drop 29 February (hours 1416-1439)
        keep = np.r_[0:1416, 1440:n]
    g = {k: np.asarray(z[k], float)[keep] for k in HOURLY}
    for k in HOURLY:                                      # rare gaps in the archive: linear fill
        v = g[k]
        if np.isnan(v).any():
            i = np.arange(len(v))
            ok = ~np.isnan(v)
            g[k] = np.interp(i, i[ok], v[ok])
    return dict(T0=g["temperature_2m"] + 273.15, Td=g["dew_point_2m"] + 273.15,
                P=g["surface_pressure"] * 100.0, ghi=g["shortwave_radiation"],
                fdir=g["direct_radiation"], dni=g["direct_normal_irradiance"],
                wind100=g["wind_speed_100m"], precip=g["precipitation"],
                lat=float(z["grid_lat"]), lon=float(z["grid_lon"]), elevation=float(z["elevation"]))


# ------------------------------------------------------------------ Koppen-Geiger
def monthly_normals(name: str):
    z = np.load(fetch(name))
    month = np.array([int(t[5:7]) for t in z["d_time"]])
    year = np.array([int(t[:4]) for t in z["d_time"]])
    T = np.array([np.nanmean(z["d_T"][month == m]) for m in range(1, 13)])
    P = np.array([np.nansum(z["d_P"][month == m]) / len(np.unique(year)) for m in range(1, 13)])
    return T, P


def koppen(T: np.ndarray, P: np.ndarray, lat: float) -> str:
    """Koppen-Geiger class from monthly normals (Peel et al. 2007 / Beck et al. 2018 rules)."""
    MAT, MAP = T.mean(), P.sum()
    Thot, Tcold = T.max(), T.min()
    summer = np.array([4, 5, 6, 7, 8, 9]) - 1 if lat >= 0 else np.array([10, 11, 12, 1, 2, 3]) - 1
    winter = np.setdiff1d(np.arange(12), summer)
    Ps, Pw = P[summer], P[winter]
    Pth = 2 * MAT + (28 if Ps.sum() >= 0.7 * MAP else 0 if Pw.sum() >= 0.7 * MAP else 14)
    if MAP < 10 * Pth:
        return ("BW" if MAP < 5 * Pth else "BS") + ("h" if MAT >= 18 else "k")
    if Tcold >= 18:
        if P.min() >= 60:
            return "Af"
        return "Am" if P.min() >= 100 - MAP / 25 else "Aw"
    if Thot < 10:
        return "ET" if Thot > 0 else "EF"
    first = "C" if Tcold > 0 else "D"
    if Ps.min() < 40 and Ps.min() < Pw.max() / 3:
        second = "s"
    elif Pw.min() < Ps.max() / 10:
        second = "w"
    else:
        second = "f"
    n10 = int((T >= 10).sum())
    if Thot >= 22:
        third = "a"
    elif n10 >= 4:
        third = "b"
    elif first == "D" and Tcold < -38:
        third = "d"
    else:
        third = "c"
    return first + second + third


def table():
    import pandas as pd
    rows = []
    for name, (lat, lon, want, desc) in SITES.items():
        fetch(name)
        w = load(name)
        T, P = monthly_normals(name)
        got = koppen(T, P, lat)
        rows.append(dict(site=name, intended=want, koppen_1991_2020=got, match=got == want,
                         grid_lat=w["lat"], grid_lon=w["lon"], elev_m=w["elevation"],
                         T_mean_2018_C=w["T0"].mean() - 273.15, Td_mean_2018_C=w["Td"].mean() - 273.15,
                         DNI_kWh_m2_yr=w["dni"].sum() / 1e3, MAP_mm=P.sum(), desc=desc))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    t = table()
    out = Path(__file__).resolve().parent.parent / "results_dac" / "study5"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "sites_koppen.csv", index=False)
    print(t.round(2).to_string(index=False))
