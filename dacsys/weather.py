"""Load the downloaded ERA5 region cubes; psychrometrics and solar geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "era5_2018"


def region_meta(region: str) -> dict:
    d = DATA / region
    m = json.loads((d / "meta.json").read_text())
    m["lat"], m["lon"] = np.array(m["lat"]), np.array(m["lon"])
    m["lsm"] = np.load(d / "lsm.npy")
    return m


def load_cells(region: str, cells: np.ndarray) -> dict:
    """Hourly series (H, n) for flat cell indices `cells` of the region grid."""
    m = region_meta(region)
    ny, nx, H = len(m["lat"]), len(m["lon"]), m["n_hours"]
    out = {}
    for v, info in m["vars"].items():
        mm = np.memmap(DATA / region / f"{v}.i16", dtype=np.int16, mode="r", shape=(H, ny * nx))
        out[v] = mm[:, cells].astype(np.float32) / info["scale"] + info["offset"]
    out["T0"] = out.pop("t2m")                      # K
    out["Td"] = out.pop("d2m")                      # K
    out["P"] = out.pop("sp")                        # Pa
    # radiation is stored as hourly accumulation (J/m2); convert to hour-mean W/m2
    out["fdir"] = np.maximum(out["fdir"], 0.0) / 3600.0   # W/m2 horizontal, hour ending at stamp
    out["ssrd"] = np.maximum(out["ssrd"], 0.0) / 3600.0
    return out


# ------------------------------- psychrometrics --------------------------------
def p_sat(T_K):
    """Saturation vapour pressure [Pa], over water above 0 C and ice below (Alduchov & Eskridge)."""
    t = np.asarray(T_K) - 273.15
    return np.where(t >= 0, 610.94 * np.exp(17.625 * t / (t + 243.04)),
                    611.21 * np.exp(22.587 * t / (t + 273.86)))


def humidity_ratio(T_dew_K, P_Pa):
    pv = p_sat(T_dew_K)
    return 0.622 * pv / (P_Pa - pv)


# ------------------------------- solar geometry ---------------------------------
def sun(hours_since_start: np.ndarray, start: str, lat_deg: np.ndarray, lon_deg: np.ndarray):
    """cos(zenith) and cos(incidence) on a N-S horizontal-axis tracking trough,
    evaluated at the middle of each accumulation hour. Shapes (H, n)."""
    import pandas as pd
    t = pd.Timestamp(start) + pd.to_timedelta(hours_since_start - 0.5, unit="h")
    doy = t.dayofyear.to_numpy()[:, None]
    utc = (t.hour + t.minute / 60).to_numpy()[:, None]
    B = np.radians((doy - 1) * 360.0 / 365.0)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                    - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B))       # min
    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0)))
    solar_time = utc + lon_deg[None, :] / 15.0 + eot / 60.0
    omega = np.radians(15.0 * (solar_time - 12.0))
    phi = np.radians(lat_deg)[None, :]
    cz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    ci = np.sqrt(np.clip(cz ** 2 + (np.cos(decl) * np.sin(omega)) ** 2, 0, 1))
    return cz, ci


def dni_from_fdir(fdir, cz):
    return np.where(cz > 0.087, np.minimum(fdir / np.maximum(cz, 0.087), 1100.0), 0.0)
