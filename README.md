# Physisorption-based direct air capture: LNG cold, renewable cooling and adsorbent targets

Model code, input data and results for the manuscript

> E. Munandar, Nasruddin. Physisorption-based direct air capture: LNG cold, renewable cooling and adsorbent targets.

The model simulates an off-grid direct air capture (DAC) plant hour by hour and calculates the levelised cost of
capture. The plant uses a physisorbent (zeolite 13X, a metal-organic framework of the PrISMa database or a model
adsorbent), an optional air dryer, and a cold source (solar GAX absorption chiller, heat pump, LNG cold or none).
Every case is compared with an amine DAC plant that uses the same energy supply at the same site.

## Contents

| Path | Content |
|---|---|
| `dacsys/plant.py` | Plant model: air treatment, adsorption, refrigeration, energy supply, cost |
| `dacsys/levers.py` | Parameters and case definition |
| `dacsys/tvsa.py` | Temperature-vacuum swing adsorption cycle (equilibrium model with IAST) |
| `dacsys/sorbents.py`, `dacsys/prisma.py` | Adsorbent models and isotherm fits |
| `dacsys/vcr.py`, `dacsys/ars_map.py` | Heat pump and GAX chiller performance maps |
| `dacsys/energy.py`, `dacsys/weather.py`, `dacsys/climate_sites.py` | Solar supply, psychrometrics, sites and weather data |
| `dacsys/study5.py` | Main case study (all adsorbents, dryers, cold sources and sites) |
| `dacsys/mc5.py` | Monte Carlo analysis |
| `dacsys/extra_runs.py`, `dacsys/extra_analysis.py` | Robustness runs (heat-exchanger cost, weather year, LNG terminals and others) |
| `dacsys/material_sens.py`, `dacsys/param_optimum.py` | Sensitivity and optimum of the adsorbent properties |
| `dacsys/decomp.py`, `dacsys/decomp_khw.py` | Shapley analysis of the adsorbent properties |
| `dacsys/lng_scale.py` | Scale of DAC with LNG cold |
| `dacsys/kim_reconcile.py` | Stepwise comparison with Kim et al. (2025) |
| `dacsys/validate_prisma.py` | Comparison of the cycle model with the PrISMa results |
| `dacsys/fig_paper1.py`, `dacsys/paper1_extra.py`, `dacsys/analysis_paper1.py`, `dacsys/database_wide.py`, `dacsys/materials_why.py` | Figures and tables |
| `lh2sys/` | Solar GAX absorption system model (chiller model and its validation) |
| `data/openmeteo_era5/` | Hourly ERA5 weather of the 21 sites for 2018 |
| `data/openmeteo_era5_extra/` | Weather of other years and of LNG terminals |
| `data/prisma/` | PrISMa material properties and fitted isotherm parameters |
| `data/naturalearth/` | Land outlines for the map |
| `tests/` | Tests |

The results of all runs are attached to the release as `results_study5.zip`. Unpack it to `results_dac/study5/` to
rebuild the figures and tables without running the model.

## Installation

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

## Use

Run the tests:

```bash
python -m pytest tests
```

Run the case study (the number of worker processes can be changed):

```bash
python -m dacsys.study5 --workers 8 --only matrix,rq5,sens,prisma1,prisma2,prismaL,db1,db2,db3,rq5cp,water
python -m dacsys.mc5 --n 200 --workers 8
python -m dacsys.extra_runs --workers 8
python -m dacsys.material_sens --workers 8
python -m dacsys.param_optimum --workers 8
python -m dacsys.decomp --workers 8 --sites all
python -m dacsys.lng_scale --workers 8
python -m dacsys.kim_reconcile
```

The complete study takes several days on a workstation. Finished cases are stored and skipped when a run is repeated.

Rebuild the figures and tables:

```bash
python -m dacsys.fig_paper1
python -m dacsys.paper1_extra
```

## Data sources

- Weather: ERA5 reanalysis (Hersbach et al., 2020), retrieved through the Open-Meteo archive.
- Adsorbent properties: PrISMa platform (Charalambous et al., 2024), Zenodo record 11244258.
- Isotherm of zeolite 13X: fitted to the data of Kim et al. (2025).
- Energy demand of the amine benchmark: fitted to the data of Wenzel et al. (2026).
- Land outlines: Natural Earth.

## Licence

The code is released under the MIT License (see `LICENSE`). The results in `results_study5.zip` are released under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0).

The input data in `data/` come from third parties and keep their own terms:

- `data/prisma/Material_Properties` and `data/prisma/Flags`: PrISMa data set, CC BY 4.0
  (https://doi.org/10.5281/zenodo.11244258).
- `data/openmeteo_era5` and `data/openmeteo_era5_extra`: weather data by Open-Meteo.com, CC BY 4.0, based on the ERA5
  reanalysis of the Copernicus Climate Change Service.
- `data/naturalearth`: Natural Earth, public domain.

## Contact

Nasruddin, Department of Mechanical Engineering, Faculty of Engineering, Universitas Indonesia (nasruddin@eng.ui.ac.id)
