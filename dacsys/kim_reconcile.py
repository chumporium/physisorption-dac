"""V3: reproduce Kim et al. (EES 2025) LNG-DAC with 13X inside our plant model, then switch
their assumptions to ours one at a time.

python -m dacsys.kim_reconcile -> results_dac/study5/kim_waterfall.csv

Kim et al. (ESI Tables S5, S8, S10): adsorption 195 K, CO2 sweep desorption at 473 K,
13X at 0.85 $/kg (as in this study), grid electricity 30 $/MWh, natural-gas heat 3.5 $/GJ (85 % efficiency),
8 % interest over 15 years, continuous operation, pressure drop 100 Pa, capture efficiency 0.6,
LNG cold free. Reported LCOC: 68.2 $/t (13X), 171.3 $/t (amine, same energy prices).
Site: Madrid (mean 15 C, the "global average" 14-15 C of the paper).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import climate_sites as cs
from . import plant
from .levers import Case, Prm, with_

OUT = Path(__file__).resolve().parent.parent / "results_dac" / "study5"
SITE = "Madrid"


def main():
    w = cs.load(SITE)
    # Kim et al. basis (ESI Tables S5-S10): their cost functions for heat exchangers, fans and vacuum pumps (x 3.06),
    # sorbent and contactor x 3, no fixed O&M, one cycle per hour, 15 years, capture 0.6, 100 Pa, fan efficiency 0.614
    kim = with_(Case(site=SITE), drying="none", cold="lng", sorbent="13X", regen="sweep200",
                energy_mode="grid", lifetime=15, eta_cap=0.6, dP_contactor=100.0, dP_coil=0.0,
                dP_recup=0.0, eta_fan=0.614, max_cycles_per_h=1.0, om_proc=0.0,
                hx_cost_model="kim", fanvac_cost_model="kim")
    steps = [
        ("K0 assumptions of Kim et al.", kim, [195.0]),
        ("plant life 15 -> 25 yr", with_(kim, lifetime=25), [195.0]),
        ("capture fraction 0.6 -> 0.85; pressure drop and fan of this model",
         with_(kim, lifetime=25, eta_cap=0.85, dP_contactor=300.0, dP_coil=100.0, dP_recup=150.0, eta_fan=0.65),
         [195.0]),
        ("cycle rate 1/h -> limited by heat transfer", None, [195.0]),
        ("heat-exchanger cost Kim -> Weiland et al. (per UA, installed)", None, [195.0]),
        ("process O&M 0 -> 3 %", None, [195.0]),
        ("adsorption temperature optimised (195-255 K)", None, None),
        ("regeneration sweep 200 C -> vacuum 100 C", None, None),
        ("fan and vacuum pump costs Kim -> NETL", None, None),
        ("grid power and gas -> off-grid solar (PTC, PV, storage)", None, None),
    ]
    rows, c = [], kim
    for label, case, T in steps:
        if case is None:
            if "heat-exchanger" in label:
                c = with_(c, hx_cost_model="netl")      # NETL correlation per UA, installed (this study)
            elif "cycle rate" in label:
                c = with_(c, max_cycles_per_h=Prm().max_cycles_per_h)
            elif "O&M" in label:
                c = with_(c, om_proc=Prm().om_proc)
            elif "NETL" in label:
                c = with_(c, fanvac_cost_model="netl")
            elif "regeneration" in label:
                c = with_(c, regen="vac100")
            elif "off-grid" in label:
                c = with_(c, energy_mode="offgrid")
        else:
            c = case
        r = plant.evaluate(c, w, T_list=T)
        a = plant.amine(w, p=c.prm)
        rows.append(dict(step=label, lcoc=r.get("lcoc"), T_design=r.get("T_design"), hours=r.get("hours"),
                         wcap_mmolg=r.get("wcap_mmolg"), heat_GJ=r.get("heat_GJ"), elec_GJ=r.get("elec_GJ"),
                         capex_process=r.get("capex_process"), amine_base=a["amine_lcoc_base"],
                         energy_mode=c.prm.energy_mode))
    # the architectures of this study, on the final (our) basis
    for label, dryers, cold in (("no LNG: best dryer and heat pump", ("none", "cond", "cond_silica", "freeze"), "vcr"),
                                ("no LNG: B0 (condensation, silica gel, solar GAX)", ("cond_silica",), "gax")):
        rs = [plant.evaluate(with_(Case(site=SITE), drying=d, cold=cold), w) for d in dryers]
        r = min((x for x in rs if x.get("status") == "ok"), key=lambda x: x["lcoc"])
        rows.append(dict(step=label, lcoc=r.get("lcoc"), T_design=r.get("T_design"), hours=r.get("hours"),
                         wcap_mmolg=r.get("wcap_mmolg"), heat_GJ=r.get("heat_GJ"), elec_GJ=r.get("elec_GJ"),
                         capex_process=r.get("capex_process"), amine_base=plant.amine(w)["amine_lcoc_base"],
                         energy_mode="offgrid"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kim_waterfall.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
