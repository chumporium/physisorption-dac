"""Full database at all 21 sites (study5 group db3): does any MOF outside the 75 stage-2 MOFs beat the stage-2 result?

python -m dacsys.db3_check
"""
from __future__ import annotations

import pandas as pd

from . import fig_paper1 as F
from .study5 import DB3_ARCH


def full_pairs():
    """Lowest relative cost of every prescreened MOF at every site over the no-LNG architectures of db3
    (from DB3, PR1, PR2 and DB1 rows)."""
    c = pd.read_csv(F.RES / "cases.csv", low_memory=False)
    a = pd.read_csv(F.RES / "amine.csv").set_index("site").amine_lcoc_base
    c = c[c.lever.isin(["DB3", "PR1", "PR2", "DB1"]) & (c.status == "ok")]
    c = c[[(d, k) in DB3_ARCH for d, k in zip(c.drying, c.cold)]].copy()
    c["ratio"] = c.lcoc / c.site.map(a)
    return c.loc[c.groupby(["sorbent", "site"]).lcoc.idxmin()]


def main():
    b = full_pairs()
    d, _ = F.load()
    s2 = F.db_best(d)
    stage2 = set(F.db(d).sorbent.unique())
    best = b.loc[b.groupby("site").lcoc.idxmin()].set_index("site")
    t = pd.DataFrame({"stage2_best": s2.ratio, "stage2_mof": s2.sorbent, "full_best": best.ratio,
                      "full_mof": best.sorbent, "full_cfg": best.drying + "/" + best.cold})
    t["new_mof_better"] = (t.full_best < t.stage2_best - 1e-9) & ~t.full_mof.isin(stage2)
    print(t.round(3).to_string())
    print("MOFs:", b.sorbent.nunique(), "pairs:", len(b), "below 1:", int((b.ratio < 1).sum()),
          "min:", round(b.ratio.min(), 3), "median:", round(b.ratio.median(), 2))
    print("sites where a MOF outside stage 2 is better:", int(t.new_mof_better.sum()))
    per = b.pivot_table(index="sorbent", columns="site", values="ratio")
    print("MOFs feasible at all 21 sites:", int(per.notna().all(axis=1).sum()))


if __name__ == "__main__":
    main()
