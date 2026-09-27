"""Synthetic control for the single treated exporter of wave 1.

Wave 1 raised the treatment-market tariff by more than one percentage point for
exactly one exporter, so no cross-sectional interaction can be estimated from it.
A synthetic control (Abadie, Diamond and Hainmueller, 2010) is the transparent
single-unit alternative.

The treated unit's exports are an order of magnitude larger than any donor's, so
no convex combination can reproduce the *level*. Each series is demeaned by its
own pre-treatment mean before weights are chosen (a unit intercept), so the
weights match the *path*. Caveats, stated in the paper: the pre-period has only
three annual observations, so a tight pre-fit is close to mechanical; with eight
donors the smallest attainable placebo p-value is 1/9; and synthetic
difference-in-differences (Arkhangelsky et al., 2021) is the estimator designed
for level mismatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import TREATED_EXPORTER, TREATMENT_MARKET


def synthetic_control(d: pd.DataFrame, treated: str = TREATED_EXPORTER,
                      dest: str = TREATMENT_MARKET, pre_end: int = 2017,
                      donors=None, outcome: str = "ln_trade"):
    """Return path, donor weights and pre-period RMSPE, or None if infeasible."""
    p = (d[(d.iso3_d == dest) & (d.trade_reported == 1)]
         .pivot_table(index="year", columns="iso3_o", values=outcome))
    if treated not in p.columns:
        return None
    donors = [c for c in (donors or p.columns) if c in p.columns and c != treated]
    p = p[[treated] + donors].dropna()
    if p.empty or p.index.min() > pre_end:
        return None
    pre_mean = p.loc[:pre_end].mean()
    q = p - pre_mean
    pre = q.loc[:pre_end]
    if len(pre) < 3:
        return None
    y, X = pre[treated].to_numpy(), pre[donors].to_numpy()
    n = len(donors)
    res = minimize(lambda w: float(((y - X @ w) ** 2).sum()), np.full(n, 1 / n),
                   method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 2000, "ftol": 1e-12})
    w = res.x
    path = pd.DataFrame({"actual": q[treated],
                         "synthetic": pd.Series(q[donors].to_numpy() @ w, index=q.index)})
    path["gap"] = path.actual - path.synthetic
    path["shortfall_pct"] = 100 * (1 - np.exp(path.gap))
    return {"path": path, "weights": pd.Series(w, index=donors).sort_values(ascending=False),
            "rmspe_pre": float(np.sqrt((path.loc[:pre_end, "gap"] ** 2).mean()))}


def in_space_placebos(d: pd.DataFrame, sc: dict, candidates, pool, pre_end: int = 2017,
                      fit_ratio: float = 5.0) -> dict:
    """Re-run the routine treating each donor in `candidates` as treated, drawing
    its donors from `pool`; keep placebos with a pre-fit within `fit_ratio` of the
    treated unit's."""
    out = {}
    for e in candidates:
        r = synthetic_control(d, treated=e, pre_end=pre_end,
                              donors=[x for x in pool if x != e])
        if r and r["rmspe_pre"] < fit_ratio * sc["rmspe_pre"]:
            out[e] = r["path"]["gap"]
    return out
