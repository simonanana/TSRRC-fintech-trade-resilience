"""Specification curve (Simonsohn, Simmons and Nelson, 2020).

Estimates the H1 interaction over every combination of
5 index standardisations x 8 sample restrictions x 2 fixed-effect structures
= 80 specifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FE_MAIN, FE_WEAK, INDEX_VARIANTS
from .estimation import coef, fit

SAMPLE_RULES = {
    "MAIN (9, no HKG, balanced)": lambda x: x[(x.iso3_o != "HKG") & x.balanced],
    "9, no HKG, unbalanced": lambda x: x[x.iso3_o != "HKG"],
    "all 10": lambda x: x,
    "no CHN": lambda x: x[x.iso3_o != "CHN"],
    "no CHN, no HKG": lambda x: x[~x.iso3_o.isin(["CHN", "HKG"])],
    "no hand-coded tariffs": lambda x: x[(x.iso3_o != "HKG") & (x.get("policy_imputed", 0) == 0)],
    "exclude 2025": lambda x: x[(x.iso3_o != "HKG") & x.balanced & (x.year <= 2024)],
    "no mirror/manual trade": lambda x: x[(x.iso3_o != "HKG") & (x.get("trade_mirror", 0) == 0)
                                          & (x.get("trade_manual_total", 0) == 0)],
}
FE_STRUCTURES = {"pair+expyr+impyr": FE_MAIN, "pair+year": FE_WEAK}


def specification_curve(all10: pd.DataFrame, min_obs: int = 300) -> pd.DataFrame:
    variants = [v for v in INDEX_VARIANTS if f"fintech_pre_{v}" in all10.columns]
    rows = []
    for v in variants:
        for sname, rule in SAMPLE_RULES.items():
            for fname, fe in FE_STRUCTURES.items():
                sub = rule(all10).copy()
                sub["_tx"] = sub.ln_tariff * sub[f"fintech_pre_{v}"]
                sub = sub.dropna(subset=["ln_tariff", "_tx", "trade_usd"])
                if len(sub) < min_obs:
                    continue
                b, s, t = coef(fit(f"trade_usd ~ ln_tariff + _tx | {fe}", sub), "_tx")
                # a NaN coefficient means the interaction was dropped as collinear with
                # the tariff term: only one unit carries tariff variation in that sample
                rows.append({"variant": v, "sample": sname, "fe": fname, "n": len(sub),
                             "coef": b, "se": s, "t": t, "identified": bool(np.isfinite(b))})
    return (pd.DataFrame(rows).sort_values(["identified", "coef"], ascending=[False, True])
            .reset_index(drop=True))


def identified(spec: pd.DataFrame) -> pd.DataFrame:
    """Specifications in which the interaction was estimable, ordered by estimate."""
    return spec[spec.identified].sort_values("coef").reset_index(drop=True)


def summarise(spec: pd.DataFrame) -> dict:
    ok = identified(spec)
    pos = int(((ok.coef > 0) & (ok.t >= 1.96)).sum())
    neg = int(((ok.coef < 0) & (ok.t <= -1.96)).sum())
    return {"n_specs": len(ok), "not_identified": int((~spec.identified).sum()),
            "positive_significant": pos, "negative_significant": neg,
            "null": len(ok) - pos - neg, "median": float(ok.coef.median()),
            "p05": float(ok.coef.quantile(.05)), "p95": float(ok.coef.quantile(.95))}
