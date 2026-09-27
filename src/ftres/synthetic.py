"""Synthetic panel with the production schema.

Used by the test suite and by ``scripts/make_synthetic_panel.py`` so that the
pipeline can be run end to end without the processed data. The generating
process mimics the structural feature the paper is about: a large tariff shock
concentrated on one exporter (and on the entrepot that shares its tariff path).

Results obtained on this panel are NOT the paper's results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EXPORTERS_ALL, INDEX_VARIANTS

#: Rank-normal index values over ten economies, as reported in the paper (Table I).
INDEX = {"CHN": 1.664, "KOR": 1.048, "MYS": 0.682, "JPN": 0.390, "THA": 0.127,
         "IDN": -0.127, "SGP": -0.390, "VNM": -0.682, "PHL": -1.048, "HKG": -1.664}

US_PATH_CHN = {2015: 3.6, 2016: 3.6, 2017: 4.0, 2018: 14.2, 2019: 21.6, 2020: 19.3,
               2021: 19.3, 2022: 19.3, 2023: 19.3, 2024: 21.0, 2025: 51.0}


def make_panel(seed: int = 7, n_other_partners: int = 34, years=range(2015, 2026)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    asia_partners = [e for e in EXPORTERS_ALL if e != "HKG"]
    partners = [f"P{j:02d}" for j in range(n_other_partners)] + ["USA"] + asia_partners
    rq = dict(zip(EXPORTERS_ALL, rng.normal(0.8, 0.6, len(EXPORTERS_ALL))))
    PAY = {e: INDEX[e] + rng.normal(0, .5) for e in EXPORTERS_ALL}
    CREDIT = {e: INDEX[e] + rng.normal(0, .5) for e in EXPORTERS_ALL}
    VARIANT_NOISE = {v: {e: (0.0 if v == "hybrid" else rng.normal(0, .3)) for e in EXPORTERS_ALL}
                     for v in INDEX_VARIANTS}
    gdp_o = dict(zip(EXPORTERS_ALL, rng.normal(26.5, 1.2, len(EXPORTERS_ALL))))
    gdp_d = {p: rng.normal(26.0, 1.3) for p in partners}
    rows = []
    for o in EXPORTERS_ALL:
        for d in partners:
            if d == o:
                continue
            dist = rng.uniform(6.5, 9.5)
            dyad_fe = rng.normal(0, 0.8)
            base_tau = rng.uniform(0, 6)
            zero_dyad = rng.random() < 0.02
            for y in years:
                if d == "USA" and o in ("CHN", "HKG"):
                    tau = US_PATH_CHN[y] if (o == "CHN" or y >= 2020) else 4.0
                elif d == "USA":
                    tau = base_tau + (rng.uniform(8, 18) if y == 2025 else 0.0)
                else:
                    tau = base_tau
                tau = max(tau, 0.0)
                lt = np.log1p(tau / 100)
                mu = np.exp(0.75 * gdp_o[o] + 1.0 * gdp_d[d] - 1.1 * dist + dyad_fe
                            - 2.6 * lt + 0.03 * (y - 2015) - 26)
                trade = 0.0 if zero_dyad else float(rng.gamma(400.0, mu / 400.0) * 1e6)
                rec = {"iso3_o": o, "iso3_d": d, "year": y, "trade_usd": trade,
                       "tariff_pct": tau, "ln_tariff": lt,
                       "trade_reported": 1, "trade_mirror": 0, "trade_manual_total": 0,
                       "policy_imputed": int(d == "USA" and y >= 2020),
                       "wgi_regquality_o": rq[o], "ln_gdp_o": gdp_o[o], "ln_gdp_d": gdp_d[d],
                       "ln_dist": dist, "contig": int(rng.random() < .05),
                       "comlang": int(rng.random() < .1), "colony": 0}
                for v in INDEX_VARIANTS:
                    rec[f"fintech_pre_{v}"] = INDEX[o] + VARIANT_NOISE[v][o]
                    rec[f"fin_pay_pre_{v}"] = PAY[o]
                    rec[f"fin_credit_pre_{v}"] = CREDIT[o]
                rows.append(rec)
    df = pd.DataFrame(rows)
    # thin coverage in late years, as in the production panel
    thin = {("CHN", 2025): 3, ("THA", 2025): 2, ("VNM", 2024): 1, ("VNM", 2025): 3}
    for (o, y), keep in thin.items():
        idx = df[(df.iso3_o == o) & (df.year == y) & (df.iso3_d != "USA")].index
        df.loc[idx[keep:], "trade_reported"] = 0
    df["pair"] = df.iso3_o + "_" + df.iso3_d
    df["exp_yr"] = df.iso3_o + "_" + df.year.astype(str)
    df["imp_yr"] = df.iso3_d + "_" + df.year.astype(str)
    return df
