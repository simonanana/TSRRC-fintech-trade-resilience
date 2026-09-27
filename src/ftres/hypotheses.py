"""Gravity benchmarks, tariff dose-response and the hypothesis table.

Hypotheses
----------
H1 (resilience)              tariff x fintech,              beta3 > 0
H2 (channel heterogeneity)   tariff x payments / x credit,  payments > credit
H3 (institutional moderation) tariff x fintech x RQ,        > 0

Every hypothesis coefficient is reported with four inference methods and with
the largest exporter deleted. A result is claimed only if it survives the cluster
jackknife, randomization inference, and deletion of the largest exporter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FE_MAIN, TARIFF_BIN_LABELS
from .data import Samples
from .estimation import coef, fit

F_H1 = f"trade_usd ~ ln_tariff + tXfin | {FE_MAIN}"
F_H2 = f"trade_usd ~ ln_tariff + tXfin_pay + tXfin_cred | {FE_MAIN}"
F_H3 = f"trade_usd ~ ln_tariff + tXfin + tXinst + tXfinXinst | {FE_MAIN}"

HYPOTHESIS_TERMS = [
    ("baseline tariff effect", F_H1, "ln_tariff"),
    ("H1 resilience interaction", F_H1, "tXfin"),
    ("H2 payments channel", F_H2, "tXfin_pay"),
    ("H2 digital-credit channel", F_H2, "tXfin_cred"),
    ("institutions, direct", F_H3, "tXinst"),
    ("H3 triple interaction", F_H3, "tXfinXinst"),
]

GRAVITY = "ln_dist + contig + comlang + colony"


def gravity_ladder(main: pd.DataFrame) -> pd.DataFrame:
    """OLS and PPML, with and without fixed effects, as a sanity check."""
    pos = main[main.trade_usd > 0]
    specs = {
        "OLS, no FE": (f"ln_trade ~ ln_gdp_o + ln_gdp_d + {GRAVITY} + ln_tariff", pos, False),
        "PPML, no FE": (f"trade_usd ~ ln_gdp_o + ln_gdp_d + {GRAVITY} + ln_tariff", main, True),
        "OLS, 3-way FE": (f"ln_trade ~ ln_tariff | {FE_MAIN}", pos, False),
        "PPML, 3-way FE": (f"trade_usd ~ ln_tariff | {FE_MAIN}", main, True),
    }
    rows = []
    for name, (f, data, pois) in specs.items():
        m = fit(f, data, poisson=pois)
        for term in ("ln_tariff", "ln_dist", "ln_gdp_o", "ln_gdp_d", "contig", "comlang"):
            b, s, t = coef(m, term)
            if np.isfinite(b):
                rows.append({"spec": name, "term": term, "coef": b, "se": s, "t": t})
    return pd.DataFrame(rows)


def dose_response(d: pd.DataFrame, label: str) -> pd.DataFrame:
    """Non-parametric tariff dose-response relative to the 0-2% bin."""
    m = fit(f"trade_usd ~ C(tau_bin) | {FE_MAIN}", d)
    base = TARIFF_BIN_LABELS[0]
    rows = [{"sample": label, "bin": base, "coef": 0.0, "se": 0.0, "t": np.nan,
             "n": int((d.tau_bin.astype(str) == base).sum())}]
    if m is not None:
        for k in m.coef().index:
            if not str(k).startswith("C(tau_bin)"):
                continue
            b_label = str(k).split("T.")[1].rstrip("]")
            b, s, t = coef(m, k)
            rows.append({"sample": label, "bin": b_label, "coef": b, "se": s, "t": t,
                         "n": int((d.tau_bin.astype(str) == b_label).sum())})
    return pd.DataFrame(rows)


def sample_definition_effects(samples: Samples) -> pd.DataFrame:
    """How the two sample restrictions move the headline coefficients."""
    rows = []
    for name, s in samples.as_dict().items():
        e, _, te = coef(fit(f"trade_usd ~ ln_tariff | {FE_MAIN}", s), "ln_tariff")
        b, _, tb = coef(fit(F_H1, s), "tXfin")
        c, _, tc = coef(fit(F_H3, s), "tXfinXinst")
        rows.append({"sample": name, "n": len(s), "tariff_elasticity": e, "t": te,
                     "beta3": b, "t_b3": tb, "H3_triple": c, "t_H3": tc})
    return pd.DataFrame(rows)


def hypothesis_row(d: pd.DataFrame, formula: str, term: str, label: str) -> dict:
    """One coefficient with CRV1 (dyad), CRV1 (exporter), CRV3 jackknife, and deletions."""
    out = {"hypothesis": label, "term": term, "n": len(d)}
    out["coef"], out["se_pair"], out["t_pair"] = coef(fit(formula, d, vcov={"CRV1": "pair"}), term)
    for tag, vc in (("exp", {"CRV1": "iso3_o"}), ("jack", {"CRV3": "pair"})):
        _, s, t = coef(fit(formula, d, vcov=vc), term)
        out[f"se_{tag}"], out[f"t_{tag}"] = s, t
    top_fin = d.drop_duplicates("iso3_o").sort_values("fintech_pre").iloc[-1].iso3_o
    for tag, sub in (("drop_CHN", d[d.iso3_o != "CHN"]),
                     ("drop_top_fintech", d[d.iso3_o != top_fin])):
        b, _, t = coef(fit(formula, sub), term)
        out[f"coef_{tag}"], out[f"t_{tag}"] = b, t
    return out


def hypothesis_table(main: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([hypothesis_row(main, f, t, lab) for lab, f, t in HYPOTHESIS_TERMS])


def leave_one_out(main: pd.DataFrame, formula: str, term: str) -> pd.DataFrame:
    """Coefficient on the full sample and with each exporter deleted in turn."""
    recs = [("full", *coef(fit(formula, main), term))]
    for e in sorted(main.iso3_o.unique()):
        recs.append((f"-{e}", *coef(fit(formula, main[main.iso3_o != e]), term)))
    return pd.DataFrame(recs, columns=["sample", "coef", "se", "t"])
