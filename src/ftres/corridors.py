"""Dyadic designs: payment corridors (Route A) and bilateral complementarity (Route B).

Route A replaces the country attribute with a bilateral, time-varying treatment:
a payment corridor that opens in a given year is absorbed by neither
exporter-year nor importer-year fixed effects.

IMPORTANT: every launch year in ``CORRIDOR_START`` is hand-coded and must be
verified against the relevant central-bank release before being cited. The
matrix pools retail fast-payment / QR linkages with wholesale CBDC platforms,
which serve different settlement layers; splitting the two is recommended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FE_MAIN, TREATMENT_MARKET
from .diagnostics import effective_units
from .estimation import coef, fit

#: First full year of each bilateral payment linkage. UNVERIFIED.
CORRIDOR_START = {
    ("SGP", "THA"): 2021,  # PayNow-PromptPay retail real-time link
    ("SGP", "IND"): 2023,  # PayNow-UPI
    ("SGP", "MYS"): 2023,  # PayNow-DuitNow
    ("SGP", "IDN"): 2023,  # QR linkage
    ("SGP", "PHL"): 2023,
    ("THA", "MYS"): 2021,  # PromptPay-DuitNow QR
    ("THA", "VNM"): 2021,
    ("THA", "IDN"): 2022,
    ("THA", "HKG"): 2024,  # wholesale CBDC platform
    ("IDN", "MYS"): 2022,  # QRIS-DuitNow
    ("CHN", "HKG"): 2021,  # e-CNY cross-boundary pilot
    ("CHN", "THA"): 2024,  # wholesale CBDC platform
    ("CHN", "ARE"): 2024,  # wholesale CBDC platform
}
CORRIDOR_VERIFIED = False
WHOLESALE_PAIRS = {("THA", "HKG"), ("CHN", "THA"), ("CHN", "ARE")}

ASIAN_DESTINATIONS = ["CHN", "HKG", "JPN", "KOR", "SGP", "IDN", "MYS", "PHL",
                      "THA", "VNM", "IND", "KHM", "BRN", "LAO", "MMR"]


def _bidirectional(pairs: dict) -> dict:
    out = dict(pairs)
    out.update({(b, a): y for (a, b), y in pairs.items()})
    return out


def add_corridor(d: pd.DataFrame, start: dict = CORRIDOR_START) -> pd.DataFrame:
    """Attach corridor status, event time and the reallocation interaction."""
    corridor = _bidirectional(start)
    wholesale = set(WHOLESALE_PAIRS) | {(b, a) for a, b in WHOLESALE_PAIRS}
    d = d.copy()
    keys = list(zip(d.iso3_o, d.iso3_d))
    st = pd.Series([corridor.get(k, np.nan) for k in keys], index=d.index)
    d["corr_start"] = st
    d["corridor"] = (st.notna() & (d.year.astype(float) >= st)).astype(int)
    d["corridor_ever"] = st.notna().astype(int)
    d["corr_rel"] = np.where(st.notna(), d.year.astype(float) - st, np.nan)
    is_ws = np.array([k in wholesale for k in keys])
    d["corridor_wholesale"] = d.corridor * is_ws
    d["corridor_retail"] = d.corridor * ~is_ws
    d["shockXcorr"] = d.us_shock_cum * d.corridor
    return d


def corridor_effects(c: pd.DataFrame) -> pd.DataFrame:
    """Level effect on all destinations and the reallocation test on third markets."""
    rows = []
    b, s, t = coef(fit(f"trade_usd ~ corridor | {FE_MAIN}", c), "corridor")
    rows.append({"spec": "level effect, all destinations", "term": "corridor",
                 "coef": b, "se": s, "t": t, "n": len(c), "dyads": c.pair.nunique()})
    nu = c[c.iso3_d != TREATMENT_MARKET]
    m = fit(f"trade_usd ~ corridor + shockXcorr | {FE_MAIN}", nu)
    for term in ("corridor", "shockXcorr"):
        b, s, t = coef(m, term)
        rows.append({"spec": "reallocation, third markets", "term": term,
                     "coef": b, "se": s, "t": t, "n": len(nu), "dyads": nu.pair.nunique()})
    out = pd.DataFrame(rows)
    out["ci_lo"], out["ci_hi"] = out.coef - 1.96 * out.se, out.coef + 1.96 * out.se
    return out


def reallocation_robustness(c: pd.DataFrame) -> pd.DataFrame:
    nu = c[c.iso3_d != TREATMENT_MARKET]
    subsets = {"full": nu, "drop CHN": nu[nu.iso3_o != "CHN"],
               "exclude 2025": nu[nu.year <= 2024],
               "non-Asian destinations": nu[~nu.iso3_d.isin(ASIAN_DESTINATIONS)]}
    rows = []
    for lab, sub in subsets.items():
        if sub.corridor.nunique() < 2 or len(sub) < 300:
            continue
        b, s, t = coef(fit(f"trade_usd ~ corridor + shockXcorr | {FE_MAIN}", sub), "shockXcorr")
        rows.append({"sample": lab, "coef": b, "se": s, "t": t, "n": len(sub)})
    return pd.DataFrame(rows)


def event_study(c: pd.DataFrame, window=(-3, 3), base: int = -1) -> pd.DataFrame:
    """Two-way fixed-effects event study around corridor launch (third markets).

    Adoption is staggered across dyads, so heterogeneity-robust estimators
    (Callaway-Sant'Anna; Sun-Abraham) are the recommended next step.
    """
    ev = c[c.iso3_d != TREATMENT_MARKET].copy()
    ks = [k for k in range(window[0], window[1] + 1) if k != base]
    names = {k: f"c_{'m' if k < 0 else 'p'}{abs(k)}" for k in ks}
    for k, nm in names.items():
        ev[nm] = (ev.corr_rel == k).astype(float)
    m = fit(f"trade_usd ~ {' + '.join(names.values())} | {FE_MAIN}", ev)
    rows = [{"k": base, "coef": 0.0, "se": 0.0, "t": np.nan}]
    for k, nm in names.items():
        b, s, t = coef(m, nm)
        rows.append({"k": k, "coef": b, "se": s, "t": t})
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def bilateral_complementarity(main: pd.DataFrame, partner_findex: pd.DataFrame,
                              pre_year: int = 2025) -> tuple[pd.DataFrame, dict]:
    """Route B: dyadic moderators from exporter and destination payment adoption.

    ``partner_findex`` needs columns iso3, year, findex_digital_pay. The
    destination index is the latest observation strictly before ``pre_year``,
    z-scored across economies.

    Note: a dyadic moderator varies across dyads, but the *tariff* still varies
    almost only for the treated exporter, so the interaction's identifying
    leverage is set by the tariff side. N_eff is returned so this can be checked
    rather than assumed.
    """
    p = partner_findex.copy()
    p["iso3"] = p.iso3.astype(str).str.upper().str.strip()
    p = p[pd.to_numeric(p.year, errors="coerce") < pre_year].dropna(subset=["findex_digital_pay"])
    pre = p.sort_values("year").groupby("iso3").tail(1)
    z = (pre.findex_digital_pay - pre.findex_digital_pay.mean()) / pre.findex_digital_pay.std()
    pf = pd.DataFrame({"iso3_d": pre.iso3.values, "fintech_d": z.values})

    b = main.merge(pf, on="iso3_d", how="left")
    b["tXmin"] = b.ln_tariff * np.minimum(b.fintech_pre, b.fintech_d)
    b["tXprod"] = b.ln_tariff * b.fintech_pre * b.fintech_d
    b = b.dropna(subset=["tXmin"])
    rows = []
    for term, lab in (("tXmin", "weakest link: min(F_o, F_d)"),
                      ("tXprod", "complementarity: F_o x F_d")):
        bb, s, t = coef(fit(f"trade_usd ~ ln_tariff + {term} | {FE_MAIN}", b), term)
        rows.append({"moderator": lab, "coef": bb, "se": s, "t": t,
                     "n": len(b), "dyads": b.pair.nunique()})
    neff = effective_units(b, term="tXmin")
    return pd.DataFrame(rows), neff
