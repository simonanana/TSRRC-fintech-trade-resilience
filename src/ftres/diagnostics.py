"""Identification diagnostics.

The paper's methodological claim is that a country-level moderator interacted
with a *targeted* trade shock is identified off very few units. This module
makes that claim measurable:

* :func:`variation_by_exporter`     where the tariff variation lives
* :func:`core_battery`              what survives deleting the index extremes
* :func:`effective_units`           N_eff, the inverse-Herfindahl count of identifying leverage
* :func:`elasticity_decomposition`  whose elasticity the headline number is
* :func:`turning_point`             where the fitted interaction implies a positive elasticity
* :func:`splice_level_shift`        a statistic for the source-splice discontinuity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FE_MAIN, TREATMENT_MARKET
from .estimation import coef, fit


# --------------------------------------------------------------------------
# Where the variation lives
# --------------------------------------------------------------------------
def variation_by_exporter(s: pd.DataFrame, market: str = TREATMENT_MARKET) -> pd.DataFrame:
    """Per-exporter summary of tariff movement in the treatment market."""
    rows = []
    for e in sorted(s.iso3_o.unique()):
        sub = s[s.iso3_o == e]
        us = sub[sub.iso3_d == market].sort_values("year")
        d_us = us.tariff_pct.diff()
        rows.append({
            "exporter": e,
            "within_dyad_sd": sub.groupby("pair")["ln_tariff"].std().mean(),
            "max_abs_d_tariff_pp": float(d_us.abs().max()) if d_us.notna().any() else np.nan,
            "n_years_dtariff_gt_1pp": int((d_us.abs() > 1).sum()),
            "tariff_2017": float(us.loc[us.year == 2017, "tariff_pct"].mean()),
            "tariff_2025": float(us.loc[us.year == 2025, "tariff_pct"].mean()),
            "fintech_index": float(sub.fintech_pre.iloc[0]),
        })
    out = pd.DataFrame(rows).sort_values("max_abs_d_tariff_pp", ascending=False)
    out["d_2017_2025_pp"] = out.tariff_2025 - out.tariff_2017
    return out.reset_index(drop=True)


def wave1_changes(s: pd.DataFrame, years=(2018, 2019), threshold_pp: float = 1.0,
                  market: str = TREATMENT_MARKET) -> pd.DataFrame:
    """Exporter-years in wave 1 with a treatment-market tariff change above threshold."""
    w = s[s.iso3_d == market].sort_values(["iso3_o", "year"]).copy()
    w["d_tariff_pp"] = w.groupby("iso3_o")["tariff_pct"].diff()
    keep = w.year.between(*years) & (w.d_tariff_pp.abs() > threshold_pp)
    return w.loc[keep, ["iso3_o", "year", "d_tariff_pp"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Stepwise deletion of the index extremes
# --------------------------------------------------------------------------
def core_battery(d: pd.DataFrame, label: str, fe: str = FE_MAIN) -> dict:
    """Tariff elasticity, H1 interaction and H3 triple interaction on one sample."""
    out = {"sample": label, "n_exporters": d.iso3_o.nunique(), "n_obs": len(d)}
    m0 = fit(f"trade_usd ~ ln_tariff | {fe}", d)
    out["b_tariff_alone"], _, out["t_tariff_alone"] = coef(m0, "ln_tariff")
    m1 = fit(f"trade_usd ~ ln_tariff + tXfin | {fe}", d)
    out["b1"], _, out["t1"] = coef(m1, "ln_tariff")
    out["b3"], _, out["t3"] = coef(m1, "tXfin")
    m3 = fit(f"trade_usd ~ ln_tariff + tXfin + tXinst + tXfinXinst | {fe}", d)
    out["triple"], _, out["t_triple"] = coef(m3, "tXfinXinst")
    return out


def stepwise_deletion(all10: pd.DataFrame) -> pd.DataFrame:
    """Delete Hong Kong, China, and both, from the unbalanced ten-exporter sample."""
    return pd.DataFrame([
        core_battery(all10, "all 10 exporters"),
        core_battery(all10[all10.iso3_o != "HKG"], "9: HKG excluded"),
        core_battery(all10[all10.iso3_o != "CHN"], "9: CHN excluded"),
        core_battery(all10[~all10.iso3_o.isin(["CHN", "HKG"])], "8: CHN+HKG excluded"),
    ])


# --------------------------------------------------------------------------
# N_eff
# --------------------------------------------------------------------------
def residualize(d: pd.DataFrame, col: str, absorb=("pair", "exp_yr", "imp_yr"),
                tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """Project `col` off several fixed-effect dimensions by alternating projections."""
    x = d[col].to_numpy(float).copy()
    for _ in range(max_iter):
        x0 = x.copy()
        for fe in absorb:
            if fe in d.columns:
                x = x - pd.Series(x, index=d.index).groupby(d[fe].values).transform("mean").to_numpy()
        if np.max(np.abs(x - x0)) < tol:
            break
    return x


def effective_units(d: pd.DataFrame, term: str = "tXfin",
                    absorb=("pair", "exp_yr", "imp_yr"), unit: str = "iso3_o") -> dict:
    r"""Effective number of identifying units, N_eff.

    Let :math:`\tilde x_{ijt}` be the interaction regressor after projecting out
    all fixed effects. Unit :math:`i`'s share of the identifying variance is

    .. math:: s_i = \sum_{j,t}\tilde x_{ijt}^2 \Big/ \sum_{i'}\sum_{j,t}\tilde x_{i'jt}^2,

    and :math:`N_{eff} = 1/\sum_i s_i^2`, the inverse Herfindahl index. It equals
    the number of units when leverage is spread evenly and falls toward one as a
    single unit supplies the variation. This is the unweighted version; weighting
    by PPML working weights gives the exact Poisson analogue.

    Returns a dict with ``N_eff``, per-unit ``shares`` and the nominal counts.
    """
    d = d.dropna(subset=[term])
    x = residualize(d, term, absorb)
    ss = pd.Series(x ** 2, index=d.index).groupby(d[unit].values).sum()
    shares = (ss / ss.sum()).sort_values(ascending=False)
    return {"N_eff": float(1.0 / (shares ** 2).sum()), "shares": shares,
            "n_units": int(d[unit].nunique()),
            "n_clusters": int(d["pair"].nunique()) if "pair" in d else None}


# --------------------------------------------------------------------------
# Elasticity decomposition and turning point
# --------------------------------------------------------------------------
def moderator_scale(main: pd.DataFrame, moderator: str = "fintech_pre") -> dict:
    """Mean and (population) standard deviation of the moderator across units.

    The index is a rank-based normal score over ten economies, so its standard
    deviation over the nine estimation-sample exporters is not one. Any
    "one-standard-deviation" statement must use this value.
    """
    per = main.drop_duplicates("iso3_o").set_index("iso3_o")[moderator].astype(float)
    return {"values": per.sort_values(ascending=False), "mean": float(per.mean()),
            "sd": float(per.std(ddof=0))}


def elasticity_decomposition(b1_interacted: float, b3: float, index_values: pd.Series,
                             b1_no_interaction: float) -> pd.DataFrame:
    """Implied elasticity beta1 + beta3 * F_i for each unit, plus the sample mean.

    When one unit supplies nearly all the identifying variance, the no-interaction
    PPML slope lands on that unit's implied elasticity rather than on the mean.
    """
    out = pd.DataFrame({"fintech_index": index_values})
    out["implied_elasticity"] = b1_interacted + b3 * out.fintech_index
    mean_row = pd.DataFrame({"fintech_index": [index_values.mean()],
                             "implied_elasticity": [b1_interacted + b3 * index_values.mean()]},
                            index=["sample mean"])
    ref = pd.DataFrame({"fintech_index": [np.nan], "implied_elasticity": [b1_no_interaction]},
                       index=["no-interaction estimate"])
    return pd.concat([out.sort_values("implied_elasticity"), mean_row, ref])


def turning_point(b1: float, b3: float) -> float:
    """Index value F* = -b1/b3 at which the fitted tariff elasticity changes sign."""
    return float(-b1 / b3) if b3 else float("nan")


# --------------------------------------------------------------------------
# Source-splice level shift
# --------------------------------------------------------------------------
def splice_level_shift(d: pd.DataFrame, boundary: int = 2024) -> pd.DataFrame:
    """Per-exporter level shift at the data-source boundary.

    Estimates ``ln(trade) ~ post + linear trend | dyad`` by OLS for each exporter.
    ``post`` is a deterministic function of the year, so it cannot be estimated
    alongside year fixed effects; a linear trend is used instead.
    """
    d = d[(d.trade_reported == 1) & (d.trade_usd > 0)].copy()
    d["ln_trade"] = np.log(d.trade_usd)
    d["post"] = (d.year.astype(int) >= boundary).astype(float)
    d["t_lin"] = d.year.astype(float) - float(d.year.min())
    rows = []
    for e in sorted(d.iso3_o.unique()):
        sub = d[d.iso3_o == e]
        if sub.post.nunique() < 2 or sub.pair.nunique() < 5:
            continue
        b, s, t = coef(fit("ln_trade ~ post + t_lin | pair", sub, poisson=False), "post")
        rows.append({"exporter": e, "delta_ln": b, "se": s, "t": t,
                     "ratio": float(np.exp(b)) if np.isfinite(b) else np.nan})
    return pd.DataFrame(rows).sort_values("delta_ln").reset_index(drop=True)
