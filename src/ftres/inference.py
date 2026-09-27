"""Randomization inference and minimum-detectable-effect calculations.

With nine exporters, cluster-robust asymptotics are unreliable. Randomization
inference (Young, 2019) permutes the moderator across exporters, rebuilds the
interaction and re-estimates, giving a finite-sample null distribution that does
not rely on the number of clusters growing. Its standard deviation is then the
input to a minimum-detectable-effect (MDE) calculation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import ALPHA, FE_MAIN, N_RI_DEFAULT, POWER, RI_SEED, TARGET_OFFSETS
from .estimation import coef, fit


def ri_interaction(d: pd.DataFrame, moderator: str = "fintech_pre", term: str = "tXfin",
                   n: int = N_RI_DEFAULT, seed: int = RI_SEED, fe: str = FE_MAIN,
                   progress: bool = False) -> dict:
    """Permutation null for the tariff x moderator interaction.

    The moderator vector is permuted across exporters (holding tariff paths fixed),
    the interaction is rebuilt, and the model is re-estimated.
    """
    base = "trade_usd ~ ln_tariff + {t} | " + fe
    rng = np.random.default_rng(seed)
    b_obs, _, _ = coef(fit(base.format(t=term), d), term)
    per = d.drop_duplicates("iso3_o").set_index("iso3_o")[moderator]
    units, vals = list(per.index), per.to_numpy(float)
    w, draws = d.copy(), []
    for k in range(n):
        w["_perm"] = w.ln_tariff * w.iso3_o.map(dict(zip(units, rng.permutation(vals))))
        b, _, _ = coef(fit(base.format(t="_perm"), w), "_perm")
        if np.isfinite(b):
            draws.append(b)
        if progress and (k + 1) % max(1, n // 10) == 0:
            print(f"    RI {k + 1}/{n}")
    draws = np.asarray(draws)
    p = float((np.abs(draws) >= abs(b_obs)).mean()) if draws.size else np.nan
    return {"b_obs": b_obs, "draws": draws, "p": p,
            "sd": float(draws.std(ddof=1)) if draws.size > 1 else np.nan,
            "n_draws": int(draws.size), "mc_se": monte_carlo_se(p, draws.size)}


def ri_triple(d: pd.DataFrame, n: int = N_RI_DEFAULT, seed: int = 11, fe: str = FE_MAIN) -> dict:
    """Permutation null for the H3 triple interaction.

    The (fintech, regulatory-quality) pair is permuted jointly across exporters so
    that the joint distribution of the two moderators is preserved.
    """
    formula = f"trade_usd ~ ln_tariff + tXfin + tXinst + tXfinXinst | {fe}"
    rng = np.random.default_rng(seed)
    b_obs, _, _ = coef(fit(formula, d), "tXfinXinst")
    per = d.drop_duplicates("iso3_o").set_index("iso3_o")[["fintech_pre", "reg_qual_z"]]
    units = list(per.index)
    fv, rv = per.fintech_pre.to_numpy(), per.reg_qual_z.to_numpy()
    w, draws = d.copy(), []
    for _ in range(n):
        pm = rng.permutation(len(units))
        f = w.iso3_o.map(dict(zip(units, fv[pm])))
        r = w.iso3_o.map(dict(zip(units, rv[pm])))
        w["_tf"], w["_ti"] = w.ln_tariff * f, w.ln_tariff * r
        w["_t3"] = w._tf * r
        b, _, _ = coef(fit(f"trade_usd ~ ln_tariff + _tf + _ti + _t3 | {fe}", w), "_t3")
        if np.isfinite(b):
            draws.append(b)
    draws = np.asarray(draws)
    p = float((np.abs(draws) >= abs(b_obs)).mean()) if draws.size else np.nan
    return {"b_obs": b_obs, "draws": draws, "p": p,
            "sd": float(draws.std(ddof=1)) if draws.size > 1 else np.nan,
            "n_draws": int(draws.size), "mc_se": monte_carlo_se(p, draws.size)}


def monte_carlo_se(p: float, n_draws: int) -> float:
    """Monte Carlo standard error of a permutation p-value."""
    return float(np.sqrt(p * (1 - p) / n_draws)) if n_draws else float("nan")


def mde(null_sd: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable effect: (z_{1-alpha/2} + z_{power}) * sd."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * null_sd)


def power_table(null_sd: float, beta1_abs: float, n_units: int, sd_moderator: float,
                offsets=TARGET_OFFSETS) -> pd.DataFrame:
    r"""Translate the resilience hypothesis into a required design size.

    The hypothesis states that a one-standard-deviation increase in the moderator
    offsets a share :math:`\pi` of the tariff effect, so

    .. math:: \beta_3^{target} = \pi\,|\beta_1| / \mathrm{sd}(F).

    The required multiple of effective identifying units is
    :math:`(\mathrm{MDE}/\beta_3^{target})^2`. ``n_units`` must be the unit count
    of the sample that produced ``beta1_abs``; ``sd_moderator`` must be the
    moderator's standard deviation in that sample.
    """
    m = mde(null_sd)
    rows = []
    for pi in offsets:
        tgt = pi * beta1_abs / sd_moderator
        ratio = m / tgt
        rows.append({"target_offset": pi, "beta3_target": tgt, "MDE": m,
                     "MDE_over_target": ratio, "units_multiple": ratio ** 2,
                     "units_needed": ratio ** 2 * n_units})
    return pd.DataFrame(rows)
