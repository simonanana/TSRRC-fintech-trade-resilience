"""Thin wrappers around pyfixest.

All gravity specifications are estimated by Poisson pseudo-maximum likelihood
(PPML; Santos Silva and Tenreyro, 2006) with high-dimensional fixed effects
(Correia, Guimaraes and Zylkin, 2020). Standard errors default to CRV1 clustered
by dyad; the diagnostics modules request exporter-level clustering and the CRV3
cluster jackknife explicitly.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pyfixest.estimation import feols, fepois

from .config import FE_MAIN

log = logging.getLogger(__name__)


def fit(formula: str, data, vcov: Optional[dict] = None, poisson: bool = True):
    """Estimate one model; return None (and log) instead of raising.

    Parameters
    ----------
    formula : pyfixest formula, e.g. ``"trade_usd ~ ln_tariff | pair + exp_yr + imp_yr"``
    data    : pandas DataFrame
    vcov    : pyfixest vcov spec; defaults to ``{"CRV1": "pair"}``
    poisson : True for PPML (fepois), False for OLS (feols)
    """
    try:
        est = fepois if poisson else feols
        return est(formula, data=data, vcov=vcov or {"CRV1": "pair"})
    except Exception as exc:  # noqa: BLE001 -- a failed fit must not stop a sweep
        log.warning("fit failed: %s ... %s: %s", formula[:70], type(exc).__name__, exc)
        return None


def coef(m, term: str) -> tuple[float, float, float]:
    """Return (coefficient, standard error, t-statistic) for one term, or NaNs."""
    if m is None or term not in m.coef().index:
        return (np.nan, np.nan, np.nan)
    b, s = float(m.coef()[term]), float(m.se()[term])
    return (b, s, b / s if s else np.nan)


def tariff_elasticity(d, fe: str = FE_MAIN) -> tuple[float, float, float]:
    """PPML tariff elasticity with the main fixed-effect structure."""
    return coef(fit(f"trade_usd ~ ln_tariff | {fe}", d), "ln_tariff")
