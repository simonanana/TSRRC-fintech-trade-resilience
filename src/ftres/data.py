"""Panel loading, variable construction, integrity checks and sample definitions.

The input is the processed dyadic panel ``panel_eastasia_v4.parquet``
(exporter x destination x year, 10 exporters x 44 destinations x 2015-2025).
See ``data/README.md`` for the data dictionary and sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (COVERAGE_THRESHOLD, EXCLUDED_EXPORTERS, INDEX_VARIANT, SPLICE_YEAR,
                     TARIFF_BIN_EDGES, TARIFF_BIN_LABELS, TREATED_EXPORTER, TREATMENT_MARKET)

REQUIRED_COLUMNS = (
    "iso3_o", "iso3_d", "year", "trade_usd", "tariff_pct", "ln_tariff",
    "trade_reported", "pair", "exp_yr", "imp_yr", "wgi_regquality_o",
)


def load_panel(path: Path) -> pd.DataFrame:
    """Read the processed panel and check that the required columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place panel_eastasia_v4.parquet in data/processed/, "
            "or run `python scripts/make_synthetic_panel.py` for a demo panel.")
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    return df


def prepare_panel(df: pd.DataFrame, variant: str = INDEX_VARIANT) -> pd.DataFrame:
    """Build every regressor used downstream from one set of definitions.

    * selects the fintech-index standardisation ``variant``;
    * builds the tariff x moderator interactions (H1-H3);
    * standardises WGI regulatory quality;
    * assigns tariff bins for the non-parametric dose-response;
    * builds cumulative US tariff pressure for the corridor designs.
    """
    d = df.copy()
    for stub in ("fintech_pre", "fin_pay_pre", "fin_credit_pre"):
        col = f"{stub}_{variant}"
        if col in d.columns:
            d[stub] = d[col]
    if "fintech_pre" not in d.columns:
        raise ValueError(f"no fintech_pre_{variant} column in panel")

    d["ln_trade"] = np.where(d.trade_usd > 0, np.log(d.trade_usd.where(d.trade_usd > 0)), np.nan)

    rq = pd.to_numeric(d["wgi_regquality_o"], errors="coerce")
    d["reg_qual_z"] = (rq - rq.mean()) / rq.std()

    d["tXfin"] = d.ln_tariff * d.fintech_pre
    if "fin_pay_pre" in d.columns:
        d["tXfin_pay"] = d.ln_tariff * d.fin_pay_pre
    if "fin_credit_pre" in d.columns:
        d["tXfin_cred"] = d.ln_tariff * d.fin_credit_pre
    d["tXinst"] = d.ln_tariff * d.reg_qual_z
    d["tXfinXinst"] = d.tXfin * d.reg_qual_z

    d["tau_bin"] = pd.cut(d.tariff_pct, list(TARIFF_BIN_EDGES), labels=list(TARIFF_BIN_LABELS))

    us = (d[d.iso3_d == TREATMENT_MARKET][["iso3_o", "year", "ln_tariff"]]
          .rename(columns={"ln_tariff": "lt_us"}).sort_values(["iso3_o", "year"]))
    us["us_shock"] = us.groupby("iso3_o")["lt_us"].diff().fillna(0.0)
    us["us_shock_cum"] = us.groupby("iso3_o")["us_shock"].cumsum()
    d = d.merge(us[["iso3_o", "year", "us_shock", "us_shock_cum"]],
                on=["iso3_o", "year"], how="left")
    d[["us_shock", "us_shock_cum"]] = d[["us_shock", "us_shock_cum"]].fillna(0.0)
    return d


# --------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------
def splice_check(d: pd.DataFrame, year: int = SPLICE_YEAR) -> pd.DataFrame:
    """Total exports either side of the BACI -> Comtrade source boundary.

    Flags ratios outside [0.67, 1.5]. The two sources differ in the treatment of
    re-exports and in mirror-flow reconciliation, so entrepot economies can show
    a level shift that is a change in what the series measures.
    """
    tot = (d[d.trade_reported == 1].groupby(["iso3_o", "year"])["trade_usd"]
           .sum().unstack() / 1e9)
    out = pd.DataFrame({f"exports_{year-1}_bn": tot[year - 1],
                        f"exports_{year}_bn": tot[year]})
    out["ratio"] = out.iloc[:, 1] / out.iloc[:, 0]
    out["verdict"] = np.where((out.ratio > 1.5) | (out.ratio < 0.67), "SPLICE BREAK", "ok")
    return out


def balance_check(d: pd.DataFrame, threshold: float = COVERAGE_THRESHOLD):
    """Destination coverage per exporter-year relative to the exporter's maximum.

    Returns (coverage table, set of balanced (exporter, year) pairs,
    list of failing (exporter, year) pairs).
    """
    rep = d[d.trade_reported == 1]
    cnt = rep.groupby(["iso3_o", "year"]).size()
    mx = cnt.groupby("iso3_o").max()
    cov = (cnt / cnt.index.get_level_values("iso3_o").map(mx)).rename("coverage")
    balanced = set(cov[cov >= threshold].index)
    failing = sorted(set(cnt.index) - balanced)
    return cov.unstack(), balanced, failing


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------
@dataclass
class Samples:
    """The estimation samples used throughout.

    all10       : every exporter, reported flows, unbalanced
    main        : nine exporters (Hong Kong excluded), balanced core  <- MAIN SAMPLE
    nine_unbal  : nine exporters, unbalanced
    eight       : Hong Kong and China excluded, balanced core
    """

    all10: pd.DataFrame
    main: pd.DataFrame
    nine_unbal: pd.DataFrame
    eight: pd.DataFrame

    @property
    def exporters_main(self) -> list[str]:
        return sorted(self.main.iso3_o.unique())

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"MAIN: 9 no HKG, balanced": self.main,
                "9 no HKG, unbalanced": self.nine_unbal,
                "all 10, unbalanced": self.all10,
                "8 no HKG no CHN, balanced": self.eight}


def build_samples(d: pd.DataFrame, balanced: set) -> Samples:
    all10 = d[d.trade_reported == 1].dropna(subset=["ln_tariff", "tXfin", "trade_usd"]).copy()
    all10["balanced"] = [(o, y) in balanced for o, y in zip(all10.iso3_o, all10.year)]
    excl = list(EXCLUDED_EXPORTERS)
    nine_unbal = all10[~all10.iso3_o.isin(excl)].copy()
    main = nine_unbal[nine_unbal.balanced].copy()
    eight = main[main.iso3_o != TREATED_EXPORTER].copy()
    return Samples(all10=all10, main=main, nine_unbal=nine_unbal, eight=eight)


def sample_summary(samples: Samples) -> pd.DataFrame:
    rows = []
    for name, s in samples.as_dict().items():
        rows.append({"sample": name, "n_obs": len(s), "n_exporters": s.iso3_o.nunique(),
                     "n_dyads": s.pair.nunique(), "n_zero_flows": int((s.trade_usd == 0).sum())})
    return pd.DataFrame(rows)
