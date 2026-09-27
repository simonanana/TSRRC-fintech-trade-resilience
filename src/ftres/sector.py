"""Sector-level exposure design (Route C) and its data loaders.

This is the design that clears the power requirement: tariff changes vary
*within* exporter-year across HS sectors, so identification no longer rests on a
comparison between the targeted economy and the others.

Required inputs (not shipped; see data/README.md):
  * CEPII BACI yearly HS6 files plus the BACI country-code table, in data/raw/baci/
  * sector tariffs, either a WITS TRAINS export at HS2, or effective rates built
    from USITC DataWeb with :func:`effective_tariff_from_dataweb`

Specification (PPML):
    X_ijst = exp[b1 ln(1+tau_ijst) + b3 ln(1+tau_ijst) x F_i
                 + a_ijs + d_it + t_jt + p_st] e_ijst
Cluster at exporter-sector as well as dyad-sector: the tariff shock is common
within exporter-year, so dyad-sector clustering alone overstates precision.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .estimation import coef, fit


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def baci_country_codes(baci_dir: Path) -> dict[int, str]:
    """Map BACI numeric country codes to ISO3 (column names vary by release)."""
    files = sorted(Path(baci_dir).glob("*ountr*.csv"))
    if not files:
        raise FileNotFoundError(f"no BACI country-code file in {baci_dir}")
    cc = pd.read_csv(files[0])
    low = {c.lower(): c for c in cc.columns}
    code = next(low[k] for k in ("country_code", "code", "i") if k in low)
    iso = next(low[k] for k in ("country_iso3", "iso_3digit_alpha", "iso3") if k in low)
    cc = cc[[code, iso]].dropna()
    return dict(zip(cc[code].astype(int), cc[iso].astype(str).str.upper()))


def sector_trade_from_baci(baci_dir: Path, exporters=None, partners=None,
                           years=(2015, 2025), hs_digits: int = 2,
                           chunksize: int = 2_000_000) -> pd.DataFrame | None:
    """Aggregate BACI HS6 flows to HS-`hs_digits` x dyad x year, in ISO3 codes.

    Reads in chunks and applies the exporter/partner filters while reading.
    """
    baci_dir = Path(baci_dir)
    files = sorted(baci_dir.glob("BACI*Y*.csv"))
    if not files:
        return None
    code2iso = baci_country_codes(baci_dir)
    exp_set, par_set = set(exporters or []), set(partners or [])
    parts = []
    for f in files:
        m = re.search(r"Y(\d{4})", f.name)
        if not m or not years[0] <= int(m.group(1)) <= years[1]:
            continue
        acc = []
        for df in pd.read_csv(f, usecols=["t", "i", "j", "k", "v"], chunksize=chunksize):
            df["iso3_o"], df["iso3_d"] = df.i.map(code2iso), df.j.map(code2iso)
            if exp_set:
                df = df[df.iso3_o.isin(exp_set)]
            if par_set:
                df = df[df.iso3_d.isin(par_set)]
            if df.empty:
                continue
            df["hs"] = df.k.astype(str).str.zfill(6).str[:hs_digits]
            acc.append(df.groupby(["iso3_o", "iso3_d", "hs"], as_index=False)["v"].sum())
        if acc:
            g = pd.concat(acc).groupby(["iso3_o", "iso3_d", "hs"], as_index=False)["v"].sum()
            g["year"] = int(m.group(1))
            parts.append(g)
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    out["trade_usd"] = out.v * 1000.0          # BACI values are in thousands of USD
    return out.drop(columns="v")


def effective_tariff_from_dataweb(df: pd.DataFrame, base_year: int = 2017,
                                  hs_digits: int = 2) -> pd.DataFrame:
    r"""Effective ad valorem tariff from customs data, with fixed pre-shock weights.

    Input columns: hts (10-digit), country, year, customs_value, calculated_duties.

    .. math:: \tau^{eff}_{ist} = \sum_{h\in s} \frac{V_{ih,base}}{V_{is,base}}
              \cdot \frac{D_{iht}}{V_{iht}}

    Fixed weights stop high-tariff lines from shrinking out of the average
    precisely because they were taxed.
    """
    d = df.rename(columns={"customs_value": "V", "calculated_duties": "D"}).copy()
    d["hts"] = d.hts.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)
    d["hs"] = d.hts.str[:hs_digits]
    d = d[(d.V > 0)]
    d["rate"] = d.D / d.V
    d = d[(d.rate >= 0) & (d.rate <= 3)]
    base = d[d.year == base_year].groupby(["country", "hs", "hts"], as_index=False)["V"].sum()
    base["share"] = base.V / base.groupby(["country", "hs"]).V.transform("sum")
    m = d.merge(base[["country", "hs", "hts", "share"]], on=["country", "hs", "hts"])
    m["contrib"] = m.share * m.rate
    out = m.groupby(["country", "hs", "year"], as_index=False)["contrib"].sum()
    out["tariff_pct"] = 100 * out.pop("contrib")
    return out


def build_sector_panel(sector_trade: pd.DataFrame, tariffs: pd.DataFrame) -> pd.DataFrame:
    """Merge sector trade (iso3_o, iso3_d, hs, year) with sector tariffs.

    ``tariffs`` must have iso3_o (exporter / WITS 'Partner'), iso3_d (importer /
    WITS 'Reporter'), hs, year, tariff_pct.
    """
    width = int(sector_trade.hs.str.len().mode().iat[0])
    t = tariffs.copy()
    t["hs"] = t.hs.astype(str).str.zfill(width).str[:width]
    p = sector_trade.merge(t[["iso3_o", "iso3_d", "hs", "year", "tariff_pct"]],
                           on=["iso3_o", "iso3_d", "hs", "year"], how="left")
    match = p.tariff_pct.notna().mean()
    if match < 0.5:
        raise ValueError(f"only {match:.0%} of rows matched a tariff; check HS width, ISO3 "
                         "codes and reporter/partner orientation")
    p["ln_tariff"] = np.log1p(p.tariff_pct / 100)
    y = p.year.astype(int).astype(str)
    p["pair_sector"] = p.iso3_o + "_" + p.iso3_d + "_" + p.hs
    p["exp_sector"] = p.iso3_o + "_" + p.hs
    p["exp_yr"], p["imp_yr"], p["sec_yr"] = p.iso3_o + "_" + y, p.iso3_d + "_" + y, p.hs + "_" + y
    return p


# --------------------------------------------------------------------------
# Exposure and estimation
# --------------------------------------------------------------------------
def shift_share_exposure(sp: pd.DataFrame, base_year: int = 2017,
                         shock_years=(2018, 2019)) -> pd.DataFrame:
    """Autor-Dorn-Hanson-style exposure: pre-shock export shares x sector tariff change.

    Inference on shift-share regressors should follow Adao, Kolesar and Morales
    (2019) and Borusyak, Hull and Jaravel (2022).
    """
    base = sp[sp.year == base_year]
    shares = (base.groupby(["iso3_o", "hs"]).trade_usd.sum()
              / base.groupby("iso3_o").trade_usd.sum()).rename("share")
    dt = (sp[sp.year.isin(shock_years)].groupby(["iso3_o", "hs"]).ln_tariff.mean()
          - base.groupby(["iso3_o", "hs"]).ln_tariff.mean()).rename("d_ln_tariff")
    ex = pd.concat([shares, dt], axis=1, join="inner").reset_index()
    ex["contrib"] = ex.share * ex.d_ln_tariff
    return ex.groupby("iso3_o", as_index=False).contrib.sum().rename(
        columns={"contrib": "exposure"})


def estimate_sector(sp: pd.DataFrame, fintech_map: dict, fin_dep: pd.DataFrame | None = None):
    """Route C, and Route D (triple difference with sector financial dependence)."""
    p = sp.copy()
    p["fintech_pre"] = p.iso3_o.map(fintech_map)
    p["tXfin"] = p.ln_tariff * p.fintech_pre
    fe = "pair_sector + exp_yr + imp_yr + sec_yr"
    specs = {"Route C: tariff x F": (f"trade_usd ~ ln_tariff + tXfin | {fe}", "tXfin")}
    if fin_dep is not None:
        p = p.merge(fin_dep[["hs", "fin_dep"]], on="hs", how="left")
        p["tXdep"] = p.ln_tariff * p.fin_dep
        p["tXfinXdep"] = p.tXfin * p.fin_dep
        specs["Route D: tariff x F x FinDep"] = (
            f"trade_usd ~ ln_tariff + tXdep + tXfinXdep | {fe}", "tXfinXdep")
    p = p.dropna(subset=["ln_tariff", "tXfin", "trade_usd"])
    rows = []
    for label, (f, term) in specs.items():
        for cl in ("pair_sector", "exp_sector"):
            b, s, t = coef(fit(f, p, vcov={"CRV1": cl}), term)
            rows.append({"spec": label, "cluster": cl, "coef": b, "se": s, "t": t,
                         "n": len(p), "clusters": p[cl].nunique()})
    return pd.DataFrame(rows)
