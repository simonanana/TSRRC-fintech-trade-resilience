"""Publication figures. Each function takes computed results and returns a Figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import TARIFF_BIN_LABELS, TREATMENT_MARKET
from .style import ACCENT, GOOD, GREY, INK, MUTED, PALETTE, WARN


def identification_anatomy(D, all10, variation, stepwise):
    """Fig. 1: where the identifying variation lives (five panels)."""
    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.28)
    hi = ("CHN", "HKG")

    ax = fig.add_subplot(gs[0, 0])
    v = variation.sort_values("max_abs_d_tariff_pp")
    ax.barh(v.exporter, v.max_abs_d_tariff_pp, edgecolor="white", lw=.5,
            color=[ACCENT if e in hi else INK for e in v.exporter])
    ax.set_xlabel("largest one-year change in US tariff, pp")
    ax.set_title("(a) Treatment is concentrated", fontsize=11)

    ax = fig.add_subplot(gs[0, 1:])
    us = D[(D.iso3_d == TREATMENT_MARKET) & (D.trade_reported == 1)]
    piv = us.pivot_table(index="year", columns="iso3_o", values="tariff_pct")
    first_other = True
    for c in piv.columns:
        if c == "CHN":
            ax.plot(piv.index, piv[c], lw=3.0, color=ACCENT, label="CHN", zorder=4)
        elif c == "HKG":
            ax.plot(piv.index, piv[c], lw=1.6, color="black", ls=(0, (2, 2)),
                    label="HKG (identical from 2020)", zorder=5)
        else:
            ax.plot(piv.index, piv[c], lw=1.1, color=MUTED, alpha=.85,
                    label="other exporters" if first_other else None)
            first_other = False
    ax.set_title("(b) CHN and HKG share one tariff path from 2020", fontsize=11)
    ax.set_ylabel("applied tariff, %"); ax.set_xlabel("year"); ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    cnt = all10.tau_bin.value_counts().reindex(list(TARIFF_BIN_LABELS)).fillna(0)
    bars = ax.bar(cnt.index.astype(str), cnt.values, edgecolor="white", lw=.5,
                  color=[INK] * 4 + [ACCENT])
    for b, n in zip(bars, cnt.values):
        ax.annotate(f"{int(n):,}", (b.get_x() + b.get_width() / 2, max(b.get_height(), 1)),
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_yscale("log"); ax.set_ylabel("observations (log scale)")
    ax.set_title(f"(c) {int(cnt.iloc[-1])} obs above 25%", fontsize=11)
    ax.tick_params(axis="x", rotation=30)

    ax = fig.add_subplot(gs[1, 1])
    lt = us.pivot_table(index="iso3_o", columns="year", values="ln_tariff")
    if {2024, 2025}.issubset(lt.columns):
        dose = (lt[2025] - lt[2024]).rename("dose")
        fin = all10.drop_duplicates("iso3_o").set_index("iso3_o")["fintech_pre"]
        P = pd.concat([dose, fin], axis=1).dropna()
        for e, r in P.iterrows():
            col = ACCENT if e in hi else INK
            ax.scatter(r.fintech_pre, r.dose, s=52, color=col, zorder=3)
            ax.annotate(e, (r.fintech_pre, r.dose), fontsize=8.5, xytext=(4, 4),
                        textcoords="offset points", color=col)
        ax.set_title(f"(d) Same dose, opposite index\ncorr = {P.dose.corr(P.fintech_pre):+.2f}",
                     fontsize=11)
    ax.set_xlabel("pre-shock fintech index"); ax.set_ylabel(r"2025 dose, $\Delta\ln(1+\tau)$")

    ax = fig.add_subplot(gs[1, 2])
    labels = ["all 10", "drop HKG", "drop CHN", "drop both"]
    b3, t3 = stepwise.b3.to_numpy(), stepwise.t3.to_numpy()
    ax.bar(labels, b3, edgecolor="white", lw=.5,
           color=[INK if abs(t) >= 1.96 else ACCENT for t in t3])
    ax.axhline(0, color="k", lw=.8)
    for i, (b, t) in enumerate(zip(b3, t3)):
        ax.annotate(f"t={t:+.1f}", (i, b), ha="center", va="top" if b < 0 else "bottom",
                    fontsize=8.5)
    ax.set_ylabel(r"$\hat\beta_3$"); ax.set_title("(e) A two-point contrast", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    fig.suptitle("The identifying variation in a country-level tariff–fintech design",
                 fontweight="bold", fontsize=13.5, y=.99)
    return fig


def chn_hkg_contrast(D, all10):
    """Fig. 2: same treatment, opposite moderator, divergent outcome."""
    sub = D[(D.iso3_d == TREATMENT_MARKET) & D.iso3_o.isin(["CHN", "HKG"])]
    t = sub.pivot_table(index="year", columns="iso3_o", values="tariff_pct")
    x = sub[sub.trade_reported == 1].pivot_table(index="year", columns="iso3_o", values="trade_usd")
    idx = x.divide(x.loc[2017]).multiply(100)
    per = all10.drop_duplicates("iso3_o").set_index("iso3_o")["fintech_pre"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    axes[0].plot(t.index, t.CHN, lw=3, color=ACCENT, label="CHN")
    axes[0].plot(t.index, t.HKG, lw=1.6, ls=(0, (2, 2)), color="black", label="HKG")
    axes[0].axvspan(2019.5, t.index.max() + .5, color=MUTED, alpha=.18)
    axes[0].set_title("Same treatment"); axes[0].set_ylabel("applied tariff, %"); axes[0].legend()
    axes[1].barh(["CHN", "HKG"], [per["CHN"], per["HKG"]], color=[ACCENT, "black"])
    axes[1].axvline(0, color="k", lw=.8)
    axes[1].set_title("Opposite moderator"); axes[1].set_xlabel("fintech index")
    axes[2].plot(idx.index, idx.CHN, marker="o", ms=4, lw=2, color=ACCENT, label="CHN")
    axes[2].plot(idx.index, idx.HKG, marker="s", ms=4, lw=2, ls=(0, (2, 2)),
                 color="black", label="HKG")
    axes[2].axhline(100, color=GREY, lw=.8, ls=":")
    axes[2].set_title("Divergent outcome"); axes[2].set_ylabel("US-bound exports, 2017 = 100")
    axes[2].legend()
    fig.suptitle(r"$\beta_3$ is the CHN–HKG contrast, and HKG is an entrepôt",
                 fontweight="bold", y=1.03)
    return fig


def dose_response(bins):
    """Fig. 3: non-parametric tariff dose-response with observation counts."""
    order = list(TARIFF_BIN_LABELS)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for lab, col, off in (("all 10", MUTED, -.10), ("9 (no HKG)", INK, .10)):
        s = bins[bins["sample"] == lab].set_index("bin").reindex(order)
        ax.errorbar(np.arange(len(order)) + off, s.coef, yerr=1.96 * s.se, fmt="o", ms=7,
                    lw=1.7, capsize=4, color=col, label=lab)
    s9 = bins[bins["sample"] == "9 (no HKG)"].set_index("bin").reindex(order)
    for i, n in enumerate(s9.n.fillna(0).astype(int)):
        ax.annotate(f"n={n:,}", (i, ax.get_ylim()[0]), ha="center", va="bottom",
                    fontsize=8, color=ACCENT if n == 0 else GREY)
    ax.axhline(0, color="k", lw=.8, ls="--")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_xlabel("applied tariff bin (base: 0–2%)")
    ax.set_ylabel("log change in exports vs base bin")
    ax.set_title("Non-parametric tariff dose–response"); ax.legend(loc="lower left")
    return fig


def power_gap(ri: dict, power: pd.DataFrame, beta1_abs: float, sd_moderator: float,
              n_units: int):
    """Fig. 7: permutation null vs the hypothesised effect, and the design gap."""
    m = float(power.MDE.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.hist(ri["draws"], bins=38, color=MUTED, edgecolor="white", lw=.4,
            label=f"permutation null ({ri['n_draws']} draws)")
    ax.axvline(ri["b_obs"], color=ACCENT, lw=2.3,
               label=fr"observed $\hat\beta_3$ = {ri['b_obs']:+.2f}")
    for pi, ls in ((.15, ":"), (.25, "-.")):
        tgt = pi * beta1_abs / sd_moderator
        ax.axvline(tgt, color=GOOD, lw=1.5, ls=ls,
                   label=fr"hypothesised effect ({pi:.0%} offset) = {tgt:+.2f}")
    ax.axvspan(-m, m, color=WARN, alpha=.10)
    ax.set_xlabel(r"$\beta_3$"); ax.set_ylabel("frequency")
    ax.set_title(f"Permutation null vs the hypothesised effect (RI p = {ri['p']:.3f})",
                 fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.2)
    ax = axes[1]
    ax.bar([f"{r:.0%} offset" for r in power.target_offset], power.units_multiple,
           color=INK, edgecolor="white", lw=.5)
    for i, v in enumerate(power.units_multiple):
        ax.annotate(f"{v:.0f}×", (i, v), ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=INK)
    ax.axhline(1, color=GREY, lw=.8, ls=":")
    ax.set_ylabel("multiple of effective units required")
    ax.set_title("How much more granularity is needed", fontsize=11.5)
    ax.annotate(f"currently {n_units} exporters\nHS2 sectors × exporters ≈ 870",
                (.97, .90), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.8, color=GREY)
    return fig


def fragility(loo: dict[str, pd.DataFrame]):
    """Fig. 4: leave-one-exporter-out estimates for every hypothesis term."""
    fig, axes = plt.subplots(1, len(loo), figsize=(15, 4.5))
    for ax, (lab, R) in zip(np.atleast_1d(axes), loo.items()):
        b0 = float(R.coef.iloc[0])
        y = np.arange(len(R))[::-1]
        for yi, r in zip(y, R.itertuples()):
            sig = abs(r.t) >= 1.96
            col = INK if sig else ACCENT
            ax.errorbar(r.coef, yi, xerr=1.96 * r.se, fmt="o", ms=6.5, color=col,
                        mfc=col if (sig or r.sample == "full") else "white", lw=1.4, capsize=3)
        ax.axvline(0, color="k", ls="--", lw=1); ax.axvline(b0, color=GREY, lw=1.1)
        ax.set_yticks(y); ax.set_yticklabels(R["sample"], fontsize=8.6)
        ax.set_title(lab, fontsize=11); ax.set_xlabel("coefficient (95% CI)")
    fig.suptitle("Every hypothesis coefficient collapses when China is deleted",
                 fontweight="bold", y=1.02)
    return fig


def synthetic_control(sc: dict, placebos: dict):
    """Fig. 6: synthetic-control fit and gap against in-space placebos."""
    path, g = sc["path"], sc["path"]["gap"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ax.plot(path.index, path.actual, lw=2.4, color=ACCENT, marker="o", ms=4, label="China, actual")
    ax.plot(path.index, path.synthetic, lw=2.0, color=INK, ls=(0, (4, 2)), marker="s", ms=3.6,
            label="synthetic China")
    for xv in (2017.5, 2024.5):
        ax.axvline(xv, color=GREY, ls=":", lw=1.2)
    ax.set_ylabel("ln exports to the US,\ndeviation from pre-2018 mean"); ax.set_xlabel("year")
    ax.set_title("Synthetic control fit"); ax.legend(loc="lower left")
    ax = axes[1]
    for s in placebos.values():
        ax.plot(s.index, s.values, color=MUTED, lw=.9, alpha=.75)
    ax.plot(g.index, g.values, color=ACCENT, lw=2.6, label="China")
    ax.axhline(0, color="k", lw=.8)
    for xv in (2017.5, 2024.5):
        ax.axvline(xv, color=GREY, ls=":", lw=1.2)
    ax.set_ylabel("gap, log points"); ax.set_xlabel("year")
    ax.set_title(f"Gap vs {len(placebos)} in-space placebos"); ax.legend(loc="lower left")
    fig.suptitle("Wave 1 has one treated exporter: a case study, not a panel test",
                 fontweight="bold", y=1.02)
    return fig


def corridor(ev: pd.DataFrame, rr: pd.DataFrame, base: int = -1):
    """Fig. 8: corridor event study and robustness of the reallocation interaction."""
    pre_fail = bool((ev[ev.k < base].t.abs() > 1.96).any())
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.fill_between(ev.k, ev.coef - 1.96 * ev.se, ev.coef + 1.96 * ev.se, color=MUTED,
                    alpha=.4, label="95% CI")
    ax.plot(ev.k, ev.coef, marker="o", ms=5.5, color=INK, lw=1.9)
    ax.axhline(0, color="k", lw=.9); ax.axvline(base, color=GREY, ls=":", lw=1.2)
    ax.axvspan(ev.k.min() - .3, base - .5, color=ACCENT if pre_fail else GOOD, alpha=.09)
    ax.set_xlabel("years relative to corridor launch (base = −1)")
    ax.set_ylabel("log change in bilateral exports")
    ax.set_title("Corridor event study"); ax.legend(loc="lower left")
    ax = axes[1]
    y = np.arange(len(rr))[::-1]
    for yi, r in zip(y, rr.itertuples()):
        sig = abs(r.t) >= 1.96
        col = INK if sig else ACCENT
        ax.errorbar(r.coef, yi, xerr=1.96 * r.se, fmt="o", ms=7, color=col,
                    mfc=col if sig else "white", lw=1.5, capsize=3.5)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(rr["sample"], fontsize=9)
    ax.set_xlabel(r"reallocation interaction, $\tau^{US}_{it}\times$corridor$_{ijt}$ (95% CI)")
    ax.set_title("Robustness of the reallocation result")
    title = ("Route A: pre-trend detected — interpret with caution" if pre_fail else
             "Route A: no pre-trend detected; post-launch coefficients are negative")
    fig.suptitle(title, fontweight="bold", y=1.03)
    return fig


def specification_curve(spec: pd.DataFrame, mde_value: float, samples, variants, fes):
    """Fig. 5: specification curve with a design-choice indicator panel."""
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 8.4), sharex=True)
    x, sig = np.arange(len(spec)), (spec.t.abs() >= 1.96).to_numpy()
    ax = axes[0]
    ax.vlines(x, spec.coef - 1.96 * spec.se, spec.coef + 1.96 * spec.se, color=MUTED, lw=1.0)
    ax.scatter(x[sig], spec.coef[sig], s=16, color=INK, zorder=3, label="|t| ≥ 1.96")
    ax.scatter(x[~sig], spec.coef[~sig], s=16, facecolor="white", edgecolor=ACCENT, lw=.9,
               zorder=3, label="|t| < 1.96")
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.axhspan(-mde_value, mde_value, color=WARN, alpha=.09)
    ax.set_ylabel(r"$\hat\beta_3$ (95% CI)")
    ax.set_title("Specification curve: index transform × sample × fixed effects")
    ax.legend(loc="upper left")
    ax = axes[1]
    groups = ([("index", "variant", v) for v in variants] + [("sample", "sample", s) for s in samples]
              + [("FE", "fe", f) for f in fes])
    colors = {"index": PALETTE[0], "sample": PALETTE[1], "FE": PALETTE[2]}
    for i, (g, col, lab) in enumerate(groups):
        on = (spec[col] == lab).to_numpy()
        ax.scatter(x[on], np.full(on.sum(), i), s=13, marker="|", color=colors[g])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g}: {l}" for g, _, l in groups], fontsize=8.4)
    ax.invert_yaxis(); ax.set_xlabel("specifications, ordered by estimate")
    return fig
