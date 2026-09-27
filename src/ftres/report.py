"""Machine-generated results summary (results/REPORT.md)."""

from __future__ import annotations

from pathlib import Path


def _f(x, fmt="+.3f"):
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return "n/a"


def write_report(r: dict, path: Path) -> Path:
    """Render the key numbers of one pipeline run as Markdown."""
    hyp = r["hypotheses"].set_index("hypothesis")
    h1, h3 = hyp.loc["H1 resilience interaction"], hyp.loc["H3 triple interaction"]
    pw = r["power"]
    sc = r.get("synth")
    spec = r["spec_summary"]
    lines = [
        "# Results summary",
        "",
        f"Main sample: {r['n_exporters']} exporters, {r['n_obs']:,} observations, "
        f"{r['n_dyads']} dyads (Hong Kong excluded; balanced core).",
        "",
        "## Identified",
        "",
        f"* Tariff elasticity of exports (PPML, dyad + exporter-year + importer-year FE): "
        f"**{_f(r['beta1_no_interaction'], '+.2f')}** (t = {_f(r['t_beta1'], '+.2f')}).",
        f"* Elasticity decomposition: {_f(r['beta1_interacted'])} + "
        f"({_f(r['beta3'])})({_f(r['index_treated'], '.3f')}) = "
        f"**{_f(r['implied_treated'])}** for the treated exporter, versus "
        f"{_f(r['implied_mean'])} at the sample mean. The no-interaction estimate is "
        + ("closer to the treated exporter's implied elasticity than to the sample mean: "
           "read it as a shock-specific elasticity, not a panel average."
           if abs(r["beta1_no_interaction"] - r["implied_treated"])
           < abs(r["beta1_no_interaction"] - r["implied_mean"])
           else "closer to the sample mean than to the treated exporter's implied elasticity."),
    ]
    if sc:
        g = sc["path"]
        yrs = [y for y in (2019, 2023) if y in g.index]
        gaps = ", ".join(f"{y}: {g.loc[y, 'gap']:+.3f} log points "
                         f"({g.loc[y, 'shortfall_pct']:.1f}% shortfall)" for y in yrs)
        lines.append(f"* Synthetic control (pre-RMSPE {sc['rmspe_pre']:.4f}): {gaps}.")
    lines += [
        "",
        "## Not identified",
        "",
        f"* Effective number of identifying units: **N_eff = {r['N_eff']:.2f}** "
        f"against {r['n_exporters']} exporters and {r['n_dyads']} dyad clusters.",
        "",
        "| Hypothesis | coef | t (dyad) | t (jackknife) | RI p | coef without CHN |",
        "|---|---|---|---|---|---|",
        f"| H1 tariff x fintech | {_f(h1.coef)} | {_f(h1.t_pair, '+.2f')} | "
        f"{_f(h1.t_jack, '+.2f')} | {_f(r.get('ri_h1_p'), '.3f')} | {_f(h1.coef_drop_CHN)} |",
        f"| H3 tariff x fintech x RQ | {_f(h3.coef)} | {_f(h3.t_pair, '+.2f')} | "
        f"{_f(h3.t_jack, '+.2f')} | {_f(r.get('ri_h3_p'), '.3f')} | {_f(h3.coef_drop_CHN)} |",
        "",
        f"* Fitted interaction implies a positive tariff elasticity for F "
        f"{'<' if r['beta3'] < 0 else '>'} {_f(r['turning_point'])}; estimation-sample "
        f"exporters in that region: {r.get('units_beyond_turning_point') or 'none'}.",
        f"* Specification curve: {spec['n_specs']} estimable specifications "
        f"({spec['not_identified']} not identified), {spec['positive_significant']} "
        f"positive-significant, {spec['negative_significant']} negative-significant, "
        f"{spec['null']} null.",
        "",
        "## Design requirement",
        "",
        f"MDE at 80% power = {pw.MDE.iloc[0]:.3f} (permutation null sd {r['ri_sd']:.3f}, "
        f"{r['ri_n_draws']} draws); sd(F) = {r['sd_moderator']:.3f}.",
        "",
        "| offset | beta3 target | MDE / target | units multiple | units needed |",
        "|---|---|---|---|---|",
    ]
    for row in pw.itertuples():
        lines.append(f"| {row.target_offset:.0%} | {row.beta3_target:.3f} | "
                     f"{row.MDE_over_target:.2f} | {row.units_multiple:.1f}x | "
                     f"{row.units_needed:.0f} |")
    path.write_text("\n".join(lines) + "\n")
    return path
