"""ftres: fintech and trade resilience under targeted tariff shocks.

Replication package for "Can Fintech Build Trade Resilience Under Tariff Shocks?
Identification Diagnostics from East Asian Economies, 2015-2025".

Modules
-------
config       paths, sample definitions and estimation constants
style        figure style and palette
estimation   thin wrappers around pyfixest (PPML / OLS with high-dimensional FE)
data         panel loading, variable construction, integrity checks, samples
diagnostics  identification diagnostics, including N_eff
inference    randomization inference and minimum-detectable-effect calculations
hypotheses   gravity benchmarks, dose-response, hypothesis table with fragility
synth        synthetic control for the single treated exporter
corridors    payment-corridor (Route A) and bilateral-complementarity (Route B) designs
sector       sector-level exposure design (Route C), data loaders and estimator
spec_curve   specification curve
plots        publication figures
report       machine-generated results summary
synthetic    synthetic panel with the production schema, for tests and demos
"""

__version__ = "1.0.0"
