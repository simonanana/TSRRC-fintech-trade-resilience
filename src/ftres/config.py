"""Project paths, sample definitions and estimation constants.

Every constant that shapes a reported number lives here, so a reader can see in
one place what the pipeline assumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Pass a different root to redirect every output."""

    root: Path = REPO_ROOT
    panel_name: str = "panel_eastasia_v4.parquet"

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def panel(self) -> Path:
        return self.data_processed / self.panel_name

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    def ensure(self) -> "Paths":
        for p in (self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------
EXPORTERS_ALL = ("CHN", "HKG", "IDN", "JPN", "KOR", "MYS", "PHL", "SGP", "THA", "VNM")

#: Hong Kong is excluded from the main sample: its US tariff path is identical to
#: China's from 2020 by construction, its trade series is entrepot re-export
#: volume with a source-splice break, and it sits at the opposite extreme of the
#: fintech index. Retaining both would reduce the interaction to a two-point
#: contrast.
EXCLUDED_EXPORTERS = ("HKG",)

TREATED_EXPORTER = "CHN"
TREATMENT_MARKET = "USA"

#: An exporter-year enters the balanced core if it reports at least this share of
#: its own maximum destination count.
COVERAGE_THRESHOLD = 0.90

#: Source boundary between CEPII BACI (<=2023) and UN Comtrade (>=2024).
SPLICE_YEAR = 2024

# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
#: Dyad, exporter-year and importer-year fixed effects (structural gravity).
FE_MAIN = "pair + exp_yr + imp_yr"
FE_WEAK = "pair + year"

#: Main fintech-index standardisation; the other four enter the specification curve.
INDEX_VARIANT = "hybrid"
INDEX_VARIANTS = ("raw_z", "rank_z", "wins_z", "resid_z", "hybrid")

TARIFF_BIN_EDGES = (-0.01, 2, 5, 10, 25, 200)
TARIFF_BIN_LABELS = ("0-2%", "2-5%", "5-10%", "10-25%", ">25%")

#: Randomization-inference draws. The paper reports 300; 5,000 or more is
#: recommended (9! = 362,880 distinct assignments exist over nine exporters).
N_RI_DEFAULT = 300
RI_SEED = 20260730

#: Minimum-detectable-effect settings.
ALPHA = 0.05
POWER = 0.80
TARGET_OFFSETS = (0.15, 0.20, 0.25)


@dataclass
class RunConfig:
    """Options for one pipeline run."""

    paths: Paths = field(default_factory=Paths)
    n_ri: int = N_RI_DEFAULT
    skip_ri: bool = False
    make_figures: bool = True
    index_variant: str = INDEX_VARIANT
