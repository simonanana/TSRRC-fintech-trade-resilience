"""Figure style shared by every exhibit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, ACCENT, MUTED, GREY = "#1B3A5C", "#C0392B", "#8FA5B8", "#6E6E6E"
GOOD, WARN = "#1E8449", "#B7950B"
PALETTE = ["#1B3A5C", "#C0392B", "#1E8449", "#B7950B", "#6C3483",
           "#117A65", "#A04000", "#5D6D7E", "#7D6608", "#4A235A"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 10.5,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
        "legend.frameon": False, "legend.fontsize": 9,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    })


def savefig(fig, directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"{stem}.png"
    fig.savefig(out)
    plt.close(fig)
    return out
