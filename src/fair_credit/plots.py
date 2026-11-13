"""Reliability diagrams per group, before and after post-processing."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402


def reliability(curves: dict, title: str, path: Path, bins: int = 10) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, key, label in [(axes[0], "p_base", "Baseline"), (axes[1], "p_ceo", "Calibrated Eq. Odds")]:
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
        for g, name, color in [(1, "privileged", "#2b6cb0"), (0, "unprivileged", "#c05621")]:
            m = curves["s"] == g
            frac, mean = calibration_curve(curves["y"][m], np.clip(curves[key][m], 0, 1),
                                           n_bins=bins, strategy="quantile")
            ax.plot(mean, frac, marker="o", ms=4, color=color, label=name)
        ax.set_title(label)
        ax.set_xlabel("Predicted P(repay)")
    axes[0].set_ylabel("Observed repayment rate")
    axes[0].legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
