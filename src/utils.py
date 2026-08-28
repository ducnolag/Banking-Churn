"""
Plotting helpers for confusion matrices and EDA figures.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path) -> None:
    """Save a pretty confusion matrix to ``out_path``."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed (0)", "Churned (1)"],
        yticklabels=["Stayed (0)", "Churned (1)"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
