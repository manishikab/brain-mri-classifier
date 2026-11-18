"""Generate figures from the leakage-safe held-out multi-class evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]
CONFUSION_MATRIX = np.array([
    [25, 57, 2],
    [0, 113, 0],
    [0, 3, 33],
])
METRICS = {
    "Precision": [1.000, 0.653, 0.943],
    "Recall": [0.298, 1.000, 0.917],
    "F1 score": [0.459, 0.790, 0.930],
}


def plot_confusion_matrix(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.8), dpi=180)
    image = axis.imshow(CONFUSION_MATRIX, cmap="YlGnBu")
    figure.colorbar(image, ax=axis, shrink=0.9)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Held-out Multi-Class Confusion Matrix",
    )

    threshold = CONFUSION_MATRIX.max() / 2
    for row, column in np.ndindex(CONFUSION_MATRIX.shape):
        axis.text(
            column,
            row,
            CONFUSION_MATRIX[row, column],
            ha="center",
            va="center",
            color="white" if CONFUSION_MATRIX[row, column] > threshold else "#111110",
            fontsize=11,
            fontweight="semibold",
        )

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_per_class_metrics(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 5.8), dpi=180)
    positions = np.arange(len(CLASS_NAMES))
    width = 0.24
    colors = ["#287fb8", "#e07b39", "#529b63"]

    for index, (name, values) in enumerate(METRICS.items()):
        bars = axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=name,
            color=colors[index],
        )
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    axis.set(
        xticks=positions,
        xticklabels=CLASS_NAMES,
        ylabel="Score",
        ylim=(0, 1.13),
        title="Held-out Per-Class Performance",
    )
    axis.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#d6d3cc",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(args.output_dir / "confusion_matrix.png")
    plot_per_class_metrics(args.output_dir / "per_class_metrics.png")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
