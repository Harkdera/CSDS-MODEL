"""Prépare le top 10 FULL de la méthodologie 2 et génère les scatter plots sur log(e-c)."""

from __future__ import annotations

import os
from pathlib import Path
from src.utils.paths import MATPLOTLIB_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final_model.final_model_utils_methodologie_2 import (
    DATASET_NAME,
    SCATTER_DIR,
    TARGET_COL,
    TARGET_LABEL,
    TOP10_DATASET_FILE,
    TOP10_FEATURES_FILE,
    build_top10_analysis_dataset,
    ensure_output_dirs,
    label,
    mirror_legacy_outputs,
    safe_slug,
)


SCATTER_INDEX_FILE = SCATTER_DIR / "scatter_index_log_e_minus_c_full.csv"


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    """Calcule la corrélation de Pearson après filtrage des NaN."""
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_num[mask], y_num[mask])[0, 1])


def plot_scatter(data: pd.DataFrame, feature: str, output_file: Path) -> float:
    """Trace un nuage de points entre une variable et log(e-c)."""
    x = pd.to_numeric(data[feature], errors="coerce")
    y = pd.to_numeric(data[TARGET_COL], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan

    x_valid = x[mask]
    y_valid = y[mask]
    r_value = pearson_r(x_valid, y_valid)

    fig = plt.figure(figsize=(7.8, 5.8))
    ax = fig.add_subplot(111)
    ax.scatter(x_valid, y_valid, alpha=0.75, s=36, edgecolor="white", linewidth=0.4)

    if x_valid.nunique() > 1:
        slope, intercept = np.polyfit(x_valid.to_numpy(dtype=float), y_valid.to_numpy(dtype=float), deg=1)
        x_line = np.linspace(float(x_valid.min()), float(x_valid.max()), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linestyle="--", linewidth=1.2)

    ax.set_title(f"Méthodologie 2 - FULL - {TARGET_LABEL} en fonction de {label(feature)}")
    ax.set_xlabel(label(feature))
    ax.set_ylabel(TARGET_LABEL)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"n = {int(mask.sum())}\nr = {r_value:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.88, edgecolor="black"),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return r_value


def main() -> None:
    ensure_output_dirs()
    top10_df, data = build_top10_analysis_dataset()

    rows: list[dict] = []
    correlations: list[float] = []

    for _, row in top10_df.iterrows():
        feature = row["feature"]
        output_file = SCATTER_DIR / f"log_e_minus_c_vs_{safe_slug(feature)}.png"
        r_value = plot_scatter(data, feature, output_file)
        correlations.append(r_value)
        rows.append({
            "dataset": DATASET_NAME,
            "target": TARGET_COL,
            "target_label": TARGET_LABEL,
            "feature": feature,
            "feature_label": row["feature_label"],
            "frequency": row["frequency"],
            "pearson_r": r_value,
            "abs_pearson_r": abs(r_value) if pd.notna(r_value) else np.nan,
            "output_file": str(output_file),
        })

    top10_df = top10_df.copy()
    top10_df["pearson_r_with_log_e_minus_c"] = correlations
    top10_df["abs_pearson_r_with_log_e_minus_c"] = top10_df["pearson_r_with_log_e_minus_c"].abs()

    top10_df.to_csv(TOP10_FEATURES_FILE, index=False)
    data.to_csv(TOP10_DATASET_FILE, index=False)
    pd.DataFrame(rows).to_csv(SCATTER_INDEX_FILE, index=False)
    mirror_legacy_outputs()

    print(f"Top 10 saved: {TOP10_FEATURES_FILE}")
    print(f"Top 10 dataset saved: {TOP10_DATASET_FILE}")
    print(f"Scatter index saved: {SCATTER_INDEX_FILE}")
    print(f"Scatter plots generated: {len(rows)}")


if __name__ == "__main__":
    main()
