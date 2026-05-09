"""Étudie la corrélation entre les top 10 variables FULL de la méthodologie 2."""

from __future__ import annotations

import os
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methodologie_2.full_interpretation_utils import (  # noqa: E402
    CORRELATION_DIR,
    TARGET_COL,
    build_top10_analysis_dataset,
    ensure_output_dirs,
    label,
)


CORRELATION_MATRIX_FILE = CORRELATION_DIR / "top10_feature_correlation_matrix_full.csv"
CORRELATION_PAIRS_FILE = CORRELATION_DIR / "top10_feature_correlation_pairs_full.csv"
HEATMAP_FILE = CORRELATION_DIR / "top10_feature_correlation_heatmap_full.png"


def build_pair_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Convertit la matrice carrée en table de paires triée."""
    rows: list[dict] = []
    columns = corr_df.columns.tolist()
    for i, left in enumerate(columns):
        for right in columns[i + 1:]:
            value = float(corr_df.loc[left, right])
            rows.append({
                "feature_left": left,
                "feature_left_label": label(left),
                "feature_right": right,
                "feature_right_label": label(right),
                "pearson_r": value,
                "abs_pearson_r": abs(value),
            })
    return pd.DataFrame(rows).sort_values(
        ["abs_pearson_r", "feature_left", "feature_right"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def save_heatmap(corr_df: pd.DataFrame) -> None:
    """Sauvegarde la heatmap de corrélation."""
    heatmap_df = corr_df.copy()
    heatmap_df.index = [label(col) for col in heatmap_df.index]
    heatmap_df.columns = [label(col) for col in heatmap_df.columns]

    fig = plt.figure(figsize=(10.5, 8.6))
    ax = fig.add_subplot(111)
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="coolwarm",
        center=0.0,
        fmt=".2f",
        square=True,
        ax=ax,
    )
    ax.set_title("FULL - Méthodologie 2 - Matrice de corrélation des top 10 variables")
    fig.tight_layout()
    fig.savefig(HEATMAP_FILE, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    _, data = build_top10_analysis_dataset()
    feature_cols = [col for col in data.columns if col not in {"sample_id", TARGET_COL}]

    corr_df = data[feature_cols].corr(numeric_only=True)
    corr_df.to_csv(CORRELATION_MATRIX_FILE, index=True)

    pair_df = build_pair_table(corr_df)
    pair_df.to_csv(CORRELATION_PAIRS_FILE, index=False)

    save_heatmap(corr_df)

    print(f"Correlation matrix saved: {CORRELATION_MATRIX_FILE}")
    print(f"Correlation pairs saved: {CORRELATION_PAIRS_FILE}")
    print(f"Heatmap saved: {HEATMAP_FILE}")


if __name__ == "__main__":
    main()
