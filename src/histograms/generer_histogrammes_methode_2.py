"""Fait l'EDA de `e`, `c`, `e-c` et `log(e-c)` pour la recherche directe sur `e`."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


from src.utils.common_methodologie_2 import (
    SPLIT_FILES,
    build_direct_e_dataset,
    dataset_slug,
    ensure_output_dirs,
    get_candidate_feature_names,
)
from src.utils.paths import HISTOGRAMS_RESULTS_DIR

METHOD_2_HISTOGRAM_ROOT = HISTOGRAMS_RESULTS_DIR / "methodologie_2"


EDA_COLUMNS = [
    "sigma_n_MPa",
    "log_sigma_n_MPa",
    "delta_peak_mm",
    "log_delta_peak_mm",
    "u_r_mm",
    "log_u_r_mm",
    "tau_peak_MPa_csds",
    "log_tau_peak_MPa_csds",
    "tau_r_MPa",
    "log_tau_r_MPa",
    "c_target",
    "d_csds",
    "log_d_csds",
    "e_csds",
    "e_minus_c_csds",
    "log_e_minus_c_csds",
]


def save_histograms(data: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    """Sauvegarde des histogrammes simples pour les variables clés."""
    hist_dir = output_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for column in EDA_COLUMNS:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        sns.histplot(data[column], kde=True, bins=25, ax=ax, color="steelblue")
        ax.set_title(f"{dataset_name} | Distribution de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequence")
        fig.tight_layout()
        fig.savefig(hist_dir / f"{column}_hist.png", dpi=180)
        plt.close(fig)


def save_scatterplots(data: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    """Sauvegarde quelques nuages de points centrés sur la cible directe."""
    scatter_dir = output_dir / "scatterplots"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("e_csds", "c_target"),
        ("e_minus_c_csds", "tau_peak_MPa_csds"),
        ("log_e_minus_c_csds", "tau_peak_MPa_csds"),
        ("log_e_minus_c_csds", "log_tau_peak_MPa_csds"),
        ("log_e_minus_c_csds", "log_tau_r_MPa"),
        ("log_e_minus_c_csds", "log_u_r_mm"),
        ("log_e_minus_c_csds", "log_delta_peak_mm"),
        ("log_e_minus_c_csds", "log_sigma_n_MPa"),
        ("log_e_minus_c_csds", "d_csds"),
        ("log_e_minus_c_csds", "log_d_csds"),
        ("log_e_minus_c_csds", "u_r_mm"),
    ]

    for y_col, x_col in pairs:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        sns.scatterplot(data=data, x=x_col, y=y_col, ax=ax, s=28, alpha=0.75)
        ax.set_title(f"{dataset_name} | {y_col} vs {x_col}")
        fig.tight_layout()
        fig.savefig(scatter_dir / f"{y_col}_vs_{x_col}.png", dpi=180)
        plt.close(fig)


def save_correlation_heatmap(data: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    """Sauvegarde une heatmap de corrélations pour les variables CSDS principales."""
    corr = data[EDA_COLUMNS].corr(numeric_only=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"{dataset_name} | Correlations des variables CSDS et des cibles")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap_core_targets.png", dpi=180)
    plt.close(fig)


def save_candidate_feature_heatmaps(data: pd.DataFrame, output_dir: Path, dataset_name: str) -> None:
    """Sauvegarde des heatmaps de corrélations pour les variables candidates de régression."""
    candidate_features = get_candidate_feature_names(data)

    target_corr = (
        data[candidate_features + ["log_e_minus_c_csds"]]
        .corr(numeric_only=True)["log_e_minus_c_csds"]
        .drop(labels=["log_e_minus_c_csds"])
        .abs()
        .sort_values(ascending=False)
    )

    top_features = target_corr.head(12).index.tolist()
    selected_cols = top_features + ["log_e_minus_c_csds", "e_csds", "e_minus_c_csds", "log_d_csds"]
    selected_cols = [col for col in selected_cols if col in data.columns]

    corr_selected = data[selected_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    sns.heatmap(corr_selected, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"{dataset_name} | Top correlations des variables candidates vers log(e-c)")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap_top_candidate_features.png", dpi=180)
    plt.close(fig)

    base_model_cols = [
        "sigma_n_MPa",
        "log_sigma_n_MPa",
        "delta_peak_mm",
        "log_delta_peak_mm",
        "u_r_mm",
        "log_u_r_mm",
        "tau_peak_MPa_csds",
        "log_tau_peak_MPa_csds",
        "tau_r_MPa",
        "log_tau_r_MPa",
        "d_csds",
        "log_d_csds",
        "e_csds",
        "e_minus_c_csds",
        "log_e_minus_c_csds",
    ]
    base_model_cols = [col for col in base_model_cols if col in data.columns]
    corr_base = data[base_model_cols].corr(numeric_only=True)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(corr_base, cmap="coolwarm", center=0.0, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f"{dataset_name} | Correlations des variables de base pour les modèles réguliers")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap_base_model_variables.png", dpi=180)
    plt.close(fig)


def save_summary_tables(data: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """Sauvegarde les statistiques descriptives et les corrélations à la cible."""
    desc = data[EDA_COLUMNS].describe().T
    desc["missing"] = data[EDA_COLUMNS].isna().sum()
    desc_file = output_dir / "descriptive_statistics.csv"
    desc.to_csv(desc_file)

    corr_target = (
        data[EDA_COLUMNS]
        .corr(numeric_only=True)[["log_e_minus_c_csds", "e_csds", "e_minus_c_csds", "log_d_csds"]]
        .sort_values(by="log_e_minus_c_csds", ascending=False)
    )
    corr_file = output_dir / "correlation_with_targets.csv"
    corr_target.to_csv(corr_file)
    return desc_file, corr_file


def main() -> None:
    ensure_output_dirs()
    sns.set_theme(style="whitegrid")

    print("=" * 100)
    print("EDA DES TARGETS DIRECTS POUR e")
    print("=" * 100)

    summary_rows = []

    for dataset_name in SPLIT_FILES:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        data = build_direct_e_dataset(dataset_name)
        output_dir = METHOD_2_HISTOGRAM_ROOT / dataset_slug(dataset_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        desc_file, corr_file = save_summary_tables(data, output_dir)
        save_histograms(data, output_dir, dataset_name)

        summary_rows.append({
            "Dataset": dataset_name,
            "Rows": len(data),
            "Mean_e": data["e_csds"].mean(),
            "Mean_c": data["c_target"].mean(),
            "Mean_log_d": data["log_d_csds"].mean(),
            "Mean_log_sigma_n": data["log_sigma_n_MPa"].mean(),
            "Mean_log_u_peak": data["log_delta_peak_mm"].mean(),
            "Mean_log_u_r": data["log_u_r_mm"].mean(),
            "Mean_log_tau_peak": data["log_tau_peak_MPa_csds"].mean(),
            "Mean_log_tau_r": data["log_tau_r_MPa"].mean(),
            "Mean_e_minus_c": data["e_minus_c_csds"].mean(),
            "Std_log_e_minus_c": data["log_e_minus_c_csds"].std(),
            "Descriptive_File": str(desc_file),
            "Correlation_File": str(corr_file),
            "Output_Dir": str(output_dir),
        })

        print(f"Rows available: {len(data)}")
        print(f"EDA saved in: {output_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = METHOD_2_HISTOGRAM_ROOT / "summary_methodologie_2_histograms.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
