"""Produit une analyse exploratoire et des graphiques descriptifs du jeu de données CSDS."""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, SCATTER_PLOTS_RESULTS_DIR

# ================================
# 1) Chemins principaux
# ================================
FILE = PROCESSED_DATA_DIR / "csds_parameters_converged_only.csv"

OUT_DIR = SCATTER_PLOTS_RESULTS_DIR / "converged-csds"
GROUPS_OUT_ROOT = SCATTER_PLOTS_RESULTS_DIR / "groups"

GROUP_INPUTS = {
    "full": PROCESSED_DATA_DIR / "csds_parameters_converged_only.csv",
    "low": INTERIM_DATA_DIR / "csds_tau_peak_low.csv",
    "high": INTERIM_DATA_DIR / "csds_tau_peak_high.csv",
    "low_1": INTERIM_DATA_DIR / "csds_tau_peak_low_1.csv",
    "low_2": INTERIM_DATA_DIR / "csds_tau_peak_low_2.csv",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
GROUPS_OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ================================
# 2) Colonnes retenues pour l'EDA
# ================================
TARGET_COLS = [
    "sigma_n_MPa",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "u_r_mm",
    "tau_r_MPa",
    "a_csds",
    "b_csds",
    "c_csds",
    "d_csds",
    "e_csds",
    "e_minus_c_csds",
    "log_d_csds",
    "log_e_minus_c_csds",
]

VARIABLE_LABELS = {
    "sigma_n_MPa": "sigma_n (MPa)",
    "delta_peak_mm": "u_p (mm)",
    "tau_peak_MPa_csds": "tau_p (MPa)",
    "u_r_mm": "u_r (mm)",
    "tau_r_MPa": "tau_r (MPa)",
    "a_csds": "a (MPa)",
    "b_csds": "b (MPa)",
    "c_csds": "c (1/mm)",
    "d_csds": "d (MPa)",
    "e_csds": "e (1/mm)",
    "e_minus_c_csds": "e - c (1/mm)",
    "log_d_csds": "log(d)",
    "log_e_minus_c_csds": "log(e - c)",
}


def variable_label(col: str) -> str:
    """Return a consistent display label for figures."""
    return VARIABLE_LABELS.get(col, col)

# ================================
# 3) Fonction de statistiques descriptives
# ================================
def descriptive_stats(series: pd.Series) -> dict:
    """Calcule les statistiques descriptives et les bornes d'outliers pour une série."""
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) == 0:
        return {
            k: np.nan for k in [
                "n", "mean", "median", "mode", "min", "max", "variance", "std",
                "cv_percent", "Q1", "Q3", "IQR", "outlier_low", "outlier_high", "n_outliers"
            ]
        }

    n = len(series)
    mean = series.mean()
    median = series.median()
    mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
    var = series.var()
    std = series.std()
    cv = (std / mean) * 100 if mean != 0 else np.nan
    xmin = series.min()
    xmax = series.max()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    low_bound = q1 - 1.5 * iqr
    high_bound = q3 + 1.5 * iqr
    n_outliers = len(series[(series < low_bound) | (series > high_bound)])

    return {
        "n": n,
        "mean": mean,
        "median": median,
        "mode": mode_val,
        "min": xmin,
        "max": xmax,
        "variance": var,
        "std": std,
        "cv_percent": cv,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "outlier_low": low_bound,
        "outlier_high": high_bound,
        "n_outliers": n_outliers,
    }


def run_eda_for_dataset(df: pd.DataFrame, dataset_label: str, output_dir: Path) -> Path:
    """Sauvegarde uniquement les nuages de points et heatmaps d'un dataset CSDS."""
    df = df.copy()
    if "d_csds" in df.columns:
        d_values = pd.to_numeric(df["d_csds"], errors="coerce")
        df["log_d_csds"] = np.where(d_values > 0, np.log(d_values), np.nan)
    if {"e_csds", "c_csds"}.issubset(df.columns):
        e_values = pd.to_numeric(df["e_csds"], errors="coerce")
        c_values = pd.to_numeric(df["c_csds"], errors="coerce")
        e_minus_c = e_values - c_values
        df["e_minus_c_csds"] = e_minus_c
        df["log_e_minus_c_csds"] = np.where(e_minus_c > 0, np.log(e_minus_c), np.nan)

    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = [c for c in TARGET_COLS if c in df.columns]

    print("\n" + "=" * 80)
    print(f"RUNNING EDA FOR {dataset_label.upper()}")
    print("=" * 80)
    print(f"Output folder: {output_dir}")
    print("Columns included in EDA:")
    for col in numeric_cols:
        print(" -", col)

    print(f"Generating scatter plots for {dataset_label}...")

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            x_col = numeric_cols[i]
            y_col = numeric_cols[j]

            x = pd.to_numeric(df[x_col], errors="coerce")
            y = pd.to_numeric(df[y_col], errors="coerce")
            mask = x.notna() & y.notna()

            if mask.sum() == 0:
                continue

            plt.figure(figsize=(6, 5))
            plt.scatter(x[mask], y[mask], alpha=0.7)
            plt.xlabel(variable_label(x_col))
            plt.ylabel(variable_label(y_col))
            plt.title(
                f"{dataset_label.upper()} - {variable_label(y_col)} en fonction de {variable_label(x_col)}"
            )
            plt.tight_layout()
            plt.savefig(output_dir / f"{y_col}_vs_{x_col}_scatter.png", dpi=300)
            plt.close()

    if len(numeric_cols) > 1:
        corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = corr_df.corr().rename(index=VARIABLE_LABELS, columns=VARIABLE_LABELS)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title(f"{dataset_label.upper()} - Matrice de corrélation")
        plt.tight_layout()
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()

    return output_dir

def main() -> None:
    print("\n" + "=" * 80)
    print("INPUT FILE")
    print("=" * 80)
    print(FILE)

    # Sortie historique conservée pour compatibilité avec les premières analyses.
    df = pd.read_csv(FILE)
    artifact_outputs = [run_eda_for_dataset(df, "full", OUT_DIR)]

    # Sorties organisées par groupe. Elles incluent aussi sigma_n_MPa, afin que
    # les figures HIGH/LOW utilisées dans le mémoire soient reproductibles.
    for group_name, group_file in GROUP_INPUTS.items():
        if not group_file.exists():
            print(f"Skipping {group_name}: missing file {group_file}")
            continue
        group_df = pd.read_csv(group_file)
        group_out_dir = GROUPS_OUT_ROOT / group_name
        artifact_outputs.append(run_eda_for_dataset(group_df, group_name, group_out_dir))

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)
    print("Scatter folders:")
    for artifact_output in artifact_outputs:
        print(f"- {artifact_output}")


if __name__ == "__main__":
    main()
