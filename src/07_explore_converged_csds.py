"""Produit une analyse exploratoire et des graphiques descriptifs du jeu de données CSDS."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1) Chemins principaux
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent

FILE = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"

OUT_DIR = BASE_DIR / "results" / "eda" / "converged_csds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
]

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

# ================================
# 4) Charger les données
# ================================
df = pd.read_csv(FILE)
numeric_cols = [c for c in TARGET_COLS if c in df.columns]

print("\n" + "=" * 80)
print("INPUT FILE")
print("=" * 80)
print(FILE)

print("\n" + "=" * 80)
print("OUTPUT FOLDER")
print("=" * 80)
print(OUT_DIR)

print("\n" + "=" * 80)
print("RUNNING EDA FOR FULL DATASET")
print("=" * 80)

print("Columns included in EDA:")
for col in numeric_cols:
    print(" -", col)

# ================================
# 5) Statistiques descriptives + histogrammes + boîtes à moustaches
# ================================
summary = {}

for col in numeric_cols:
    print(f"Processing: {col}")
    data_col = pd.to_numeric(df[col], errors="coerce").dropna()

    stats = descriptive_stats(data_col)
    summary[col] = stats

    # Générer l'histogramme de la variable.
    plt.figure(figsize=(7, 5))
    sns.histplot(data_col, kde=True)
    plt.title(f"FULL - Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.text(
        0.70, 0.95,
        f"mean = {stats['mean']:.3g}\nstd = {stats['std']:.3g}",
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black")
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_hist.png", dpi=300)
    plt.close()

    # Générer la boîte à moustaches de la variable.
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x=data_col)
    plt.title(f"FULL - Boxplot - {col}")
    ax.text(
        0.01, 0.95,
        f"min = {stats['min']:.3g}",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.6)
    )
    ax.text(
        0.01, 0.85,
        f"max = {stats['max']:.3g}",
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.6)
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_box.png", dpi=300)
    plt.close()

# ================================
# 6) Nuages de points par paire de variables
# ================================
print("\nGenerating scatter plots...")

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
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"FULL - {y_col} vs {x_col}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{y_col}_vs_{x_col}_scatter.png", dpi=300)
        plt.close()

# ================================
# 7) Carte de chaleur des corrélations
# ================================
if len(numeric_cols) > 1:
    corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("FULL - Correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

# ================================
# 8) Exporter les statistiques descriptives
# ================================
stats_df = pd.DataFrame(summary).T
stats_output = OUT_DIR / "descriptive_statistics.csv"
stats_df.to_csv(stats_output, index=True)

# ================================
# 9) Message final
# ================================
print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
print(f"Figures and statistics saved in: {OUT_DIR}")
print("\nMain stats file:")
print(stats_output)
