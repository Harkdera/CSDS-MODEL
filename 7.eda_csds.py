import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================================
# 1) LOAD DATA
# ================================
FILE = "CSDS_parameters_CONVERGED_ONLY.csv"
df = pd.read_csv(FILE)

# ================================
# 2) SELECT ONLY PHYSICAL + TRUE CSDS PARAMETERS
# ================================
numeric_cols = [
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

# keep only columns that exist in the dataset
numeric_cols = [c for c in numeric_cols if c in df.columns]

print("✔ Columns included in EDA:")
for c in numeric_cols:
    print(" -", c)

# ================================
# 3) OUTPUT FOLDER
# ================================
OUT = "EDA_CSDS_CONVERGED"
os.makedirs(OUT, exist_ok=True)

# ================================
# 4) DESCRIPTIVE STATS FUNCTION
# ================================
def descriptive_stats(series: pd.Series) -> dict:
    series = series.dropna()

    if len(series) == 0:
        return {k: np.nan for k in [
            "n","mean","median","mode","min","max","variance","std",
            "cv(%)","Q1","Q3","IQR","outlier_low","outlier_high","n_outliers"
        ]}

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
        "cv(%)": cv,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "outlier_low": low_bound,
        "outlier_high": high_bound,
        "n_outliers": n_outliers,
    }


# ================================
# 5) MAIN LOOP FOR HIST + BOXPLOT
# ================================
summary = {}

for col in numeric_cols:
    print(f"Processing: {col}")
    data = pd.to_numeric(df[col], errors="coerce").dropna()

    # compute stats
    stats = descriptive_stats(data)
    summary[col] = stats

    # --- Histogram ---
    plt.figure(figsize=(7, 5))
    sns.histplot(data, kde=True)
    plt.title(f"Histogram - {col}")
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
    plt.savefig(f"{OUT}/{col}_hist.png", dpi=300)
    plt.close()

    # --- Boxplot ---
    plt.figure(figsize=(6, 4))
    ax = sns.boxplot(x=data)
    plt.title(f"Boxplot - {col}")

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
    plt.savefig(f"{OUT}/{col}_box.png", dpi=300)
    plt.close()


# ================================
# 6) PAIRWISE SCATTER PLOTS
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
        plt.title(f"{y_col} vs {x_col}")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{y_col}_vs_{x_col}_scatter.png", dpi=300)
        plt.close()

# ================================
# 7) EXPORT SUMMARY STATS
# ================================
stats_df = pd.DataFrame(summary).T
stats_df.to_csv(f"{OUT}/CSDS_descriptive_statistics.csv", index=True)

print("\n✅ EDA complete. Results saved to:", OUT)
