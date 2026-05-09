"""Sépare le jeu de données convergé en groupes LOW et HIGH selon `tau_peak_MPa_csds`."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import (
    DATABASE_SPLIT_DATASETS_DIR,
    DATABASE_SPLIT_DIAGNOSTICS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ================================
# 1) Charger le jeu de données
# ================================
INPUT_FILE = PROCESSED_DATA_DIR / "csds_parameters_converged_only.csv"

OUTPUT_LOW = INTERIM_DATA_DIR / "csds_tau_peak_low.csv"
OUTPUT_HIGH = INTERIM_DATA_DIR / "csds_tau_peak_high.csv"
OUTPUT_DIR = DATABASE_SPLIT_DIAGNOSTICS_DIR / "tau_peak_low_high"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_PEAK_LABEL = "tau_p (MPa)"
Y_LABEL = "Fréquence"

df = pd.read_csv(INPUT_FILE)

if "sample_id" not in df.columns:
    raise ValueError("Column sample_id not found in converged file.")

if df["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample_id values found in converged file.")

# Vérifier que la colonne de découpage est disponible.
if "tau_peak_MPa_csds" not in df.columns:
    raise ValueError("Column tau_peak_MPa_csds not found in file.")

df["tau_peak_MPa_csds"] = pd.to_numeric(df["tau_peak_MPa_csds"], errors="coerce")
df = df.dropna(subset=["tau_peak_MPa_csds"]).reset_index(drop=True)

tau = df["tau_peak_MPa_csds"]

# ================================
# 2) Choisir le seuil de séparation
# ================================
SPLIT = 7.0 # MPa

# Définir les deux sous-groupes à partir du seuil.
df_low = df[df["tau_peak_MPa_csds"] < SPLIT].copy()
df_high = df[df["tau_peak_MPa_csds"] >= SPLIT].copy()

# Conserver explicitement l'ordre du fichier convergé de référence.
df_low = df_low.sort_values(by="sample_id", key=lambda s: pd.Categorical(s, categories=df["sample_id"], ordered=True)).reset_index(drop=True)
df_high = df_high.sort_values(by="sample_id", key=lambda s: pd.Categorical(s, categories=df["sample_id"], ordered=True)).reset_index(drop=True)

# Calculer la taille de chaque groupe.
count_low = len(df_low)
count_high = len(df_high)

print(f"Group LOW (< {SPLIT} MPa): {count_low}")
print(f"Group HIGH (>= {SPLIT} MPa): {count_high}")

# ================================
# 3) Enregistrer les deux sous-jeux de données
# ================================
OUTPUT_LOW.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_HIGH.parent.mkdir(parents=True, exist_ok=True)

df_low.to_csv(OUTPUT_LOW, index=False)
df_high.to_csv(OUTPUT_HIGH, index=False)
DATABASE_SPLIT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
df_low.to_csv(DATABASE_SPLIT_DATASETS_DIR / OUTPUT_LOW.name, index=False)
df_high.to_csv(DATABASE_SPLIT_DATASETS_DIR / OUTPUT_HIGH.name, index=False)

# ================================
# 4) Tracer l'histogramme global avec le seuil
# ================================
full_hist_path = OUTPUT_DIR / "tau_peak_full_hist_split_7_0.png"

plt.figure(figsize=(9, 5))
sns.histplot(tau, kde=True, bins=30, color="skyblue")

plt.axvline(
    SPLIT,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Split threshold = {SPLIT} MPa"
)

plt.text(
    0.02, 0.95,
    f"LOW group (< {SPLIT}): {count_low}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.text(
    0.02, 0.85,
    f"HIGH group (>= {SPLIT}): {count_high}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.xlabel(TAU_PEAK_LABEL)
plt.ylabel(Y_LABEL)
plt.title(f"Histogramme de tau_p avec seuil de séparation à {SPLIT} MPa")
plt.legend()
plt.tight_layout()
plt.savefig(full_hist_path, dpi=300)
plt.close()

# ================================
# 5) Histogramme et boîte à moustaches pour LOW
# ================================
low_tau = df_low["tau_peak_MPa_csds"]

low_hist_path = OUTPUT_DIR / "tau_peak_low_hist.png"
low_box_path = OUTPUT_DIR / "tau_peak_low_box.png"

plt.figure(figsize=(8, 5))
sns.histplot(low_tau, kde=True, bins=20)
plt.title("LOW - Histogramme - tau_p")
plt.xlabel(TAU_PEAK_LABEL)
plt.ylabel(Y_LABEL)
plt.text(
    0.72, 0.95,
    f"mean = {low_tau.mean():.3g}\nstd = {low_tau.std():.3g}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)
plt.tight_layout()
plt.savefig(low_hist_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
ax = sns.boxplot(x=low_tau)
plt.title("LOW - Boîte à moustaches - tau_p")
ax.text(
    0.01, 0.95,
    f"min = {low_tau.min():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
ax.text(
    0.01, 0.83,
    f"max = {low_tau.max():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig(low_box_path, dpi=300)
plt.close()

# ================================
# 6) Histogramme et boîte à moustaches pour HIGH
# ================================
high_tau = df_high["tau_peak_MPa_csds"]

high_hist_path = OUTPUT_DIR / "tau_peak_high_hist.png"
high_box_path = OUTPUT_DIR / "tau_peak_high_box.png"

plt.figure(figsize=(8, 5))
sns.histplot(high_tau, kde=True, bins=20)
plt.title("HIGH - Histogramme - tau_p")
plt.xlabel(TAU_PEAK_LABEL)
plt.ylabel(Y_LABEL)
plt.text(
    0.72, 0.95,
    f"mean = {high_tau.mean():.3g}\nstd = {high_tau.std():.3g}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)
plt.tight_layout()
plt.savefig(high_hist_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
ax = sns.boxplot(x=high_tau)
plt.title("HIGH - Boîte à moustaches - tau_p")
ax.text(
    0.01, 0.95,
    f"min = {high_tau.min():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
ax.text(
    0.01, 0.83,
    f"max = {high_tau.max():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig(high_box_path, dpi=300)
plt.close()

# ================================
# 7) Message final
# ================================
print("\nDone!")
print("Saved files:")
print(f"- {OUTPUT_LOW}")
print(f"- {OUTPUT_HIGH}")
print(f"- {full_hist_path}")
print(f"- {low_hist_path}")
print(f"- {low_box_path}")
print(f"- {high_hist_path}")
print(f"- {high_box_path}")
