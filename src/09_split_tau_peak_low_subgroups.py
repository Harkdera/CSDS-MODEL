"""Sépare le sous-ensemble LOW en groupes LOW_1 et LOW_2 selon un second seuil."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 1) Charger le jeu de données LOW
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "interim" / "csds_tau_peak_low.csv"
CONVERGED_FILE = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"

OUTPUT_LOW_1 = BASE_DIR / "data" / "interim" / "csds_tau_peak_low_1.csv"
OUTPUT_LOW_2 = BASE_DIR / "data" / "interim" / "csds_tau_peak_low_2.csv"

OUTPUT_DIR = BASE_DIR / "results" / "figures" / "splits" / "tau_peak_low_subgroups"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df_converged = pd.read_csv(CONVERGED_FILE)

if "sample_id" not in df.columns:
    raise ValueError("Column sample_id not found in LOW file.")
if "sample_id" not in df_converged.columns:
    raise ValueError("Column sample_id not found in converged file.")
if df["sample_id"].duplicated().any():
    raise ValueError("Duplicate sample_id values found in LOW file.")
if df_converged["sample_id"].duplicated().any():
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
SPLIT = 3.5  # MPa

# Définir les deux sous-groupes issus de LOW.
df_low_1 = df[df["tau_peak_MPa_csds"] < SPLIT].copy()
df_low_2 = df[df["tau_peak_MPa_csds"] >= SPLIT].copy()

# Reconstruire les sous-groupes depuis le fichier convergé pour garder
# exactement le même ordre que la base convergée de référence.
low_1_ids = df_low_1["sample_id"].tolist()
low_2_ids = df_low_2["sample_id"].tolist()

df_low_1 = df_converged[df_converged["sample_id"].isin(low_1_ids)].copy().reset_index(drop=True)
df_low_2 = df_converged[df_converged["sample_id"].isin(low_2_ids)].copy().reset_index(drop=True)

if len(df_low_1) != len(low_1_ids):
    raise ValueError("Mismatch while rebuilding LOW_1 from converged sample_id values.")
if len(df_low_2) != len(low_2_ids):
    raise ValueError("Mismatch while rebuilding LOW_2 from converged sample_id values.")

# Calculer la taille de chaque groupe.
count_low_1 = len(df_low_1)
count_low_2 = len(df_low_2)

print(f"Group LOW_1 (< {SPLIT} MPa): {count_low_1}")
print(f"Group LOW_2 (>= {SPLIT} MPa): {count_low_2}")

# ================================
# 3) Enregistrer les deux sous-jeux de données
# ================================
OUTPUT_LOW_1.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_LOW_2.parent.mkdir(parents=True, exist_ok=True)

df_low_1.to_csv(OUTPUT_LOW_1, index=False)
df_low_2.to_csv(OUTPUT_LOW_2, index=False)

# ================================
# 4) Tracer l'histogramme global avec le seuil
# ================================
full_hist_path = OUTPUT_DIR / "tau_peak_low_hist_split_2_5.png"

plt.figure(figsize=(9, 5))
sns.histplot(tau, kde=True, bins=25, color="skyblue")

plt.axvline(
    SPLIT,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Split threshold = {SPLIT} MPa"
)

plt.text(
    0.02, 0.95,
    f"LOW_1 (< {SPLIT}): {count_low_1}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.text(
    0.02, 0.85,
    f"LOW_2 (>= {SPLIT}): {count_low_2}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.xlabel("tau_peak_MPa_csds (MPa)")
plt.ylabel("Frequency")
plt.title(f"LOW group: tau_peak_MPa_csds split at {SPLIT} MPa")
plt.legend()
plt.tight_layout()
plt.savefig(full_hist_path, dpi=300)
plt.close()

# ================================
# 5) Histogramme et boîte à moustaches pour LOW_1
# ================================
low_1_tau = df_low_1["tau_peak_MPa_csds"]

low_1_hist_path = OUTPUT_DIR / "tau_peak_low_1_hist.png"
low_1_box_path = OUTPUT_DIR / "tau_peak_low_1_box.png"

plt.figure(figsize=(8, 5))
sns.histplot(low_1_tau, kde=True, bins=20)
plt.title("LOW_1 - Histogram - tau_peak_MPa_csds")
plt.xlabel("tau_peak_MPa_csds (MPa)")
plt.ylabel("Frequency")
plt.text(
    0.72, 0.95,
    f"mean = {low_1_tau.mean():.3g}\nstd = {low_1_tau.std():.3g}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)
plt.tight_layout()
plt.savefig(low_1_hist_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
ax = sns.boxplot(x=low_1_tau)
plt.title("LOW_1 - Boxplot - tau_peak_MPa_csds")
ax.text(
    0.01, 0.95,
    f"min = {low_1_tau.min():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
ax.text(
    0.01, 0.83,
    f"max = {low_1_tau.max():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig(low_1_box_path, dpi=300)
plt.close()

# ================================
# 6) Histogramme et boîte à moustaches pour LOW_2
# ================================
low_2_tau = df_low_2["tau_peak_MPa_csds"]

low_2_hist_path = OUTPUT_DIR / "tau_peak_low_2_hist.png"
low_2_box_path = OUTPUT_DIR / "tau_peak_low_2_box.png"

plt.figure(figsize=(8, 5))
sns.histplot(low_2_tau, kde=True, bins=20)
plt.title("LOW_2 - Histogram - tau_peak_MPa_csds")
plt.xlabel("tau_peak_MPa_csds (MPa)")
plt.ylabel("Frequency")
plt.text(
    0.72, 0.95,
    f"mean = {low_2_tau.mean():.3g}\nstd = {low_2_tau.std():.3g}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)
plt.tight_layout()
plt.savefig(low_2_hist_path, dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
ax = sns.boxplot(x=low_2_tau)
plt.title("LOW_2 - Boxplot - tau_peak_MPa_csds")
ax.text(
    0.01, 0.95,
    f"min = {low_2_tau.min():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
ax.text(
    0.01, 0.83,
    f"max = {low_2_tau.max():.3g}",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)
plt.tight_layout()
plt.savefig(low_2_box_path, dpi=300)
plt.close()

# ================================
# 7) Message final
# ================================
print("\nDone!")
print("Saved files:")
print(f"- {OUTPUT_LOW_1}")
print(f"- {OUTPUT_LOW_2}")
print(f"- {full_hist_path}")
print(f"- {low_1_hist_path}")
print(f"- {low_1_box_path}")
print(f"- {low_2_hist_path}")
print(f"- {low_2_box_path}")
