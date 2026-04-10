"""Réduit la table des paramètres CSDS aux colonnes utiles pour la suite du pipeline."""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "interim" / "csds_parameters.csv"
OUTPUT_FILE = BASE_DIR / "data" / "interim" / "csds_parameters_cleaned.csv"
RESULTS_DATASETS_DIR = BASE_DIR / "results" / "datasets"


# ---------------------------------------
# 1) Charger le fichier complet des paramètres CSDS
# ---------------------------------------
df = pd.read_csv(INPUT_FILE)


# ----------------------------------------------------
# 2) Ajouter une colonne `sample_id` si elle n'existe pas
# ----------------------------------------------------
if "sample_id" not in df.columns:
    df.insert(0, "sample_id", range(1, len(df) + 1))


# ----------------------------------------------------
# 3) Colonnes à conserver
# ----------------------------------------------------
keep_cols = [
    "sample_id",
    "sigma_n_MPa",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "u_r_mm",
    "tau_r_MPa",
    "tau_peak_estimated",
    "u_r_estimated",
    "tau_r_estimated"
]

keep_cols = [c for c in keep_cols if c in df.columns]

print("Columns kept:", keep_cols)

df_clean = df[keep_cols].copy()


# ----------------------------------------------------
# 4) Exporter le jeu de données nettoyé
# ----------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(OUTPUT_FILE, index=False)
RESULTS_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(RESULTS_DATASETS_DIR / OUTPUT_FILE.name, index=False)

print(f"\nFile saved: {OUTPUT_FILE}")
print(df_clean.head())
