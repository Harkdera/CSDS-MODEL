"""Assemble et nettoie les colonnes brutes pour former la table CSDS intermédiaire."""

import pandas as pd

from src.utils.paths import DATABASE_MERGED_DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR


# ------------------------------
# 1) Chemins vers les fichiers CSV d'entrée
# ------------------------------
paths = {
    "No": RAW_DATA_DIR / "no.csv",
    "L_m": RAW_DATA_DIR / "l.csv",
    "JRC": RAW_DATA_DIR / "jrc.csv",
    "JCS_MPa": RAW_DATA_DIR / "jcs.csv",
    "delta_peak_mm": RAW_DATA_DIR / "u_peak.csv",
    "sigma_n_MPa": RAW_DATA_DIR / "sigma_n.csv",
    "tau_peak_MPa": RAW_DATA_DIR / "tau_peak.csv",
    "phi_deg": RAW_DATA_DIR / "phi.csv"
}


# ----------------------------------
# 2) Fonction de nettoyage des colonnes standard
# ----------------------------------
def clean_column(df, col):
    """Convertit une colonne en numérique, supprime les valeurs invalides et réindexe le résultat."""
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    out = out[out[col] >= 0]
    out = out.reset_index(drop=True)
    return out


# --------------------------------------------------------
# 3) Charger et nettoyer les colonnes principales
# --------------------------------------------------------
clean = {}

for col, path in paths.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, header=None, names=[col])
    df = clean_column(df, col)
    clean[col] = df

print("\nSizes of columns after cleaning:")
for col, df in clean.items():
    print(f"{col}: {len(df)} rows")


# ========================================================
# 4) Charger normalement ur_tau_r.csv (8 colonnes)
# ========================================================
ur_tau_r_file = RAW_DATA_DIR / "ur_tau_r.csv"

if not ur_tau_r_file.exists():
    raise FileNotFoundError(f"Missing file: {ur_tau_r_file}")

df_ur_taur = pd.read_csv(
    ur_tau_r_file,
    encoding="utf-8-sig",
    usecols=range(8)
)

df_ur_taur.columns = [
    "Ur_1", "Ur_2", "Ur_3", "Ur_4",
    "tau_r_1", "tau_r_2", "tau_r_3", "tau_r_4"
]

df_ur_taur = df_ur_taur.reset_index(drop=True)

print("\nColumns in ur_tau_r:", df_ur_taur.columns.tolist())
print("Number of rows in ur_tau_r:", len(df_ur_taur))


# --------------------------------------------------------
# 5) Assembler la table finale
# --------------------------------------------------------
dfs = [df for df in clean.values()]
dfs.append(df_ur_taur)

df_final = pd.concat(dfs, axis=1)


# --------------------------------------------------------
# 6) Exporter le résultat
# --------------------------------------------------------
output_file = INTERIM_DATA_DIR / "csds_full_table_clean.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(output_file, index=False)
DATABASE_MERGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(DATABASE_MERGED_DATA_DIR / output_file.name, index=False)

print(f"\nSaved to: {output_file}")
print(df_final.head())
print("\nFinal shape:", df_final.shape)
