import pandas as pd

# ------------------------------
# 1) Paths to your CSV files
# ------------------------------
paths = {
    "No":              "No.csv",
    "L_m":             "L.csv",
    "JRC":             "JRC.csv",
    "JCS_MPa":         "JCS.csv",
    "delta_peak_mm":   "u_peak.csv",
    "sigma_n_MPa":     "sigma_n.csv",
    "tau_peak_MPa":    "taux_peak.csv",
    "phi_deg":         "phi.csv"
}

# ----------------------------------
# 2) Cleaning function for standard columns
# ----------------------------------
def clean_column(df, col):
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")  # texte -> NaN
    out = out.dropna(subset=[col])                       # enlève texte
    out = out[out[col] >= 0]                             # enlève négatifs
    out = out.reset_index(drop=True)
    return out

# --------------------------------------------------------
# 3) Load + clean main columns
# --------------------------------------------------------
clean = {}

for col, path in paths.items():
    df = pd.read_csv(path, header=None, names=[col])

    # ⭐ tau_peak_MPa now normal like other columns
    df = clean_column(df, col)
    clean[col] = df

print("\nTailles des colonnes après nettoyage:")
for col, df in clean.items():
    print(f"{col}: {len(df)} lignes")

# ========================================================
# 4) Load ur_tau_r.csv normally (8 columns)
# ========================================================
df_ur_taur = pd.read_csv(
    "ur_tau_r.csv",
    encoding="utf-8-sig",
    usecols=range(8)   # ignore colonne vide
)

df_ur_taur.columns = [
    "Ur_1", "Ur_2", "Ur_3", "Ur_4",
    "tau_r_1", "tau_r_2", "tau_r_3", "tau_r_4"
]

df_ur_taur = df_ur_taur.reset_index(drop=True)

print("\nColonnes ur_taur :", df_ur_taur.columns.tolist())
print("Nombre de lignes ur_taur :", len(df_ur_taur))

# --------------------------------------------------------
# 5) Assemble final table
# --------------------------------------------------------
dfs = [df for df in clean.values()]
dfs.append(df_ur_taur)

df_final = pd.concat(dfs, axis=1)

# --------------------------------------------------------
# 6) Export
# --------------------------------------------------------
df_final.to_csv("Table_CSDS_FULL_CLEAN.csv", index=False)

print("\n✅ Sauvegardé sous: Table_CSDS_FULL_CLEAN.csv")
print(df_final.head())
print("\nShape final:", df_final.shape)
