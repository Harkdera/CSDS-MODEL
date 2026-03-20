from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

# ------------------------------
# 1) Paths to your CSV files
# ------------------------------
paths = {
    "No": RAW_DIR / "no.csv",
    "L_m": RAW_DIR / "l.csv",
    "JRC": RAW_DIR / "jrc.csv",
    "JCS_MPa": RAW_DIR / "jcs.csv",
    "delta_peak_mm": RAW_DIR / "u_peak.csv",
    "sigma_n_MPa": RAW_DIR / "sigma_n.csv",
    "tau_peak_MPa": RAW_DIR / "tau_peak.csv",
    "phi_deg": RAW_DIR / "phi.csv"
}


# ----------------------------------
# 2) Cleaning function for standard columns
# ----------------------------------
def clean_column(df, col):
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[col])
    out = out[out[col] >= 0]
    out = out.reset_index(drop=True)
    return out


# --------------------------------------------------------
# 3) Load + clean main columns
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
# 4) Load ur_tau_r.csv normally (8 columns)
# ========================================================
ur_tau_r_file = RAW_DIR / "ur_tau_r.csv"

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
# 5) Assemble final table
# --------------------------------------------------------
dfs = [df for df in clean.values()]
dfs.append(df_ur_taur)

df_final = pd.concat(dfs, axis=1)


# --------------------------------------------------------
# 6) Export
# --------------------------------------------------------
output_file = INTERIM_DIR / "csds_full_table_clean.csv"
output_file.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")
print(df_final.head())
print("\nFinal shape:", df_final.shape)