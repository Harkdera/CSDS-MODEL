import pandas as pd

# ---------------------------------------
# 1) Load the full CSDS parameter file
# ---------------------------------------
df = pd.read_csv("CSDS_parameters.csv")

# ----------------------------------------------------
# 2) Add a "sample_id" column if none exists
# ----------------------------------------------------
if "sample_id" not in df.columns:
    df.insert(0, "sample_id", range(1, len(df) + 1))

# ----------------------------------------------------
# 3) Columns to keep
# ----------------------------------------------------
keep_cols = [
    "sample_id",
    "delta_peak_mm",       # u_peak
    "tau_peak_MPa_csds",   # tau_peak
    "u_r_mm",              # u_r
    "tau_r_MPa",           # tau_r
    "tau_peak_estimated",  # flags
    "u_r_estimated",
    "tau_r_estimated"
]

# Keep only existing columns
keep_cols = [c for c in keep_cols if c in df.columns]

print("✔ Columns kept:", keep_cols)

df_clean = df[keep_cols].copy()

# ----------------------------------------------------
# 4) Export cleaned dataset
# ----------------------------------------------------
df_clean.to_csv("CSDS_parameters_CLEANED.csv", index=False)

print("\n✅ File saved: CSDS_parameters_CLEANED.csv")
print(df_clean.head())
