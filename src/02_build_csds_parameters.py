"""Calcule les paramètres CSDS manquants ou estimés à partir de la table nettoyée."""

from pathlib import Path
import pandas as pd
import numpy as np
import math


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "interim" / "csds_full_table_clean.csv"
OUTPUT_FILE = BASE_DIR / "data" / "interim" / "csds_parameters.csv"
RESULTS_DATASETS_DIR = BASE_DIR / "results" / "datasets"


# --------------------------------------------------
# 1) Utilitaire : conversion numérique facultative
# --------------------------------------------------
def parse_optional_float(x):
    """Retourne un flottant si la valeur est exploitable, sinon `None`."""
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if s == "" or all(ch in "-–" for ch in s):
        return None

    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------
# 2) Charger le fichier combiné
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)


# Listes qui stockent les valeurs calculées pour chaque ligne
tau_peak_list = []
u_r_list = []
tau_r_list = []

# Colonnes indicatrices : 1 = estimé, 0 = mesuré
tau_peak_flag = []   # 1 = estimated, 0 = measured
u_r_flag = []        # 1 = estimated, 0 = measured
tau_r_flag = []      # 1 = estimated, 0 = measured

# Facteur empirique utilisé pour estimer `u_r`
factor_ur = 0.5 ** (-1 / 0.381)


# --------------------------------------------------
# 3) Parcourir les lignes et compléter les paramètres
# --------------------------------------------------
for _, row in df.iterrows():

    # -----------------------
    # tau_p : résistance au cisaillement au pic
    # -----------------------
    tau_peak_val = parse_optional_float(row.get("tau_peak_MPa"))

    if tau_peak_val is not None:
        tau_peak_flag.append(0)
    else:
        sigma_n = parse_optional_float(row.get("sigma_n_MPa"))
        JRC = parse_optional_float(row.get("JRC"))
        JCS = parse_optional_float(row.get("JCS_MPa"))
        phi_b = parse_optional_float(row.get("phi_deg"))

        if None in (sigma_n, JRC, JCS, phi_b) or sigma_n <= 0 or JCS <= 0:
            tau_peak_val = np.nan
            tau_peak_flag.append(1)
        else:
            inside = JRC * math.log10(JCS / sigma_n) + phi_b
            tau_peak_val = sigma_n * math.tan(math.radians(inside))
            tau_peak_flag.append(1)

    tau_peak_list.append(tau_peak_val)

    # -----------------------
    # u_r : déplacement résiduel
    # -----------------------
    ur_vals = []
    for col in ["Ur_1", "Ur_2", "Ur_3", "Ur_4"]:
        v = parse_optional_float(row.get(col))
        if v is not None:
            ur_vals.append(v)

    if len(ur_vals) >= 1:
        u_r_val = sum(ur_vals) / len(ur_vals)
        u_r_flag.append(0)
    else:
        u_peak = parse_optional_float(row.get("delta_peak_mm"))
        if u_peak is None:
            u_r_val = np.nan
        else:
            u_r_val = u_peak * factor_ur
        u_r_flag.append(1)

    u_r_list.append(u_r_val)

    # -----------------------
    # tau_r : résistance résiduelle au cisaillement
    # -----------------------
    taur_vals = []
    for col in ["tau_r_1", "tau_r_2", "tau_r_3", "tau_r_4"]:
        v = parse_optional_float(row.get(col))
        if v is not None:
            taur_vals.append(v)

    if len(taur_vals) >= 1:
        tau_r_val = sum(taur_vals) / len(taur_vals)
        tau_r_flag.append(0)
    else:
        sigma_n = parse_optional_float(row.get("sigma_n_MPa"))
        phi_r = parse_optional_float(row.get("phi_deg"))
        if None in (sigma_n, phi_r):
            tau_r_val = np.nan
        else:
            tau_r_val = sigma_n * math.tan(math.radians(phi_r))
        tau_r_flag.append(1)

    tau_r_list.append(tau_r_val)


# --------------------------------------------------
# 4) Ajouter les colonnes calculées
# --------------------------------------------------
df["tau_peak_MPa_csds"] = tau_peak_list
df["u_r_mm"] = u_r_list
df["tau_r_MPa"] = tau_r_list

df["tau_peak_estimated"] = tau_peak_flag
df["u_r_estimated"] = u_r_flag
df["tau_r_estimated"] = tau_r_flag


# --------------------------------------------------
# 5) Exporter le fichier final
# --------------------------------------------------
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
RESULTS_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DATASETS_DIR / OUTPUT_FILE.name, index=False)

print(f"Created: {OUTPUT_FILE}")
