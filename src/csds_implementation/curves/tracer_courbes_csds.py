"""Trace les courbes du modèle CSDS pour chaque ligne convergée."""

import os
import numpy as np
import pandas as pd

from src.utils.paths import CSDS_RECONSTRUCTED_CURVES_DIR, MATPLOTLIB_CACHE_DIR, PROCESSED_DATA_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# CSDS model from Simon (1999) - exact formulation used:
#
# tau(u) = tau_r * [1 - exp(-5u / u_r)]
#        + d * [exp(-5u / u_r) - exp(-e * u)]
#
# with:
#   - u in mm
#   - tau in MPa
#   - tau_r : residual shear stress
#   - u_r   : residual displacement
#   - d, e  : fitted model parameters
#
# Input file:
#   data/processed/csds_parameters_converged_only.csv
#
# Output folder:
#   results/figures/curves/csds_tau_vs_u/
#
# Note:
#   This script keeps sigma_n_MPa from the input file because it does
#   not remove any original columns. The script only reads the table,
#   uses the required columns for plotting, and preserves sigma_n_MPa
#   in the dataframe if it is present.
# =====================================================================


INPUT_CSV = PROCESSED_DATA_DIR / "csds_parameters_converged_only.csv"
OUTPUT_DIR = CSDS_RECONSTRUCTED_CURVES_DIR


def csds_tau_simon(u, tau_r, u_r, d, e):
    """Calcule la courbe CSDS de Simon pour une série de déplacements `u`."""
    u = np.asarray(u, dtype=float)
    exp_5u_ur = np.exp(-5.0 * u / u_r)
    return tau_r * (1.0 - exp_5u_ur) + d * (exp_5u_ur - np.exp(-e * u))


def main():
    """Lit les paramètres ajustés puis génère une figure par échantillon."""
    # -------------------------------------------------------------
    # 1) Paramètre général de tracé
    # -------------------------------------------------------------
    displacement_factor = 1.5
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 2) Lire le fichier d'entrée
    # -------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)

    # Vérifier que `sigma_n_MPa` est encore disponible
    if "sigma_n_MPa" in df.columns:
        print("sigma_n_MPa is present in the input file and preserved in the dataframe.")
    else:
        print("Warning: sigma_n_MPa is missing from the input file.")

    # -------------------------------------------------------------
    # 3) Parcourir tous les échantillons
    # -------------------------------------------------------------
    for row_index, row in df.iterrows():
        try:
            tau_r = float(row["tau_r_MPa"])
            u_r = float(row["u_r_mm"])
            d = float(row["d_csds"])
            e = float(row["e_csds"])
            u_p = float(row["delta_peak_mm"])
            tau_p = float(row["tau_peak_MPa_csds"])

            # `sigma_n` est optionnel pour le tracé, mais on l'affiche si disponible.
            sigma_n = row["sigma_n_MPa"] if "sigma_n_MPa" in df.columns else np.nan

            if any(np.isnan([tau_r, u_r, d, e, u_p, tau_p])) or u_r <= 0 or u_p <= 0:
                print(f"Skipping row {row_index}: invalid values")
                continue

            # ---------------------------------------------------------
            # 4) Définir la plage de déplacements à tracer
            # ---------------------------------------------------------
            u_max = displacement_factor * max(u_p, u_r)
            u_vals = np.linspace(0.0, u_max, 300)

            tau_vals = csds_tau_simon(u_vals, tau_r, u_r, d, e)
            tau_r_model = csds_tau_simon(u_r, tau_r, u_r, d, e)

            # ---------------------------------------------------------
            # 5) Construire la figure
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 6))

            plt.plot(u_vals, tau_vals, label="CSDS model")
            plt.scatter([u_p], [tau_p], marker="o", label="Peak (u_p, tau_p)")
            plt.scatter([u_r], [tau_r], marker="s", label="Residual data")
            plt.scatter([u_r], [tau_r_model], marker="x", label="Residual model")

            # Construire l'encadré récapitulatif des paramètres.
            if pd.notna(sigma_n):
                param_text = (
                    f"sigma_n = {float(sigma_n):.3f} MPa\n"
                    f"tau_r = {tau_r:.3f} MPa\n"
                    f"u_r = {u_r:.3f} mm\n"
                    f"d = {d:.3f}\n"
                    f"e = {e:.3f}\n"
                    f"u_p = {u_p:.3f} mm\n"
                    f"tau_p = {tau_p:.3f} MPa"
                )
            else:
                param_text = (
                    f"tau_r = {tau_r:.3f} MPa\n"
                    f"u_r = {u_r:.3f} mm\n"
                    f"d = {d:.3f}\n"
                    f"e = {e:.3f}\n"
                    f"u_p = {u_p:.3f} mm\n"
                    f"tau_p = {tau_p:.3f} MPa"
                )

            plt.text(
                0.65, 0.97,
                param_text,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
            )

            plt.xlabel("Shear displacement u (mm)")
            plt.ylabel("Shear stress tau (MPa)")
            plt.title(f"CSDS curve (row {row_index})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # ---------------------------------------------------------
            # 6) Enregistrer la figure
            # ---------------------------------------------------------
            filename = f"tau_vs_u_row_{row_index}.png"
            filepath = OUTPUT_DIR / filename
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"Saved figure for row {row_index} -> {filepath}")

        except Exception as exc:
            print(f"Error on row {row_index}: {exc}")
            plt.close()


if __name__ == "__main__":
    main()
    
