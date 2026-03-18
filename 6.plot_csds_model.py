import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# Modèle CSDS de Simon (1999) - Formulation exacte utilisée :
#
# (5.10)  τ(u) = τ_r[1 - exp(-5u / u_r)]
#               + d[exp(-5u / u_r) - exp(-e·u)]
#
# avec :
#   - u en mm
#   - τ en MPa
#   - τ_r  : contrainte résiduelle           (tau_r_MPa)
#   - u_r  : déplacement résiduel            (u_r_mm)
#   - d,e  : paramètres ajustés du modèle    (d_csds, e_csds)
#
# On suppose que le fichier suivant existe dans le même dossier :
#   CSDS_parameters_with_CSDS_model.csv
#
# et contient au moins les colonnes :
#   - tau_r_MPa
#   - u_r_mm
#   - d_csds
#   - e_csds
#   - delta_peak_mm       (u_p)
#   - tau_peak_MPa_csds   (τ_p)
# =====================================================================

def csds_tau_simon(u, tau_r, u_r, d, e):
    """
    Modèle CSDS selon Simon (Eq. 5.10) :

        τ(u) = τ_r[1 - exp(-5u / u_r)]
               + d[exp(-5u / u_r) - exp(-e·u)]

    u : déplacement en mm (scalaire ou array)
    """
    u = np.asarray(u, dtype=float)
    exp_5u_ur = np.exp(-5.0 * u / u_r)
    return tau_r * (1.0 - exp_5u_ur) + d * (exp_5u_ur - np.exp(-e * u))


def main():
    # -------------------------------------------------------------
    # 1) Paramètres généraux
    # -------------------------------------------------------------
    input_csv = "CSDS_parameters_CONVERGED_ONLY.csv"

    # Facteur pour la plage max de déplacement (1.5 × max(u_p, u_r))
    displacement_factor = 1.5

    # Dossier où sauvegarder toutes les figures
    output_dir = "tau vs u"
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 2) Lecture du fichier complet
    # -------------------------------------------------------------
    df = pd.read_csv(input_csv)

    # -------------------------------------------------------------
    # 3) Boucle sur toutes les lignes / échantillons
    # -------------------------------------------------------------
    for row_index, row in df.iterrows():
        try:
            # ----- Récupération des paramètres CSDS (formulation Simon) ----
            tau_r = float(row["tau_r_MPa"])
            u_r = float(row["u_r_mm"])
            d = float(row["d_csds"])
            e = float(row["e_csds"])

            # Points caractéristiques (pour marquer le pic et le résiduel)
            u_p = float(row["delta_peak_mm"])
            tau_p = float(row["tau_peak_MPa_csds"])

            # Vérification basique (éviter NaN ou valeurs non physiques)
            if any(np.isnan([tau_r, u_r, d, e, u_p, tau_p])) or u_r <= 0 or u_p <= 0:
                print(f"Skipping row {row_index}: invalid or missing values.")
                continue

            # ---------------------------------------------------------
            # 4) Plage de déplacement u (en mm)
            # ---------------------------------------------------------
            u_max = displacement_factor * max(u_p, u_r)
            u_vals = np.linspace(0.0, u_max, 300)

            # Calcul du modèle CSDS (formulation exacte Eq. 5.10)
            tau_vals = csds_tau_simon(u_vals, tau_r, u_r, d, e)

            # Valeur du modèle au résiduel pour vérifier cohérence
            tau_r_model = csds_tau_simon(u_r, tau_r, u_r, d, e)

            # ---------------------------------------------------------
            # 5) Tracé
            # ---------------------------------------------------------
            plt.figure()

            # Courbe du modèle CSDS
            plt.plot(u_vals, tau_vals, label="CSDS model (Eq. 5.10, Simon 1999)")

            # Marquer le pic (données)
            plt.scatter([u_p], [tau_p], marker="o", zorder=5, label="Peak (u_p, τ_p)")

            # Marquer le résiduel (données + modèle au même point)
            plt.scatter([u_r], [tau_r], marker="s", zorder=5,
                        label="Residual data (u_r, τ_r)")
            plt.scatter([u_r], [tau_r_model], marker="x", zorder=5,
                        label="Residual model τ(u_r)")

            # Mise en forme
            plt.xlabel("Shear displacement u (mm)")
            plt.ylabel("Shear stress τ (MPa)")
            plt.title(f"CSDS shear stress–displacement curve (row {row_index})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # ---------------------------------------------------------
            # 6) Sauvegarde de la figure dans le dossier "tau vs u"
            # ---------------------------------------------------------
            # Nom de fichier basé sur l’index de ligne
            filename = f"tau_vs_u_row_{row_index}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"Saved figure for row {row_index} → {filepath}")

        except Exception as exc:
            print(f"Error on row {row_index}: {exc}")
            plt.close()


if __name__ == "__main__":
    main()
