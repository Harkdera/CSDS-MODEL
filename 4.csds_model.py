import numpy as np
import pandas as pd


# =====================================================================
# Implémentation du modèle CSDS de Simon (1999)
#
# Modèle général (Eq. 5.1) :
#   τ(u) = a + b·exp(-c·u) - d·exp(-e·u)
#
# Conditions et relations utilisées :
#
# (5.2) τ(0) = 0  →  a + b - d = 0
# (5.3) a + b = d
#
# (5.4–5.5) à grand déplacement u >> 0 :
#   τ(u) → τ_r  →  a = τ_r
#
# (5.7) à u = u_r, τ = τ_r :
#   b·exp(-c·u_r) - d·exp(-e·u_r) = 0
#
# (5.8–5.9) on impose exp(-c·u_r) ≈ 0.07  →  c ≈ 5 / u_r
#
# En remplaçant a = τ_r et c = 5 / u_r on obtient :
# (5.10) τ(u) = τ_r[1 - exp(-5u/u_r)]
#              + d[exp(-5u/u_r) - exp(-e·u)]
#
# La dérivée est :
# (5.11) dτ/du = -5/u_r · (d - τ_r) · exp(-5u/u_r) + d·e·exp(-e·u)
#
# Au pic (u = u_p) on a un maximum :
# (5.12–5.13) dτ/du |_{u_p} = 0  →  relation non-linéaire entre d et e
#
# En manipulant on obtient une équation non-linéaire en e :
# (5.15) F(e) = d·e/(b·c) - exp[u_p·(e - c)] = 0
#
# Avec :
#   b = d - τ_r               (5.3 avec a = τ_r)
#   c = 5 / u_r               (5.9)
#
# Sa dérivée :
# (5.23) F'(e) = d/(b·c) - u_p·exp[u_p·(e - c)]
#
# Newton (5.22) :
#   e_{i+1} = e_i - F(e_i) / F'(e_i)
#
# Pour choisir e_init :
# (5.24) e_p   = ln(d/(b·c·u_p)) / u_p + c   (valeur au maximum de F(e))
# (5.25) e_init = e_p + 1
#
# Une fois e trouvé, on recalcule d :
# (5.19) d = [τ_p - τ_r(1 - exp(-5u_p/u_r))] /
#            [exp(-5u_p/u_r) - exp(-e·u_p)]
#
# Paramètres d’entrée nécessaires par échantillon :
#   delta_peak_mm      → u_p
#   tau_peak_MPa_csds  → τ_p
#   u_r_mm             → u_r
#   tau_r_MPa          → τ_r
#
# Le script suppose que CSDS_parameters_CLEANED.csv
# se trouve dans le même dossier que ce script.
# =====================================================================


def csds_tau(u, a, b, c, d, e):
    """Fonction τ(u) = a + b·exp(-c·u) - d·exp(-e·u)."""
    u = np.asarray(u, dtype=float)
    return a + b * np.exp(-c * u) - d * np.exp(-e * u)


def fit_csds_one_row(row, max_iter=50, tol=1e-6):
    """
    Ajuste les paramètres (a,b,c,d,e) du modèle CSDS pour UNE ligne.

    Colonnes utilisées dans la ligne :
        - 'delta_peak_mm'       → u_p
        - 'tau_peak_MPa_csds'   → τ_p
        - 'u_r_mm'              → u_r
        - 'tau_r_MPa'           → τ_r
    """

    tau_p = float(row["tau_peak_MPa_csds"])
    u_p = float(row["delta_peak_mm"])
    u_r = float(row["u_r_mm"])
    tau_r = float(row["tau_r_MPa"])

    # Vérifications de base (évite NaN/bogue)
    if u_p <= 0 or u_r <= 0 or tau_p <= 0 or tau_r <= 0:
        return {
            "a_csds": np.nan,
            "b_csds": np.nan,
            "c_csds": np.nan,
            "d_csds": np.nan,
            "e_csds": np.nan,
            "csds_converged": False,
            "csds_iterations": 0,
        }

    # --- Étape 1 : paramètres simples --------------------------------
    # (5.4–5.5) a = τ_r
    a = tau_r

    # (5.9) c ≈ 5 / u_r
    c = 5.0 / u_r

    # --- Étape 2 : première approximation de d (Eqs. 5.20–5.21) ------
    exp_minus_c_up = np.exp(-c * u_p)

    # num = τ_p - a[1 - exp(-c u_p)]
    num = tau_p - a * (1.0 - exp_minus_c_up)
    den = exp_minus_c_up

    if den == 0.0:
        return {
            "a_csds": a,
            "b_csds": np.nan,
            "c_csds": c,
            "d_csds": np.nan,
            "e_csds": np.nan,
            "csds_converged": False,
            "csds_iterations": 0,
        }

    # (5.21) d_init
    d = num / den

    # (5.3) avec a = τ_r → b = d - a
    b = d - a

    # Vérification de signes pour éviter des logs bizarres
    bc = b * c
    if bc <= 0.0 or d <= 0.0:
        return {
            "a_csds": a,
            "b_csds": np.nan,
            "c_csds": c,
            "d_csds": np.nan,
            "e_csds": np.nan,
            "csds_converged": False,
            "csds_iterations": 0,
        }

    # --- Étape 3 : choix de e_init (Eqs. 5.24–5.25) ------------------
    try:
        # (5.24) e_p = ln(d / (b c u_p)) / u_p + c
        e_p = np.log(d / (bc * u_p)) / u_p + c
    except (FloatingPointError, ValueError):
        return {
            "a_csds": a,
            "b_csds": np.nan,
            "c_csds": c,
            "d_csds": np.nan,
            "e_csds": np.nan,
            "csds_converged": False,
            "csds_iterations": 0,
        }

    # (5.25) e_init = e_p + 1
    e = e_p + 1.0

    # --- Étape 4 : Newton pour résoudre F(e) = 0 (Eqs. 5.15, 5.22) ---
    def F(e_val):
        # (5.15) F(e) = d·e/(b·c) - exp[u_p(e - c)]
        return d * e_val / bc - np.exp(u_p * (e_val - c))

    def dF(e_val):
        # (5.23) F'(e) = d/(b·c) - u_p·exp[u_p(e - c)]
        return d / bc - u_p * np.exp(u_p * (e_val - c))

    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        Fe = F(e)
        dFe = dF(e)

        # Condition de convergence sur F(e)
        if abs(Fe) < tol:
            converged = True
            break

        # Évite division par zéro / NaN
        if dFe == 0.0 or not np.isfinite(dFe):
            break

        # (5.22) e_{i+1} = e_i - F(e_i)/F'(e_i)
        e_new = e - Fe / dFe

        if not np.isfinite(e_new):
            break

        # Convergence sur le changement
        if abs(e_new - e) < tol:
            e = e_new
            converged = True
            break

        e = e_new

    # --- Étape 5 : recalcul de d avec le e final (Eq. 5.19) ----------
    exp_5up_ur = np.exp(-5.0 * u_p / u_r)
    den = exp_5up_ur - np.exp(-e * u_p)
    num = tau_p - tau_r * (1.0 - exp_5up_ur)

    if den == 0.0:
        d_final = np.nan
        b_final = np.nan
    else:
        # (5.19) d = [...]
        d_final = num / den
        b_final = d_final - a

    return {
        "a_csds": a,
        "b_csds": b_final,
        "c_csds": c,
        "d_csds": d_final,
        "e_csds": e,
        "csds_converged": converged,
        "csds_iterations": it,
    }


def main():
    # -----------------------------------------------------------------
    # 1) Lecture du CSV (même dossier que ce script)
    # -----------------------------------------------------------------
    input_csv = "CSDS_parameters_CLEANED.csv"
    #input_csv = "csdscalibreationtest.csv"
    df = pd.read_csv(input_csv)

    # -----------------------------------------------------------------
    # 2) Ajustement CSDS ligne par ligne
    # -----------------------------------------------------------------
    params_df = df.apply(
        fit_csds_one_row,
        axis=1,
        result_type="expand"
    )

    # -----------------------------------------------------------------
    # 3) Fusion des paramètres au tableau d’origine
    # -----------------------------------------------------------------
    df_out = pd.concat([df, params_df], axis=1)

    # -----------------------------------------------------------------
    # 4) Sauvegarde
    # -----------------------------------------------------------------
    output_csv = "CSDS_parameters_with_CSDS_model.csv"
    df_out.to_csv(output_csv, index=False)

    print("✔ CSDS model parameters saved to:", output_csv)


if __name__ == "__main__":
    main()

