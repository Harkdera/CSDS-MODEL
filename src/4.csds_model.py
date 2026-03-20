from pathlib import Path
import numpy as np
import pandas as pd


# =====================================================================
# Implementation of the CSDS model from Simon (1999)
#
# General model:
#   tau(u) = a + b * exp(-c * u) - d * exp(-e * u)
#
# Using the model relations:
#   a = tau_r
#   b = d - a
#   c = 5 / u_r
#
# Required input columns:
#   - delta_peak_mm
#   - tau_peak_MPa_csds
#   - u_r_mm
#   - tau_r_MPa
#
# Input file:
#   data/interim/csds_parameters_cleaned.csv
#
# Output file:
#   data/processed/csds_parameters_with_model.csv
# =====================================================================


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "interim" / "csds_parameters_cleaned.csv"
OUTPUT_CSV = BASE_DIR / "data" / "processed" / "csds_parameters_with_model.csv"


def csds_tau(u, a, b, c, d, e):
    """Return tau(u) = a + b * exp(-c * u) - d * exp(-e * u)."""
    u = np.asarray(u, dtype=float)
    return a + b * np.exp(-c * u) - d * np.exp(-e * u)


def fit_csds_one_row(row, max_iter=50, tol=1e-6):
    """
    Fit the CSDS parameters (a, b, c, d, e) for one row.
    """

    tau_p = float(row["tau_peak_MPa_csds"])
    u_p = float(row["delta_peak_mm"])
    u_r = float(row["u_r_mm"])
    tau_r = float(row["tau_r_MPa"])

    # Basic checks
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

    # Step 1: direct parameters
    a = tau_r
    c = 5.0 / u_r

    # Step 2: first approximation of d
    exp_minus_c_up = np.exp(-c * u_p)
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

    d = num / den
    b = d - a

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

    # Step 3: initialize e
    try:
        e_p = np.log(d / (bc * u_p)) / u_p + c
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return {
            "a_csds": a,
            "b_csds": np.nan,
            "c_csds": c,
            "d_csds": np.nan,
            "e_csds": np.nan,
            "csds_converged": False,
            "csds_iterations": 0,
        }

    e = e_p + 1.0

    # Step 4: Newton method to solve F(e) = 0
    def F(e_val):
        return d * e_val / bc - np.exp(u_p * (e_val - c))

    def dF(e_val):
        return d / bc - u_p * np.exp(u_p * (e_val - c))

    converged = False
    it = 0

    for it in range(1, max_iter + 1):
        Fe = F(e)
        dFe = dF(e)

        if abs(Fe) < tol:
            converged = True
            break

        if dFe == 0.0 or not np.isfinite(dFe):
            break

        e_new = e - Fe / dFe

        if not np.isfinite(e_new):
            break

        if abs(e_new - e) < tol:
            e = e_new
            converged = True
            break

        e = e_new

    # Step 5: recalculate d with final e
    exp_5up_ur = np.exp(-5.0 * u_p / u_r)
    den = exp_5up_ur - np.exp(-e * u_p)
    num = tau_p - tau_r * (1.0 - exp_5up_ur)

    if den == 0.0:
        d_final = np.nan
        b_final = np.nan
    else:
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
    # 1) Read input CSV
    df = pd.read_csv(INPUT_CSV)

    # 2) Fit CSDS row by row
    params_df = df.apply(
        fit_csds_one_row,
        axis=1,
        result_type="expand"
    )

    # 3) Merge fitted parameters with original table
    df_out = pd.concat([df, params_df], axis=1)

    # 4) Save output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print("CSDS model parameters saved to:", OUTPUT_CSV)


if __name__ == "__main__":
    main()