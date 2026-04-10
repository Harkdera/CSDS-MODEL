"""Reconstruit et résume les meilleurs modèles retenus sur `log(e-c)`."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    GROUP_DIR,
    REGRESSION_DIR,
    SPLIT_FILES,
    TOP5_DIR,
    build_direct_e_dataset,
    dataset_group,
    dataset_slug,
    deduplicate_top_rows,
    ensure_output_dirs,
    parse_feature_list,
)


TOP_N = 5
RIDGE_ALPHA = 1.0
POLY_DEGREE = 2


RESULT_FILES_EXP = {
    dataset_name: REGRESSION_DIR / f"exp_log_e_gap_selection_{dataset_slug(dataset_name)}.csv"
    for dataset_name in SPLIT_FILES
}
RESULT_FILES_POLY = {
    dataset_name: REGRESSION_DIR / f"poly_log_e_gap_selection_{dataset_slug(dataset_name)}.csv"
    for dataset_name in SPLIT_FILES
}


def build_exponential_model():
    """Construit le pipeline exponentiel utilisé pour `log(e-c)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def build_polynomial_model():
    """Construit le pipeline polynomial utilisé pour `log(e-c)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def convert_scaled_linear_to_raw(model, feature_cols):
    """Convertit les coefficients standardisés vers l'espace brut."""
    scaler = model.named_steps["scaler"]
    reg = model.named_steps["reg"]
    means = scaler.mean_
    scales = scaler.scale_
    coef_scaled = reg.coef_
    intercept_scaled = reg.intercept_

    coef_raw = coef_scaled / scales
    intercept_raw = intercept_scaled - np.sum(coef_scaled * means / scales)
    return intercept_raw, coef_raw


def make_exp_equations(intercept_raw, coef_raw, feature_cols, digits=10):
    """Construit les équations textuelles pour `z = log(e-c)` et `e` reconstruit."""
    pieces = [f"log(e-c) = {intercept_raw:.{digits}f}"]
    for beta, feat in zip(coef_raw, feature_cols):
        sign = "+" if beta >= 0 else "-"
        pieces.append(f" {sign} {abs(beta):.{digits}f}*{feat}")
    eq_z = "".join(pieces)
    eq_e = f"e = c + exp({eq_z.split('=', 1)[1].strip()})"
    return eq_z, eq_e


def polynomial_equation_from_pipeline(model, feature_cols, digits=10):
    """Construit l'équation polynomiale dans l'espace standardisé."""
    reg = model.named_steps["reg"]
    poly = model.named_steps["poly"]
    poly_feature_names = poly.get_feature_names_out(feature_cols)

    pieces = [f"log(e-c) = {reg.intercept_:.{digits}f}"]
    for coef, term in zip(reg.coef_, poly_feature_names):
        if abs(coef) < 1e-12:
            continue
        sign = "+" if coef >= 0 else "-"
        pieces.append(f" {sign} {abs(coef):.{digits}f}*{term}")

    eq_z = "".join(pieces)
    eq_e = f"e = c + exp({eq_z.split('=', 1)[1].strip()})"
    return eq_z, eq_e, list(poly_feature_names), reg.coef_


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("RESUME DES MEILLEURS MODELES POUR log(e-c)")
    print("=" * 100)

    for dataset_name in SPLIT_FILES:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        data = build_direct_e_dataset(dataset_name)
        group_dir = TOP5_DIR / dataset_group(dataset_name)
        group_dir.mkdir(parents=True, exist_ok=True)
        slug = dataset_slug(dataset_name)

        exp_summary_rows = []
        exp_coef_rows = []
        poly_summary_rows = []
        poly_coef_rows = []

        exp_file = RESULT_FILES_EXP[dataset_name]
        if exp_file.exists():
            exp_df = pd.read_csv(exp_file)
            kept_exp = deduplicate_top_rows(exp_df, TOP_N)
            for rank, (row, feature_list) in enumerate(kept_exp, start=1):
                model = build_exponential_model()
                model.fit(data[feature_list], data["log_e_minus_c_csds"])
                intercept_raw, coef_raw = convert_scaled_linear_to_raw(model, feature_list)
                eq_z, eq_e = make_exp_equations(intercept_raw, coef_raw, feature_list)

                exp_summary_rows.append({
                    "Dataset": dataset_name,
                    "Selection_Mode": "log_gap",
                    "Rank_in_saved_results": rank,
                    "N_Features": len(feature_list),
                    "Features": " + ".join(feature_list),
                    "Intercept_log_e_minus_c": intercept_raw,
                    "Equation_log_e_minus_c": eq_z,
                    "Equation_e": eq_e,
                    "Saved_R2_val_z": row.get("R2_val_z", np.nan),
                    "Saved_R2_cv_mean_z": row.get("R2_cv_mean_z", np.nan),
                    "Saved_R2_val_e": row.get("R2_val_e", np.nan),
                    "Saved_RMSE_val_e": row.get("RMSE_val_e", np.nan),
                    "Saved_Selection_Score": row.get("Selection_Score", np.nan),
                })

                for feat, coef in zip(feature_list, coef_raw):
                    exp_coef_rows.append({
                        "Dataset": dataset_name,
                        "Selection_Mode": "log_gap",
                        "Rank_in_saved_results": rank,
                        "Feature": feat,
                        "Coefficient_in_log_e_minus_c_equation": coef,
                    })

        poly_file = RESULT_FILES_POLY[dataset_name]
        if poly_file.exists():
            poly_df = pd.read_csv(poly_file)
            kept_poly = deduplicate_top_rows(poly_df, TOP_N)
            for rank, (row, feature_list) in enumerate(kept_poly, start=1):
                model = build_polynomial_model()
                model.fit(data[feature_list], data["log_e_minus_c_csds"])
                eq_z, eq_e, poly_terms, poly_coefs = polynomial_equation_from_pipeline(model, feature_list)

                poly_summary_rows.append({
                    "Dataset": dataset_name,
                    "Rank_in_saved_results": rank,
                    "N_Input_Features": len(feature_list),
                    "Input_Features": " + ".join(feature_list),
                    "Equation_log_e_minus_c_scaled_space": eq_z,
                    "Equation_e_scaled_space": eq_e,
                    "Saved_R2_val_z": row.get("R2_val_z", np.nan),
                    "Saved_R2_cv_mean_z": row.get("R2_cv_mean_z", np.nan),
                    "Saved_R2_val_e": row.get("R2_val_e", np.nan),
                    "Saved_RMSE_val_e": row.get("RMSE_val_e", np.nan),
                    "Saved_Selection_Score": row.get("Selection_Score", np.nan),
                })

                for term, coef in zip(poly_terms, poly_coefs):
                    poly_coef_rows.append({
                        "Dataset": dataset_name,
                        "Rank_in_saved_results": rank,
                        "Polynomial_Term": term,
                        "Coefficient_in_scaled_space": coef,
                    })

        pd.DataFrame(exp_summary_rows).to_csv(
            group_dir / f"{slug}_exp_top5_models_log_e_gap_summary.csv",
            index=False,
        )
        pd.DataFrame(exp_coef_rows).to_csv(
            group_dir / f"{slug}_exp_top5_models_log_e_gap_coefficients.csv",
            index=False,
        )
        pd.DataFrame(poly_summary_rows).to_csv(
            group_dir / f"{slug}_poly_top5_models_log_e_gap_summary.csv",
            index=False,
        )
        pd.DataFrame(poly_coef_rows).to_csv(
            group_dir / f"{slug}_poly_top5_models_log_e_gap_coefficients.csv",
            index=False,
        )

        print(f"Top-5 files saved in: {group_dir}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
