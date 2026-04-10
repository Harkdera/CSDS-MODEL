"""Reconstruit et résume les meilleures équations exponentielles et polynomiales sauvegardées."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

try:
    from methodologie_1.common import REGRESSION_DIR, TOP5_DIR, SPLIT_FILES, build_d_dataset, parse_feature_list, deduplicate_top_rows, dataset_group, dataset_slug
except ModuleNotFoundError:
    from common import REGRESSION_DIR, TOP5_DIR, SPLIT_FILES, build_d_dataset, parse_feature_list, deduplicate_top_rows, dataset_group, dataset_slug


# ============================================================
# 1) Chemins des données et des résultats
# ============================================================
REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
TOP5_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = dict(SPLIT_FILES)

RESULT_FILES_EXP = {
    "FULL": {
        "log": REGRESSION_DIR / "exp_regression_log_selection_full.csv",
        "d": REGRESSION_DIR / "exp_regression_d_selection_full.csv",
    },
    "LOW_1": {
        "log": REGRESSION_DIR / "exp_regression_log_selection_low_1.csv",
        "d": REGRESSION_DIR / "exp_regression_d_selection_low_1.csv",
    },
    "LOW_2": {
        "log": REGRESSION_DIR / "exp_regression_log_selection_low_2.csv",
        "d": REGRESSION_DIR / "exp_regression_d_selection_low_2.csv",
    },
    "HIGH": {
        "log": REGRESSION_DIR / "exp_regression_log_selection_high.csv",
        "d": REGRESSION_DIR / "exp_regression_d_selection_high.csv",
    },
}

RESULT_FILES_POLY = {
    "FULL": REGRESSION_DIR / "log_d_csds_genetic_algorithm_diverse_full.csv",
    "LOW_1": REGRESSION_DIR / "log_d_csds_genetic_algorithm_diverse_low_1.csv",
    "LOW_2": REGRESSION_DIR / "log_d_csds_genetic_algorithm_diverse_low_2.csv",
    "HIGH": REGRESSION_DIR / "log_d_csds_genetic_algorithm_diverse_high.csv",
}


# ============================================================
# 2) Paramètres globaux
# ============================================================
TOP_N = 5
RIDGE_ALPHA = 1.0

required_cols = [
    "sigma_n_MPa",
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
    "d_csds",
]


# ============================================================
# 3) Fonctions utilitaires
# ============================================================
def load_and_prepare(dataset_name):
    """Recharge un jeu de données via le helper commun."""
    return build_d_dataset(dataset_name, include_targets=("d_csds",))


# ============================================================
# 4) Aides pour les modèles exponentiels
# ============================================================
def build_exponential_model():
    """Construit le pipeline Ridge utilisé pour la régression exponentielle sur `log(d)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA))
    ])


def convert_scaled_linear_to_raw(model, feature_cols):
    """Convertit les coefficients appris après standardisation vers l'espace brut des variables."""
    scaler = model.named_steps["scaler"]
    reg = model.named_steps["reg"]

    means = scaler.mean_
    scales = scaler.scale_
    coef_scaled = reg.coef_
    intercept_scaled = reg.intercept_

    coef_raw = coef_scaled / scales
    intercept_raw = intercept_scaled - np.sum(coef_scaled * means / scales)

    return intercept_raw, coef_raw


def make_exponential_equations(intercept_raw, coef_raw, feature_cols, digits=10):
    """Construit les équations textuelles de `log(d_csds)` et `d_csds`."""
    pieces = [f"log(d_csds) = {intercept_raw:.{digits}f}"]
    for beta, feat in zip(coef_raw, feature_cols):
        sign = "+" if beta >= 0 else "-"
        pieces.append(f" {sign} {abs(beta):.{digits}f}*{feat}")

    eq_log = "".join(pieces)
    eq_d = "d_csds = exp(" + eq_log.replace("log(d_csds) = ", "") + ")"

    return eq_log, eq_d


# ============================================================
# 5) Aides pour les modèles polynomiaux
# ============================================================
def build_polynomial_model():
    """Construit le pipeline polynomial régularisé utilisé sur `log(d_csds)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg", Ridge(alpha=RIDGE_ALPHA))
    ])


def polynomial_equation_from_pipeline(model, feature_cols, digits=10):
    """Extrait l'équation polynomiale estimée dans l'espace standardisé."""
    scaler = model.named_steps["scaler"]
    poly = model.named_steps["poly"]
    reg = model.named_steps["reg"]

    feature_names = poly.get_feature_names_out(feature_cols)
    coef = reg.coef_
    intercept = reg.intercept_

    # L'équation est écrite dans l'espace standardisé après expansion polynomiale.
    pieces = [f"log(d_csds) = {intercept:.{digits}f}"]
    for c, name in zip(coef, feature_names):
        sign = "+" if c >= 0 else "-"
        pieces.append(f" {sign} {abs(c):.{digits}f}*{name}")

    eq_log_scaled = "".join(pieces)
    eq_d_scaled = "d_csds = exp(" + eq_log_scaled.replace("log(d_csds) = ", "") + ")"

    return eq_log_scaled, eq_d_scaled, feature_names, coef


# ============================================================
# 6) Traitement principal
# ============================================================
for dataset_name in DATA_FILES:
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    data = load_and_prepare(dataset_name)
    dataset_top5_dir = TOP5_DIR / dataset_group(dataset_name)
    dataset_top5_dir.mkdir(parents=True, exist_ok=True)

    summary_rows_exp = []
    coef_rows_exp = []
    summary_rows_poly = []
    coef_rows_poly = []

    # ------------------------------------------------------------
    # A) Rejouer les 5 meilleurs modèles exponentiels
    # ------------------------------------------------------------
    for selection_mode in ["log", "d"]:
        results_file = RESULT_FILES_EXP[dataset_name][selection_mode]

        if not results_file.exists():
            print(f"Missing exponential file: {results_file}")
            continue

        results_df = pd.read_csv(results_file)
        if results_df.empty:
            print(f"No rows in: {results_file}")
            continue

        kept_models = deduplicate_top_rows(results_df, TOP_N)
        print(f"Exponential | {selection_mode} | models kept: {len(kept_models)}")

        for rank, (row, feature_list) in enumerate(kept_models, start=1):
            missing_features = [f for f in feature_list if f not in data.columns]
            if missing_features:
                print(f"Skipped exponential rank {rank}: missing {missing_features}")
                continue

            X = data[feature_list]
            y = data["log_d_csds"]

            model = build_exponential_model()
            model.fit(X, y)

            intercept_raw, coef_raw = convert_scaled_linear_to_raw(model, feature_list)
            eq_log, eq_d = make_exponential_equations(intercept_raw, coef_raw, feature_list)

            summary_rows_exp.append({
                "Dataset": dataset_name,
                "Selection_Mode": selection_mode,
                "Rank_in_saved_results": rank,
                "N_Features": len(feature_list),
                "Features": " + ".join(feature_list),
                "Intercept_log_d": intercept_raw,
                "Equation_log_d": eq_log,
                "Equation_d": eq_d,
                "Saved_R2_val_log": row.get("R2_val_log", np.nan),
                "Saved_R2_cv_mean_log": row.get("R2_cv_mean_log", np.nan),
                "Saved_R2_cv_std_log": row.get("R2_cv_std_log", np.nan),
                "Saved_R2_val_d": row.get("R2_val_d", np.nan),
                "Saved_RMSE_val_d": row.get("RMSE_val_d", np.nan),
                "Saved_Selection_Score": row.get("Selection_Score", np.nan),
            })

            for feat, beta in zip(feature_list, coef_raw):
                coef_rows_exp.append({
                    "Dataset": dataset_name,
                    "Selection_Mode": selection_mode,
                    "Rank_in_saved_results": rank,
                    "Feature": feat,
                    "Coefficient_in_log_d_equation": beta,
                })

    # ------------------------------------------------------------
    # B) Rejouer les 5 meilleurs modèles polynomiaux
    # ------------------------------------------------------------
    poly_file = RESULT_FILES_POLY[dataset_name]

    if poly_file.exists():
        poly_df = pd.read_csv(poly_file)

        if not poly_df.empty:
            kept_poly = deduplicate_top_rows(poly_df, TOP_N)
            print(f"Polynomial | models kept: {len(kept_poly)}")

            for rank, (row, feature_list) in enumerate(kept_poly, start=1):
                missing_features = [f for f in feature_list if f not in data.columns]
                if missing_features:
                    print(f"Skipped polynomial rank {rank}: missing {missing_features}")
                    continue

                X = data[feature_list]
                y = data["log_d_csds"]

                model = build_polynomial_model()
                model.fit(X, y)

                eq_log_scaled, eq_d_scaled, poly_feature_names, poly_coefs = polynomial_equation_from_pipeline(
                    model, feature_list
                )

                summary_rows_poly.append({
                    "Dataset": dataset_name,
                    "Rank_in_saved_results": rank,
                    "N_Input_Features": len(feature_list),
                    "Input_Features": " + ".join(feature_list),
                    "Equation_log_d_scaled_space": eq_log_scaled,
                    "Equation_d_scaled_space": eq_d_scaled,
                    "Saved_R2_val_log": row.get("R2_val_log", np.nan),
                    "Saved_R2_cv_mean_log": row.get("R2_cv_mean_log", np.nan),
                    "Saved_R2_cv_std_log": row.get("R2_cv_std_log", np.nan),
                    "Saved_R2_val_d": row.get("R2_val_d", np.nan),
                    "Saved_RMSE_val_d": row.get("RMSE_val_d", np.nan),
                    "Saved_Selection_Score": row.get("Selection_Score", np.nan),
                })

                for term, coef in zip(poly_feature_names, poly_coefs):
                    coef_rows_poly.append({
                        "Dataset": dataset_name,
                        "Rank_in_saved_results": rank,
                        "Polynomial_Term": term,
                        "Coefficient_in_scaled_space": coef,
                    })
    else:
        print(f"Missing polynomial file: {poly_file}")


    # ------------------------------------------------------------
    # C) Sauvegarder les sorties du dataset courant
    # ------------------------------------------------------------
    summary_exp_df = pd.DataFrame(summary_rows_exp)
    coef_exp_df = pd.DataFrame(coef_rows_exp)
    summary_poly_df = pd.DataFrame(summary_rows_poly)
    coef_poly_df = pd.DataFrame(coef_rows_poly)

    slug = dataset_slug(dataset_name)
    summary_exp_output = dataset_top5_dir / f"{slug}_exp_top5_models_equations_summary.csv"
    coef_exp_output = dataset_top5_dir / f"{slug}_exp_top5_models_coefficients_detail.csv"
    summary_poly_output = dataset_top5_dir / f"{slug}_poly_top5_models_equations_summary.csv"
    coef_poly_output = dataset_top5_dir / f"{slug}_poly_top5_models_coefficients_detail.csv"

    summary_exp_df.to_csv(summary_exp_output, index=False)
    coef_exp_df.to_csv(coef_exp_output, index=False)
    summary_poly_df.to_csv(summary_poly_output, index=False)
    coef_poly_df.to_csv(coef_poly_output, index=False)

    print(f"Saved exponential summary: {summary_exp_output}")
    print(f"Saved exponential coefficients: {coef_exp_output}")
    print(f"Saved polynomial summary: {summary_poly_output}")
    print(f"Saved polynomial coefficients: {coef_poly_output}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Top-5 outputs saved in: {TOP5_DIR}")
