"""Ajuste un modèle polynomial sur `log(e-c)` avec les 5 variables de base.

Le script :
1. construit le dataset FULL de la méthodologie 2 ;
2. limite les prédicteurs aux 5 variables de base ;
3. cherche le meilleur lambda Ridge par validation croisée sur FULL ;
4. réentraîne le meilleur modèle sur tout FULL ;
5. applique ce modèle au fichier `data/test/csds_calibration_test.csv` ;
6. compare les estimations directes obtenues à la calibration itérative CSDS.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methodologie_2.common import (  # noqa: E402
    BASE_PREDICTOR_COLS,
    CV_FOLDS_BY_DATASET,
    EVALUATION_DIR,
    RANDOM_SEED,
    build_direct_e_dataset,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
    ensure_output_dirs,
)


DATASET_NAME = "FULL"
POLY_DEGREE = 2
LAMBDA_GRID = np.logspace(-4, 4, 25)
CALIBRATION_TEST_FILE = PROJECT_ROOT / "data" / "test" / "csds_calibration_test.csv"
OUTPUT_DIR = EVALUATION_DIR / "base5_poly_log_e_gap_calibration_test"
CV_CANDIDATES_FILE = OUTPUT_DIR / "cv_lambda_candidates_full_base5_poly_log_e_gap.csv"
CV_BEST_FILE = OUTPUT_DIR / "cv_best_lambda_full_base5_poly_log_e_gap.csv"
CV_OOF_FILE = OUTPUT_DIR / "cv_oof_predictions_full_base5_poly_log_e_gap.csv"
CALIBRATION_DETAILED_FILE = OUTPUT_DIR / "calibration_test_direct_vs_iterative_detailed.csv"
CALIBRATION_SUMMARY_FILE = OUTPUT_DIR / "calibration_test_direct_vs_iterative_summary.csv"
FINAL_COEFFICIENTS_FILE = OUTPUT_DIR / "final_model_coefficients_full_base5_poly_log_e_gap.csv"


def build_model(alpha: float) -> Pipeline:
    """Construit le pipeline polynomial régularisé."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=float(alpha))),
    ])


def build_calibration_test_dataset() -> pd.DataFrame:
    """Prépare le dataset `csds_calibration_test` pour la comparaison externe."""
    fit_module = import_module("src.04_fit_csds_model")

    raw = pd.read_csv(CALIBRATION_TEST_FILE).copy()
    raw["sample_id"] = np.arange(1, len(raw) + 1)

    numeric_cols = [
        "sigma_n_MPa",
        "delta_peak_mm",
        "tau_peak_MPa_csds",
        "u_r_mm",
        "tau_r_MPa",
        "d",
        "e",
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=numeric_cols).reset_index(drop=True)
    raw = raw[
        (raw["sigma_n_MPa"] > 0) &
        (raw["delta_peak_mm"] > 0) &
        (raw["tau_peak_MPa_csds"] > 0) &
        (raw["u_r_mm"] > 0) &
        (raw["tau_r_MPa"] > 0) &
        (raw["d"] > 0) &
        (raw["e"] > 0)
    ].reset_index(drop=True)

    iterative = raw.apply(
        fit_module.fit_csds_one_row,
        axis=1,
        result_type="expand",
    )

    data = raw.copy()
    data["a_csds"] = pd.to_numeric(iterative["a_csds"], errors="coerce")
    data["b_csds"] = pd.to_numeric(iterative["b_csds"], errors="coerce")
    data["c_target"] = pd.to_numeric(iterative["c_csds"], errors="coerce")
    data["d_csds"] = pd.to_numeric(iterative["d_csds"], errors="coerce")
    data["e_csds"] = pd.to_numeric(iterative["e_csds"], errors="coerce")
    data["csds_converged"] = iterative["csds_converged"].fillna(False).astype(bool)
    data["iterative_iterations"] = pd.to_numeric(iterative["csds_iterations"], errors="coerce")
    data["d_csv_reference"] = pd.to_numeric(raw["d"], errors="coerce")
    data["e_csv_reference"] = pd.to_numeric(raw["e"], errors="coerce")

    data = data.dropna(subset=["a_csds", "b_csds", "c_target", "d_csds", "e_csds"]).reset_index(drop=True)
    data = data[(data["d_csds"] > 0) & (data["e_csds"] > 0)].reset_index(drop=True)
    data["e_minus_c_csds"] = data["e_csds"] - data["c_target"]
    data = data[data["e_minus_c_csds"] > 0].reset_index(drop=True)
    data["log_e_minus_c_csds"] = np.log(data["e_minus_c_csds"])

    return data


def predict_direct_outputs(model: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Prévoit `z`, `e`, `d` et `b` pour un dataframe donné."""
    work = df.copy()
    z_pred = model.predict(work[feature_cols])
    e_pred = work["c_target"].to_numpy(dtype=float) + np.exp(z_pred)
    work["z_pred"] = z_pred
    work["e_pred"] = e_pred
    work["d_pred"] = compute_d_from_e_peak_equation(work, e_col="e_pred")
    work["b_pred"] = work["d_pred"] - work["a_csds"]
    work["e_constraint_respected"] = work["e_pred"] > work["c_target"]
    return work


def summarize_predictions(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    """Calcule les métriques de comparaison pour un tableau de prédictions."""
    work, curve_metrics = compute_curve_metrics_for_direct_prediction(df)
    z_metrics = compute_metrics(work["log_e_minus_c_csds"], work["z_pred"])
    e_metrics = compute_metrics(work["e_csds"], work["e_pred"])
    d_metrics = compute_metrics(work["d_csds"], work["d_pred"])
    b_metrics = compute_metrics(work["b_csds"], work["b_pred"])

    return {
        f"{prefix}_n_rows": int(len(work)),
        f"{prefix}_rmse_z": z_metrics["RMSE"],
        f"{prefix}_r2_z": z_metrics["R2"],
        f"{prefix}_rmse_e": e_metrics["RMSE"],
        f"{prefix}_r2_e": e_metrics["R2"],
        f"{prefix}_rmse_d": d_metrics["RMSE"],
        f"{prefix}_r2_d": d_metrics["R2"],
        f"{prefix}_rmse_b": b_metrics["RMSE"],
        f"{prefix}_r2_b": b_metrics["R2"],
        f"{prefix}_rmse_tau_u": curve_metrics["rmse_tau_u"],
        f"{prefix}_r2_tau_u": curve_metrics["r2_tau_u"],
        f"{prefix}_constraint_ok_count": int(work["e_constraint_respected"].fillna(False).sum()),
        f"{prefix}_valid_curve_count": int(work["curve_valid"].fillna(False).sum()),
    }


def run_cross_validation(full_data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Évalue plusieurs lambdas par validation croisée sur FULL."""
    cv = KFold(
        n_splits=CV_FOLDS_BY_DATASET[DATASET_NAME],
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    candidate_rows: list[dict] = []

    for alpha in LAMBDA_GRID:
        fold_rows: list[dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(full_data), start=1):
            train_df = full_data.iloc[train_idx].reset_index(drop=True)
            val_df = full_data.iloc[val_idx].reset_index(drop=True)

            model = build_model(alpha=float(alpha))
            model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
            fold_pred = predict_direct_outputs(model, val_df, feature_cols)
            fold_metrics = summarize_predictions(fold_pred, prefix="fold")
            fold_rows.append({
                "fold": fold_idx,
                **fold_metrics,
            })

        fold_df = pd.DataFrame(fold_rows)
        candidate_rows.append({
            "lambda": float(alpha),
            "n_features": len(feature_cols),
            "cv_folds": CV_FOLDS_BY_DATASET[DATASET_NAME],
            "mean_rmse_z": float(fold_df["fold_rmse_z"].mean()),
            "std_rmse_z": float(fold_df["fold_rmse_z"].std(ddof=0)),
            "mean_r2_z": float(fold_df["fold_r2_z"].mean()),
            "std_r2_z": float(fold_df["fold_r2_z"].std(ddof=0)),
            "mean_rmse_e": float(fold_df["fold_rmse_e"].mean()),
            "std_rmse_e": float(fold_df["fold_rmse_e"].std(ddof=0)),
            "mean_r2_e": float(fold_df["fold_r2_e"].mean()),
            "mean_rmse_d": float(fold_df["fold_rmse_d"].mean()),
            "mean_r2_d": float(fold_df["fold_r2_d"].mean()),
            "mean_rmse_b": float(fold_df["fold_rmse_b"].mean()),
            "mean_r2_b": float(fold_df["fold_r2_b"].mean()),
            "mean_rmse_tau_u": float(fold_df["fold_rmse_tau_u"].mean()),
            "std_rmse_tau_u": float(fold_df["fold_rmse_tau_u"].std(ddof=0)),
            "mean_r2_tau_u": float(fold_df["fold_r2_tau_u"].mean()),
            "mean_constraint_ok_count": float(fold_df["fold_constraint_ok_count"].mean()),
            "mean_valid_curve_count": float(fold_df["fold_valid_curve_count"].mean()),
        })

    candidates = pd.DataFrame(candidate_rows).sort_values("lambda").reset_index(drop=True)
    best = (
        candidates
        .sort_values(
            by=["mean_rmse_z", "std_rmse_z", "mean_r2_z", "mean_rmse_tau_u", "lambda"],
            ascending=[True, True, False, True, True],
        )
        .head(1)
        .copy()
    )
    best["selection_rule"] = (
        "Best lambda selected on FULL with 5-fold CV using only the 5 base variables: "
        "lowest mean RMSE on z = log(e-c), then lower RMSE std, then higher mean R2 on z, "
        "then lower mean RMSE on tau(u)."
    )
    best_lambda = float(best.iloc[0]["lambda"])

    oof_rows: list[pd.DataFrame] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(full_data), start=1):
        train_df = full_data.iloc[train_idx].reset_index(drop=True)
        val_df = full_data.iloc[val_idx].copy().reset_index(drop=True)
        model = build_model(alpha=best_lambda)
        model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
        fold_pred = predict_direct_outputs(model, val_df, feature_cols)
        fold_pred["cv_fold"] = fold_idx
        oof_rows.append(fold_pred)

    oof_df = pd.concat(oof_rows, ignore_index=True)
    return candidates, best, best_lambda, oof_df


def export_final_coefficients(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Exporte les coefficients du modèle final dans l'espace polynomial standardisé."""
    poly = model.named_steps["poly"]
    reg = model.named_steps["reg"]
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    coef_df = pd.DataFrame({
        "term": poly_feature_names,
        "coefficient": reg.coef_,
    })
    intercept_df = pd.DataFrame({
        "term": ["intercept"],
        "coefficient": [float(reg.intercept_)],
    })
    return pd.concat([intercept_df, coef_df], ignore_index=True)


def main() -> None:
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = list(BASE_PREDICTOR_COLS)
    full_data = build_direct_e_dataset(DATASET_NAME).copy()
    full_data = full_data.dropna(subset=feature_cols + ["log_e_minus_c_csds", "e_csds", "d_csds", "b_csds", "c_target"]).reset_index(drop=True)

    candidates, best, best_lambda, oof_df = run_cross_validation(full_data, feature_cols)
    oof_metrics = summarize_predictions(oof_df, prefix="cv_oof")

    final_model = build_model(alpha=best_lambda)
    final_model.fit(full_data[feature_cols], full_data["log_e_minus_c_csds"])

    calibration_data = build_calibration_test_dataset()
    calibration_pred = predict_direct_outputs(final_model, calibration_data, feature_cols)
    calibration_metrics = summarize_predictions(calibration_pred, prefix="calibration_test")

    comparison_df = calibration_pred.copy()
    comparison_df["z_reference"] = comparison_df["log_e_minus_c_csds"]
    comparison_df["e_reference_iterative"] = comparison_df["e_csds"]
    comparison_df["d_reference_iterative"] = comparison_df["d_csds"]
    comparison_df["b_reference_iterative"] = comparison_df["b_csds"]
    comparison_df["e_error_vs_iterative"] = comparison_df["e_pred"] - comparison_df["e_csds"]
    comparison_df["d_error_vs_iterative"] = comparison_df["d_pred"] - comparison_df["d_csds"]
    comparison_df["b_error_vs_iterative"] = comparison_df["b_pred"] - comparison_df["b_csds"]

    summary_row = {
        "dataset_train": DATASET_NAME,
        "dataset_test": "csds_calibration_test",
        "model_family": "polynomial",
        "target_mode": "log_e_minus_c",
        "feature_set": " + ".join(feature_cols),
        "n_features": len(feature_cols),
        "degree": POLY_DEGREE,
        "best_lambda": best_lambda,
        **best.iloc[0].to_dict(),
        **oof_metrics,
        **calibration_metrics,
    }

    candidates.to_csv(CV_CANDIDATES_FILE, index=False)
    best.to_csv(CV_BEST_FILE, index=False)
    oof_df.to_csv(CV_OOF_FILE, index=False)
    comparison_df.to_csv(CALIBRATION_DETAILED_FILE, index=False)
    pd.DataFrame([summary_row]).to_csv(CALIBRATION_SUMMARY_FILE, index=False)
    export_final_coefficients(final_model, feature_cols).to_csv(FINAL_COEFFICIENTS_FILE, index=False)

    print("=" * 100)
    print("BASE-5 POLYNOMIAL MODEL ON log(e-c)")
    print("=" * 100)
    print(f"Training dataset: {DATASET_NAME}")
    print(f"Base features: {feature_cols}")
    print(f"Rows in FULL: {len(full_data)}")
    print(f"Best lambda: {best_lambda:.8f}")
    print(f"CV mean RMSE(z): {best.iloc[0]['mean_rmse_z']:.6f}")
    print(f"CV mean R2(z): {best.iloc[0]['mean_r2_z']:.6f}")
    print(f"CV mean RMSE(tau(u)): {best.iloc[0]['mean_rmse_tau_u']:.6f}")
    print("-" * 100)
    print(f"Calibration-test rows: {len(calibration_data)}")
    print(f"RMSE(e) vs iterative: {calibration_metrics['calibration_test_rmse_e']:.6f}")
    print(f"R2(e) vs iterative:   {calibration_metrics['calibration_test_r2_e']:.6f}")
    print(f"RMSE(d) vs iterative: {calibration_metrics['calibration_test_rmse_d']:.6f}")
    print(f"R2(d) vs iterative:   {calibration_metrics['calibration_test_r2_d']:.6f}")
    print(f"RMSE(b) vs iterative: {calibration_metrics['calibration_test_rmse_b']:.6f}")
    print(f"R2(b) vs iterative:   {calibration_metrics['calibration_test_r2_b']:.6f}")
    print(f"RMSE(tau(u)) vs iterative: {calibration_metrics['calibration_test_rmse_tau_u']:.6f}")
    print(f"R2(tau(u)) vs iterative:   {calibration_metrics['calibration_test_r2_tau_u']:.6f}")
    print(f"Constraint e>c respected: {calibration_metrics['calibration_test_constraint_ok_count']}/{len(calibration_data)}")
    print("-" * 100)
    print(f"Lambda candidates saved: {CV_CANDIDATES_FILE}")
    print(f"Best lambda saved: {CV_BEST_FILE}")
    print(f"OOF CV predictions saved: {CV_OOF_FILE}")
    print(f"Calibration detailed comparison saved: {CALIBRATION_DETAILED_FILE}")
    print(f"Calibration summary saved: {CALIBRATION_SUMMARY_FILE}")
    print(f"Final model coefficients saved: {FINAL_COEFFICIENTS_FILE}")


if __name__ == "__main__":
    main()
