"""Compare trois modèles polynomiaux sur `log(e-c)` avec test final sur `csds_calibration_test`.

Modèles comparés :
1. variables de base seulement ;
2. variables de base + variables combinées ;
3. meilleur modèle polynomial FULL déjà retenu.
"""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import sys

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methodologie_2.common import (  # noqa: E402
    ALLOWED_PREDICTOR_COLS,
    BASE_PREDICTOR_COLS,
    CV_FOLDS_BY_DATASET,
    EVALUATION_DIR,
    RANDOM_SEED,
    add_engineered_features,
    build_direct_e_dataset,
    csds_tau,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
    ensure_output_dirs,
    make_u_grid,
    parse_feature_list,
)


DATASET_NAME = "FULL"
POLY_DEGREE = 2
LAMBDA_GRID = np.logspace(-4, 4, 25)
CALIBRATION_TEST_FILE = PROJECT_ROOT / "data" / "test" / "csds_calibration_test.csv"
BEST_FULL_TOP5_FILE = PROJECT_ROOT / "results" / "methodologie_2" / "regressions" / "top5" / "full" / "full_poly_top5_models_log_e_gap_summary.csv"

OUTPUT_DIR = EVALUATION_DIR / "model_complexity_comparison_calibration_test"
CV_CANDIDATES_FILE = OUTPUT_DIR / "cv_lambda_candidates_model_complexity_comparison.csv"
CV_BEST_FILE = OUTPUT_DIR / "cv_best_lambda_model_complexity_comparison.csv"
CV_OOF_FILE = OUTPUT_DIR / "cv_oof_predictions_model_complexity_comparison.csv"
CALIBRATION_DETAILED_FILE = OUTPUT_DIR / "calibration_test_direct_vs_iterative_by_model.csv"
CALIBRATION_SUMMARY_FILE = OUTPUT_DIR / "calibration_test_direct_vs_iterative_summary_by_model.csv"
MODEL_DEFINITION_FILE = OUTPUT_DIR / "model_definitions_model_complexity_comparison.csv"
COEFFICIENTS_FILE = OUTPUT_DIR / "model_coefficients_model_complexity_comparison.csv"
CURVE_PLOTS_DIR = OUTPUT_DIR / "curve_plots_by_sample"
CURVE_PLOT_INDEX_FILE = OUTPUT_DIR / "curve_plot_index.csv"


def build_model(alpha: float) -> Pipeline:
    """Construit le pipeline polynomial Ridge."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=float(alpha))),
    ])


def load_model_specs(full_data: pd.DataFrame) -> list[dict]:
    """Construit les trois définitions de modèles à comparer."""
    all_features = [
        col for col in ALLOWED_PREDICTOR_COLS
        if col in full_data.columns and pd.api.types.is_numeric_dtype(full_data[col])
    ]

    top5_df = pd.read_csv(BEST_FULL_TOP5_FILE)
    best_row = top5_df.iloc[0]
    best_features = parse_feature_list(best_row["Input_Features"])

    return [
        {
            "model_id": "base5_only",
            "model_label": "Base 5 variables",
            "feature_cols": list(BASE_PREDICTOR_COLS),
            "feature_origin": "5 variables de base uniquement",
        },
        {
            "model_id": "base5_plus_engineered",
            "model_label": "Base 5 + variables combinees",
            "feature_cols": all_features,
            "feature_origin": "toutes les variables candidates autorisées",
        },
        {
            "model_id": "best_full_retained",
            "model_label": "Meilleur modele FULL retenu",
            "feature_cols": best_features,
            "feature_origin": f"top-1 FULL retenu ({best_row['Rank_in_saved_results']})",
        },
    ]


def build_calibration_test_dataset() -> pd.DataFrame:
    """Prépare `csds_calibration_test` avec référence itérative et variables dérivées."""
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
    data["log_d_csds"] = np.log(data["d_csds"])

    data = add_engineered_features(data)
    data = data.replace([np.inf, -np.inf], np.nan)
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


def summarize_predictions(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcule les métriques sur `z`, `e`, `d`, `b` et `tau(u)`."""
    work, curve_metrics = compute_curve_metrics_for_direct_prediction(df)
    z_metrics = compute_metrics(work["log_e_minus_c_csds"], work["z_pred"])
    e_metrics = compute_metrics(work["e_csds"], work["e_pred"])
    d_metrics = compute_metrics(work["d_csds"], work["d_pred"])
    b_metrics = compute_metrics(work["b_csds"], work["b_pred"])

    metrics = {
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
    return work, metrics


def run_cross_validation(full_data: pd.DataFrame, feature_cols: list[str], model_id: str) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    """Évalue plusieurs lambdas par validation croisée sur FULL pour un modèle donné."""
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
            _, fold_metrics = summarize_predictions(fold_pred, prefix="fold")
            fold_rows.append({
                "fold": fold_idx,
                **fold_metrics,
            })

        fold_df = pd.DataFrame(fold_rows)
        candidate_rows.append({
            "model_id": model_id,
            "lambda": float(alpha),
            "n_features": len(feature_cols),
            "cv_folds": CV_FOLDS_BY_DATASET[DATASET_NAME],
            "mean_rmse_z": float(fold_df["fold_rmse_z"].mean()),
            "std_rmse_z": float(fold_df["fold_rmse_z"].std(ddof=0)),
            "mean_r2_z": float(fold_df["fold_r2_z"].mean()),
            "std_r2_z": float(fold_df["fold_r2_z"].std(ddof=0)),
            "mean_rmse_e": float(fold_df["fold_rmse_e"].mean()),
            "mean_r2_e": float(fold_df["fold_r2_e"].mean()),
            "mean_rmse_d": float(fold_df["fold_rmse_d"].mean()),
            "mean_r2_d": float(fold_df["fold_r2_d"].mean()),
            "mean_rmse_b": float(fold_df["fold_rmse_b"].mean()),
            "mean_r2_b": float(fold_df["fold_r2_b"].mean()),
            "mean_rmse_tau_u": float(fold_df["fold_rmse_tau_u"].mean()),
            "std_rmse_tau_u": float(fold_df["fold_rmse_tau_u"].std(ddof=0)),
            "mean_r2_tau_u": float(fold_df["fold_r2_tau_u"].mean()),
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
    best_lambda = float(best.iloc[0]["lambda"])
    best["selection_rule"] = (
        "Best lambda selected on FULL with 5-fold CV: lowest mean RMSE on z = log(e-c), "
        "then lower RMSE std, then higher mean R2 on z, then lower mean RMSE on tau(u)."
    )

    oof_rows: list[pd.DataFrame] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(full_data), start=1):
        train_df = full_data.iloc[train_idx].reset_index(drop=True)
        val_df = full_data.iloc[val_idx].copy().reset_index(drop=True)
        model = build_model(alpha=best_lambda)
        model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
        fold_pred = predict_direct_outputs(model, val_df, feature_cols)
        fold_pred["cv_fold"] = fold_idx
        fold_pred["model_id"] = model_id
        oof_rows.append(fold_pred)

    oof_df = pd.concat(oof_rows, ignore_index=True)
    return candidates, best, best_lambda, oof_df


def export_final_coefficients(model: Pipeline, feature_cols: list[str], model_id: str, model_label: str) -> pd.DataFrame:
    """Exporte les coefficients du modèle final."""
    poly = model.named_steps["poly"]
    reg = model.named_steps["reg"]
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    coef_df = pd.DataFrame({
        "model_id": model_id,
        "model_label": model_label,
        "term": poly_feature_names,
        "coefficient": reg.coef_,
    })
    intercept_df = pd.DataFrame({
        "model_id": [model_id],
        "model_label": [model_label],
        "term": ["intercept"],
        "coefficient": [float(reg.intercept_)],
    })
    return pd.concat([intercept_df, coef_df], ignore_index=True)


def safe_slug(text: str | int) -> str:
    """Produit un nom de fichier stable."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def plot_curves_by_sample(calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Trace une figure par échantillon avec la courbe itérative et les trois modèles."""
    CURVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rows: list[dict] = []

    for sample_id, sample_rows in calibration_df.groupby("sample_id", sort=True):
        first_row = sample_rows.iloc[0]
        u = make_u_grid(first_row)
        tau_true = csds_tau(
            u=u,
            a=float(first_row["a_csds"]),
            b=float(first_row["b_csds"]),
            c=float(first_row["c_target"]),
            d=float(first_row["d_csds"]),
            e=float(first_row["e_csds"]),
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(u, tau_true, color="black", linewidth=2.5, label="Courbe itérative de référence")
        ax.scatter([0.0], [0.0], color="black", s=35, marker="o", label="Origine")
        ax.scatter([float(first_row["delta_peak_mm"])], [float(first_row["tau_peak_MPa_csds"])], color="black", s=70, marker="x", label="Point de pic")
        ax.scatter([float(first_row["u_r_mm"])], [float(first_row["tau_r_MPa"])], color="black", s=55, marker="s", label="Point résiduel")

        for _, row in sample_rows.sort_values("model_label").iterrows():
            tau_pred = csds_tau(
                u=u,
                a=float(row["a_csds"]),
                b=float(row["b_pred"]),
                c=float(row["c_target"]),
                d=float(row["d_pred"]),
                e=float(row["e_pred"]),
            )
            label = (
                f"{row['model_label']} | "
                f"RMSE τ(u)={row['curve_rmse_tau_u']:.4f} | "
                f"RMSE(e)={abs(row['e_error_vs_iterative']):.4f}"
            )
            ax.plot(u, tau_pred, linestyle="--", linewidth=1.8, label=label)

        ax.set_title(f"Calibration test | sample_id={sample_id} | comparaison des courbes")
        ax.set_xlabel("u")
        ax.set_ylabel("τ(u)")
        ax.grid(True, alpha=0.30)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        output_file = CURVE_PLOTS_DIR / f"sample_{safe_slug(sample_id)}_curve_comparison.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        plot_rows.append({
            "sample_id": sample_id,
            "plot_file": str(output_file),
            "n_models_shown": int(sample_rows["model_id"].nunique()),
        })

    return pd.DataFrame(plot_rows)


def main() -> None:
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_data = build_direct_e_dataset(DATASET_NAME).copy()
    calibration_data = build_calibration_test_dataset()
    model_specs = load_model_specs(full_data)

    cv_candidate_frames: list[pd.DataFrame] = []
    cv_best_rows: list[pd.DataFrame] = []
    cv_oof_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    coefficient_frames: list[pd.DataFrame] = []
    definition_rows: list[dict] = []

    print("=" * 100)
    print("MODEL COMPLEXITY COMPARISON ON CALIBRATION TEST")
    print("=" * 100)

    for spec in model_specs:
        feature_cols = [col for col in spec["feature_cols"] if col in full_data.columns and col in calibration_data.columns]
        if not feature_cols:
            raise ValueError(f"No usable features found for {spec['model_id']}.")

        train_df = full_data.dropna(subset=feature_cols + ["log_e_minus_c_csds", "e_csds", "d_csds", "b_csds", "c_target"]).reset_index(drop=True)
        test_df = calibration_data.dropna(subset=feature_cols + ["log_e_minus_c_csds", "e_csds", "d_csds", "b_csds", "c_target"]).reset_index(drop=True)

        definition_rows.append({
            "model_id": spec["model_id"],
            "model_label": spec["model_label"],
            "feature_origin": spec["feature_origin"],
            "n_features": len(feature_cols),
            "feature_cols": " + ".join(feature_cols),
        })

        candidates, best, best_lambda, oof_df = run_cross_validation(train_df, feature_cols, spec["model_id"])
        _, oof_metrics = summarize_predictions(oof_df, prefix="cv_oof")

        final_model = build_model(alpha=best_lambda)
        final_model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
        calibration_pred = predict_direct_outputs(final_model, test_df, feature_cols)
        calibration_pred["model_id"] = spec["model_id"]
        calibration_pred["model_label"] = spec["model_label"]
        calibration_pred["e_error_vs_iterative"] = calibration_pred["e_pred"] - calibration_pred["e_csds"]
        calibration_pred["d_error_vs_iterative"] = calibration_pred["d_pred"] - calibration_pred["d_csds"]
        calibration_pred["b_error_vs_iterative"] = calibration_pred["b_pred"] - calibration_pred["b_csds"]
        calibration_work, calibration_metrics = summarize_predictions(calibration_pred, prefix="calibration_test")

        cv_candidate_frames.append(candidates)
        cv_best_rows.append(best.assign(model_id=spec["model_id"], model_label=spec["model_label"]))
        cv_oof_frames.append(oof_df.assign(model_label=spec["model_label"]))
        calibration_frames.append(calibration_work)
        coefficient_frames.append(export_final_coefficients(final_model, feature_cols, spec["model_id"], spec["model_label"]))

        summary_rows.append({
            "model_id": spec["model_id"],
            "model_label": spec["model_label"],
            "feature_origin": spec["feature_origin"],
            "feature_cols": " + ".join(feature_cols),
            "n_features": len(feature_cols),
            "best_lambda": best_lambda,
            **best.iloc[0].to_dict(),
            **oof_metrics,
            **calibration_metrics,
        })

        print("-" * 100)
        print(f"Model: {spec['model_label']}")
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        print(f"Best lambda: {best_lambda:.8f}")
        print(f"CV mean RMSE(z): {best.iloc[0]['mean_rmse_z']:.6f}")
        print(f"CV mean RMSE(tau(u)): {best.iloc[0]['mean_rmse_tau_u']:.6f}")
        print(f"Calibration RMSE(e): {calibration_metrics['calibration_test_rmse_e']:.6f}")
        print(f"Calibration RMSE(d): {calibration_metrics['calibration_test_rmse_d']:.6f}")
        print(f"Calibration RMSE(tau(u)): {calibration_metrics['calibration_test_rmse_tau_u']:.6f}")

    pd.concat(cv_candidate_frames, ignore_index=True).to_csv(CV_CANDIDATES_FILE, index=False)
    pd.concat(cv_best_rows, ignore_index=True).to_csv(CV_BEST_FILE, index=False)
    pd.concat(cv_oof_frames, ignore_index=True).to_csv(CV_OOF_FILE, index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(CALIBRATION_DETAILED_FILE, index=False)
    pd.DataFrame(summary_rows).sort_values(
        by=["calibration_test_rmse_tau_u", "calibration_test_rmse_e", "n_features"],
        ascending=[True, True, True],
    ).to_csv(CALIBRATION_SUMMARY_FILE, index=False)
    pd.DataFrame(definition_rows).to_csv(MODEL_DEFINITION_FILE, index=False)
    pd.concat(coefficient_frames, ignore_index=True).to_csv(COEFFICIENTS_FILE, index=False)
    calibration_all_df = pd.concat(calibration_frames, ignore_index=True)
    curve_plot_index = plot_curves_by_sample(calibration_all_df)
    curve_plot_index.to_csv(CURVE_PLOT_INDEX_FILE, index=False)

    print("=" * 100)
    print(f"CV candidates saved: {CV_CANDIDATES_FILE}")
    print(f"CV best lambda saved: {CV_BEST_FILE}")
    print(f"CV OOF predictions saved: {CV_OOF_FILE}")
    print(f"Calibration detailed comparison saved: {CALIBRATION_DETAILED_FILE}")
    print(f"Calibration summary saved: {CALIBRATION_SUMMARY_FILE}")
    print(f"Model definitions saved: {MODEL_DEFINITION_FILE}")
    print(f"Model coefficients saved: {COEFFICIENTS_FILE}")
    print(f"Curve plots saved in: {CURVE_PLOTS_DIR}")
    print(f"Curve plot index saved: {CURVE_PLOT_INDEX_FILE}")


if __name__ == "__main__":
    main()
