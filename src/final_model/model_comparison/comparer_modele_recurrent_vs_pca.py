"""Compare un modèle Top-5 récurrent régularisé au modèle final PCA.

Le script :
1. récupère les 5 variables les plus récurrentes des meilleurs modèles polynomiaux FULL ;
2. construit le même split que les méthodologies 1 et 2 : train/validation/test = 64/16/20 ;
3. ajuste les deux modèles sur `z = log(e-c)` avec validation croisée sur train ;
4. sélectionne le meilleur lambda sur validation ;
5. évalue les deux modèles sur le test interne issu de FULL ;
6. réentraîne les deux modèles sur FULL complet ;
7. compare leurs prédictions au résultat de la calibration itérative sur `csds_calibration_test` ;
8. sauvegarde des tableaux comparatifs et des courbes par échantillon.
"""

from __future__ import annotations

from importlib import import_module
import os
from src.final_model.final_model_utils_methodologie_2 import load_full_dataset, load_top10_features
from src.utils.paths import FINAL_LAMBDA_SEARCH_DIR, FINAL_MODEL_COMPARISON_DIR, MATPLOTLIB_CACHE_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.utils.common_methodologie_2 import (
    CV_FOLDS_BY_DATASET,
    RANDOM_SEED,
    add_engineered_features,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
    csds_tau,
    ensure_output_dirs,
    make_u_grid,
)


DATASET_NAME = "FULL"
POLY_DEGREE = 2
N_COMPONENTS = 5
LAMBDA_GRID = np.logspace(-4, 4, 25)
CALIBRATION_TEST_FILE = PROJECT_ROOT / "data" / "test" / "csds_calibration_test.csv"
BEST_PCA_LAMBDA_FILE = FINAL_LAMBDA_SEARCH_DIR / "lambda_best_top5_pca_full.csv"

OUTPUT_DIR = FINAL_MODEL_COMPARISON_DIR / "top5_recurrent_vs_pca_calibration_test"
TOP5_FEATURES_FILE = OUTPUT_DIR / "top5_recurrent_features.csv"
LAMBDA_CANDIDATES_FILE = OUTPUT_DIR / "lambda_candidates_top5_recurrent_vs_pca.csv"
BEST_LAMBDA_FILE = OUTPUT_DIR / "best_lambda_top5_recurrent_vs_pca.csv"
CV_COMPARE_FOLDS_FILE = OUTPUT_DIR / "cv_train_fold_metrics_top5_recurrent_vs_pca.csv"
CV_COMPARE_SUMMARY_FILE = OUTPUT_DIR / "cv_train_summary_top5_recurrent_vs_pca.csv"
CV_COMPARE_OOF_FILE = OUTPUT_DIR / "cv_train_oof_predictions_top5_recurrent_vs_pca.csv"
SPLIT_TEST_DETAILED_FILE = OUTPUT_DIR / "full_split_test_top5_recurrent_vs_pca_detailed.csv"
SPLIT_TEST_SUMMARY_FILE = OUTPUT_DIR / "full_split_test_top5_recurrent_vs_pca_summary.csv"
CALIBRATION_DETAILED_FILE = OUTPUT_DIR / "calibration_test_top5_recurrent_vs_pca_detailed.csv"
CALIBRATION_SUMMARY_FILE = OUTPUT_DIR / "calibration_test_top5_recurrent_vs_pca_summary.csv"
MODEL_DEFINITIONS_FILE = OUTPUT_DIR / "model_definitions_top5_recurrent_vs_pca.csv"
TOP5_COEFFICIENTS_FILE = OUTPUT_DIR / "top5_recurrent_model_coefficients.csv"
SPLIT_TEST_CURVE_PLOTS_DIR = OUTPUT_DIR / "curve_plots_full_split_test"
SPLIT_TEST_CURVE_PLOT_INDEX_FILE = OUTPUT_DIR / "curve_plot_index_full_split_test.csv"
CALIBRATION_CURVE_PLOTS_DIR = OUTPUT_DIR / "curve_plots_calibration_test"
CALIBRATION_CURVE_PLOT_INDEX_FILE = OUTPUT_DIR / "curve_plot_index_calibration_test.csv"


def build_top5_model(alpha: float) -> Pipeline:
    """Construit le pipeline polynomial Ridge pour les 5 variables récurrentes."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=float(alpha))),
    ])


def build_pca_model(alpha: float) -> Pipeline:
    """Construit le pipeline PCA + polynôme + Ridge."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS)),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=float(alpha))),
    ])


def safe_slug(text: str | int) -> str:
    """Produit un nom de fichier stable."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def load_best_pca_lambda() -> float:
    """Charge le meilleur lambda du modèle PCA déjà sélectionné sur FULL."""
    if not BEST_PCA_LAMBDA_FILE.exists():
        raise FileNotFoundError(
            f"Best-lambda file not found: {BEST_PCA_LAMBDA_FILE}. "
            "Run 22_search_best_lambda_top5_pca_full.py first."
        )
    best_df = pd.read_csv(BEST_PCA_LAMBDA_FILE)
    if best_df.empty:
        raise ValueError(f"Best-lambda file is empty: {BEST_PCA_LAMBDA_FILE}")
    return float(best_df.iloc[0]["lambda"])


def build_top5_recurrent_features() -> list[str]:
    """Retourne les 5 variables les plus récurrentes dans FULL pour la méthodologie 2."""
    top5_df = load_top10_features(top_n=5).copy()
    top5_df.to_csv(TOP5_FEATURES_FILE, index=False)
    return [str(feature) for feature in top5_df["feature"].tolist()]


def build_pca_feature_list() -> list[str]:
    """Retourne les 10 variables utilisées pour le modèle PCA final."""
    top10_df = load_top10_features().copy()
    return [str(feature) for feature in top10_df["feature"].tolist()]


def build_training_dataset(required_feature_cols: list[str]) -> pd.DataFrame:
    """Construit le dataset FULL en imposant les colonnes requises pour les deux modèles."""
    data = load_full_dataset().copy()
    required_cols = required_feature_cols + [
        "sample_id",
        "log_e_minus_c_csds",
        "e_minus_c_csds",
        "e_csds",
        "d_csds",
        "b_csds",
        "a_csds",
        "c_target",
        "delta_peak_mm",
        "u_r_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
    ]
    return data.dropna(subset=required_cols).reset_index(drop=True)


def build_calibration_test_dataset(required_feature_cols: list[str]) -> pd.DataFrame:
    """Prépare `csds_calibration_test` avec référence itérative et variables dérivées."""
    fit_module = import_module("src.csds_implementation.calibration.calibrer_modele_csds")

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

    data = add_engineered_features(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required_feature_cols).reset_index(drop=True)
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


def split_full_dataset(full_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applique le split 64/16/20 utilisé dans les méthodologies 1 et 2."""
    train_val_df, test_df = train_test_split(
        full_data,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def search_best_lambda_on_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_builder,
    model_id: str,
    model_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Cherche le meilleur lambda avec CV sur train et sélection sur validation."""
    cv = KFold(
        n_splits=CV_FOLDS_BY_DATASET[DATASET_NAME],
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    candidate_rows: list[dict] = []
    all_cv_fold_rows: list[pd.DataFrame] = []
    best_oof_df = pd.DataFrame()

    for alpha in LAMBDA_GRID:
        fold_rows: list[dict] = []
        oof_rows: list[pd.DataFrame] = []

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(cv.split(train_df), start=1):
            fold_train_df = train_df.iloc[fold_train_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[fold_val_idx].copy().reset_index(drop=True)

            model = model_builder(alpha=float(alpha))
            model.fit(fold_train_df[feature_cols], fold_train_df["log_e_minus_c_csds"])
            fold_pred = predict_direct_outputs(model, fold_val_df, feature_cols)
            fold_work, fold_metrics = summarize_predictions(fold_pred, prefix="fold")
            fold_rows.append({
                "model_id": model_id,
                "model_label": model_label,
                "lambda": float(alpha),
                "fold": fold_idx,
                "n_features_before_model": len(feature_cols),
                **fold_metrics,
            })
            fold_work["cv_fold"] = fold_idx
            fold_work["lambda"] = float(alpha)
            fold_work["model_id"] = model_id
            fold_work["model_label"] = model_label
            oof_rows.append(fold_work)

        fold_df = pd.DataFrame(fold_rows)
        all_cv_fold_rows.append(fold_df)

        model = model_builder(alpha=float(alpha))
        model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
        val_pred = predict_direct_outputs(model, val_df, feature_cols)
        _, val_metrics = summarize_predictions(val_pred, prefix="val")
        test_pred = predict_direct_outputs(model, test_df, feature_cols)
        _, test_metrics = summarize_predictions(test_pred, prefix="test")

        candidate_rows.append({
            "model_id": model_id,
            "model_label": model_label,
            "lambda": float(alpha),
            "n_features": len(feature_cols),
            "cv_folds": CV_FOLDS_BY_DATASET[DATASET_NAME],
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "cv_rmse_mean_z": float(fold_df["fold_rmse_z"].mean()),
            "cv_rmse_std_z": float(fold_df["fold_rmse_z"].std(ddof=0)),
            "cv_r2_mean_z": float(fold_df["fold_r2_z"].mean()),
            "cv_r2_std_z": float(fold_df["fold_r2_z"].std(ddof=0)),
            "cv_rmse_mean_tau_u": float(fold_df["fold_rmse_tau_u"].mean()),
            "cv_rmse_std_tau_u": float(fold_df["fold_rmse_tau_u"].std(ddof=0)),
            "cv_r2_mean_tau_u": float(fold_df["fold_r2_tau_u"].mean()),
            "cv_r2_std_tau_u": float(fold_df["fold_r2_tau_u"].std(ddof=0)),
            **val_metrics,
            **test_metrics,
        })

        if best_oof_df.empty:
            best_oof_df = pd.concat(oof_rows, ignore_index=True)

    candidates = pd.DataFrame(candidate_rows)
    best = (
        candidates
        .sort_values(
            by=["val_rmse_z", "cv_rmse_std_z", "cv_r2_mean_z", "val_rmse_tau_u", "lambda"],
            ascending=[True, True, False, True, True],
        )
        .head(1)
        .copy()
    )
    best["selection_rule"] = (
        "Best lambda selected with the same train/validation/test logic as methodologies 1 and 2: "
        "lowest RMSE on validation for z = log(e-c), then lower CV std on training folds, "
        "then higher mean CV R2, then lower validation RMSE on tau(u)."
    )
    best_lambda = float(best.iloc[0]["lambda"])

    best_oof_rows: list[pd.DataFrame] = []
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(cv.split(train_df), start=1):
        fold_train_df = train_df.iloc[fold_train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[fold_val_idx].copy().reset_index(drop=True)
        model = model_builder(alpha=best_lambda)
        model.fit(fold_train_df[feature_cols], fold_train_df["log_e_minus_c_csds"])
        fold_pred = predict_direct_outputs(model, fold_val_df, feature_cols)
        fold_pred["cv_fold"] = fold_idx
        fold_pred["lambda"] = best_lambda
        fold_pred["model_id"] = model_id
        fold_pred["model_label"] = model_label
        best_oof_rows.append(fold_pred)

    best_model = model_builder(alpha=best_lambda)
    best_model.fit(train_df[feature_cols], train_df["log_e_minus_c_csds"])
    val_pred = predict_direct_outputs(best_model, val_df, feature_cols)
    val_work, val_metrics = summarize_predictions(val_pred, prefix="val")
    val_work["model_id"] = model_id
    val_work["model_label"] = model_label

    test_pred = predict_direct_outputs(best_model, test_df, feature_cols)
    test_work, test_metrics = summarize_predictions(test_pred, prefix="test")
    test_work["model_id"] = model_id
    test_work["model_label"] = model_label

    cv_fold_df = pd.concat(all_cv_fold_rows, ignore_index=True)
    cv_fold_df = cv_fold_df[cv_fold_df["lambda"] == best_lambda].reset_index(drop=True)
    best_oof_df = pd.concat(best_oof_rows, ignore_index=True)

    cv_summary = {
        "model_id": model_id,
        "model_label": model_label,
        "best_lambda": best_lambda,
        "n_features_before_model": len(feature_cols),
        "cv_folds": CV_FOLDS_BY_DATASET[DATASET_NAME],
        "cv_rmse_mean_z": float(cv_fold_df["fold_rmse_z"].mean()),
        "cv_rmse_std_z": float(cv_fold_df["fold_rmse_z"].std(ddof=0)),
        "cv_r2_mean_z": float(cv_fold_df["fold_r2_z"].mean()),
        "cv_r2_std_z": float(cv_fold_df["fold_r2_z"].std(ddof=0)),
        "cv_rmse_mean_tau_u": float(cv_fold_df["fold_rmse_tau_u"].mean()),
        "cv_rmse_std_tau_u": float(cv_fold_df["fold_rmse_tau_u"].std(ddof=0)),
        "cv_r2_mean_tau_u": float(cv_fold_df["fold_r2_tau_u"].mean()),
        "cv_r2_std_tau_u": float(cv_fold_df["fold_r2_tau_u"].std(ddof=0)),
    }
    split_summary = {
        "model_id": model_id,
        "model_label": model_label,
        "best_lambda": best_lambda,
        "n_features_before_model": len(feature_cols),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        **val_metrics,
        **test_metrics,
    }
    return candidates.sort_values(["model_id", "lambda"]).reset_index(drop=True), best, best_lambda, cv_fold_df, best_oof_df, cv_summary, val_work, test_work, split_summary


def export_top5_coefficients(model: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Exporte les coefficients du modèle final Top-5 dans l'espace polynomial standardisé."""
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


def plot_curves_by_sample(
    prediction_df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str,
    file_suffix: str,
) -> pd.DataFrame:
    """Trace une figure par échantillon avec la courbe itérative et les deux modèles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rows: list[dict] = []

    color_map = {
        "Top 5 recurrentes + Ridge": "#1f77b4",
        "PCA 5 composantes": "#d62728",
    }

    for sample_id, sample_rows in prediction_df.groupby("sample_id", sort=True):
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
                f"R² τ(u)={row['curve_r2_tau_u']:.4f}"
            )
            ax.plot(
                u,
                tau_pred,
                linestyle="--",
                linewidth=2.0,
                color=color_map.get(str(row["model_label"]), None),
                label=label,
            )

        ax.set_title(f"{title_prefix} | sample_id={sample_id} | Top 5 recurrentes vs PCA")
        ax.set_xlabel("u")
        ax.set_ylabel("τ(u)")
        ax.grid(True, alpha=0.30)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        output_file = output_dir / f"sample_{safe_slug(sample_id)}_{file_suffix}.png"
        fig.tight_layout()
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        plot_rows.append({
            "sample_id": sample_id,
            "plot_file": str(output_file),
        })

    return pd.DataFrame(plot_rows)


def main() -> None:
    ensure_output_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    top5_features = build_top5_recurrent_features()
    pca_features = build_pca_feature_list()
    required_feature_cols = sorted(set(top5_features) | set(pca_features))

    full_data = build_training_dataset(required_feature_cols)
    train_df, val_df, test_df = split_full_dataset(full_data)
    calibration_data = build_calibration_test_dataset(required_feature_cols)

    (
        top5_candidates_df,
        top5_best_df,
        top5_lambda,
        top5_cv_folds_df,
        top5_cv_oof_df,
        top5_cv_summary,
        top5_val_work,
        top5_test_work,
        top5_split_summary,
    ) = search_best_lambda_on_split(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=top5_features,
        model_id="top5_recurrent_ridge",
        model_label="Top 5 recurrentes + Ridge",
        model_builder=build_top5_model,
    )

    (
        pca_candidates_df,
        pca_best_df,
        pca_lambda,
        pca_cv_folds_df,
        pca_cv_oof_df,
        pca_cv_summary,
        pca_val_work,
        pca_test_work,
        pca_split_summary,
    ) = search_best_lambda_on_split(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=pca_features,
        model_id="top5_pca_full",
        model_label="PCA 5 composantes",
        model_builder=build_pca_model,
    )

    lambda_candidates_df = pd.concat([top5_candidates_df, pca_candidates_df], ignore_index=True)
    best_lambda_df = pd.concat([top5_best_df, pca_best_df], ignore_index=True)

    top5_model = build_top5_model(alpha=top5_lambda)
    top5_model.fit(full_data[top5_features], full_data["log_e_minus_c_csds"])
    top5_pred = predict_direct_outputs(top5_model, calibration_data, top5_features)
    top5_pred["model_id"] = "top5_recurrent_ridge"
    top5_pred["model_label"] = "Top 5 recurrentes + Ridge"
    top5_pred["e_error_vs_iterative"] = top5_pred["e_pred"] - top5_pred["e_csds"]
    top5_pred["d_error_vs_iterative"] = top5_pred["d_pred"] - top5_pred["d_csds"]
    top5_pred["b_error_vs_iterative"] = top5_pred["b_pred"] - top5_pred["b_csds"]
    top5_work, top5_metrics = summarize_predictions(top5_pred, prefix="calibration_test")

    pca_model = build_pca_model(alpha=pca_lambda)
    pca_model.fit(full_data[pca_features], full_data["log_e_minus_c_csds"])
    pca_pred = predict_direct_outputs(pca_model, calibration_data, pca_features)
    pca_pred["model_id"] = "top5_pca_full"
    pca_pred["model_label"] = "PCA 5 composantes"
    pca_pred["e_error_vs_iterative"] = pca_pred["e_pred"] - pca_pred["e_csds"]
    pca_pred["d_error_vs_iterative"] = pca_pred["d_pred"] - pca_pred["d_csds"]
    pca_pred["b_error_vs_iterative"] = pca_pred["b_pred"] - pca_pred["b_csds"]
    pca_work, pca_metrics = summarize_predictions(pca_pred, prefix="calibration_test")

    split_test_df = pd.concat([top5_test_work, pca_test_work], ignore_index=True)
    split_test_summary_df = pd.DataFrame([top5_split_summary, pca_split_summary])
    calibration_detailed_df = pd.concat([top5_work, pca_work], ignore_index=True)
    cv_fold_df = pd.concat([top5_cv_folds_df, pca_cv_folds_df], ignore_index=True)
    cv_oof_compare_df = pd.concat([top5_cv_oof_df, pca_cv_oof_df], ignore_index=True)
    cv_summary_df = pd.DataFrame([top5_cv_summary, pca_cv_summary])
    calibration_summary_df = pd.DataFrame([
        {
            "model_id": "top5_recurrent_ridge",
            "model_label": "Top 5 recurrentes + Ridge",
            "degree": POLY_DEGREE,
            "best_lambda": top5_lambda,
            "n_features_before_model": len(top5_features),
            "feature_cols": " + ".join(top5_features),
            **top5_metrics,
        },
        {
            "model_id": "top5_pca_full",
            "model_label": "PCA 5 composantes",
            "degree": POLY_DEGREE,
            "best_lambda": pca_lambda,
            "n_features_before_model": len(pca_features),
            "n_components": N_COMPONENTS,
            "feature_cols": " + ".join(pca_features),
            **pca_metrics,
        },
    ])

    model_definitions_df = pd.DataFrame([
        {
            "model_id": "top5_recurrent_ridge",
            "model_label": "Top 5 recurrentes + Ridge",
            "model_family": "PolynomialFeatures + Ridge",
            "degree": POLY_DEGREE,
            "best_lambda": top5_lambda,
            "feature_cols": " + ".join(top5_features),
        },
        {
            "model_id": "top5_pca_full",
            "model_label": "PCA 5 composantes",
            "model_family": "StandardScaler + PCA(5) + PolynomialFeatures + Ridge",
            "degree": POLY_DEGREE,
            "best_lambda": pca_lambda,
            "n_components": N_COMPONENTS,
            "feature_cols": " + ".join(pca_features),
        },
    ])

    top5_coef_df = export_top5_coefficients(top5_model, top5_features)

    lambda_candidates_df.to_csv(LAMBDA_CANDIDATES_FILE, index=False)
    best_lambda_df.to_csv(BEST_LAMBDA_FILE, index=False)
    cv_fold_df.to_csv(CV_COMPARE_FOLDS_FILE, index=False)
    cv_summary_df.to_csv(CV_COMPARE_SUMMARY_FILE, index=False)
    cv_oof_compare_df.to_csv(CV_COMPARE_OOF_FILE, index=False)
    split_test_df.to_csv(SPLIT_TEST_DETAILED_FILE, index=False)
    split_test_summary_df.to_csv(SPLIT_TEST_SUMMARY_FILE, index=False)
    calibration_detailed_df.to_csv(CALIBRATION_DETAILED_FILE, index=False)
    calibration_summary_df.to_csv(CALIBRATION_SUMMARY_FILE, index=False)
    model_definitions_df.to_csv(MODEL_DEFINITIONS_FILE, index=False)
    top5_coef_df.to_csv(TOP5_COEFFICIENTS_FILE, index=False)
    split_curve_index_df = plot_curves_by_sample(
        prediction_df=split_test_df,
        output_dir=SPLIT_TEST_CURVE_PLOTS_DIR,
        title_prefix="FULL split test",
        file_suffix="top5_recurrent_vs_pca_full_split_test_curve",
    )
    split_curve_index_df.to_csv(SPLIT_TEST_CURVE_PLOT_INDEX_FILE, index=False)
    calibration_curve_index_df = plot_curves_by_sample(
        prediction_df=calibration_detailed_df,
        output_dir=CALIBRATION_CURVE_PLOTS_DIR,
        title_prefix="Calibration test",
        file_suffix="top5_recurrent_vs_pca_calibration_test_curve",
    )
    calibration_curve_index_df.to_csv(CALIBRATION_CURVE_PLOT_INDEX_FILE, index=False)

    print("=" * 100)
    print("TOP-5 RECURRENT VARIABLES VS PCA FINAL MODEL")
    print("=" * 100)
    print(f"Top-5 recurrent features: {top5_features}")
    print(
        "Split FULL utilise : "
        f"train={len(train_df)}, validation={len(val_df)}, test_interne={len(test_df)}"
    )
    print(f"Calibration-test rows: {len(calibration_data)}")
    print("-" * 100)
    print("Cross-validation sur TRAIN (meme logique que methodologie 1 et 2)")
    print(
        "Top 5 recurrentes + Ridge | "
        f"RMSE(z)={top5_cv_summary['cv_rmse_mean_z']:.6f} ± {top5_cv_summary['cv_rmse_std_z']:.6f} | "
        f"RMSE(tau(u))={top5_cv_summary['cv_rmse_mean_tau_u']:.6f} ± {top5_cv_summary['cv_rmse_std_tau_u']:.6f}"
    )
    print(
        "PCA 5 composantes | "
        f"RMSE(z)={pca_cv_summary['cv_rmse_mean_z']:.6f} ± {pca_cv_summary['cv_rmse_std_z']:.6f} | "
        f"RMSE(tau(u))={pca_cv_summary['cv_rmse_mean_tau_u']:.6f} ± {pca_cv_summary['cv_rmse_std_tau_u']:.6f}"
    )
    print("-" * 100)
    print("Test interne issu de FULL")
    print(
        "Top 5 recurrentes + Ridge | "
        f"RMSE(z)={top5_split_summary['test_rmse_z']:.6f} | "
        f"RMSE(e)={top5_split_summary['test_rmse_e']:.6f} | "
        f"RMSE(tau(u))={top5_split_summary['test_rmse_tau_u']:.6f}"
    )
    print(
        "PCA 5 composantes | "
        f"RMSE(z)={pca_split_summary['test_rmse_z']:.6f} | "
        f"RMSE(e)={pca_split_summary['test_rmse_e']:.6f} | "
        f"RMSE(tau(u))={pca_split_summary['test_rmse_tau_u']:.6f}"
    )
    print("-" * 100)
    print("Test externe sur csds_calibration_test.csv")
    print("Top 5 recurrentes + Ridge")
    print(f"Best lambda: {top5_lambda:.8f}")
    print(f"RMSE(z) vs iterative: {top5_metrics['calibration_test_rmse_z']:.6f}")
    print(f"R2(z) vs iterative:   {top5_metrics['calibration_test_r2_z']:.6f}")
    print(f"RMSE(e) vs iterative: {top5_metrics['calibration_test_rmse_e']:.6f}")
    print(f"R2(e) vs iterative:   {top5_metrics['calibration_test_r2_e']:.6f}")
    print(f"RMSE(d) vs iterative: {top5_metrics['calibration_test_rmse_d']:.6f}")
    print(f"R2(d) vs iterative:   {top5_metrics['calibration_test_r2_d']:.6f}")
    print(f"RMSE(b) vs iterative: {top5_metrics['calibration_test_rmse_b']:.6f}")
    print(f"R2(b) vs iterative:   {top5_metrics['calibration_test_r2_b']:.6f}")
    print(f"RMSE(tau(u)) vs iterative: {top5_metrics['calibration_test_rmse_tau_u']:.6f}")
    print(f"R2(tau(u)) vs iterative:   {top5_metrics['calibration_test_r2_tau_u']:.6f}")
    print(
        "Constraint e>c respected: "
        f"{top5_metrics['calibration_test_constraint_ok_count']}/{len(calibration_data)}"
    )
    print("-" * 100)
    print("PCA 5 composantes")
    print(f"Best lambda: {pca_lambda:.8f}")
    print(f"RMSE(z) vs iterative: {pca_metrics['calibration_test_rmse_z']:.6f}")
    print(f"R2(z) vs iterative:   {pca_metrics['calibration_test_r2_z']:.6f}")
    print(f"RMSE(e) vs iterative: {pca_metrics['calibration_test_rmse_e']:.6f}")
    print(f"R2(e) vs iterative:   {pca_metrics['calibration_test_r2_e']:.6f}")
    print(f"RMSE(d) vs iterative: {pca_metrics['calibration_test_rmse_d']:.6f}")
    print(f"R2(d) vs iterative:   {pca_metrics['calibration_test_r2_d']:.6f}")
    print(f"RMSE(b) vs iterative: {pca_metrics['calibration_test_rmse_b']:.6f}")
    print(f"R2(b) vs iterative:   {pca_metrics['calibration_test_r2_b']:.6f}")
    print(f"RMSE(tau(u)) vs iterative: {pca_metrics['calibration_test_rmse_tau_u']:.6f}")
    print(f"R2(tau(u)) vs iterative:   {pca_metrics['calibration_test_r2_tau_u']:.6f}")
    print(
        "Constraint e>c respected: "
        f"{pca_metrics['calibration_test_constraint_ok_count']}/{len(calibration_data)}"
    )
    print("-" * 100)
    print(f"Top-5 feature list saved: {TOP5_FEATURES_FILE}")
    print(f"Lambda candidates saved: {LAMBDA_CANDIDATES_FILE}")
    print(f"Best lambda summary saved: {BEST_LAMBDA_FILE}")
    print(f"CV fold metrics saved: {CV_COMPARE_FOLDS_FILE}")
    print(f"CV summary saved: {CV_COMPARE_SUMMARY_FILE}")
    print(f"CV OOF comparison saved: {CV_COMPARE_OOF_FILE}")
    print(f"FULL split-test detailed saved: {SPLIT_TEST_DETAILED_FILE}")
    print(f"FULL split-test summary saved: {SPLIT_TEST_SUMMARY_FILE}")
    print(f"Calibration-test detailed saved: {CALIBRATION_DETAILED_FILE}")
    print(f"Calibration-test summary saved: {CALIBRATION_SUMMARY_FILE}")
    print(f"Model definitions saved: {MODEL_DEFINITIONS_FILE}")
    print(f"Top-5 coefficients saved: {TOP5_COEFFICIENTS_FILE}")
    print(f"FULL split-test curves saved in: {SPLIT_TEST_CURVE_PLOTS_DIR}")
    print(f"FULL split-test curve index saved: {SPLIT_TEST_CURVE_PLOT_INDEX_FILE}")
    print(f"Calibration-test curves saved in: {CALIBRATION_CURVE_PLOTS_DIR}")
    print(f"Calibration-test curve index saved: {CALIBRATION_CURVE_PLOT_INDEX_FILE}")


if __name__ == "__main__":
    main()
