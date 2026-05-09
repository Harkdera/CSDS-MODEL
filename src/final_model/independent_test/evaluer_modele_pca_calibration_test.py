"""Évalue le meilleur modèle PCA (5 composantes) de FULL sur `csds_calibration_test`.

Le script charge :
- les 10 variables retenues pour FULL ;
- le meilleur lambda trouvé par `22_search_best_lambda_top5_pca_full.py` ;
- le dataset FULL complet pour réentraîner le modèle final ;
- le fichier `csds_calibration_test.csv` pour le test externe.

La comparaison finale est faite contre la calibration itérative CSDS.
"""

from __future__ import annotations

import os
from importlib import import_module
from src.final_model.final_model_utils_methodologie_2 import load_full_dataset, load_top10_features
from src.utils.paths import FINAL_INDEPENDENT_DATASET_DIR, FINAL_LAMBDA_SEARCH_DIR, MATPLOTLIB_CACHE_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.utils.common_methodologie_2 import (
    add_engineered_features,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
    csds_tau,
    ensure_output_dirs,
    make_u_grid,
)


N_COMPONENTS = 5
DEGREE = 2
BEST_LAMBDA_FILE = FINAL_LAMBDA_SEARCH_DIR / "lambda_best_top5_pca_full.csv"
CALIBRATION_TEST_FILE = PROJECT_ROOT / "data" / "test" / "csds_calibration_test.csv"

OUTPUT_DIR = FINAL_INDEPENDENT_DATASET_DIR / "top5_pca_full_on_calibration_test"
DETAILED_FILE = OUTPUT_DIR / "top5_pca_full_calibration_test_detailed.csv"
SUMMARY_FILE = OUTPUT_DIR / "top5_pca_full_calibration_test_summary.csv"
COEFFICIENT_INFO_FILE = OUTPUT_DIR / "top5_pca_full_model_definition.csv"
CURVE_PLOTS_DIR = OUTPUT_DIR / "curve_plots_by_sample"
CURVE_PLOT_INDEX_FILE = OUTPUT_DIR / "curve_plot_index.csv"


def build_model(alpha: float) -> Pipeline:
    """Construit le pipeline PCA + polynôme + Ridge."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS)),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=float(alpha))),
    ])


def safe_slug(text: str | int) -> str:
    """Produit un nom de fichier stable."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def load_best_lambda() -> float:
    """Charge le meilleur lambda déjà trouvé sur FULL."""
    if not BEST_LAMBDA_FILE.exists():
        raise FileNotFoundError(
            f"Best-lambda file not found: {BEST_LAMBDA_FILE}. "
            "Run 22_search_best_lambda_top5_pca_full.py first."
        )
    best_df = pd.read_csv(BEST_LAMBDA_FILE)
    if best_df.empty:
        raise ValueError(f"Best-lambda file is empty: {BEST_LAMBDA_FILE}")
    return float(best_df.iloc[0]["lambda"])


def build_feature_list() -> list[str]:
    """Charge les 10 variables FULL utilisées pour le modèle PCA."""
    top10_df = load_top10_features()
    return [str(feature) for feature in top10_df["feature"].tolist()]


def build_training_dataset(feature_cols: list[str]) -> pd.DataFrame:
    """Construit le dataset FULL complet pour réentraîner le modèle final."""
    data = load_full_dataset().copy()
    required_cols = feature_cols + [
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
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    return data


def build_calibration_test_dataset(feature_cols: list[str]) -> pd.DataFrame:
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
    data = data.dropna(subset=feature_cols).reset_index(drop=True)
    return data


def predict_direct_outputs(model: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Prévoit `z`, `e`, `d` et `b` avec le modèle PCA final."""
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


def plot_curves_by_sample(calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Trace une figure par échantillon contre la courbe itérative de référence."""
    CURVE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rows: list[dict] = []

    for sample_id, sample_rows in calibration_df.groupby("sample_id", sort=True):
        row = sample_rows.iloc[0]
        u = make_u_grid(row)
        tau_true = csds_tau(
            u=u,
            a=float(row["a_csds"]),
            b=float(row["b_csds"]),
            c=float(row["c_target"]),
            d=float(row["d_csds"]),
            e=float(row["e_csds"]),
        )
        tau_pred = csds_tau(
            u=u,
            a=float(row["a_csds"]),
            b=float(row["b_pred"]),
            c=float(row["c_target"]),
            d=float(row["d_pred"]),
            e=float(row["e_pred"]),
        )

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(u, tau_true, color="black", linewidth=2.5, label="Courbe itérative de référence")
        ax.plot(
            u,
            tau_pred,
            color="#1f77b4",
            linestyle="--",
            linewidth=2.0,
            label=(
                "Modèle PCA 5 composantes | "
                f"RMSE τ(u)={row['curve_rmse_tau_u']:.4f} | "
                f"R² τ(u)={row['curve_r2_tau_u']:.4f}"
            ),
        )
        ax.scatter([0.0], [0.0], color="black", s=35, marker="o", label="Origine")
        ax.scatter([float(row["delta_peak_mm"])], [float(row["tau_peak_MPa_csds"])], color="black", s=70, marker="x", label="Point de pic")
        ax.scatter([float(row["u_r_mm"])], [float(row["tau_r_MPa"])], color="black", s=55, marker="s", label="Point résiduel")
        ax.set_title(f"Calibration test | sample_id={sample_id} | PCA 5 composantes")
        ax.set_xlabel("u")
        ax.set_ylabel("τ(u)")
        ax.grid(True, alpha=0.30)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        output_file = CURVE_PLOTS_DIR / f"sample_{safe_slug(sample_id)}_top5_pca_curve.png"
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

    feature_cols = build_feature_list()
    best_lambda = load_best_lambda()

    train_data = build_training_dataset(feature_cols)
    calibration_data = build_calibration_test_dataset(feature_cols)

    final_model = build_model(alpha=best_lambda)
    final_model.fit(train_data[feature_cols], train_data["log_e_minus_c_csds"])

    calibration_pred = predict_direct_outputs(final_model, calibration_data, feature_cols)
    calibration_pred["model_label"] = "PCA 5 composantes FULL"
    calibration_pred["e_error_vs_iterative"] = calibration_pred["e_pred"] - calibration_pred["e_csds"]
    calibration_pred["d_error_vs_iterative"] = calibration_pred["d_pred"] - calibration_pred["d_csds"]
    calibration_pred["b_error_vs_iterative"] = calibration_pred["b_pred"] - calibration_pred["b_csds"]
    calibration_work, metrics = summarize_predictions(calibration_pred, prefix="calibration_test")

    summary_df = pd.DataFrame([{
        "model_label": "PCA 5 composantes FULL",
        "n_components": N_COMPONENTS,
        "degree": DEGREE,
        "best_lambda": best_lambda,
        "feature_cols": " + ".join(feature_cols),
        "n_features_before_pca": len(feature_cols),
        **metrics,
    }])

    definition_df = pd.DataFrame([{
        "model_label": "PCA 5 composantes FULL",
        "best_lambda": best_lambda,
        "n_components": N_COMPONENTS,
        "degree": DEGREE,
        "feature_cols": " + ".join(feature_cols),
    }])

    calibration_work.to_csv(DETAILED_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    definition_df.to_csv(COEFFICIENT_INFO_FILE, index=False)
    curve_index_df = plot_curves_by_sample(calibration_work)
    curve_index_df.to_csv(CURVE_PLOT_INDEX_FILE, index=False)

    print("=" * 100)
    print("PCA 5-COMPONENT MODEL ON CALIBRATION TEST")
    print("=" * 100)
    print(f"Best lambda loaded from: {BEST_LAMBDA_FILE}")
    print(f"Best lambda: {best_lambda:.8f}")
    print(f"Training rows (FULL): {len(train_data)}")
    print(f"Calibration-test rows: {len(calibration_data)}")
    print(f"RMSE(z) vs iterative: {metrics['calibration_test_rmse_z']:.6f}")
    print(f"R2(z) vs iterative:   {metrics['calibration_test_r2_z']:.6f}")
    print(f"RMSE(e) vs iterative: {metrics['calibration_test_rmse_e']:.6f}")
    print(f"R2(e) vs iterative:   {metrics['calibration_test_r2_e']:.6f}")
    print(f"RMSE(d) vs iterative: {metrics['calibration_test_rmse_d']:.6f}")
    print(f"R2(d) vs iterative:   {metrics['calibration_test_r2_d']:.6f}")
    print(f"RMSE(b) vs iterative: {metrics['calibration_test_rmse_b']:.6f}")
    print(f"R2(b) vs iterative:   {metrics['calibration_test_r2_b']:.6f}")
    print(f"RMSE(tau(u)) vs iterative: {metrics['calibration_test_rmse_tau_u']:.6f}")
    print(f"R2(tau(u)) vs iterative:   {metrics['calibration_test_r2_tau_u']:.6f}")
    print(f"Constraint e>c respected: {metrics['calibration_test_constraint_ok_count']}/{len(calibration_data)}")
    print("-" * 100)
    print(f"Detailed file saved: {DETAILED_FILE}")
    print(f"Summary file saved: {SUMMARY_FILE}")
    print(f"Model definition saved: {COEFFICIENT_INFO_FILE}")
    print(f"Curve plots saved in: {CURVE_PLOTS_DIR}")
    print(f"Curve plot index saved: {CURVE_PLOT_INDEX_FILE}")


if __name__ == "__main__":
    main()
