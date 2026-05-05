"""Recherche du meilleur lambda de régularisation Ridge sur les 5 premières composantes PCA."""

from __future__ import annotations

import os
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from methodologie_2.common import (  # noqa: E402
    RANDOM_SEED,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
)
from methodologie_2.full_interpretation_utils import (  # noqa: E402
    INTERPRETATION_ROOT,
    TARGET_COL,
    ensure_output_dirs,
    load_full_dataset,
    load_top10_features,
)


N_COMPONENTS = 5
DEGREE = 2
CV_FOLDS = 10
TEST_SIZE = 0.20
LAMBDA_GRID = np.logspace(-4, 4, 25)

LAMBDA_DIR = INTERPRETATION_ROOT / "lambda_search"
CANDIDATE_FILE = LAMBDA_DIR / "lambda_candidates_top5_pca_full.csv"
BEST_FILE = LAMBDA_DIR / "lambda_best_top5_pca_full.csv"
TEST_PREDICTIONS_FILE = LAMBDA_DIR / "best_lambda_test_predictions_top5_pca_full.csv"
PLOT_FILE = LAMBDA_DIR / "lambda_cv_test_summary_top5_pca_full.png"


def ensure_lambda_output_dir() -> None:
    """Crée le dossier de sortie de la recherche en lambda."""
    ensure_output_dirs()
    LAMBDA_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    """Construit le dataset FULL avec top 10 variables et colonnes utiles à la reconstruction."""
    top10_df = load_top10_features()
    data = load_full_dataset().copy()
    feature_cols = [feature for feature in top10_df["feature"] if feature in data.columns]

    required_cols = {
        "sample_id",
        TARGET_COL,
        "delta_peak_mm",
        "u_r_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
        "a_csds",
        "b_csds",
        "c_target",
        "d_csds",
        "e_csds",
        "e_minus_c_csds",
        *feature_cols,
    }
    dataset = data[[col for col in data.columns if col in required_cols]].dropna().reset_index(drop=True)
    return dataset, feature_cols


def build_model(alpha: float) -> Pipeline:
    """Construit le pipeline PCA + polynôme + Ridge pour un lambda donné."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=N_COMPONENTS)),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=alpha)),
    ])


def evaluate_split(
    model: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Évalue le modèle sur un split donné."""
    x_data = df[feature_cols]
    z_pred = model.predict(x_data)
    gap_pred = np.exp(z_pred)
    e_pred = df["c_target"].to_numpy(dtype=float) + gap_pred

    z_metrics = compute_metrics(df[TARGET_COL], z_pred)
    gap_metrics = compute_metrics(df["e_minus_c_csds"], gap_pred)
    e_metrics = compute_metrics(df["e_csds"], e_pred)

    work = df.copy()
    work["z_pred"] = z_pred
    work["e_minus_c_pred"] = gap_pred
    work["e_pred"] = e_pred
    work["d_pred"] = compute_d_from_e_peak_equation(work, e_col="e_pred")
    work["b_pred"] = work["d_pred"] - work["a_csds"]
    work, curve_metrics = compute_curve_metrics_for_direct_prediction(work)

    metrics = {
        f"R2_{split_name}_z": z_metrics["R2"],
        f"RMSE_{split_name}_z": z_metrics["RMSE"],
        f"R2_{split_name}_e_gap": gap_metrics["R2"],
        f"RMSE_{split_name}_e_gap": gap_metrics["RMSE"],
        f"R2_{split_name}_e": e_metrics["R2"],
        f"RMSE_{split_name}_e": e_metrics["RMSE"],
        f"RMSE_{split_name}_tau_u": curve_metrics["rmse_tau_u"],
        f"R2_{split_name}_tau_u": curve_metrics["r2_tau_u"],
        f"Valid_curve_count_{split_name}": int(work["curve_valid"].sum()),
    }
    return metrics, work


def evaluate_external_test(
    model: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Évalue le modèle final sur le test externe."""
    return evaluate_split(model, test_df, feature_cols, split_name="test")


def search_best_lambda(data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recherche le meilleur lambda avec CV sur entraînement et test externe séparé."""
    train_val_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    rows: list[dict] = []
    best_predictions = pd.DataFrame()

    for alpha in LAMBDA_GRID:
        model = build_model(alpha=float(alpha))
        cv_results = cross_validate(
            model,
            x_train,
            y_train,
            cv=cv,
            scoring={
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=None,
            return_train_score=False,
        )

        model.fit(x_train, y_train)
        val_metrics, _ = evaluate_split(model, val_df, feature_cols, split_name="val")
        test_metrics, test_predictions = evaluate_external_test(model, test_df, feature_cols)

        row = {
            "lambda": float(alpha),
            "n_components": N_COMPONENTS,
            "degree": DEGREE,
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "cv_rmse_mean_z": float(-np.mean(cv_results["test_rmse"])),
            "cv_rmse_std_z": float(np.std(-cv_results["test_rmse"])),
            "cv_r2_mean_z": float(np.mean(cv_results["test_r2"])),
            "cv_r2_std_z": float(np.std(cv_results["test_r2"])),
            **val_metrics,
            **test_metrics,
        }
        rows.append(row)

    candidates = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    best = (
        candidates
        .sort_values(
            by=["RMSE_val_z", "cv_rmse_std_z", "cv_r2_mean_z", "RMSE_val_tau_u", "lambda"],
            ascending=[True, True, False, True, True],
        )
        .head(1)
        .copy()
    )
    best["selection_rule"] = (
        "Best lambda selected with the same train/validation/test logic as methodologies 1 and 2: "
        "lowest RMSE on validation for z = log(e-c), then lower CV std on training folds, "
        "then higher mean CV R2, then lower validation RMSE on tau(u). External test is reported separately."
    )

    best_lambda = float(best.iloc[0]["lambda"])
    best_model = build_model(alpha=best_lambda)
    best_model.fit(x_train, y_train)
    _, best_predictions = evaluate_external_test(best_model, test_df, feature_cols)
    best_predictions["lambda"] = best_lambda

    return candidates, best, best_predictions


def save_summary_plot(candidates: pd.DataFrame) -> None:
    """Trace l'évolution des métriques CV et test selon lambda."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axes[0].plot(candidates["lambda"], candidates["cv_rmse_mean_z"], marker="o", color="#4C78A8")
    axes[0].fill_between(
        candidates["lambda"],
        candidates["cv_rmse_mean_z"] - candidates["cv_rmse_std_z"],
        candidates["cv_rmse_mean_z"] + candidates["cv_rmse_std_z"],
        color="#4C78A8",
        alpha=0.20,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("CV RMSE sur log(e-c)")
    axes[0].set_title("Validation croisée")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(candidates["lambda"], candidates["RMSE_test_z"], marker="o", label="RMSE test z", color="#E45756")
    axes[1].plot(
        candidates["lambda"],
        candidates["RMSE_test_tau_u"],
        marker="s",
        label=r"RMSE test $\tau(u)$",
        color="#72B7B2",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Erreur sur test externe")
    axes[1].set_title("Test externe")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_lambda_output_dir()
    data, feature_cols = build_dataset()
    candidates, best, predictions = search_best_lambda(data, feature_cols)

    candidates.to_csv(CANDIDATE_FILE, index=False)
    best.to_csv(BEST_FILE, index=False)
    predictions.to_csv(TEST_PREDICTIONS_FILE, index=False)
    save_summary_plot(candidates)

    print(f"Lambda candidates saved: {CANDIDATE_FILE}")
    print(f"Best lambda saved: {BEST_FILE}")
    print(f"Best-lambda test predictions saved: {TEST_PREDICTIONS_FILE}")
    print(f"Summary plot saved: {PLOT_FILE}")
    print(f"Best lambda: {float(best.iloc[0]['lambda']):.8f}")


if __name__ == "__main__":
    main()
