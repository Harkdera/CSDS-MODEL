"""Analyse en composantes principales sur le top 10 FULL de la méthodologie 2."""

from __future__ import annotations

import os
from src.utils.paths import MATPLOTLIB_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src.final_model.final_model_utils_methodologie_2 import (
    DATASET_NAME,
    PCA_DIR,
    TARGET_COL,
    TOP10_DATASET_FILE,
    ensure_output_dirs,
    label,
    load_full_dataset,
    load_top10_features,
    mirror_legacy_outputs,
)
from src.utils.common_methodologie_2 import (
    RANDOM_SEED,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
)


DEGREE = 2
RIDGE_ALPHA = 1.0
CV_FOLDS = 5
VARIANCE_THRESHOLD = 0.85

CANDIDATE_FILE = PCA_DIR / "pca_polynomial_candidates_full.csv"
EXPLAINED_VARIANCE_FILE = PCA_DIR / "pca_explained_variance_full.csv"
LOADINGS_FILE = PCA_DIR / "pca_loadings_full.csv"
SCORES_FILE = PCA_DIR / "pca_scores_best_model_full.csv"
BEST_MODEL_FILE = PCA_DIR / "pca_best_polynomial_model_full.csv"
SCREE_PLOT_FILE = PCA_DIR / "pca_scree_plot_full.png"
CUMULATIVE_PLOT_FILE = PCA_DIR / "pca_cumulative_variance_full.png"
SCORES_PLOT_FILE = PCA_DIR / "pca_scores_pc1_pc2_full.png"
LOADINGS_PLOT_FILE = PCA_DIR / "pca_loadings_heatmap_full.png"


def build_pca_polynomial_model(n_components: int) -> Pipeline:
    """Construit le pipeline ACP + régression polynomiale."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("poly", PolynomialFeatures(degree=DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def fit_and_evaluate_pca_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int,
) -> dict[str, float]:
    """Évalue un pipeline ACP + régression polynomiale sur z, e-c, e et tau(u)."""
    model = build_pca_polynomial_model(n_components)

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    x_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    model.fit(x_train, y_train)

    z_train_pred = model.predict(x_train)
    z_val_pred = model.predict(x_val)
    z_test_pred = model.predict(x_test)

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="r2")

    gap_train_pred = np.exp(z_train_pred)
    gap_val_pred = np.exp(z_val_pred)
    gap_test_pred = np.exp(z_test_pred)

    e_train_pred = train_df["c_target"].to_numpy(dtype=float) + gap_train_pred
    e_val_pred = val_df["c_target"].to_numpy(dtype=float) + gap_val_pred
    e_test_pred = test_df["c_target"].to_numpy(dtype=float) + gap_test_pred

    z_train_metrics = compute_metrics(train_df[TARGET_COL], z_train_pred)
    z_val_metrics = compute_metrics(val_df[TARGET_COL], z_val_pred)
    z_test_metrics = compute_metrics(test_df[TARGET_COL], z_test_pred)
    gap_train_metrics = compute_metrics(train_df["e_minus_c_csds"], gap_train_pred)
    gap_val_metrics = compute_metrics(val_df["e_minus_c_csds"], gap_val_pred)
    gap_test_metrics = compute_metrics(test_df["e_minus_c_csds"], gap_test_pred)
    e_train_metrics = compute_metrics(train_df["e_csds"], e_train_pred)
    e_val_metrics = compute_metrics(val_df["e_csds"], e_val_pred)
    e_test_metrics = compute_metrics(test_df["e_csds"], e_test_pred)

    train_curve_df = train_df.copy()
    train_curve_df["e_pred"] = e_train_pred
    train_curve_df["d_pred"] = compute_d_from_e_peak_equation(train_curve_df, e_col="e_pred")
    train_curve_df["b_pred"] = train_curve_df["d_pred"] - train_curve_df["a_csds"]
    train_curve_df, train_curve_metrics = compute_curve_metrics_for_direct_prediction(train_curve_df)

    val_curve_df = val_df.copy()
    val_curve_df["e_pred"] = e_val_pred
    val_curve_df["d_pred"] = compute_d_from_e_peak_equation(val_curve_df, e_col="e_pred")
    val_curve_df["b_pred"] = val_curve_df["d_pred"] - val_curve_df["a_csds"]
    val_curve_df, val_curve_metrics = compute_curve_metrics_for_direct_prediction(val_curve_df)

    test_curve_df = test_df.copy()
    test_curve_df["e_pred"] = e_test_pred
    test_curve_df["d_pred"] = compute_d_from_e_peak_equation(test_curve_df, e_col="e_pred")
    test_curve_df["b_pred"] = test_curve_df["d_pred"] - test_curve_df["a_csds"]
    test_curve_df, test_curve_metrics = compute_curve_metrics_for_direct_prediction(test_curve_df)

    return {
        "target_mode": "pca_then_polynomial_on_log_e_minus_c",
        "n_features": len(feature_cols),
        "R2_train_z": z_train_metrics["R2"],
        "RMSE_train_z": z_train_metrics["RMSE"],
        "R2_val_z": z_val_metrics["R2"],
        "RMSE_val_z": z_val_metrics["RMSE"],
        "R2_test_z": z_test_metrics["R2"],
        "RMSE_test_z": z_test_metrics["RMSE"],
        "R2_cv_mean_z": float(np.mean(cv_scores)),
        "R2_cv_std_z": float(np.std(cv_scores)),
        "R2_train_e_gap": gap_train_metrics["R2"],
        "RMSE_train_e_gap": gap_train_metrics["RMSE"],
        "R2_val_e_gap": gap_val_metrics["R2"],
        "RMSE_val_e_gap": gap_val_metrics["RMSE"],
        "R2_test_e_gap": gap_test_metrics["R2"],
        "RMSE_test_e_gap": gap_test_metrics["RMSE"],
        "R2_train_e": e_train_metrics["R2"],
        "RMSE_train_e": e_train_metrics["RMSE"],
        "R2_val_e": e_val_metrics["R2"],
        "RMSE_val_e": e_val_metrics["RMSE"],
        "R2_test_e": e_test_metrics["R2"],
        "RMSE_test_e": e_test_metrics["RMSE"],
        "RMSE_train_tau_u": train_curve_metrics["rmse_tau_u"],
        "R2_train_tau_u": train_curve_metrics["r2_tau_u"],
        "RMSE_val_tau_u": val_curve_metrics["rmse_tau_u"],
        "R2_val_tau_u": val_curve_metrics["r2_tau_u"],
        "RMSE_test_tau_u": test_curve_metrics["rmse_tau_u"],
        "R2_test_tau_u": test_curve_metrics["r2_tau_u"],
        "Valid_curve_count_train": int(train_curve_df["curve_valid"].sum()),
        "Valid_curve_count_val": int(val_curve_df["curve_valid"].sum()),
        "Valid_curve_count_test": int(test_curve_df["curve_valid"].sum()),
    }


def evaluate_component_counts(data: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Teste différents nombres de composantes avec la régression polynomiale."""
    train_val_df, test_df = train_test_split(
        data,
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

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[feature_cols])
    pca_full = PCA().fit(x_scaled)

    explained_df = pd.DataFrame({
        "component": np.arange(1, len(feature_cols) + 1),
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
    })
    min_components = int(np.searchsorted(
        explained_df["cumulative_explained_variance"].to_numpy(),
        VARIANCE_THRESHOLD,
        side="left",
    ) + 1)
    explained_df["meets_variance_threshold"] = explained_df["component"] >= min_components

    candidate_rows: list[dict] = []
    for n_components in range(1, len(feature_cols) + 1):
        metrics = fit_and_evaluate_pca_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
            n_components=n_components,
        )
        component_info = explained_df.iloc[n_components - 1]
        candidate_rows.append({
            "dataset": DATASET_NAME,
            "n_original_features": len(feature_cols),
            "n_components": n_components,
            "explained_variance_ratio_component": component_info["explained_variance_ratio"],
            "cumulative_explained_variance": component_info["cumulative_explained_variance"],
            "meets_variance_threshold": bool(component_info["meets_variance_threshold"]),
            **metrics,
        })

    candidates = pd.DataFrame(candidate_rows)
    eligible = candidates[candidates["meets_variance_threshold"]].copy()
    if eligible.empty:
        eligible = candidates.copy()

    best = (
        eligible
        .sort_values(
            by=["RMSE_val_tau_u", "R2_cv_std_z", "R2_cv_mean_z", "n_components"],
            ascending=[True, True, False, True],
        )
        .head(1)
        .copy()
    )
    best["selection_rule"] = (
        "Best eligible model by RMSE_val_tau_u, then lower CV std on z, "
        "then higher CV mean on z, then fewer components."
    )
    return explained_df, candidates.sort_values("n_components").reset_index(drop=True), best


def save_explained_variance_plots(explained_df: pd.DataFrame) -> None:
    """Sauvegarde les figures principales de l'ACP."""
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    ax.bar(explained_df["component"], explained_df["explained_variance_ratio"], color="#4C78A8")
    ax.set_xlabel("Composante principale")
    ax.set_ylabel("Variance expliquée")
    ax.set_title("ACP FULL - Variance expliquée par composante")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(SCREE_PLOT_FILE, dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    ax.plot(
        explained_df["component"],
        explained_df["cumulative_explained_variance"],
        marker="o",
        color="#E45756",
    )
    ax.axhline(VARIANCE_THRESHOLD, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Variance cumulée expliquée")
    ax.set_title("ACP FULL - Variance cumulée expliquée")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(CUMULATIVE_PLOT_FILE, dpi=300)
    plt.close(fig)


def fit_best_model_and_save_outputs(
    data: pd.DataFrame,
    feature_cols: list[str],
    best_n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajuste le meilleur modèle sur FULL et sauvegarde scores et loadings."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[feature_cols])

    pca = PCA(n_components=best_n_components)
    scores = pca.fit_transform(x_scaled)

    model = build_pca_polynomial_model(best_n_components)
    model.fit(data[feature_cols], data[TARGET_COL])
    z_pred = model.predict(data[feature_cols])

    scores_df = pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, best_n_components + 1)])
    scores_df.insert(0, "sample_id", data["sample_id"].to_numpy())
    scores_df[TARGET_COL] = data[TARGET_COL].to_numpy()
    scores_df["z_pred"] = z_pred
    scores_df["z_residual"] = scores_df[TARGET_COL] - scores_df["z_pred"]

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(
        loadings,
        index=feature_cols,
        columns=[f"PC{i}" for i in range(1, best_n_components + 1)],
    ).reset_index().rename(columns={"index": "feature"})
    loadings_df["feature_label"] = loadings_df["feature"].map(label)

    scores_df.to_csv(SCORES_FILE, index=False)
    loadings_df.to_csv(LOADINGS_FILE, index=False)
    return scores_df, loadings_df


def save_component_plots(scores_df: pd.DataFrame, loadings_df: pd.DataFrame) -> None:
    """Sauvegarde les figures de scores et de loadings."""
    if {"PC1", "PC2"}.issubset(scores_df.columns):
        fig = plt.figure(figsize=(7.4, 6.0))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            scores_df["PC1"],
            scores_df["PC2"],
            c=scores_df[TARGET_COL],
            cmap="viridis",
            alpha=0.8,
            s=34,
            edgecolor="white",
            linewidth=0.35,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(r"$\log(e - c)$")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("ACP FULL - Projection des observations sur PC1 et PC2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(SCORES_PLOT_FILE, dpi=300)
        plt.close(fig)

    loading_cols = [col for col in loadings_df.columns if col.startswith("PC")]
    if loading_cols:
        heatmap_df = loadings_df.set_index("feature_label")[loading_cols]
        fig = plt.figure(figsize=(8.8, max(4.6, 0.45 * len(heatmap_df))))
        ax = fig.add_subplot(111)
        sns.heatmap(heatmap_df, annot=True, cmap="coolwarm", center=0.0, fmt=".2f", ax=ax)
        ax.set_title("ACP FULL - Loadings des composantes retenues")
        fig.tight_layout()
        fig.savefig(LOADINGS_PLOT_FILE, dpi=300)
        plt.close(fig)


def main() -> None:
    ensure_output_dirs()
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
        "b_csds",
        "c_target",
        "d_csds",
        "e_minus_c_csds",
        "e_csds",
        "a_csds",
        *feature_cols,
    }
    data = data[[col for col in data.columns if col in required_cols]].dropna().reset_index(drop=True)

    explained_df, candidates_df, best_df = evaluate_component_counts(data, feature_cols)
    best_n_components = int(best_df.iloc[0]["n_components"])

    explained_df.to_csv(EXPLAINED_VARIANCE_FILE, index=False)
    candidates_df.to_csv(CANDIDATE_FILE, index=False)
    best_df.to_csv(BEST_MODEL_FILE, index=False)
    save_explained_variance_plots(explained_df)

    scores_df, loadings_df = fit_best_model_and_save_outputs(
        data=data,
        feature_cols=feature_cols,
        best_n_components=best_n_components,
    )
    save_component_plots(scores_df, loadings_df)

    if not TOP10_DATASET_FILE.exists():
        data.to_csv(TOP10_DATASET_FILE, index=False)

    mirror_legacy_outputs()

    print(f"Candidate models saved: {CANDIDATE_FILE}")
    print(f"Best PCA model saved: {BEST_MODEL_FILE}")
    print(f"Explained variance saved: {EXPLAINED_VARIANCE_FILE}")
    print(f"Scores saved: {SCORES_FILE}")
    print(f"Loadings saved: {LOADINGS_FILE}")
    print(f"Best number of components: {best_n_components}")


if __name__ == "__main__":
    main()
