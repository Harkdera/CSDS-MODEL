"""Analyse des residus pour les modeles finaux deja exportes dans le projet."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

from src.utils.paths import (
    COMPARISON_RESULTS_DIR,
    FINAL_INDEPENDENT_DATASET_DIR,
    MATPLOTLIB_CACHE_DIR,
    METHODOLOGIE_1_RESULTS_DIR,
    METHODOLOGIE_2_RESULTS_DIR,
    PROJECT_ROOT,
    RESIDUAL_ANALYSIS_RESULTS_DIR,
)

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, shapiro, t as student_t


BEST_BY_METHOD_FILE = COMPARISON_RESULTS_DIR / "inter-methodologies" / "direct_vs_indirect_best_by_method.csv"
PCA_FINAL_DETAILED_FILE = (
    FINAL_INDEPENDENT_DATASET_DIR
    / "top5_pca_full_on_calibration_test"
    / "top5_pca_full_calibration_test_detailed.csv"
)
DEFAULT_OUTPUT_ROOT = RESIDUAL_ANALYSIS_RESULTS_DIR

TARGET_SPECS = {
    "z": {
        "label": "z = log(e-c)",
        "true_candidates": ["log_e_minus_c_csds", "true_z", "z_true"],
        "pred_candidates": ["z_pred", "estimated_z", "predicted_z", "z_estimated"],
    },
    "d": {
        "label": "d",
        "true_candidates": ["d_csds", "true_d", "d_true", "d_csv_reference", "d"],
        "pred_candidates": ["d_pred", "estimated_d", "d_estimated", "predicted_d", "d_predicted"],
    },
    "e": {
        "label": "e",
        "true_candidates": ["e_csds", "true_e", "e_true", "e_csv_reference", "e"],
        "pred_candidates": ["e_pred", "estimated_e", "e_estimated", "predicted_e", "e_predicted"],
    },
    "b": {
        "label": "b",
        "true_candidates": ["b_csds", "true_b", "b_true", "b"],
        "pred_candidates": ["b_pred", "estimated_b", "b_estimated", "predicted_b", "b_predicted"],
    },
}
CURVE_COLUMNS = ["curve_rmse_tau_u", "curve_mae_tau_u", "curve_r2_tau_u", "curve_valid"]
DATASET_GROUPS = {
    "FULL": "full",
    "HIGH": "high",
    "LOW_1": "low",
    "LOW_2": "low",
    "CALIBRATION_TEST": "calibration_test",
}


@dataclass(frozen=True)
class ModelSpec:
    source: str
    dataset: str
    method: str
    model_name: str
    model_label: str
    model_family: str
    selection_mode: str
    prediction_file: Path

    @property
    def model_key(self) -> str:
        return safe_slug(f"{self.source}_{self.dataset}_{self.method}_{self.model_name}")


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construit une analyse de residus a partir des sorties detaillees des "
            "modeles finaux deja exportes dans le depot."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Dossier de sortie (defaut: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--skip-best-by-method",
        action="store_true",
        help="Ignore les modeles de `direct_vs_indirect_best_by_method.csv`.",
    )
    parser.add_argument(
        "--skip-pca-final",
        action="store_true",
        help="Ignore le modele final PCA `top5_pca_full` sur `csds_calibration_test`.",
    )
    return parser


def build_best_by_method_specs() -> list[ModelSpec]:
    best_by_method_file = BEST_BY_METHOD_FILE
    if not best_by_method_file.exists():
        raise FileNotFoundError(f"Best-by-method file not found: {BEST_BY_METHOD_FILE}")

    df = pd.read_csv(best_by_method_file)
    if df.empty:
        raise ValueError(f"Best-by-method file is empty: {best_by_method_file}")

    specs: list[ModelSpec] = []
    for row in df.itertuples(index=False):
        prediction_file = resolve_prediction_file(
            dataset=str(row.dataset),
            method=str(row.method),
            model_name=str(row.model_name),
            raw_prediction_file=getattr(row, "detailed_output_file"),
        )
        if not prediction_file.exists():
            raise FileNotFoundError(
                f"Detailed prediction file not found for {row.model_name}: {prediction_file}"
            )

        specs.append(
            ModelSpec(
                source="best_by_method",
                dataset=str(row.dataset),
                method=str(row.method),
                model_name=str(row.model_name),
                model_label=str(row.model_name),
                model_family=str(row.model_family),
                selection_mode=str(row.selection_mode),
                prediction_file=prediction_file,
            )
        )

    return specs


def resolve_prediction_file(
    dataset: str,
    method: str,
    model_name: str,
    raw_prediction_file: object,
) -> Path:
    if isinstance(raw_prediction_file, str) and raw_prediction_file.strip():
        raw_path = Path(raw_prediction_file)
        if raw_path.exists():
            return raw_path

        raw_text = str(raw_path)
        replacements = [
            (
                str(PROJECT_ROOT / "results" / "06_methodologie_1"),
                str(METHODOLOGIE_1_RESULTS_DIR),
            ),
            (
                str(PROJECT_ROOT / "results" / "07_methodologie_2"),
                str(METHODOLOGIE_2_RESULTS_DIR),
            ),
            (
                str(PROJECT_ROOT / "results" / "08_comparaison_methodologies"),
                str(COMPARISON_RESULTS_DIR),
            ),
        ]
        for old_prefix, new_prefix in replacements:
            if raw_text.startswith(old_prefix):
                candidate = Path(raw_text.replace(old_prefix, new_prefix, 1))
                if candidate.exists():
                    return candidate
        reverse_replacements = [(new_prefix, old_prefix) for old_prefix, new_prefix in replacements]
        for new_prefix, old_prefix in reverse_replacements:
            if raw_text.startswith(new_prefix):
                candidate = Path(raw_text.replace(new_prefix, old_prefix, 1))
                if candidate.exists():
                    return candidate
        return raw_path

    group = DATASET_GROUPS.get(dataset)
    if group is None:
        raise KeyError(f"Unknown dataset group for {dataset}")

    if method == "methodologie_1":
        canonical = (
            METHODOLOGIE_1_RESULTS_DIR
            / "reconstructed_parameters"
            / "tau_u"
            / group
            / model_name
            / f"{model_name}_comparison_d_b_e_tau_u.csv"
        )
        legacy = (
            PROJECT_ROOT
            / "results"
            / "06_methodologie_1"
            / "compare_d_b_e_tau_u"
            / group
            / model_name
            / f"{model_name}_comparison_d_b_e_tau_u.csv"
        )
        return canonical if canonical.exists() else legacy

    if method == "methodologie_2":
        canonical = (
            METHODOLOGIE_2_RESULTS_DIR
            / "reconstructed_parameters"
            / "from_z"
            / group
            / f"{model_name}_b_d_from_e_predictions.csv"
        )
        legacy = (
            PROJECT_ROOT
            / "results"
            / "07_methodologie_2"
            / "evaluations"
            / group
            / f"{model_name}_b_d_from_e_predictions.csv"
        )
        return canonical if canonical.exists() else legacy

    raise ValueError(f"Unsupported method for residual analysis: {method}")


def build_pca_final_spec() -> ModelSpec:
    if not PCA_FINAL_DETAILED_FILE.exists():
        raise FileNotFoundError(f"PCA final detailed file not found: {PCA_FINAL_DETAILED_FILE}")

    return ModelSpec(
        source="pca_final_calibration_test",
        dataset="CALIBRATION_TEST",
        method="methodologie_2",
        model_name="top5_pca_full",
        model_label="PCA 5 composantes FULL",
        model_family="pca_ridge",
        selection_mode="calibration_test",
        prediction_file=PCA_FINAL_DETAILED_FILE,
    )


def build_model_specs(skip_best_by_method: bool, skip_pca_final: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if not skip_best_by_method:
        specs.extend(build_best_by_method_specs())
    if not skip_pca_final:
        specs.append(build_pca_final_spec())
    if not specs:
        raise ValueError("No model source selected. Remove at least one skip flag.")
    return specs


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_scalar_targets(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    for target_name, spec in TARGET_SPECS.items():
        true_col = find_first_existing(df, spec["true_candidates"])
        pred_col = find_first_existing(df, spec["pred_candidates"])
        if true_col and pred_col:
            targets[target_name] = (true_col, pred_col)
    return targets


def build_scalar_residual_frame(
    df: pd.DataFrame,
    model_spec: ModelSpec,
    target_name: str,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    sample_id = df["sample_id"] if "sample_id" in df.columns else pd.Series(np.arange(1, len(df) + 1))

    work = pd.DataFrame(
        {
            "sample_id": sample_id,
            "observed": pd.to_numeric(df[true_col], errors="coerce"),
            "predicted": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    ).dropna(subset=["observed", "predicted"]).reset_index(drop=True)

    work["residual"] = work["predicted"] - work["observed"]
    work["abs_residual"] = work["residual"].abs()

    residual_std = float(work["residual"].std(ddof=1)) if len(work) > 1 else np.nan
    if np.isfinite(residual_std) and residual_std > 0:
        work["standardized_residual"] = work["residual"] / residual_std
    else:
        work["standardized_residual"] = 0.0

    work["source"] = model_spec.source
    work["dataset"] = model_spec.dataset
    work["method"] = model_spec.method
    work["model_name"] = model_spec.model_name
    work["model_label"] = model_spec.model_label
    work["model_family"] = model_spec.model_family
    work["selection_mode"] = model_spec.selection_mode
    work["model_key"] = model_spec.model_key
    work["target"] = target_name
    work["target_label"] = TARGET_SPECS[target_name]["label"]
    work["true_column"] = true_col
    work["pred_column"] = pred_col
    return work


def rmse_from_frame(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    return float(np.sqrt(np.mean((df["predicted"] - df["observed"]) ** 2)))


def r2_from_frame(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return np.nan
    y_true = df["observed"].to_numpy(dtype=float)
    y_pred = df["predicted"].to_numpy(dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return np.nan
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (ss_res / ss_tot)


def summarize_scalar_residuals(df: pd.DataFrame) -> dict[str, object]:
    residual = df["residual"]
    abs_residual = df["abs_residual"]
    return {
        "source": df["source"].iloc[0],
        "dataset": df["dataset"].iloc[0],
        "method": df["method"].iloc[0],
        "model_name": df["model_name"].iloc[0],
        "model_label": df["model_label"].iloc[0],
        "model_family": df["model_family"].iloc[0],
        "selection_mode": df["selection_mode"].iloc[0],
        "model_key": df["model_key"].iloc[0],
        "target": df["target"].iloc[0],
        "target_label": df["target_label"].iloc[0],
        "true_column": df["true_column"].iloc[0],
        "pred_column": df["pred_column"].iloc[0],
        "n_rows": int(len(df)),
        "mean_observed": float(df["observed"].mean()),
        "mean_predicted": float(df["predicted"].mean()),
        "mean_residual": float(residual.mean()),
        "median_residual": float(residual.median()),
        "std_residual": float(residual.std(ddof=1)) if len(df) > 1 else np.nan,
        "mae": float(abs_residual.mean()),
        "rmse": rmse_from_frame(df),
        "r2": r2_from_frame(df),
        "residual_q05": float(residual.quantile(0.05)),
        "residual_q95": float(residual.quantile(0.95)),
        "max_abs_residual": float(abs_residual.max()),
        "overprediction_share": float((residual > 0).mean()),
    }


def status_from_value(value: object) -> str:
    try:
        if pd.isna(value):
            return "non_evalue"
    except Exception:
        pass
    return "raisonnablement_respectee" if bool(value) else "a_surveiller"


def fit_auxiliary_ols(y: np.ndarray, x: np.ndarray) -> dict[str, object]:
    """Ajuste une OLS simple avec `numpy` et retourne p-values et R2 si estimables."""
    n_rows, n_cols = x.shape
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - (sse / sst) if sst > 0 else np.nan

    if rank < n_cols or n_rows <= n_cols:
        return {
            "beta": beta,
            "pvalues": np.full(n_cols, np.nan, dtype=float),
            "r2": float(r2),
        }

    sigma2 = sse / float(n_rows - n_cols)
    xtx_inv = np.linalg.pinv(x.T @ x)
    std_err = np.sqrt(np.diag(sigma2 * xtx_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / std_err
    pvalues = 2.0 * student_t.sf(np.abs(t_stats), df=n_rows - n_cols)
    return {
        "beta": beta,
        "pvalues": pvalues,
        "r2": float(r2),
    }


def check_regression_assumptions(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Vérifie les hypothèses de régression à partir des résidus déjà exportés."""
    work = df.dropna(subset=["observed", "predicted", "residual"]).copy()
    n_rows = len(work)

    base: dict[str, object] = {
        "n_rows": int(n_rows),
        "alpha": float(alpha),
        "linearity_pvalue_quad": np.nan,
        "linearity_aux_r2": np.nan,
        "linearity_ok": np.nan,
        "durbin_watson": np.nan,
        "lag1_autocorr": np.nan,
        "independence_ok": np.nan,
        "breusch_pagan_pvalue": np.nan,
        "homoscedasticity_ok": np.nan,
        "shapiro_w": np.nan,
        "shapiro_pvalue": np.nan,
        "normality_ok": np.nan,
        "share_abs_std_resid_gt_2": np.nan,
        "share_abs_std_resid_gt_3": np.nan,
        "linearity_status": "non_evalue",
        "independence_status": "non_evalue",
        "homoscedasticity_status": "non_evalue",
        "normality_status": "non_evalue",
        "independence_note": "a interpreter seulement si l'ordre des observations a un sens",
        "linearity_note": "verification approximative basee sur les valeurs predites",
    }
    if n_rows < 3:
        return base

    fitted = work["predicted"].to_numpy(dtype=float)
    residual = work["residual"].to_numpy(dtype=float)
    standardized = work["standardized_residual"].to_numpy(dtype=float)

    try:
        x_linear = np.column_stack([np.ones(n_rows, dtype=float), fitted, fitted**2])
        linear_fit = fit_auxiliary_ols(residual, x_linear)
        p_quad = float(linear_fit["pvalues"][2]) if len(linear_fit["pvalues"]) > 2 else np.nan
        base["linearity_pvalue_quad"] = p_quad
        base["linearity_aux_r2"] = float(linear_fit["r2"])
        if np.isfinite(p_quad):
            base["linearity_ok"] = bool(p_quad > alpha)
    except Exception:
        pass

    try:
        resid_diff = np.diff(residual)
        resid_sq_sum = float(np.sum(residual**2))
        if resid_sq_sum > 0:
            base["durbin_watson"] = float(np.sum(resid_diff**2) / resid_sq_sum)

        if n_rows >= 4 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
            base["lag1_autocorr"] = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])

        if np.isfinite(base["durbin_watson"]):
            base["independence_ok"] = bool(1.5 <= float(base["durbin_watson"]) <= 2.5)
    except Exception:
        pass

    try:
        exog_bp = np.column_stack([np.ones(n_rows, dtype=float), fitted])
        bp_fit = fit_auxiliary_ols(residual**2, exog_bp)
        dof = exog_bp.shape[1] - 1
        if dof > 0 and np.isfinite(bp_fit["r2"]):
            lm_stat = float(n_rows * float(bp_fit["r2"]))
            bp_pvalue = float(chi2.sf(lm_stat, dof))
            base["breusch_pagan_pvalue"] = bp_pvalue
            base["homoscedasticity_ok"] = bool(bp_pvalue > alpha)
    except Exception:
        pass

    try:
        if 3 <= n_rows <= 5000:
            shapiro_w, shapiro_pvalue = shapiro(residual)
            base["shapiro_w"] = float(shapiro_w)
            base["shapiro_pvalue"] = float(shapiro_pvalue)
            base["normality_ok"] = bool(shapiro_pvalue > alpha)
    except Exception:
        pass

    try:
        base["share_abs_std_resid_gt_2"] = float(np.mean(np.abs(standardized) > 2.0))
        base["share_abs_std_resid_gt_3"] = float(np.mean(np.abs(standardized) > 3.0))
    except Exception:
        pass

    base["linearity_status"] = status_from_value(base["linearity_ok"])
    base["independence_status"] = status_from_value(base["independence_ok"])
    base["homoscedasticity_status"] = status_from_value(base["homoscedasticity_ok"])
    base["normality_status"] = status_from_value(base["normality_ok"])
    return base


def plot_scalar_diagnostics(df: pd.DataFrame, output_file: Path, title: str) -> None:
    residual = df["residual"].to_numpy(dtype=float)
    predicted = df["predicted"].to_numpy(dtype=float)
    observed = df["observed"].to_numpy(dtype=float)
    standardized = df["standardized_residual"].to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_pred, ax_obs, ax_hist, ax_qq = axes.flatten()

    ax_pred.scatter(predicted, residual, alpha=0.75, edgecolor="none", color="#1f77b4")
    ax_pred.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_pred.set_xlabel("Valeur predite")
    ax_pred.set_ylabel("Residuel (pred - obs)")
    ax_pred.set_title("Residus vs predictions")

    ax_obs.scatter(observed, residual, alpha=0.75, edgecolor="none", color="#ff7f0e")
    ax_obs.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_obs.set_xlabel("Valeur observee")
    ax_obs.set_ylabel("Residuel (pred - obs)")
    ax_obs.set_title("Residus vs observations")

    ax_hist.hist(residual, bins=min(20, max(5, len(df) // 3)), color="#2ca02c", alpha=0.85)
    ax_hist.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_hist.set_xlabel("Residuel")
    ax_hist.set_ylabel("Frequence")
    ax_hist.set_title("Histogramme des residus")

    sorted_std = np.sort(standardized)
    if len(sorted_std) > 0:
        probs = (np.arange(1, len(sorted_std) + 1) - 0.5) / len(sorted_std)
        theoretical = np.array([NormalDist().inv_cdf(float(p)) for p in probs], dtype=float)
        qq_min = float(min(theoretical.min(), sorted_std.min()))
        qq_max = float(max(theoretical.max(), sorted_std.max()))
        ax_qq.scatter(theoretical, sorted_std, alpha=0.75, edgecolor="none", color="#d62728")
        ax_qq.plot([qq_min, qq_max], [qq_min, qq_max], color="black", linewidth=1.0, linestyle="--")
    ax_qq.set_xlabel("Quantiles normaux theoriques")
    ax_qq.set_ylabel("Residus standardises")
    ax_qq.set_title("QQ-plot")

    summary_text = (
        f"n = {len(df)}\n"
        f"RMSE = {rmse_from_frame(df):.4g}\n"
        f"MAE = {df['abs_residual'].mean():.4g}\n"
        f"Biais moyen = {df['residual'].mean():.4g}"
    )
    fig.suptitle(title, fontsize=13)
    fig.text(0.985, 0.02, summary_text, ha="right", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_order(df: pd.DataFrame, output_file: Path, title: str) -> None:
    """Trace les résidus selon l'ordre des observations."""
    residual = df["residual"].to_numpy(dtype=float)
    order = np.arange(1, len(df) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(order, residual, marker="o", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Ordre des observations")
    ax.set_ylabel("Residuel (pred - obs)")
    ax.set_title("Residus vs ordre des observations")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_curve_frame(df: pd.DataFrame, model_spec: ModelSpec) -> pd.DataFrame:
    available = [col for col in CURVE_COLUMNS if col in df.columns]
    if not available:
        return pd.DataFrame()

    sample_id = df["sample_id"] if "sample_id" in df.columns else pd.Series(np.arange(1, len(df) + 1))
    work = pd.DataFrame({"sample_id": sample_id})
    for col in available:
        if col == "curve_valid":
            work[col] = df[col].fillna(False).astype(bool)
        else:
            work[col] = pd.to_numeric(df[col], errors="coerce")

    for key, value in asdict(model_spec).items():
        if key == "prediction_file":
            work[key] = str(value)
        else:
            work[key] = value
    work["model_key"] = model_spec.model_key
    return work


def summarize_curve_metrics(df: pd.DataFrame) -> dict[str, object]:
    curve_valid = df["curve_valid"] if "curve_valid" in df.columns else pd.Series([True] * len(df))
    summary = {
        "source": df["source"].iloc[0],
        "dataset": df["dataset"].iloc[0],
        "method": df["method"].iloc[0],
        "model_name": df["model_name"].iloc[0],
        "model_label": df["model_label"].iloc[0],
        "model_family": df["model_family"].iloc[0],
        "selection_mode": df["selection_mode"].iloc[0],
        "model_key": df["model_key"].iloc[0],
        "n_rows": int(len(df)),
        "valid_curve_count": int(curve_valid.fillna(False).sum()),
    }

    for metric in ("curve_rmse_tau_u", "curve_mae_tau_u", "curve_r2_tau_u"):
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_median"] = float(values.median())
        summary[f"{metric}_q95"] = float(values.quantile(0.95))
        summary[f"{metric}_max"] = float(values.max())

    return summary


def plot_curve_diagnostics(df: pd.DataFrame, output_file: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    metric_specs = [
        ("curve_rmse_tau_u", "RMSE de la courbe", "#1f77b4"),
        ("curve_mae_tau_u", "MAE de la courbe", "#ff7f0e"),
        ("curve_r2_tau_u", "R2 de la courbe", "#2ca02c"),
    ]

    for ax, (metric, label, color) in zip(axes, metric_specs):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            ax.set_visible(False)
            continue
        ax.hist(values, bins=min(20, max(5, len(values) // 3)), color=color, alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel(metric)
        ax.set_ylabel("Frequence")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze_model(
    model_spec: ModelSpec,
    output_root: Path,
) -> tuple[
    list[pd.DataFrame],
    list[dict[str, object]],
    list[dict[str, object]],
    pd.DataFrame,
    dict[str, object] | None,
    list[dict[str, object]],
]:
    df = pd.read_csv(model_spec.prediction_file)

    scalar_frames: list[pd.DataFrame] = []
    scalar_summaries: list[dict[str, object]] = []
    assumption_summaries: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []

    scalar_targets = resolve_scalar_targets(df)
    scalar_plot_dir = output_root / "plots" / model_spec.model_key
    for target_name, (true_col, pred_col) in scalar_targets.items():
        residual_df = build_scalar_residual_frame(df, model_spec, target_name, true_col, pred_col)
        if residual_df.empty:
            continue

        scalar_frames.append(residual_df)
        scalar_summaries.append(summarize_scalar_residuals(residual_df))
        assumption_summaries.append(
            {
                "source": model_spec.source,
                "dataset": model_spec.dataset,
                "method": model_spec.method,
                "model_name": model_spec.model_name,
                "model_label": model_spec.model_label,
                "model_family": model_spec.model_family,
                "selection_mode": model_spec.selection_mode,
                "model_key": model_spec.model_key,
                "target": target_name,
                "target_label": TARGET_SPECS[target_name]["label"],
                **check_regression_assumptions(residual_df, alpha=0.05),
            }
        )

        plot_file = scalar_plot_dir / f"{target_name}_residual_diagnostics.png"
        title = (
            f"{model_spec.model_label} | {model_spec.dataset} | {TARGET_SPECS[target_name]['label']}"
        )
        plot_scalar_diagnostics(residual_df, plot_file, title)
        plot_records.append(
            {
                "model_key": model_spec.model_key,
                "dataset": model_spec.dataset,
                "method": model_spec.method,
                "model_name": model_spec.model_name,
                "plot_type": "scalar_residual_diagnostics",
                "target": target_name,
                "plot_file": str(plot_file),
            }
        )
        order_plot_file = scalar_plot_dir / f"{target_name}_residual_order.png"
        plot_residual_order(residual_df, order_plot_file, title)
        plot_records.append(
            {
                "model_key": model_spec.model_key,
                "dataset": model_spec.dataset,
                "method": model_spec.method,
                "model_name": model_spec.model_name,
                "plot_type": "residual_order_plot",
                "target": target_name,
                "plot_file": str(order_plot_file),
            }
        )

    curve_df = build_curve_frame(df, model_spec)
    curve_summary: dict[str, object] | None = None
    if not curve_df.empty:
        curve_summary = summarize_curve_metrics(curve_df)
        curve_plot_file = scalar_plot_dir / "curve_metric_distributions.png"
        curve_title = f"{model_spec.model_label} | {model_spec.dataset} | metriques tau(u)"
        plot_curve_diagnostics(curve_df, curve_plot_file, curve_title)
        plot_records.append(
            {
                "model_key": model_spec.model_key,
                "dataset": model_spec.dataset,
                "method": model_spec.method,
                "model_name": model_spec.model_name,
                "plot_type": "curve_metric_distributions",
                "target": "tau_u",
                "plot_file": str(curve_plot_file),
            }
        )

    return scalar_frames, scalar_summaries, assumption_summaries, curve_df, curve_summary, plot_records


def write_outputs(
    output_root: Path,
    specs: list[ModelSpec],
    scalar_frames: list[pd.DataFrame],
    scalar_summaries: list[dict[str, object]],
    assumption_summaries: list[dict[str, object]],
    curve_frames: list[pd.DataFrame],
    curve_summaries: list[dict[str, object]],
    plot_records: list[dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for spec in specs:
        manifest_rows.append(
            {
                "source": spec.source,
                "dataset": spec.dataset,
                "method": spec.method,
                "model_name": spec.model_name,
                "model_label": spec.model_label,
                "model_family": spec.model_family,
                "selection_mode": spec.selection_mode,
                "prediction_file": str(spec.prediction_file),
                "model_key": spec.model_key,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(output_root / "model_manifest.csv", index=False)

    if scalar_frames:
        pd.concat(scalar_frames, ignore_index=True).to_csv(
            output_root / "scalar_residuals_long.csv",
            index=False,
        )
    if scalar_summaries:
        pd.DataFrame(scalar_summaries).sort_values(
            ["source", "dataset", "method", "model_name", "target"]
        ).to_csv(output_root / "scalar_residual_summary.csv", index=False)
    if assumption_summaries:
        pd.DataFrame(assumption_summaries).sort_values(
            ["source", "dataset", "method", "model_name", "target"]
        ).to_csv(output_root / "assumption_checks_summary.csv", index=False)

    if curve_frames:
        pd.concat(curve_frames, ignore_index=True).to_csv(
            output_root / "curve_metrics_long.csv",
            index=False,
        )
    if curve_summaries:
        pd.DataFrame(curve_summaries).sort_values(
            ["source", "dataset", "method", "model_name"]
        ).to_csv(output_root / "curve_metrics_summary.csv", index=False)

    if plot_records:
        pd.DataFrame(plot_records).to_csv(output_root / "plot_index.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    specs = build_model_specs(
        skip_best_by_method=args.skip_best_by_method,
        skip_pca_final=args.skip_pca_final,
    )

    scalar_frames: list[pd.DataFrame] = []
    scalar_summaries: list[dict[str, object]] = []
    assumption_summaries: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    curve_summaries: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []

    for spec in specs:
        (
            model_scalar_frames,
            model_scalar_summaries,
            model_assumption_summaries,
            curve_df,
            curve_summary,
            model_plot_records,
        ) = analyze_model(spec, args.output_root)
        scalar_frames.extend(model_scalar_frames)
        scalar_summaries.extend(model_scalar_summaries)
        assumption_summaries.extend(model_assumption_summaries)
        if not curve_df.empty:
            curve_frames.append(curve_df)
        if curve_summary is not None:
            curve_summaries.append(curve_summary)
        plot_records.extend(model_plot_records)

    write_outputs(
        output_root=args.output_root,
        specs=specs,
        scalar_frames=scalar_frames,
        scalar_summaries=scalar_summaries,
        assumption_summaries=assumption_summaries,
        curve_frames=curve_frames,
        curve_summaries=curve_summaries,
        plot_records=plot_records,
    )
    print(f"Residual analysis written to: {args.output_root}")
    print(f"Models analyzed: {len(specs)}")
    print(f"Scalar residual summaries: {len(scalar_summaries)}")
    print(f"Assumption summaries: {len(assumption_summaries)}")
    print(f"Curve summaries: {len(curve_summaries)}")
    print(f"Diagnostic plots: {len(plot_records)}")


if __name__ == "__main__":
    main()
