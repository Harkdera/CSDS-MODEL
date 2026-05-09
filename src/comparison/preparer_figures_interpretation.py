from __future__ import annotations

from collections import Counter
import os
from pathlib import Path

from src.utils.common_methodologie_1 import build_d_dataset
from src.utils.common_methodologie_2 import build_direct_e_dataset
from src.utils.paths import COMPARISON_RESULTS_DIR, MATPLOTLIB_CACHE_DIR, PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RANKING_FILE = (
    COMPARISON_RESULTS_DIR
    / "model-ranking"
    / "composite_model_ranking_all_retained_models.csv"
)
OUTPUT_ROOT = COMPARISON_RESULTS_DIR / "interpretation"
SCATTER_ROOT = OUTPUT_ROOT / "scatter_plots"
TABLE_ROOT = OUTPUT_ROOT / "model_selection"

DATASETS = ["FULL", "LOW_1", "LOW_2", "HIGH"]
METHODS = ["methodologie_1", "methodologie_2"]
SCORE_VARIANTS = ["balanced", "performance_only", "stability_reinforced"]
TOP_N_MODELS = 2
TOP_N_FEATURES = 5
TOP_N_COMBINED_FEATURES = 10
TOP_N_METHOD_FEATURES = 10
BASE_PREDICTOR_COLS = {
    "sigma_n_MPa",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "u_r_mm",
    "tau_r_MPa",
}
ENGINEERED_PREDICTOR_COLS = {
    "sigma_n_x_u_r",
    "sigma_n_x_u_p",
    "sigma_n_x_tau_p",
    "sigma_n_x_tau_r",
    "u_r_x_u_p",
    "u_r_x_tau_p",
    "u_r_x_tau_r",
    "u_p_x_tau_p",
    "u_p_x_tau_r",
    "tau_p_x_tau_r",
    "u_p_div_u_r",
    "u_r_div_u_p",
    "tau_p_div_tau_r",
    "tau_r_div_tau_p",
    "u_r_div_tau_p",
    "u_r_div_tau_r",
    "u_p_div_tau_p",
    "u_p_div_tau_r",
    "tau_p_div_u_r",
    "tau_p_div_u_p",
    "tau_r_div_u_r",
    "tau_r_div_u_p",
    "tau_p_div_sigma_n",
    "tau_r_div_sigma_n",
    "sigma_n_div_tau_p",
    "sigma_n_div_tau_r",
    "sigma_n_div_u_r",
    "sigma_n_div_u_p",
}
ALLOWED_PREDICTOR_COLS = BASE_PREDICTOR_COLS | ENGINEERED_PREDICTOR_COLS

METHOD_TARGET_COLUMNS = {
    "methodologie_1": "d_csds",
    "methodologie_2": "e_csds",
}

TARGET_COLUMNS = {
    "d_csds": "d",
    "e_csds": "e",
}

DISPLAY_LABELS = {
    "sigma_n_MPa": r"$\sigma_n$ (MPa)",
    "u_r_mm": r"$u_r$ (mm)",
    "delta_peak_mm": r"$u_p$ (mm)",
    "tau_peak_MPa_csds": r"$\tau_p$ (MPa)",
    "tau_r_MPa": r"$\tau_r$ (MPa)",
    "tau_peak_estimated": r"$\tau_p$ estimé (MPa)",
    "u_r_estimated": r"$u_r$ estimé (mm)",
    "tau_r_estimated": r"$\tau_r$ estimé (MPa)",
    "u_p_div_u_r": r"$u_p/u_r$",
    "u_r_div_u_p": r"$u_r/u_p$",
    "tau_p_div_tau_r": r"$\tau_p/\tau_r$",
    "tau_r_div_tau_p": r"$\tau_r/\tau_p$",
    "u_r_div_tau_p": r"$u_r/\tau_p$",
    "u_r_div_tau_r": r"$u_r/\tau_r$",
    "u_p_div_tau_p": r"$u_p/\tau_p$",
    "u_p_div_tau_r": r"$u_p/\tau_r$",
    "tau_p_div_u_r": r"$\tau_p/u_r$",
    "tau_p_div_u_p": r"$\tau_p/u_p$",
    "tau_r_div_u_r": r"$\tau_r/u_r$",
    "tau_r_div_u_p": r"$\tau_r/u_p$",
    "tau_p_div_sigma_n": r"$\tau_p/\sigma_n$",
    "tau_r_div_sigma_n": r"$\tau_r/\sigma_n$",
    "sigma_n_div_tau_p": r"$\sigma_n/\tau_p$",
    "sigma_n_div_tau_r": r"$\sigma_n/\tau_r$",
    "sigma_n_div_u_r": r"$\sigma_n/u_r$",
    "sigma_n_div_u_p": r"$\sigma_n/u_p$",
    "sigma_n_x_u_r": r"$\sigma_n u_r$",
    "sigma_n_x_u_p": r"$\sigma_n u_p$",
    "sigma_n_x_tau_p": r"$\sigma_n \tau_p$",
    "sigma_n_x_tau_r": r"$\sigma_n \tau_r$",
    "u_r_x_u_p": r"$u_r u_p$",
    "u_r_x_tau_p": r"$u_r \tau_p$",
    "u_r_x_tau_r": r"$u_r \tau_r$",
    "u_p_x_tau_p": r"$u_p \tau_p$",
    "u_p_x_tau_r": r"$u_p \tau_r$",
    "tau_p_x_tau_r": r"$\tau_p \tau_r$",
    "d_csds": r"$d$",
    "e_csds": r"$e$",
}


def parse_features(value: object) -> list[str]:
    """Split the stored model feature string into a clean feature list."""
    if pd.isna(value):
        return []
    return [
        part.strip()
        for part in str(value).split("+")
        if part.strip() and part.strip() in ALLOWED_PREDICTOR_COLS
    ]


def parse_features_raw(value: object) -> list[str]:
    """Split the stored model feature string without applying predictor exclusions."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("+") if part.strip()]


def label(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def load_ranking() -> pd.DataFrame:
    if not RANKING_FILE.exists():
        raise FileNotFoundError(
            f"Missing ranking file: {RANKING_FILE}. Run src/comparaison/19_rank_models_composite_score.py first."
        )
    ranking = pd.read_csv(RANKING_FILE)
    required = {"methodologie", "dataset", "model_name", "features"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"Ranking file is missing required columns: {sorted(missing)}")
    return ranking


def select_top_models(ranking: pd.DataFrame) -> pd.DataFrame:
    """Return the top two models per methodology, dataset and score variant."""
    rows: list[pd.DataFrame] = []
    ranking = ranking.copy()
    ranking["_has_only_allowed_predictors"] = ranking["features"].apply(
        lambda value: all(feature in ALLOWED_PREDICTOR_COLS for feature in parse_features_raw(value))
    )
    ranking = ranking[ranking["_has_only_allowed_predictors"]].copy()

    for score_variant in SCORE_VARIANTS:
        score_col = f"score_{score_variant}"
        if score_col not in ranking.columns:
            raise ValueError(f"Ranking file is missing score column: {score_col}")

        sorted_ranking = ranking.sort_values(
            ["methodologie", "dataset", score_col, "rmse_tau_u", "aicc_tau_u", "cv_std"],
            ascending=[True, True, True, True, True, True],
            na_position="last",
        )
        top = (
            sorted_ranking
            .groupby(["methodologie", "dataset"], as_index=False)
            .head(TOP_N_MODELS)
            .copy()
        )
        top["score_variant"] = score_variant
        top["rank_within_score_variant"] = (
            top.groupby(["methodologie", "dataset"]).cumcount() + 1
        )
        top = top.drop(columns=["_has_only_allowed_predictors"], errors="ignore")
        rows.append(top)

    return pd.concat(rows, ignore_index=True)


def compute_feature_frequency(top_models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for (methodologie, dataset), group in top_models.groupby(["methodologie", "dataset"], sort=True):
        counter: Counter[str] = Counter()
        model_counter: Counter[str] = Counter()

        for _, row in group.iterrows():
            features = parse_features(row["features"])
            counter.update(features)
            model_counter.update(set(features))

        for feature, count in counter.most_common():
            rows.append({
                "methodologie": methodologie,
                "dataset": dataset,
                "feature": feature,
                "feature_label": label(feature),
                "total_occurrences": count,
                "models_containing_feature": model_counter[feature],
                "n_top_model_rows_considered": len(group),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["methodologie", "dataset", "total_occurrences", "models_containing_feature", "feature"],
        ascending=[True, True, False, False, True],
    )


def load_dataset(methodologie: str, dataset: str) -> pd.DataFrame:
    if methodologie == "methodologie_1":
        return build_d_dataset(dataset)
    if methodologie == "methodologie_2":
        return build_direct_e_dataset(dataset)
    raise ValueError(f"Unsupported methodology: {methodologie}")


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_num[mask], y_num[mask])[0, 1])


def plot_scatter(data: pd.DataFrame, x_col: str, y_col: str, methodologie: str, dataset: str, output_file: Path) -> None:
    x = pd.to_numeric(data[x_col], errors="coerce")
    y = pd.to_numeric(data[y_col], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return

    x_valid = x[mask]
    y_valid = y[mask]
    r_value = pearson_r(x_valid, y_valid)

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    ax.scatter(x_valid, y_valid, alpha=0.72, s=34, edgecolor="white", linewidth=0.35)

    if mask.sum() >= 3 and x_valid.nunique() > 1:
        slope, intercept = np.polyfit(x_valid.to_numpy(dtype=float), y_valid.to_numpy(dtype=float), deg=1)
        x_line = np.linspace(float(x_valid.min()), float(x_valid.max()), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linewidth=1.3, linestyle="--", label="Tendance linéaire")
        ax.legend(loc="best")

    method_label = "Méthodologie 1" if methodologie == "methodologie_1" else "Méthodologie 2"
    target_label = TARGET_COLUMNS.get(y_col, y_col)
    ax.set_title(f"{method_label} - {dataset} - {target_label} en fonction de {label(x_col)}")
    ax.set_xlabel(label(x_col))
    ax.set_ylabel(label(y_col))
    ax.grid(True, alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"n = {int(mask.sum())}\nr = {r_value:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def create_scatter_plots(feature_frequency: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for methodologie in METHODS:
        for dataset in DATASETS:
            subset = feature_frequency[
                (feature_frequency["methodologie"] == methodologie) &
                (feature_frequency["dataset"] == dataset)
            ].copy()
            if subset.empty:
                continue

            data = load_dataset(methodologie, dataset)
            features = [
                feature
                for feature in subset["feature"].head(TOP_N_FEATURES).tolist()
                if feature in data.columns
            ]

            for feature in features:
                for target_col in TARGET_COLUMNS:
                    if target_col not in data.columns:
                        continue
                    output_file = (
                        SCATTER_ROOT
                        / methodologie
                        / dataset.lower()
                        / f"{TARGET_COLUMNS[target_col]}_vs_{safe_slug(feature)}.png"
                    )
                    plot_scatter(data, feature, target_col, methodologie, dataset, output_file)
                    rows.append({
                        "methodologie": methodologie,
                        "dataset": dataset,
                        "target": target_col,
                        "target_label": TARGET_COLUMNS[target_col],
                        "feature": feature,
                        "feature_label": label(feature),
                        "pearson_r": pearson_r(data[feature], data[target_col]),
                        "output_file": str(output_file),
                    })

    return pd.DataFrame(rows)


def compute_combined_feature_frequency(feature_frequency: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature frequency across both methodologies and all datasets."""
    if feature_frequency.empty:
        return feature_frequency

    rows: list[dict] = []
    for feature, group in feature_frequency.groupby("feature", sort=True):
        rows.append({
            "feature": feature,
            "feature_label": label(feature),
            "total_occurrences": int(group["total_occurrences"].sum()),
            "models_containing_feature": int(group["models_containing_feature"].sum()),
            "n_methodologies": int(group["methodologie"].nunique()),
            "methodologies": ", ".join(sorted(group["methodologie"].unique())),
            "n_datasets": int(group["dataset"].nunique()),
            "datasets": ", ".join(sorted(group["dataset"].unique())),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(
        [
            "total_occurrences",
            "models_containing_feature",
            "n_methodologies",
            "n_datasets",
            "feature",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def create_combined_top_feature_scatter_plots(combined_frequency: pd.DataFrame) -> pd.DataFrame:
    """Create d/e scatter plots for the top combined variables from both methodologies."""
    rows: list[dict] = []
    if combined_frequency.empty:
        return pd.DataFrame(rows)

    top_features = combined_frequency["feature"].head(TOP_N_COMBINED_FEATURES).tolist()

    for methodologie in METHODS:
        for dataset in DATASETS:
            data = load_dataset(methodologie, dataset)
            available_features = [feature for feature in top_features if feature in data.columns]

            for feature in available_features:
                for target_col in TARGET_COLUMNS:
                    if target_col not in data.columns:
                        continue

                    output_file = (
                        SCATTER_ROOT
                        / "top10_combined_variables"
                        / methodologie
                        / dataset.lower()
                        / f"{TARGET_COLUMNS[target_col]}_vs_{safe_slug(feature)}.png"
                    )
                    plot_scatter(data, feature, target_col, methodologie, dataset, output_file)
                    rows.append({
                        "methodologie": methodologie,
                        "dataset": dataset,
                        "target": target_col,
                        "target_label": TARGET_COLUMNS[target_col],
                        "feature": feature,
                        "feature_label": label(feature),
                        "pearson_r": pearson_r(data[feature], data[target_col]),
                        "abs_pearson_r": abs(pearson_r(data[feature], data[target_col])),
                        "output_file": str(output_file),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["target_label", "abs_pearson_r", "methodologie", "dataset", "feature"],
        ascending=[True, False, True, True, True],
        na_position="last",
    )


def compute_method_feature_frequency(feature_frequency: pd.DataFrame) -> pd.DataFrame:
    """Aggregate feature frequency separately for each methodology."""
    if feature_frequency.empty:
        return feature_frequency

    rows: list[dict] = []
    for (methodologie, feature), group in feature_frequency.groupby(["methodologie", "feature"], sort=True):
        rows.append({
            "methodologie": methodologie,
            "feature": feature,
            "feature_label": label(feature),
            "total_occurrences": int(group["total_occurrences"].sum()),
            "models_containing_feature": int(group["models_containing_feature"].sum()),
            "n_datasets": int(group["dataset"].nunique()),
            "datasets": ", ".join(sorted(group["dataset"].unique())),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["methodologie", "total_occurrences", "models_containing_feature", "n_datasets", "feature"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)


def create_method_target_scatter_plots(method_frequency: pd.DataFrame) -> pd.DataFrame:
    """
    Create target-specific scatter plots:
    - methodology 1: recurrent variables against d
    - methodology 2: recurrent variables against e
    """
    rows: list[dict] = []
    if method_frequency.empty:
        return pd.DataFrame(rows)

    for methodologie in METHODS:
        target_col = METHOD_TARGET_COLUMNS[methodologie]
        method_features = (
            method_frequency[method_frequency["methodologie"] == methodologie]
            ["feature"]
            .head(TOP_N_METHOD_FEATURES)
            .tolist()
        )
        for dataset in DATASETS:
            data = load_dataset(methodologie, dataset)
            if target_col not in data.columns:
                continue

            for feature in method_features:
                if feature not in data.columns:
                    continue

                target_label = TARGET_COLUMNS[target_col]
                output_file = (
                    SCATTER_ROOT
                    / "method_specific_target"
                    / methodologie
                    / dataset.lower()
                    / f"{target_label}_vs_{safe_slug(feature)}.png"
                )
                plot_scatter(data, feature, target_col, methodologie, dataset, output_file)
                r_value = pearson_r(data[feature], data[target_col])
                rows.append({
                    "methodologie": methodologie,
                    "dataset": dataset,
                    "target": target_col,
                    "target_label": target_label,
                    "feature": feature,
                    "feature_label": label(feature),
                    "pearson_r": r_value,
                    "abs_pearson_r": abs(r_value),
                    "output_file": str(output_file),
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["methodologie", "abs_pearson_r", "dataset", "feature"],
        ascending=[True, False, True, True],
        na_position="last",
    )


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    SCATTER_ROOT.mkdir(parents=True, exist_ok=True)

    ranking = load_ranking()
    top_models = select_top_models(ranking)
    feature_frequency = compute_feature_frequency(top_models)
    scatter_index = create_scatter_plots(feature_frequency)
    method_frequency = compute_method_feature_frequency(feature_frequency)
    method_target_scatter_index = create_method_target_scatter_plots(method_frequency)

    top_models_file = TABLE_ROOT / "top2_models_by_score_variant.csv"
    frequency_file = TABLE_ROOT / "top2_feature_frequency_by_method_dataset.csv"
    method_frequency_file = TABLE_ROOT / "top10_feature_frequency_by_method.csv"
    scatter_index_file = SCATTER_ROOT / "scatter_plot_index.csv"
    method_target_scatter_index_file = SCATTER_ROOT / "method_specific_target_scatter_plot_index.csv"

    top_models.to_csv(top_models_file, index=False)
    feature_frequency.to_csv(frequency_file, index=False)
    method_frequency.to_csv(method_frequency_file, index=False)
    scatter_index.to_csv(scatter_index_file, index=False)
    method_target_scatter_index.to_csv(method_target_scatter_index_file, index=False)
    print("=" * 100)
    print("INTERPRETATION SCATTER PLOTS")
    print("=" * 100)
    print(f"Top 2 models saved: {top_models_file}")
    print(f"Feature frequency saved: {frequency_file}")
    print(f"Method-specific feature frequency saved: {method_frequency_file}")
    print(f"Scatter plot index saved: {scatter_index_file}")
    print(f"Method-specific target scatter plot index saved: {method_target_scatter_index_file}")
    print(f"Scatter plot folder: {SCATTER_ROOT}")
    print(f"Figures generated: {len(scatter_index)}")
    print(f"Method-specific target figures generated: {len(method_target_scatter_index)}")


if __name__ == "__main__":
    main()
