from __future__ import annotations

import ast
import math
import numpy as np
import pandas as pd

from src.utils.paths import COMPARISON_RESULTS_DIR, METHODOLOGIE_1_RESULTS_DIR, METHODOLOGIE_2_RESULTS_DIR


OUTPUT_DIR = COMPARISON_RESULTS_DIR / "model-ranking"
N_CURVE_POINTS = 100

SCORE_VARIANTS = {
    "balanced": {
        "rank_rmse_tau_u": 0.45,
        "rank_neg_r2_tau_u": 0.20,
        "rank_aicc_tau_u": 0.10,
        "rank_cv_std": 0.25,
    },
    "performance_only": {
        "rank_rmse_tau_u": 0.70,
        "rank_neg_r2_tau_u": 0.30,
        "rank_aicc_tau_u": 0.00,
        "rank_cv_std": 0.00,
    },
    "stability_reinforced": {
        "rank_rmse_tau_u": 0.40,
        "rank_neg_r2_tau_u": 0.15,
        "rank_aicc_tau_u": 0.05,
        "rank_cv_std": 0.40,
    },
}

DATASET_TO_FOLDER = {
    "FULL": "full",
    "LOW_1": "low",
    "LOW_2": "low",
    "HIGH": "high",
}


METHODS = {
    "methodologie_1": {
        "compare_summary": METHODOLOGIE_1_RESULTS_DIR / "reconstructed_parameters" / "tau_u" / "summary_all_models_compare_d_b_e_tau_u.csv",
        "top5_dir": METHODOLOGIE_1_RESULTS_DIR / "cross_validation" / "top5",
    },
    "methodologie_2": {
        "compare_summary": METHODOLOGIE_2_RESULTS_DIR / "reconstructed_parameters" / "tau_u" / "summary_all_models_compare_b_d_e_tau_u.csv",
        "top5_dir": METHODOLOGIE_2_RESULTS_DIR / "cross_validation" / "top5",
        "selection_dir": METHODOLOGIE_2_RESULTS_DIR / "heuristic",
    },
}


def parse_feature_list(value: object) -> list[str]:
    """Return a stable feature list from either a Python-list string or a plus-separated string."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return [str(item).strip() for item in parsed]
        except (SyntaxError, ValueError):
            pass
    return [part.strip() for part in text.split("+") if part.strip()]


def normalize_features(value: object) -> str:
    return " + ".join(parse_feature_list(value))


def polynomial_parameter_count(n_input_features: int, degree: int = 2) -> int:
    """Intercept + number of polynomial terms up to `degree` without bias."""
    if n_input_features <= 0:
        return 1
    # For degree 2: p linear terms + p*(p+1)/2 quadratic/interaction terms + intercept.
    if degree != 2:
        return 1 + sum(math.comb(n_input_features + d - 1, d) for d in range(1, degree + 1))
    return 1 + n_input_features + (n_input_features * (n_input_features + 1)) // 2


def linear_parameter_count(n_features: int) -> int:
    return max(1, int(n_features) + 1)


def compute_aicc_from_tau_rmse(rmse_tau_u: float, n_rows: int, n_parameters: int) -> float:
    """Approximate AICc from the pooled tau(u) RMSE over all curve points.

    The constant term is omitted because only relative model ranking is needed.
    """
    if not np.isfinite(n_rows) or not np.isfinite(n_parameters):
        return np.nan
    n_obs = int(n_rows) * N_CURVE_POINTS
    k = int(n_parameters)
    if n_obs <= k + 1 or not np.isfinite(rmse_tau_u) or rmse_tau_u <= 0:
        return np.nan
    rss = float(rmse_tau_u) ** 2 * n_obs
    aic = n_obs * np.log(rss / n_obs) + 2 * k
    correction = (2 * k * (k + 1)) / (n_obs - k - 1)
    return float(aic + correction)


def load_methodologie_1_metadata() -> pd.DataFrame:
    rows: list[dict] = []
    top5_dir = METHODS["methodologie_1"]["top5_dir"]
    for path in sorted(top5_dir.rglob("*_summary.csv")):
        df = pd.read_csv(path)
        is_poly = "_poly_" in path.name
        is_exp = "_exp_" in path.name
        for _, row in df.iterrows():
            dataset = row["Dataset"]
            rank = int(row["Rank_in_saved_results"])
            if is_poly:
                model_name = f"{dataset}_poly_rank_{rank}"
                n_input_features = int(row.get("N_Input_Features", row.get("N_Features", np.nan)))
                n_parameters = polynomial_parameter_count(n_input_features)
                features = row.get("Input_Features", row.get("Features", ""))
                cv_std = row.get("Saved_R2_cv_std_log", np.nan)
                family = "polynomial"
            elif is_exp:
                selection_mode = row.get("Selection_Mode", "")
                model_name = f"{dataset}_exp_{selection_mode}_rank_{rank}"
                n_input_features = int(row.get("N_Features", row.get("N_Input_Features", np.nan)))
                n_parameters = linear_parameter_count(n_input_features)
                features = row.get("Features", row.get("Input_Features", ""))
                cv_std = row.get("Saved_R2_cv_std_log", np.nan)
                family = "exponential"
            else:
                continue
            rows.append({
                "methodologie": "methodologie_1",
                "dataset": dataset,
                "model_name": model_name,
                "model_family": family,
                "n_input_features": n_input_features,
                "n_parameters_for_aicc": n_parameters,
                "features": normalize_features(features),
                "cv_std": cv_std,
            })
    return pd.DataFrame(rows)


def load_methodologie_2_selection_cv_lookup() -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    selection_dir = METHODS["methodologie_2"]["selection_dir"]
    for path in sorted(selection_dir.glob("*log_e_gap_selection_*.csv")):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            dataset = str(row.get("Dataset", ""))
            family = str(row.get("Model_Family", ""))
            features = normalize_features(row.get("Features", row.get("Feature_List", "")))
            lookup[(dataset, family, features)] = row.get("R2_cv_std_z", np.nan)
    return lookup


def load_methodologie_2_metadata() -> pd.DataFrame:
    rows: list[dict] = []
    lookup = load_methodologie_2_selection_cv_lookup()
    top5_dir = METHODS["methodologie_2"]["top5_dir"]
    for path in sorted(top5_dir.rglob("*_summary.csv")):
        df = pd.read_csv(path)
        is_poly = "_poly_" in path.name
        is_exp = "_exp_" in path.name
        for _, row in df.iterrows():
            dataset = row["Dataset"]
            rank = int(row["Rank_in_saved_results"])
            if is_poly:
                model_name = f"{dataset}_direct_poly_rank_{rank}"
                family = "polynomial"
                n_input_features = int(row.get("N_Input_Features", row.get("N_Features", np.nan)))
                n_parameters = polynomial_parameter_count(n_input_features)
                features = normalize_features(row.get("Input_Features", row.get("Features", "")))
            elif is_exp:
                model_name = f"{dataset}_direct_exp_rank_{rank}"
                family = "exponential"
                n_input_features = int(row.get("N_Features", row.get("N_Input_Features", np.nan)))
                n_parameters = linear_parameter_count(n_input_features)
                features = normalize_features(row.get("Features", row.get("Input_Features", "")))
            else:
                continue
            rows.append({
                "methodologie": "methodologie_2",
                "dataset": dataset,
                "model_name": model_name,
                "model_family": family,
                "n_input_features": n_input_features,
                "n_parameters_for_aicc": n_parameters,
                "features": features,
                "cv_std": lookup.get((dataset, family, features), np.nan),
            })
    return pd.DataFrame(rows)


def load_compare_summary(methodologie: str) -> pd.DataFrame:
    df = pd.read_csv(METHODS[methodologie]["compare_summary"])
    df = df.copy()
    df["methodologie"] = methodologie
    return df


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["negative_r2_tau_u"] = -pd.to_numeric(out["r2_tau_u"], errors="coerce")
    out["n_parameters_for_aicc"] = pd.to_numeric(out["n_parameters_for_aicc"], errors="coerce")
    out["cv_std"] = pd.to_numeric(out["cv_std"], errors="coerce")
    out["aicc_tau_u"] = [
        compute_aicc_from_tau_rmse(rmse, n_rows, n_params)
        for rmse, n_rows, n_params in zip(
            pd.to_numeric(out["rmse_tau_u"], errors="coerce"),
            pd.to_numeric(out["n_rows"], errors="coerce"),
            pd.to_numeric(out["n_parameters_for_aicc"], errors="coerce"),
        )
    ]

    scored = []
    for (_, _), group in out.groupby(["methodologie", "dataset"], sort=True):
        g = group.copy()
        g["rank_rmse_tau_u"] = g["rmse_tau_u"].rank(method="min", ascending=True, na_option="bottom")
        g["rank_neg_r2_tau_u"] = g["negative_r2_tau_u"].rank(method="min", ascending=True, na_option="bottom")
        g["rank_aicc_tau_u"] = g["aicc_tau_u"].rank(method="min", ascending=True, na_option="bottom")
        g["rank_cv_std"] = g["cv_std"].rank(method="min", ascending=True, na_option="bottom")
        for score_name, weights in SCORE_VARIANTS.items():
            score_col = f"score_{score_name}"
            rank_col = f"rank_{score_name}"
            g[score_col] = sum(g[col] * weight for col, weight in weights.items())
            g[rank_col] = g[score_col].rank(method="min", ascending=True)
        scored.append(g)
    return pd.concat(scored, ignore_index=True).sort_values(
        ["methodologie", "dataset", "score_balanced", "rmse_tau_u"],
        ascending=[True, True, True, True],
    )


def select_best_models(ranked: pd.DataFrame, score_name: str) -> pd.DataFrame:
    """Select one best model per methodology/dataset for one score variant."""
    score_col = f"score_{score_name}"
    return (
        ranked.sort_values(
            [
                "methodologie",
                "dataset",
                score_col,
                "rmse_tau_u",
                "negative_r2_tau_u",
                "aicc_tau_u",
                "cv_std",
            ],
            ascending=[True, True, True, True, True, True, True],
        )
        .groupby(["methodologie", "dataset"], as_index=False)
        .head(1)
        .copy()
        .assign(score_variant=score_name)
        .sort_values(["methodologie", "dataset"])
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compare = pd.concat([
        load_compare_summary("methodologie_1"),
        load_compare_summary("methodologie_2"),
    ], ignore_index=True)
    metadata = pd.concat([
        load_methodologie_1_metadata(),
        load_methodologie_2_metadata(),
    ], ignore_index=True)

    merged = compare.merge(
        metadata,
        on=["methodologie", "dataset", "model_name"],
        how="left",
        validate="one_to_one",
    )

    missing = merged[merged["n_parameters_for_aicc"].isna() | merged["cv_std"].isna()]
    if not missing.empty:
        print("WARNING: missing metadata for some models; they will rank last for missing fields.")
        print(missing[["methodologie", "dataset", "model_name", "n_parameters_for_aicc", "cv_std"]].to_string(index=False))
        merged["cv_std"] = merged["cv_std"].fillna(merged.groupby(["methodologie", "dataset"])["cv_std"].transform("max") * 10)

    ranked = add_composite_score(merged)
    best_by_variant = pd.concat(
        [select_best_models(ranked, score_name) for score_name in SCORE_VARIANTS],
        ignore_index=True,
    )
    best_balanced = best_by_variant[best_by_variant["score_variant"] == "balanced"].copy()

    detailed_file = OUTPUT_DIR / "composite_model_ranking_all_retained_models.csv"
    best_file = OUTPUT_DIR / "best_models_by_composite_score.csv"
    best_by_variant_file = OUTPUT_DIR / "best_models_by_score_variant.csv"
    weights_file = OUTPUT_DIR / "composite_score_weights.csv"

    ranked.to_csv(detailed_file, index=False)
    best_balanced.to_csv(best_file, index=False)
    best_by_variant.to_csv(best_by_variant_file, index=False)
    pd.DataFrame.from_dict(SCORE_VARIANTS, orient="index").rename_axis("score_variant").to_csv(weights_file)

    print("=" * 100)
    print("COMPOSITE MODEL RANKING")
    print("=" * 100)
    print(f"Detailed ranking saved: {detailed_file}")
    print(f"Best balanced-score models saved: {best_file}")
    print(f"Best models by score variant saved: {best_by_variant_file}")
    print(f"Weights saved: {weights_file}")
    print("\nBest models by score variant:")
    display_cols = [
        "score_variant", "methodologie", "dataset", "model_name", "model_family", "n_rows",
        "rmse_tau_u", "r2_tau_u", "aicc_tau_u", "cv_std",
        "score_balanced", "score_performance_only", "score_stability_reinforced",
    ]
    print(best_by_variant[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
