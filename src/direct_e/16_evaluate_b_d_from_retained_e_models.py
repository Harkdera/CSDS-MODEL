"""Évalue `b` et `d` reconstruits à partir des meilleurs modèles retenus pour l'estimation directe de `e`."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    EVALUATION_DIR,
    SPLIT_FILES,
    TOP5_DIR,
    build_direct_e_dataset,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
    dataset_group,
    dataset_slug,
    ensure_output_dirs,
    parse_feature_list,
)


RIDGE_ALPHA = 1.0
POLY_DEGREE = 2


def build_exponential_model():
    """Construit le pipeline exponentiel utilisé sur `log(e-c)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def build_polynomial_model():
    """Construit le pipeline polynomial utilisé sur `log(e-c)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=RIDGE_ALPHA)),
    ])


def load_retained_models_for_dataset(dataset_name: str) -> list[dict]:
    """Charge les modèles retenus à partir des fichiers top-5 résumés."""
    slug = dataset_slug(dataset_name)
    group_dir = TOP5_DIR / dataset_group(dataset_name)

    model_specs: list[dict] = []

    exp_summary_file = group_dir / f"{slug}_exp_top5_models_log_e_gap_summary.csv"
    if exp_summary_file.exists():
        exp_df = pd.read_csv(exp_summary_file)
        for _, row in exp_df.iterrows():
            model_specs.append({
                "Dataset": dataset_name,
                "Model_Family": "exponential",
                "Selection_Mode": row.get("Selection_Mode", "log_gap"),
                "Rank_in_saved_results": int(row["Rank_in_saved_results"]),
                "Model_Label": f"{dataset_name}_direct_exp_rank_{int(row['Rank_in_saved_results'])}",
                "Features": parse_feature_list(row["Features"]),
                "Saved_R2_val_z": row.get("Saved_R2_val_z", np.nan),
                "Saved_R2_cv_mean_z": row.get("Saved_R2_cv_mean_z", np.nan),
                "Saved_R2_val_e": row.get("Saved_R2_val_e", np.nan),
                "Saved_RMSE_val_e": row.get("Saved_RMSE_val_e", np.nan),
                "Saved_Selection_Score": row.get("Saved_Selection_Score", np.nan),
            })

    poly_summary_file = group_dir / f"{slug}_poly_top5_models_log_e_gap_summary.csv"
    if poly_summary_file.exists():
        poly_df = pd.read_csv(poly_summary_file)
        for _, row in poly_df.iterrows():
            model_specs.append({
                "Dataset": dataset_name,
                "Model_Family": "polynomial",
                "Selection_Mode": "log_gap",
                "Rank_in_saved_results": int(row["Rank_in_saved_results"]),
                "Model_Label": f"{dataset_name}_direct_poly_rank_{int(row['Rank_in_saved_results'])}",
                "Features": parse_feature_list(row["Input_Features"]),
                "Saved_R2_val_z": row.get("Saved_R2_val_z", np.nan),
                "Saved_R2_cv_mean_z": row.get("Saved_R2_cv_mean_z", np.nan),
                "Saved_R2_val_e": row.get("Saved_R2_val_e", np.nan),
                "Saved_RMSE_val_e": row.get("Saved_RMSE_val_e", np.nan),
                "Saved_Selection_Score": row.get("Saved_Selection_Score", np.nan),
            })

    return model_specs


def fit_and_predict(model_spec: dict, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Réentraîne le modèle retenu sur le dataset courant puis prédit `z` et `e`."""
    feature_cols = model_spec["Features"]
    X = data[feature_cols]
    y = data["log_e_minus_c_csds"]

    if model_spec["Model_Family"] == "exponential":
        model = build_exponential_model()
    else:
        model = build_polynomial_model()

    model.fit(X, y)
    z_pred = model.predict(X)
    e_pred = data["c_target"].to_numpy(dtype=float) + np.exp(z_pred)
    return z_pred, e_pred


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("EVALUATION DE b ET d A PARTIR DE e ESTIME")
    print("=" * 100)

    all_summary_rows = []

    for dataset_name in SPLIT_FILES:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        data = build_direct_e_dataset(dataset_name)
        model_specs = load_retained_models_for_dataset(dataset_name)
        group_dir = EVALUATION_DIR / dataset_group(dataset_name)
        group_dir.mkdir(parents=True, exist_ok=True)

        print(f"Rows available: {len(data)}")
        print(f"Retained models found: {len(model_specs)}")

        dataset_summary_rows = []

        for model_spec in model_specs:
            model_label = model_spec["Model_Label"]
            print(f"\nProcessing model: {model_label}")

            z_pred, e_pred = fit_and_predict(model_spec, data)

            temp = data.copy()
            temp["z_pred"] = z_pred
            temp["e_pred"] = e_pred
            temp["d_pred"] = compute_d_from_e_peak_equation(temp, e_col="e_pred")
            temp["b_pred"] = temp["d_pred"] - temp["a_csds"]
            temp["e_constraint_respected"] = temp["e_pred"] > temp["c_target"]
            temp["d_positive"] = temp["d_pred"] > 0
            temp["b_positive"] = temp["b_pred"] > 0
            temp["Dataset"] = dataset_name
            temp["Model_Label"] = model_label
            temp["Model_Family"] = model_spec["Model_Family"]
            temp["Selection_Mode"] = model_spec["Selection_Mode"]
            temp["Rank_in_saved_results"] = model_spec["Rank_in_saved_results"]
            temp["Model_Features"] = " + ".join(model_spec["Features"])

            temp, pooled_curve = compute_curve_metrics_for_direct_prediction(temp)
            z_metrics = compute_metrics(temp["log_e_minus_c_csds"], temp["z_pred"])
            e_metrics = compute_metrics(temp["e_csds"], temp["e_pred"])
            d_metrics = compute_metrics(temp["d_csds"], temp["d_pred"])
            b_metrics = compute_metrics(temp["b_csds"], temp["b_pred"])

            detailed_file = group_dir / f"{model_label}_b_d_from_e_predictions.csv"
            temp.to_csv(detailed_file, index=False)

            summary_row = {
                "dataset": dataset_name,
                "model_name": model_label,
                "model_family": model_spec["Model_Family"],
                "selection_mode": model_spec["Selection_Mode"],
                "rank_in_saved_results": model_spec["Rank_in_saved_results"],
                "model_features": " + ".join(model_spec["Features"]),
                "saved_r2_val_z": model_spec["Saved_R2_val_z"],
                "saved_r2_cv_mean_z": model_spec["Saved_R2_cv_mean_z"],
                "saved_r2_val_e": model_spec["Saved_R2_val_e"],
                "saved_rmse_val_e": model_spec["Saved_RMSE_val_e"],
                "saved_selection_score": model_spec["Saved_Selection_Score"],
                "n_rows": len(temp),
                "rmse_z": z_metrics["RMSE"],
                "r2_z": z_metrics["R2"],
                "rmse_d": d_metrics["RMSE"],
                "r2_d": d_metrics["R2"],
                "rmse_b": b_metrics["RMSE"],
                "r2_b": b_metrics["R2"],
                "rmse_e": e_metrics["RMSE"],
                "r2_e": e_metrics["R2"],
                "rmse_tau_u": pooled_curve["rmse_tau_u"],
                "r2_tau_u": pooled_curve["r2_tau_u"],
                "mean_curve_rmse_tau_u": float(temp["curve_rmse_tau_u"].mean()),
                "median_curve_rmse_tau_u": float(temp["curve_rmse_tau_u"].median()),
                "max_curve_rmse_tau_u": float(temp["curve_rmse_tau_u"].max()),
                "valid_curve_count": int(temp["curve_valid"].sum()),
                "constraint_ok_count": int(temp["e_constraint_respected"].sum()),
                "d_positive_count": int(temp["d_positive"].fillna(False).sum()),
                "b_positive_count": int(temp["b_positive"].fillna(False).sum()),
                "detailed_output_file": str(detailed_file),
            }

            dataset_summary_rows.append(summary_row)
            all_summary_rows.append(summary_row)

            print(f"Saved detailed file: {detailed_file}")
            print(f"RMSE(z) = {summary_row['rmse_z']:.6f} | R2(z) = {summary_row['r2_z']:.6f}")
            print(f"RMSE(d) = {summary_row['rmse_d']:.6f} | R2(d) = {summary_row['r2_d']:.6f}")
            print(f"RMSE(b) = {summary_row['rmse_b']:.6f} | R2(b) = {summary_row['r2_b']:.6f}")
            print(f"RMSE(e) = {summary_row['rmse_e']:.6f} | R2(e) = {summary_row['r2_e']:.6f}")
            print(f"RMSE tau(u) = {summary_row['rmse_tau_u']:.6f} | R2 tau(u) = {summary_row['r2_tau_u']:.6f}")

        dataset_summary_df = pd.DataFrame(dataset_summary_rows).sort_values(
            by=["rmse_tau_u", "rmse_d", "rmse_e", "r2_e"],
            ascending=[True, True, True, False],
        )
        dataset_summary_file = group_dir / f"{dataset_slug(dataset_name)}_summary_b_d_from_e_models.csv"
        dataset_summary_df.to_csv(dataset_summary_file, index=False)
        print(f"\nDataset summary saved: {dataset_summary_file}")

    global_summary_df = pd.DataFrame(all_summary_rows).sort_values(
        by=["dataset", "rmse_tau_u", "rmse_d", "rmse_e", "r2_e"],
        ascending=[True, True, True, True, False],
    )
    global_summary_file = EVALUATION_DIR / "summary_all_b_d_from_e_models.csv"
    global_summary_df.to_csv(global_summary_file, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Global summary saved: {global_summary_file}")


if __name__ == "__main__":
    main()
