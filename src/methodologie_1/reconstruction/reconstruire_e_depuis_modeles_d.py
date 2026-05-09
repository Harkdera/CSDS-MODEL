"""Evaluate retained exponential and polynomial d-models, then derive e with Simon's equation."""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

from src.utils.common_methodologie_1 import (
    E_FROM_D_DIR,
    SPLIT_FILES,
    TOP5_DIR,
    build_d_dataset,
    compute_metrics,
    dataset_group,
    parse_feature_list,
)


# ============================================================
# 1) CONFIGURATION
# ============================================================
DATA_FILES = dict(SPLIT_FILES)

RIDGE_ALPHA = 1.0
POLY_DEGREE = 2

MAX_ITER = 100
TOL_F = 1e-10
TOL_X = 1e-10
EPS = 1e-12

OUTPUT_DIR = E_FROM_D_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) REQUIRED COLUMNS
# ============================================================
required_cols = [
    "sigma_n_MPa",
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
]


# ============================================================
# 3) HELPERS
# ============================================================
def load_and_prepare(dataset_name):
    """Load a dataset and rebuild the engineered variables used by the saved models."""
    return build_d_dataset(dataset_name, include_targets=("d_csds", "e_csds"))


def build_exponential_model():
    """Build the exponential Ridge pipeline on log(d)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=RIDGE_ALPHA))
    ])


def build_polynomial_model():
    """Build the polynomial Ridge pipeline on log(d)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)),
        ("reg", Ridge(alpha=RIDGE_ALPHA))
    ])


# ============================================================
# 4) LOAD RETAINED MODELS
# ============================================================
def load_exp_models_for_dataset(dataset_name):
    """Load the retained top-5 exponential model specifications for one dataset."""
    slug = dataset_name.lower()
    dataset_top5_dir = TOP5_DIR / dataset_group(dataset_name)

    summary_file = dataset_top5_dir / f"{slug}_exp_top5_models_equations_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    summary_df = pd.read_csv(summary_file)
    summary_df = summary_df[summary_df["Dataset"] == dataset_name].copy()

    model_specs = []
    for _, row in summary_df.iterrows():
        model_specs.append({
            "Dataset": dataset_name,
            "Model_Family": "exponential",
            "Selection_Mode": row["Selection_Mode"],
            "Rank_in_saved_results": int(row["Rank_in_saved_results"]),
            "Model_Label": f"{dataset_name}_exp_{row['Selection_Mode']}_rank_{int(row['Rank_in_saved_results'])}",
            "Features": parse_feature_list(row["Features"]),
            "Saved_R2_val_log": row.get("Saved_R2_val_log", np.nan),
            "Saved_R2_cv_mean_log": row.get("Saved_R2_cv_mean_log", np.nan),
            "Saved_R2_val_d": row.get("Saved_R2_val_d", np.nan),
            "Saved_RMSE_val_d": row.get("Saved_RMSE_val_d", np.nan),
            "Saved_R2_test_log": row.get("Saved_R2_test_log", np.nan),
            "Saved_RMSE_test_log": row.get("Saved_RMSE_test_log", np.nan),
            "Saved_R2_test_d": row.get("Saved_R2_test_d", np.nan),
            "Saved_RMSE_test_d": row.get("Saved_RMSE_test_d", np.nan),
            "Saved_Selection_Score": row.get("Saved_Selection_Score", np.nan),
        })

    return model_specs


def load_poly_models_for_dataset(dataset_name):
    """Load the retained top-5 polynomial model specifications for one dataset."""
    slug = dataset_name.lower()
    dataset_top5_dir = TOP5_DIR / dataset_group(dataset_name)

    summary_file = dataset_top5_dir / f"{slug}_poly_top5_models_equations_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_file}")

    summary_df = pd.read_csv(summary_file)
    summary_df = summary_df[summary_df["Dataset"] == dataset_name].copy()

    model_specs = []
    for _, row in summary_df.iterrows():
        rank = int(row["Rank_in_saved_results"])
        model_specs.append({
            "Dataset": dataset_name,
            "Model_Family": "polynomial",
            "Selection_Mode": "log",
            "Rank_in_saved_results": rank,
            "Model_Label": f"{dataset_name}_poly_rank_{rank}",
            "Features": parse_feature_list(row["Input_Features"]),
            "Saved_R2_val_log": row.get("Saved_R2_val_log", np.nan),
            "Saved_R2_cv_mean_log": row.get("Saved_R2_cv_mean_log", np.nan),
            "Saved_R2_val_d": row.get("Saved_R2_val_d", np.nan),
            "Saved_RMSE_val_d": row.get("Saved_RMSE_val_d", np.nan),
            "Saved_R2_test_log": row.get("Saved_R2_test_log", np.nan),
            "Saved_RMSE_test_log": row.get("Saved_RMSE_test_log", np.nan),
            "Saved_R2_test_d": row.get("Saved_R2_test_d", np.nan),
            "Saved_RMSE_test_d": row.get("Saved_RMSE_test_d", np.nan),
            "Saved_Selection_Score": row.get("Saved_Selection_Score", np.nan),
        })

    return model_specs


def fit_and_predict_d_from_model(data, model_spec):
    """Refit one retained model on the full dataset and return d predictions."""
    feature_cols = model_spec["Features"]
    missing = [feat for feat in feature_cols if feat not in data.columns]
    if missing:
        raise ValueError(f"Missing features for {model_spec['Model_Label']}: {missing}")
    if "log_d_csds" not in data.columns:
        raise ValueError("The dataset does not contain d_csds/log_d_csds needed to refit the model.")

    X = data[feature_cols]
    y_log = data["log_d_csds"]

    if model_spec["Model_Family"] == "exponential":
        model = build_exponential_model()
    else:
        model = build_polynomial_model()

    model.fit(X, y_log)
    log_d_pred = model.predict(X)
    d_pred = np.exp(log_d_pred)
    return d_pred, log_d_pred


# ============================================================
# 5) SOLVE e FROM d WITH SIMON
# ============================================================
def solve_e_from_d_simon(d_value, u_p, u_r, tau_r, tau_p,
                         max_iter=100, tol_f=1e-10, tol_x=1e-10, eps=1e-12):
    """Solve e from estimated d using Simon's equation and Newton's method."""
    invalid_result = (np.nan, False, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    if not np.isfinite(d_value) or not np.isfinite(u_p) or not np.isfinite(u_r):
        return invalid_result
    if not np.isfinite(tau_r) or not np.isfinite(tau_p):
        return invalid_result
    if d_value <= 0 or u_p <= 0 or u_r <= 0 or tau_r <= 0 or tau_p <= 0:
        return invalid_result

    a = tau_r
    c = 5.0 / u_r
    b = d_value - a

    if b <= 0:
        return np.nan, False, 0, a, b, c, np.nan, np.nan, np.nan

    base = d_value / (b * c * u_p)
    if base <= 0:
        return np.nan, False, 0, a, b, c, np.nan, np.nan, np.nan

    e_p = np.log(base) / u_p + c
    e = e_p + 1.0

    lower_bound = max(c + eps, e_p + eps)
    if e <= lower_bound:
        e = lower_bound + 1.0

    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        arg = np.clip(u_p * (e - c), -700, 700)
        exp_term = np.exp(arg)

        F = d_value * e / (b * c) - exp_term
        dF = d_value / (b * c) - u_p * exp_term

        if not np.isfinite(F) or not np.isfinite(dF):
            break
        if abs(F) < tol_f:
            converged = True
            break
        if abs(dF) < eps:
            break

        step = F / dF
        e_new = e - step

        n_backtrack = 0
        while (not np.isfinite(e_new)) or (e_new <= lower_bound):
            step *= 0.5
            e_new = e - step
            n_backtrack += 1
            if n_backtrack > 50:
                e_new = np.nan
                break

        if not np.isfinite(e_new):
            break
        if abs(e_new - e) < tol_x:
            e = e_new
            converged = True
            break

        e = e_new

    if np.isfinite(e):
        arg = np.clip(u_p * (e - c), -700, 700)
        residual_515 = d_value * e / (b * c) - np.exp(arg)

        exp_5up_ur = np.exp(-5.0 * u_p / u_r)
        exp_eup = np.exp(-e * u_p)
        residual_518 = tau_p - tau_r * (1.0 - exp_5up_ur) - d_value * (exp_5up_ur - exp_eup)
    else:
        residual_515 = np.nan
        residual_518 = np.nan

    if np.isfinite(e) and not (e > c):
        converged = False

    return e, converged, n_iter, a, b, c, e_p, residual_515, residual_518


# ============================================================
# 6) MAIN LOOP
# ============================================================
all_summary_rows = []

for dataset_name in DATA_FILES:
    print("\n" + "=" * 90)
    print(f"DATASET: {dataset_name}")
    print("=" * 90)

    data = load_and_prepare(dataset_name)
    model_specs = load_exp_models_for_dataset(dataset_name) + load_poly_models_for_dataset(dataset_name)

    group_output_dir = OUTPUT_DIR / dataset_group(dataset_name)
    group_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rows available: {len(data)}")
    print(f"Retained models found: {len(model_specs)}")

    dataset_summary_rows = []

    for model_spec in model_specs:
        model_label = model_spec["Model_Label"]
        print(f"\nProcessing model: {model_label}")

        d_pred, log_d_pred = fit_and_predict_d_from_model(data, model_spec)

        temp = data.copy()
        temp["log_d_pred"] = log_d_pred
        temp["d_pred"] = d_pred

        e_results = temp.apply(
            lambda row: solve_e_from_d_simon(
                d_value=float(row["d_pred"]),
                u_p=float(row["delta_peak_mm"]),
                u_r=float(row["u_r_mm"]),
                tau_r=float(row["tau_r_MPa"]),
                tau_p=float(row["tau_peak_MPa_csds"]),
                max_iter=MAX_ITER,
                tol_f=TOL_F,
                tol_x=TOL_X,
                eps=EPS,
            ),
            axis=1,
            result_type="expand"
        )

        e_results.columns = [
            "e_pred",
            "e_converged",
            "e_iterations",
            "a_from_tau_r",
            "b_from_d_minus_a",
            "c_from_5_over_u_r",
            "e_p_initial",
            "residual_eq_515",
            "residual_eq_518",
        ]

        temp = pd.concat([temp, e_results], axis=1)
        temp["condition_c_lt_e"] = temp["e_pred"] > temp["c_from_5_over_u_r"]
        temp["Dataset"] = dataset_name
        temp["Model_Label"] = model_label
        temp["Model_Family"] = model_spec["Model_Family"]
        temp["Selection_Mode"] = model_spec["Selection_Mode"]
        temp["Rank_in_saved_results"] = model_spec["Rank_in_saved_results"]
        temp["Model_Features"] = " + ".join(model_spec["Features"])

        d_metrics = compute_metrics(temp.get("d_csds"), temp["d_pred"]) if "d_csds" in temp.columns else {"N": 0, "RMSE": np.nan, "R2": np.nan}
        e_metrics = compute_metrics(temp.get("e_csds"), temp["e_pred"]) if "e_csds" in temp.columns else {"N": 0, "RMSE": np.nan, "R2": np.nan}

        model_output_file = group_output_dir / f"{model_label}_e_predictions.csv"
        temp.to_csv(model_output_file, index=False)

        n_total = len(temp)
        n_conv = int(temp["e_converged"].fillna(False).sum())
        n_cond = int(temp["condition_c_lt_e"].fillna(False).sum())
        both_mask = temp["e_converged"].fillna(False) & temp["condition_c_lt_e"].fillna(False)
        n_both = int(both_mask.sum())

        row_summary = {
            "Dataset": dataset_name,
            "Model_Label": model_label,
            "Model_Family": model_spec["Model_Family"],
            "Selection_Mode": model_spec["Selection_Mode"],
            "Rank_in_saved_results": model_spec["Rank_in_saved_results"],
            "Model_Features": " + ".join(model_spec["Features"]),
            "Saved_R2_val_log": model_spec["Saved_R2_val_log"],
            "Saved_R2_cv_mean_log": model_spec["Saved_R2_cv_mean_log"],
            "Saved_R2_val_d": model_spec["Saved_R2_val_d"],
            "Saved_RMSE_val_d": model_spec["Saved_RMSE_val_d"],
            "Saved_R2_test_log": model_spec["Saved_R2_test_log"],
            "Saved_RMSE_test_log": model_spec["Saved_RMSE_test_log"],
            "Saved_R2_test_d": model_spec["Saved_R2_test_d"],
            "Saved_RMSE_test_d": model_spec["Saved_RMSE_test_d"],
            "Saved_Selection_Score": model_spec["Saved_Selection_Score"],
            "N_rows": n_total,
            "N_e_converged": n_conv,
            "N_c_lt_e": n_cond,
            "N_converged_and_c_lt_e": n_both,
            "Pct_converged_and_c_lt_e": 100.0 * n_both / n_total if n_total > 0 else np.nan,
            "RMSE_d_pred_vs_d_csds": d_metrics["RMSE"],
            "R2_d_pred_vs_d_csds": d_metrics["R2"],
            "N_d_metric_rows": d_metrics["N"],
            "RMSE_e_pred_vs_e_csds": e_metrics["RMSE"],
            "R2_e_pred_vs_e_csds": e_metrics["R2"],
            "N_e_metric_rows": e_metrics["N"],
            "Mean_e_pred": temp["e_pred"].mean(),
            "Median_e_pred": temp["e_pred"].median(),
            "Mean_residual_eq_515_abs": temp["residual_eq_515"].abs().mean(),
            "Mean_residual_eq_518_abs": temp["residual_eq_518"].abs().mean(),
            "Detailed_Output_File": str(model_output_file),
        }

        dataset_summary_rows.append(row_summary)
        all_summary_rows.append(row_summary)

        print(f"Saved detailed file: {model_output_file}")
        print(f"RMSE(d) = {d_metrics['RMSE']:.6f} | R2(d) = {d_metrics['R2']:.6f}" if d_metrics["N"] > 0 else "No valid d metrics.")
        print(f"RMSE(e) = {e_metrics['RMSE']:.6f} | R2(e) = {e_metrics['R2']:.6f}" if e_metrics["N"] > 0 else "No valid e metrics.")
        print(f"Converged and c<e: {n_both}/{n_total}")

    dataset_summary_df = pd.DataFrame(dataset_summary_rows)
    dataset_summary_file = group_output_dir / f"{dataset_name.lower()}_summary_e_from_d_models.csv"
    dataset_summary_df.to_csv(dataset_summary_file, index=False)
    print(f"\nDataset summary saved: {dataset_summary_file}")


summary_df = pd.DataFrame(all_summary_rows)
summary_output = OUTPUT_DIR / "summary_all_retained_models_e_from_d.csv"
summary_df.to_csv(summary_output, index=False)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Global summary saved: {summary_output}")
