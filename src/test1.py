from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1) PATHS
# ============================================================
base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

low_path = (project_dir / "data/interim/csds_tau_peak_low.csv").resolve()
high_path = (project_dir / "data/interim/csds_tau_peak_high.csv").resolve()
output_dir = (project_dir / "data/processed").resolve()

print("Low file :", low_path)
print("High file:", high_path)

# ============================================================
# 2) REQUIRED COLUMNS
# ============================================================
required_cols = [
    "u_r_mm",
    "tau_r_MPa",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "e_csds"
]

feature_sets = [
    ["u_r_mm"],
    ["u_r_mm", "delta_peak_mm"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_r_MPa"]
]

degrees = [2, 3, 4]

# ============================================================
# 3) HELPERS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {missing}")

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    data = data[
        (data["u_r_mm"] > 0) &
        (data["delta_peak_mm"] > 0) &
        (data["tau_r_MPa"] >= 0) &
        (data["tau_peak_MPa_csds"] >= 0) &
        (data["e_csds"] > 0)
    ].reset_index(drop=True)

    return data

def fit_and_evaluate_polynomial_model(train_df, val_df, feature_cols, degree, target_col="e_csds"):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    return {
        "features": feature_cols,
        "degree": degree,
        "R2_train": r2_score(y_train, y_train_pred),
        "RMSE_train": rmse(y_train, y_train_pred),
        "R2_val": r2_score(y_val, y_val_pred),
        "RMSE_val": rmse(y_val, y_val_pred),
        "model": model
    }

def compute_d_from_e(u_r, tau_r, u_p, tau_p, e):
    a = tau_r
    c = 5.0 / u_r

    numerator = tau_p - a * (1.0 - np.exp(-c * u_p))
    denominator = np.exp(-c * u_p) - np.exp(-e * u_p)

    if np.isclose(denominator, 0.0):
        return np.nan

    return numerator / denominator

def run_component_analysis(data, dataset_name):
    print("\n" + "=" * 90)
    print(f"DATASET: {dataset_name}")
    print("=" * 90)
    print(f"Number of usable rows: {len(data)}")

    train_df, val_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    results_all = []
    model_id = 0

    print("\n" + "=" * 90)
    print(f"POLYNOMIAL REGRESSION ANALYSIS FOR e_csds ({dataset_name})")
    print("=" * 90)

    for degree in degrees:
        print(f"\n########## DEGREE = {degree} ##########")

        for features in feature_sets:
            model_id += 1
            print(f"\n--- Model {model_id}: degree={degree}, features={features} ---")

            result = fit_and_evaluate_polynomial_model(
                train_df=train_df,
                val_df=val_df,
                feature_cols=features,
                degree=degree,
                target_col="e_csds"
            )

            results_all.append(result)

            print(f"R2_train   = {result['R2_train']:.6f}")
            print(f"RMSE_train = {result['RMSE_train']:.6f}")
            print(f"R2_val     = {result['R2_val']:.6f}")
            print(f"RMSE_val   = {result['RMSE_val']:.6f}")

    summary_rows = []
    for res in results_all:
        summary_rows.append({
            "Dataset": dataset_name,
            "Features": " + ".join(res["features"]),
            "Degree": res["degree"],
            "R2_train": res["R2_train"],
            "RMSE_train": res["RMSE_train"],
            "R2_val": res["R2_val"],
            "RMSE_val": res["RMSE_val"]
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="R2_val", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    best_features_str = summary_df.loc[0, "Features"]
    best_degree = int(summary_df.loc[0, "Degree"])

    best_result = None
    for res in results_all:
        if " + ".join(res["features"]) == best_features_str and res["degree"] == best_degree:
            best_result = res
            break

    best_model = best_result["model"]
    best_features = best_result["features"]

    print("\n" + "=" * 90)
    print("BEST MODEL")
    print("=" * 90)
    print(f"Best feature set      : {best_features}")
    print(f"Best polynomial degree: {best_degree}")
    print(f"Best validation R2    : {best_result['R2_val']:.6f}")
    print(f"Best validation RMSE  : {best_result['RMSE_val']:.6f}")

    data_out = data.copy()
    X_full = data_out[best_features]
    data_out["e_pred"] = best_model.predict(X_full)
    data_out["e_pred"] = np.maximum(data_out["e_pred"], 1e-6)

    data_out["d_from_e_pred"] = data_out.apply(
        lambda row: compute_d_from_e(
            u_r=row["u_r_mm"],
            tau_r=row["tau_r_MPa"],
            u_p=row["delta_peak_mm"],
            tau_p=row["tau_peak_MPa_csds"],
            e=row["e_pred"]
        ),
        axis=1
    )

    print("\n" + "=" * 90)
    print("SAMPLE PREDICTIONS")
    print("=" * 90)
    cols_to_show = [
        "u_r_mm",
        "delta_peak_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
        "e_csds",
        "e_pred",
        "d_from_e_pred"
    ]
    print(data_out[cols_to_show].head(15).to_string(index=False))

    return summary_df, data_out

# ============================================================
# 4) RUN LOW
# ============================================================
low_data = load_and_clean(low_path)
summary_low, pred_low = run_component_analysis(low_data, "LOW")

# ============================================================
# 5) RUN HIGH
# ============================================================
high_data = load_and_clean(high_path)
summary_high, pred_high = run_component_analysis(high_data, "HIGH")

# ============================================================
# 6) SAVE
# ============================================================
summary_low_path = output_dir / "polynomial_component_analysis_e_low_deg234.csv"
summary_high_path = output_dir / "polynomial_component_analysis_e_high_deg234.csv"

pred_low_path = output_dir / "predicted_e_and_d_low_deg234.csv"
pred_high_path = output_dir / "predicted_e_and_d_high_deg234.csv"

summary_low.to_csv(summary_low_path, index=False)
summary_high.to_csv(summary_high_path, index=False)

pred_low.to_csv(pred_low_path, index=False)
pred_high.to_csv(pred_high_path, index=False)

print("\nSaved files:")
print(summary_low_path)
print(summary_high_path)
print(pred_low_path)
print(pred_high_path)