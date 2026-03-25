from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1) LOAD DATA
# ============================================================
csv_path = (Path(__file__).resolve().parent / "../data/processed/csds_parameters_converged_only.csv").resolve()
print("Using file:", csv_path)

df = pd.read_csv(csv_path)

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

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

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

print(f"Number of usable rows: {len(data)}")

# ============================================================
# 3) TRAIN / VALIDATION SPLIT
# ============================================================
train_df, val_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# ============================================================
# 4) HELPERS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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

# ============================================================
# 5) FEATURE SETS + DEGREES
# ============================================================
feature_sets = [
    ["u_r_mm"],
    ["u_r_mm", "delta_peak_mm"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_r_MPa"]
]

degrees = [2, 3, 4]

# ============================================================
# 6) RUN ALL MODELS
# ============================================================
results_all = []

print("\n" + "=" * 90)
print("POLYNOMIAL REGRESSION ANALYSIS FOR e_csds")
print("=" * 90)

model_id = 0
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

# ============================================================
# 7) SUMMARY TABLE
# ============================================================
summary_rows = []
for res in results_all:
    summary_rows.append({
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

# ============================================================
# 8) BEST MODEL
# ============================================================
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
print(f"Best feature set     : {best_features}")
print(f"Best polynomial degree: {best_degree}")
print(f"Best validation R2   : {best_result['R2_val']:.6f}")
print(f"Best validation RMSE : {best_result['RMSE_val']:.6f}")

# ============================================================
# 9) PREDICT e ON FULL DATASET
# ============================================================
X_full_best = data[best_features]
data["e_pred"] = best_model.predict(X_full_best)
data["e_pred"] = np.maximum(data["e_pred"], 1e-6)

data["d_from_e_pred"] = data.apply(
    lambda row: compute_d_from_e(
        u_r=row["u_r_mm"],
        tau_r=row["tau_r_MPa"],
        u_p=row["delta_peak_mm"],
        tau_p=row["tau_peak_MPa_csds"],
        e=row["e_pred"]
    ),
    axis=1
)

# ============================================================
# 10) SHOW SAMPLE OUTPUT
# ============================================================
cols_to_show = [
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
    "e_csds",
    "e_pred",
    "d_from_e_pred"
]

print("\n" + "=" * 90)
print("SAMPLE PREDICTIONS")
print("=" * 90)
print(data[cols_to_show].head(15).to_string(index=False))

# ============================================================
# 11) SAVE OUTPUTS
# ============================================================
summary_output = (Path(__file__).resolve().parent / "../data/processed/polynomial_component_analysis_e_deg234.csv").resolve()
pred_output = (Path(__file__).resolve().parent / "../data/processed/predicted_e_and_computed_d_best_deg234.csv").resolve()

summary_df.to_csv(summary_output, index=False)
data.to_csv(pred_output, index=False)

print("\nSaved:")
print(f"  - {summary_output}")
print(f"  - {pred_output}")