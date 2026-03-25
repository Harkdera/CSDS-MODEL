from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

def compute_d_from_e(u_r, tau_r, u_p, tau_p, e):
    a = tau_r
    c = 5.0 / u_r

    numerator = tau_p - a * (1.0 - np.exp(-c * u_p))
    denominator = np.exp(-c * u_p) - np.exp(-e * u_p)

    if np.isclose(denominator, 0.0):
        return np.nan

    return numerator / denominator

def fit_and_evaluate_exponential_model(train_df, val_df, feature_cols, target_col="e_csds"):
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    # Exponential regression:
    # log(y) = linear model
    y_train_log = np.log(y_train)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    model.fit(X_train, y_train_log)

    # Predictions in log-space
    y_train_log_pred = model.predict(X_train)
    y_val_log_pred = model.predict(X_val)

    # Back-transform to original space
    y_train_pred = np.exp(y_train_log_pred)
    y_val_pred = np.exp(y_val_log_pred)

    return {
        "features": feature_cols,
        "R2_train": r2_score(y_train, y_train_pred),
        "RMSE_train": rmse(y_train, y_train_pred),
        "R2_val": r2_score(y_val, y_val_pred),
        "RMSE_val": rmse(y_val, y_val_pred),
        "model": model
    }

def print_exponential_equation(model, feature_cols):
    scaler = model.named_steps["scaler"]
    reg = model.named_steps["reg"]

    coefs_scaled = reg.coef_
    intercept_scaled = reg.intercept_

    print("\nApproximate exponential form:")
    print("log(e_csds) = b0 + b1*x1 + b2*x2 + ...   (after scaling)")
    print("e_csds = exp(b0 + b1*x1 + b2*x2 + ...)")

    print("\nScaled coefficients:")
    print(f"Intercept: {intercept_scaled:.6f}")
    for name, coef in zip(feature_cols, coefs_scaled):
        print(f"{name}: {coef:.6f}")

# ============================================================
# 5) FEATURE SETS
# ============================================================
feature_sets = [
    ["u_r_mm"],
    ["u_r_mm", "delta_peak_mm"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds"],
    ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_r_MPa"]
]

# ============================================================
# 6) RUN ALL EXPONENTIAL MODELS
# ============================================================
results_all = []

print("\n" + "=" * 90)
print("EXPONENTIAL REGRESSION ANALYSIS FOR e_csds")
print("=" * 90)

for i, features in enumerate(feature_sets, start=1):
    print(f"\n--- Model {i}: features={features} ---")

    result = fit_and_evaluate_exponential_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=features,
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

best_result = None
for res in results_all:
    if " + ".join(res["features"]) == best_features_str:
        best_result = res
        break

best_model = best_result["model"]
best_features = best_result["features"]

print("\n" + "=" * 90)
print("BEST MODEL")
print("=" * 90)
print(f"Best feature set    : {best_features}")
print(f"Best validation R2  : {best_result['R2_val']:.6f}")
print(f"Best validation RMSE: {best_result['RMSE_val']:.6f}")

print_exponential_equation(best_model, best_features)

# ============================================================
# 9) PREDICT e ON FULL DATASET
# ============================================================
X_full_best = data[best_features]
data["e_pred"] = np.exp(best_model.predict(X_full_best))
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
summary_output = (Path(__file__).resolve().parent / "../data/processed/exponential_component_analysis_e.csv").resolve()
pred_output = (Path(__file__).resolve().parent / "../data/processed/predicted_e_and_computed_d_best_exponential.csv").resolve()

summary_df.to_csv(summary_output, index=False)
data.to_csv(pred_output, index=False)

print("\nSaved:")
print(f"  - {summary_output}")
print(f"  - {pred_output}")