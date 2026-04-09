"""Teste des régressions exponentielles pour prédire `d_csds` sur plusieurs sous-jeux."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1) Chemins des données
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

FILES = {
    "FULL": BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv",
    "HIGH": BASE_DIR / "data" / "interim" / "csds_tau_peak_high.csv",
    "LOW": BASE_DIR / "data" / "interim" / "csds_tau_peak_low.csv",
    "LOW_1": BASE_DIR / "data" / "interim" / "csds_tau_peak_low_1.csv",
    "LOW_2": BASE_DIR / "data" / "interim" / "csds_tau_peak_low_2.csv",
}

OUTPUT_DIR = BASE_DIR / "data" / "processed"
REGRESSION_DIR = OUTPUT_DIR / "regressions"
REGRESSION_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2) Colonnes nécessaires
# Régression exponentielle de `d_csds` à partir de paramètres mesurables
# ============================================================
required_cols = [
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
    "d_csds",
]

# ============================================================
# 3) Fonctions utilitaires
# ============================================================
def rmse(y_true, y_pred):
    """Calcule la racine de l'erreur quadratique moyenne."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_clean(file_path):
    """Charge un fichier, garde les colonnes utiles et supprime les lignes invalides."""
    df = pd.read_csv(file_path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing columns: {missing}")

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    data = data[
        (data["u_r_mm"] > 0) &
        (data["delta_peak_mm"] > 0) &
        (data["tau_peak_MPa_csds"] >= 0) &
        (data["tau_r_MPa"] >= 0) &
        (data["d_csds"] > 0)
    ].reset_index(drop=True)

    return data

def build_exponential_equation_raw(model_pipeline, feature_cols, target_name="d_csds", digits=6):
    """Réécrit l'équation ajustée dans l'espace original des variables explicatives."""
    scaler = model_pipeline.named_steps["scaler"]
    reg = model_pipeline.named_steps["reg"]

    beta_scaled = reg.coef_
    intercept_scaled = reg.intercept_

    means = scaler.mean_
    scales = scaler.scale_

    beta_raw = beta_scaled / scales
    intercept_raw = intercept_scaled - np.sum(beta_scaled * means / scales)

    log_eq = f"log({target_name}) = {intercept_raw:.{digits}f}"
    exp_eq = f"{target_name} = exp({intercept_raw:.{digits}f}"

    for coef, name in zip(beta_raw, feature_cols):
        sign = "+" if coef >= 0 else "-"
        log_eq += f" {sign} {abs(coef):.{digits}f}*{name}"
        exp_eq += f" {sign} {abs(coef):.{digits}f}*{name}"

    exp_eq += ")"
    return log_eq, exp_eq

def fit_and_evaluate_exponential_model(train_df, val_df, feature_cols, target_col="d_csds"):
    """Ajuste un modèle exponentiel et retourne ses métriques sur train et validation."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    y_train_log = np.log(y_train)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ])

    model.fit(X_train, y_train_log)

    y_train_log_pred = model.predict(X_train)
    y_val_log_pred = model.predict(X_val)

    y_train_pred = np.exp(y_train_log_pred)
    y_val_pred = np.exp(y_val_log_pred)

    log_eq, exp_eq = build_exponential_equation_raw(
        model_pipeline=model,
        feature_cols=feature_cols,
        target_name=target_col,
        digits=6
    )

    return {
        "features": feature_cols,
        "R2_train": r2_score(y_train, y_train_pred),
        "RMSE_train": rmse(y_train, y_train_pred),
        "R2_val": r2_score(y_val, y_val_pred),
        "RMSE_val": rmse(y_val, y_val_pred),
        "log_equation": log_eq,
        "exp_equation": exp_eq,
        "model": model
    }

def run_component_analysis_for_dataset(data, dataset_name):
    """Teste plusieurs combinaisons de variables et conserve le meilleur modèle d'un sous-jeu."""
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

    feature_sets = [
        ["u_r_mm"],
        ["u_r_mm", "delta_peak_mm"],
        ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds"],
        ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_r_MPa"]
    ]

    results_all = []

    print("\n" + "=" * 90)
    print(f"EXPONENTIAL COMPONENT ANALYSIS FOR d_csds ({dataset_name})")
    print("=" * 90)

    for i, features in enumerate(feature_sets, start=1):
        print(f"\n--- Model {i}: features={features} ---")

        result = fit_and_evaluate_exponential_model(
            train_df=train_df,
            val_df=val_df,
            feature_cols=features,
            target_col="d_csds"
        )

        results_all.append(result)

        print(f"R2_train   = {result['R2_train']:.6f}")
        print(f"RMSE_train = {result['RMSE_train']:.6f}")
        print(f"R2_val     = {result['R2_val']:.6f}")
        print(f"RMSE_val   = {result['RMSE_val']:.6f}")

    summary_rows = []
    for i, res in enumerate(results_all, start=1):
        summary_rows.append({
            "Dataset": dataset_name,
            "Model_ID": i,
            "Features": " + ".join(res["features"]),
            "R2_train": res["R2_train"],
            "RMSE_train": res["RMSE_train"],
            "R2_val": res["R2_val"],
            "RMSE_val": res["RMSE_val"],
            "Log_Equation": res["log_equation"],
            "Exponential_Equation": res["exp_equation"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="R2_val", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(summary_df[[
        "Dataset", "Features", "R2_train", "RMSE_train", "R2_val", "RMSE_val"
    ]].to_string(index=False))

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
    print("\nBest equation:")
    print(best_result["exp_equation"])

    data_out = data.copy()
    X_full = data_out[best_features]
    data_out["d_pred"] = np.exp(best_model.predict(X_full))
    data_out["d_pred"] = np.maximum(data_out["d_pred"], 1e-12)

    print("\n" + "=" * 90)
    print("SAMPLE PREDICTIONS")
    print("=" * 90)

    cols_to_show = [
        "u_r_mm",
        "delta_peak_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
        "d_csds",
        "d_pred",
    ]

    print(data_out[cols_to_show].head(15).to_string(index=False))

    return summary_df, data_out

# ============================================================
# 4) Exécuter l'analyse sur tous les sous-jeux
# ============================================================
all_summary = []
all_predictions = {}

for dataset_name, file_path in FILES.items():
    data = load_and_clean(file_path)
    summary_df, pred_df = run_component_analysis_for_dataset(data, dataset_name)

    all_summary.append(summary_df)
    all_predictions[dataset_name] = pred_df

    summary_path = REGRESSION_DIR / f"exponential_component_analysis_d_{dataset_name.lower()}.csv"
    pred_path = REGRESSION_DIR / f"predicted_d_{dataset_name.lower()}_exponential.csv"

    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(pred_path)

# ============================================================
# 5) Enregistrer le résumé global
# ============================================================
global_summary_df = pd.concat(all_summary, ignore_index=True)
global_summary_path = REGRESSION_DIR / "exponential_component_analysis_d_all_groups.csv"
global_summary_df.to_csv(global_summary_path, index=False)

print("\n" + "=" * 90)
print("GLOBAL SUMMARY SAVED")
print("=" * 90)
print(global_summary_path)
