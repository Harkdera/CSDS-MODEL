"""Évalue une sélection de modèles polynomiaux pour prédire `d_csds`."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
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
    """Charge un jeu de données et reconstruit les variables dérivées nécessaires."""
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
        (data["tau_peak_MPa_csds"] > 0) &
        (data["tau_r_MPa"] >= 0)
    ].reset_index(drop=True)

    # Variable dérivée utilisée par les modèles 147 et 148.
    data["tau_p_div_u_r"] = data["tau_peak_MPa_csds"] / data["u_r_mm"]

    data = data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    return data

def polynomial_equation_to_string(model_pipeline, feature_cols, target_name="d_csds", digits=6):
    """Transforme un pipeline polynomial en équation lisible."""
    poly = model_pipeline.named_steps["poly"]
    reg = model_pipeline.named_steps["reg"]

    terms = poly.get_feature_names_out(feature_cols)
    intercept = reg.intercept_
    coefs = reg.coef_

    pieces = [f"{target_name} = {intercept:.{digits}f}"]
    for coef, term in zip(coefs, terms):
        sign = "+" if coef >= 0 else "-"
        pieces.append(f" {sign} {abs(coef):.{digits}f}*{term}")

    return "".join(pieces)

def fit_and_evaluate(train_df, val_df, feature_cols, degree=2, target_col="d_csds"):
    """Ajuste une régression polynomiale et retourne les métriques associées."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
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
        "equation": polynomial_equation_to_string(model, feature_cols, target_name=target_col),
        "model": model
    }

# ============================================================
# 4) Modèles à tester
# Basés sur les combinaisons déjà sélectionnées
# ============================================================
MODELS_TO_TEST = {
    "Model_147": ["u_r_mm", "delta_peak_mm", "tau_p_div_u_r"],
    "Model_148": ["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_p_div_u_r"],
}

DEGREE = 2

# ============================================================
# 5) Lancer l'analyse pour chaque sous-jeu
# ============================================================
all_summary = []
all_predictions = []

for dataset_name, file_path in FILES.items():
    data = load_and_clean(file_path)

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

    results = []

    print("\n" + "=" * 90)
    print(f"TEST OF SELECTED COMBINATION MODELS FOR d_csds ({dataset_name})")
    print("=" * 90)

    for model_name, feature_cols in MODELS_TO_TEST.items():
        print(f"\n--- {model_name}: {feature_cols} ---")

        result = fit_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            degree=DEGREE,
            target_col="d_csds"
        )

        results.append({
            "Dataset": dataset_name,
            "Model_Name": model_name,
            "Features": " + ".join(feature_cols),
            "Degree": DEGREE,
            "R2_train": result["R2_train"],
            "RMSE_train": result["RMSE_train"],
            "R2_val": result["R2_val"],
            "RMSE_val": result["RMSE_val"],
            "Equation": result["equation"],
            "model_obj": result["model"],
            "feature_cols": feature_cols,
        })

        print(f"R2_train   = {result['R2_train']:.6f}")
        print(f"RMSE_train = {result['RMSE_train']:.6f}")
        print(f"R2_val     = {result['R2_val']:.6f}")
        print(f"RMSE_val   = {result['RMSE_val']:.6f}")

    summary_df = pd.DataFrame([
        {
            "Dataset": r["Dataset"],
            "Model_Name": r["Model_Name"],
            "Features": r["Features"],
            "Degree": r["Degree"],
            "R2_train": r["R2_train"],
            "RMSE_train": r["RMSE_train"],
            "R2_val": r["R2_val"],
            "RMSE_val": r["RMSE_val"],
            "Equation": r["Equation"],
        }
        for r in results
    ]).sort_values(by="R2_val", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(summary_df[[
        "Dataset", "Model_Name", "Features", "Degree",
        "R2_train", "RMSE_train", "R2_val", "RMSE_val"
    ]].to_string(index=False))

    best_row = summary_df.iloc[0]
    best_model_name = best_row["Model_Name"]

    best_result = None
    for r in results:
        if r["Model_Name"] == best_model_name:
            best_result = r
            break

    print("\n" + "=" * 90)
    print("BEST MODEL")
    print("=" * 90)
    print(f"Best model         : {best_result['Model_Name']}")
    print(f"Best feature set   : {best_result['feature_cols']}")
    print(f"Best validation R2 : {best_result['R2_val']:.6f}")
    print(f"Best validation RMSE: {best_result['RMSE_val']:.6f}")
    print("\nBest equation:")
    print(best_result["Equation"])

    # Appliquer le meilleur modèle à l'ensemble du sous-jeu.
    pred_df = data.copy()
    X_full = pred_df[best_result["feature_cols"]]
    pred_df["d_pred"] = best_result["model_obj"].predict(X_full)
    pred_df["Best_Model_Name"] = best_result["Model_Name"]

    print("\n" + "=" * 90)
    print("SAMPLE PREDICTIONS")
    print("=" * 90)
    cols_to_show = [
        "u_r_mm",
        "delta_peak_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
        "tau_p_div_u_r",
        "d_csds",
        "d_pred",
        "Best_Model_Name"
    ]
    print(pred_df[cols_to_show].head(15).to_string(index=False))

    # Enregistrer les résultats spécifiques à ce sous-jeu.
    summary_path = REGRESSION_DIR / f"selected_combination_models_{dataset_name.lower()}.csv"
    pred_path = REGRESSION_DIR / f"predicted_d_selected_models_{dataset_name.lower()}.csv"

    summary_df.to_csv(summary_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(pred_path)

    all_summary.append(summary_df)
    all_predictions.append(pred_df)

# ============================================================
# 6) Résumé global
# ============================================================
global_summary_df = pd.concat(all_summary, ignore_index=True)
global_summary_path = REGRESSION_DIR / "selected_combination_models_all_groups.csv"
global_summary_df.to_csv(global_summary_path, index=False)

print("\n" + "=" * 90)
print("GLOBAL SUMMARY SAVED")
print("=" * 90)
print(global_summary_path)
