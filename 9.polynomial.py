"""
CSDS – Régression polynomiale avec règle de Montgomery (15 observations / paramètre)
Version FINALISÉE :
- Implémentation de la règle de Montgomery pour le split Train/Val
- Affichage R² et RMSE sur Train/Val
- Affichage de l'équation polynomiale
- Analyse des résidus sur Validation (avec Test de Shapiro-Wilk)
- Calcul du BIC (Bayesian Information Criterion) pour la sélection de modèle
- Tableau de synthèse final des performances
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ============================================================
# 1) Chargement des données (simulé)
# ============================================================
# ATTENTION: Simuler le chargement des données. REMPLACER PAR LE CHARGEMENT RÉEL.
try:
    FILE = "CSDS_parameters_CONVERGED_ONLY.csv"
    df = pd.read_csv(FILE)

    if "csds_converged" in df.columns:
        df = df[df["csds_converged"] == 1].copy()

    required_cols = [
        "delta_peak_mm", "tau_peak_MPa_csds", "u_r_mm", "tau_r_MPa",
        "a_csds", "b_csds", "c_csds", "d_csds", "e_csds",
    ]
    df = df.dropna(subset=required_cols).copy()

    y_d = df["d_csds"]
    y_e = df["e_csds"]
    
    N_total = len(df)
    if N_total == 0:
        raise ValueError("Le DataFrame est vide après le nettoyage des données.")

except FileNotFoundError:
    print("ERREUR: Fichier de données non trouvé. Impossible de continuer.")
    exit()
except Exception as e:
    print(f"ERREUR lors du chargement/nettoyage des données : {e}")
    exit()

# ============================================================
# 2) Fonctions utilitaires
# ============================================================

def split_train_val_by_params(X, y, degree=2, random_state=42):
    """
    Règle de Montgomery : n_train ≈ 15 × (nombre de paramètres estimés).
    """
    n_samples = X.shape[0]

    poly_tmp = PolynomialFeatures(degree=degree, include_bias=False)
    poly_tmp.fit(X)
    n_params = poly_tmp.n_output_features_  # Nombre de termes X^k (sans l'intercept)

    desired_train = 15 * n_params

    if n_samples > desired_train:
        n_train = desired_train
    else:
        # Fallback: 70% train si pas assez de données pour la règle
        n_train = max(int(0.7 * n_samples), 1)

    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    return X_train, X_val, y_train, y_val, n_train, len(val_idx), n_params, desired_train


def fit_poly_model(X_train, y_train, X_val, y_val, degree=2):
    """
    Ajuste et évalue le modèle, et calcule le BIC.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    model = LinearRegression().fit(X_train_poly, y_train)

    # Métriques Train
    y_train_pred = model.predict(X_train_poly)
    train_res = y_train - y_train_pred
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = float(np.sqrt(np.mean(train_res**2)))
    SSE_train = np.sum(train_res**2)

    # Métriques Validation
    y_val_pred = model.predict(X_val_poly)
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = float(np.sqrt(np.mean((y_val - y_val_pred)**2)))

    # BIC (Bayesian Information Criterion)
    n_params_bic = poly.n_output_features_ + 1 # Nombre de Bêtas: termes + intercept
    N_train = len(y_train)
    # BIC = N*ln(SSE/N) + k*ln(N)
    bic = N_train * np.log(SSE_train / N_train) + n_params_bic * np.log(N_train)

    return {
        "model": model, "poly": poly, 
        "y_val_pred": y_val_pred, 
        "r2_train": float(r2_train), "rmse_train": rmse_train, 
        "r2_val": float(r2_val), "rmse_val": rmse_val,
        "val_residuals": y_val - y_val_pred,
        "BIC": float(bic),
        "n_params_estimated": n_params_bic,
    }


def print_polynomial_equation(model, poly, feature_names, title="Équation polynomiale"):
    coefs = model.coef_
    intercept = model.intercept_
    poly_features = poly.get_feature_names_out(feature_names)

    print(f"\n=================== {title} ===================")
    print(f"Intercept = {intercept:.6f}")
    print("Équation (y = Intercept + Σ β_j * terme_j) :")
    for name, coef in zip(poly_features, coefs):
        print(f"  {coef:.6f} * {name}")


def analyze_residuals(y_true, y_pred, title="Résidus – validation"):
    residuals = y_true - y_pred

    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    skew = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    
    # Test de Normalité (Shapiro-Wilk)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    print(f"\n===== {title} =====")
    print(f"Moyenne   : {mean:.6f}")
    print(f"Écart-type: {std:.6f}")
    print(f"Asymétrie : {skew:.6f}")
    print(f"Kurtosis  : {kurt:.6f}")
    print(f"RMSE      : {rmse:.6f}")
    print(f"Shapiro-Wilk p : {shapiro_p:.6f}")
    if shapiro_p < 0.05:
        print("    -> Normalité rejetée (p < 0.05).")
    else:
        print("    -> Normalité acceptée.")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].scatter(y_pred, residuals, alpha=0.6)
    ax[0].axhline(0, linestyle="--")
    ax[0].set_title("Résidus vs valeurs prédites (Homoscédasticité)")
    ax[0].set_xlabel("y_prédit")
    ax[0].set_ylabel("Résidu")

    ax[1].hist(residuals, bins=30)
    ax[1].set_title("Histogramme des résidus")

    stats.probplot(residuals, dist="norm", plot=ax[2])
    ax[2].set_title("Q-Q plot (Normalité)")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Définition des 3 modèles
# ============================================================

X1 = df[[
    "a_csds", "b_csds", "c_csds",
    "u_r_mm", "delta_peak_mm",
    "tau_peak_MPa_csds", "tau_r_MPa",
]]
X2 = df[["a_csds", "b_csds", "c_csds"]]
X3 = df[["u_r_mm", "delta_peak_mm", "tau_peak_MPa_csds", "tau_r_MPa"]]
DEGREE = 2


# ============================================================
# 4) Fonction pour lancer et afficher un modèle
# ============================================================

def run_one_model(X, y, model_label, base_feature_names, random_state=42):
    # Split selon la règle 15 observations / paramètre
    X_train, X_val, y_train, y_val, n_train, n_val, n_params, desired = \
        split_train_val_by_params(X, y, degree=DEGREE, random_state=random_state)

    print(f"\n========== {model_label} ==========")
    print(f"Nombre total d’observations : {len(X)}")
    print(f"Nombre de paramètres (polynôme deg={DEGREE}) : {n_params + 1} (dont Intercept)")
    print(f"Taille théorique entraînement (15 × params) : {desired}")
    print(f"Taille réelle entraînement : {n_train}")
    print(f"Taille validation          : {n_val}")

    # Ajustement et évaluation
    res = fit_poly_model(X_train, y_train, X_val, y_val, degree=DEGREE)

    print(f"\n{model_label} – métriques entraînement :")
    print(f"  R²   = {res['r2_train']:.4f}")
    print(f"  RMSE = {res['rmse_train']:.6f}")

    print(f"\n{model_label} – métriques validation :")
    print(f"  R²   = {res['r2_val']:.4f}")
    print(f"  RMSE = {res['rmse_val']:.6f}")
    print(f"  BIC  = {res['BIC']:.2f}")

    # Équation polynomiale
    print_polynomial_equation(res["model"], res["poly"], base_feature_names,
                              title=f"{model_label} – équation polynomiale")

    # Analyse des résidus sur la validation
    analyze_residuals(y_val, res["y_val_pred"], title=f"{model_label} – résidus validation")

    return res


# ============================================================
# 5) Lancement des 6 modèles et collection des résultats
# ============================================================

# Pour d_csds
res1_d = run_one_model(X1, y_d, "Modèle 1 – d_csds",
                       ["a", "b", "c", "u_r", "u_peak", "tau_peak", "tau_r"])
res2_d = run_one_model(X2, y_d, "Modèle 2 – d_csds",
                       ["a", "b", "c"])
res3_d = run_one_model(X3, y_d, "Modèle 3 – d_csds",
                       ["u_r", "u_peak", "tau_peak", "tau_r"])

# Pour e_csds
res1_e = run_one_model(X1, y_e, "Modèle 1 – e_csds",
                       ["a", "b", "c", "u_r", "u_peak", "tau_peak", "tau_r"])
res2_e = run_one_model(X2, y_e, "Modèle 2 – e_csds",
                       ["a", "b", "c"])
res3_e = run_one_model(X3, y_e, "Modèle 3 – e_csds",
                       ["u_r", "u_peak", "tau_peak", "tau_r"])


# ============================================================
# 6) Synthèse des performances (Tableau récapitulatif)
# ============================================================

all_results = [
    {"Modèle": "M1 (Complet) d_csds", **res1_d, "n_params_base": 7},
    {"Modèle": "M2 (Interne) d_csds", **res2_d, "n_params_base": 3},
    {"Modèle": "M3 (Terrain) d_csds", **res3_d, "n_params_base": 4},
    {"Modèle": "M1 (Complet) e_csds", **res1_e, "n_params_base": 7},
    {"Modèle": "M2 (Interne) e_csds", **res2_e, "n_params_base": 3},
    {"Modèle": "M3 (Terrain) e_csds", **res3_e, "n_params_base": 4},
]

summary_df = pd.DataFrame([
    {
        "Modèle": res["Modèle"],
        "Target": res["Modèle"].split(" ")[-1],
        "n_coeffs": res["n_params_estimated"],
        "R2_Train": res["r2_train"],
        "R2_Val": res["r2_val"],
        "RMSE_Val": res["rmse_val"],
        "BIC": res["BIC"],
    } for res in all_results
])

print("\n\n===================== SYNTHÈSE DES PERFORMANCES =====================")
print(f"Basée sur N={N_total} observations totales.")
print("Règle de Montgomery (n_train = 15 x n_coeffs) appliquée.\n")

# Affichage du tableau de synthèse
print(summary_df.to_string(index=False, float_format="%.4f"))

print("\n* BIC (Bayesian Information Criterion) : Le plus bas est le meilleur modèle.")
print("* R2_Val vs R2_Train : Une grande différence indique un overfitting.")