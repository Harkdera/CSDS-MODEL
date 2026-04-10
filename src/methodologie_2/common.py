"""Fonctions partagées pour la recherche sur la cible contrainte `log(e-c)`."""

from __future__ import annotations

from pathlib import Path
import ast
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONVERGED_FILE = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"
RESULTS_DIR = BASE_DIR / "results"
METHOD_2_ROOT = RESULTS_DIR / "methodologie_2"
METHOD_2_FIGURES_DIR = METHOD_2_ROOT / "figures"
METHOD_2_MODEL_FIGURES_DIR = METHOD_2_FIGURES_DIR / "models"
METHOD_2_COMPARE_DIR = METHOD_2_ROOT / "compare_b_d_e_tau_u"
METHOD_COMPARISON_DIR = RESULTS_DIR / "comparaison" / "methodologie_2_vs_methodologie_1"
SPLIT_FILES = {
    "FULL": CONVERGED_FILE,
    "LOW_1": BASE_DIR / "data" / "interim" / "csds_tau_peak_low_1.csv",
    "LOW_2": BASE_DIR / "data" / "interim" / "csds_tau_peak_low_2.csv",
    "HIGH": BASE_DIR / "data" / "interim" / "csds_tau_peak_high.csv",
}
GROUP_DIR = {
    "FULL": "full",
    "LOW_1": "low",
    "LOW_2": "low",
    "HIGH": "high",
}
DATASET_DIR = METHOD_2_ROOT / "datasets"
EDA_DIR = METHOD_2_ROOT / "eda"
REGRESSION_DIR = METHOD_2_ROOT / "regressions"
TOP5_DIR = REGRESSION_DIR / "top5"
EVALUATION_DIR = METHOD_2_ROOT / "evaluations"
N_CURVE_POINTS = 100
RANDOM_SEED = 42
CV_FOLDS_BY_DATASET = {
    "FULL": 5,
    "LOW_1": 5,
    "LOW_2": 3,
    "HIGH": 3,
}


RAW_FEATURE_COLS = [
    "sample_id",
    "sigma_n_MPa",
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
    "a_csds",
    "b_csds",
    "c_csds",
    "d_csds",
    "e_csds",
]


def ensure_output_dirs() -> None:
    """Crée les dossiers de sortie utilisés par la nouvelle branche de recherche."""
    for path in [
        RESULTS_DIR,
        METHOD_2_ROOT,
        METHOD_2_FIGURES_DIR,
        METHOD_2_MODEL_FIGURES_DIR,
        DATASET_DIR,
        EDA_DIR,
        REGRESSION_DIR,
        TOP5_DIR,
        EVALUATION_DIR,
        METHOD_2_COMPARE_DIR,
        METHOD_COMPARISON_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def dataset_slug(dataset_name: str) -> str:
    """Produit un identifiant de fichier stable pour un dataset."""
    return dataset_name.lower()


def dataset_group(dataset_name: str) -> str:
    """Associe un dataset à son dossier logique."""
    return GROUP_DIR[dataset_name]


def find_sample_id_column(df: pd.DataFrame) -> str:
    """Retrouve la colonne identifiant les échantillons."""
    candidates = [
        "sample_id",
        "Sample_ID",
        "sampleID",
        "SampleID",
        "id",
        "ID",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Impossible de trouver la colonne d'identifiant parmi {candidates}.")


def load_dataset_in_converged_order(dataset_name: str) -> pd.DataFrame:
    """Recharge un dataset en imposant l'ordre de la base convergée via `sample_id`."""
    converged = pd.read_csv(CONVERGED_FILE)
    sample_col = find_sample_id_column(converged)

    if converged[sample_col].duplicated().any():
        raise ValueError("Duplicate sample_id values found in converged file.")

    if dataset_name == "FULL":
        return converged.copy().reset_index(drop=True)

    split_df = pd.read_csv(SPLIT_FILES[dataset_name])
    split_sample_col = find_sample_id_column(split_df)

    if split_df[split_sample_col].duplicated().any():
        raise ValueError(f"Duplicate sample_id values found in split file for {dataset_name}.")

    split_ids = split_df[split_sample_col].tolist()
    data = converged[converged[sample_col].isin(split_ids)].copy().reset_index(drop=True)

    if len(data) != len(split_ids):
        missing_ids = sorted(set(split_ids) - set(data[sample_col].tolist()))
        raise ValueError(f"Missing sample_id values for {dataset_name}: {missing_ids[:10]}")

    return data


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables produits et ratios utilisées par les recherches précédentes."""
    data = df.copy()

    sigma_n = data["sigma_n_MPa"]
    u_r = data["u_r_mm"]
    u_p = data["delta_peak_mm"]
    tau_p = data["tau_peak_MPa_csds"]
    tau_r = data["tau_r_MPa"]

    data["sigma_n_x_u_r"] = sigma_n * u_r
    data["sigma_n_x_u_p"] = sigma_n * u_p
    data["sigma_n_x_tau_p"] = sigma_n * tau_p
    data["sigma_n_x_tau_r"] = sigma_n * tau_r
    data["u_r_x_u_p"] = u_r * u_p
    data["u_r_x_tau_p"] = u_r * tau_p
    data["u_r_x_tau_r"] = u_r * tau_r
    data["u_p_x_tau_p"] = u_p * tau_p
    data["u_p_x_tau_r"] = u_p * tau_r
    data["tau_p_x_tau_r"] = tau_p * tau_r

    data["u_p_div_u_r"] = u_p / u_r
    data["u_r_div_u_p"] = u_r / u_p
    data["tau_p_div_tau_r"] = tau_p / tau_r
    data["tau_r_div_tau_p"] = tau_r / tau_p
    data["u_r_div_tau_p"] = u_r / tau_p
    data["u_r_div_tau_r"] = u_r / tau_r
    data["u_p_div_tau_p"] = u_p / tau_p
    data["u_p_div_tau_r"] = u_p / tau_r
    data["tau_p_div_u_r"] = tau_p / u_r
    data["tau_p_div_u_p"] = tau_p / u_p
    data["tau_r_div_u_r"] = tau_r / u_r
    data["tau_r_div_u_p"] = tau_r / u_p
    data["tau_p_div_sigma_n"] = tau_p / sigma_n
    data["tau_r_div_sigma_n"] = tau_r / sigma_n
    data["sigma_n_div_tau_p"] = sigma_n / tau_p
    data["sigma_n_div_tau_r"] = sigma_n / tau_r
    data["sigma_n_div_u_r"] = sigma_n / u_r
    data["sigma_n_div_u_p"] = sigma_n / u_p

    return data.replace([np.inf, -np.inf], np.nan)


def build_direct_e_dataset(dataset_name: str) -> pd.DataFrame:
    """Construit le dataset complet pour la cible `z = log(e-c)`."""
    df = load_dataset_in_converged_order(dataset_name)

    missing = [col for col in RAW_FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {dataset_name}: {missing}")

    data = df.copy()

    numeric_cols = [col for col in data.columns if col != "sample_id"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "sigma_n_MPa",
        "u_r_mm",
        "delta_peak_mm",
        "tau_peak_MPa_csds",
        "tau_r_MPa",
        "d_csds",
        "e_csds",
    ]).reset_index(drop=True)

    data = data[
        (data["sigma_n_MPa"] > 0) &
        (data["u_r_mm"] > 0) &
        (data["delta_peak_mm"] > 0) &
        (data["tau_peak_MPa_csds"] > 0) &
        (data["tau_r_MPa"] > 0) &
        (data["d_csds"] > 0) &
        (data["e_csds"] > 0)
    ].reset_index(drop=True)

    data["c_target"] = data["c_csds"].where(data["c_csds"].notna(), 5.0 / data["u_r_mm"])
    data["e_minus_c_csds"] = data["e_csds"] - data["c_target"]
    data = data[data["e_minus_c_csds"] > 0].reset_index(drop=True)
    data["log_e_minus_c_csds"] = np.log(data["e_minus_c_csds"])
    data["log_d_csds"] = np.log(data["d_csds"])
    data["log_sigma_n_MPa"] = np.log(data["sigma_n_MPa"])
    data["log_delta_peak_mm"] = np.log(data["delta_peak_mm"])
    data["log_u_r_mm"] = np.log(data["u_r_mm"])
    data["log_tau_peak_MPa_csds"] = np.log(data["tau_peak_MPa_csds"])
    data["log_tau_r_MPa"] = np.log(data["tau_r_MPa"])

    data = add_engineered_features(data)
    data = data.dropna().reset_index(drop=True)
    return data


def get_candidate_feature_names(data: pd.DataFrame) -> list[str]:
    """Liste les variables explicatives autorisées pour la recherche."""
    excluded = {
        "sample_id",
        "a_csds",
        "b_csds",
        "c_csds",
        "d_csds",
        "e_csds",
        "c_target",
        "e_minus_c_csds",
        "log_e_minus_c_csds",
        "log_d_csds",
        "log_sigma_n_MPa",
        "log_delta_peak_mm",
        "log_u_r_mm",
        "log_tau_peak_MPa_csds",
        "log_tau_r_MPa",
        "csds_converged",
        "csds_iterations",
    }
    return [col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])]


def parse_feature_list(value) -> list[str]:
    """Convertit une représentation texte de features en vraie liste."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass

    if " + " in text:
        return [piece.strip() for piece in text.split(" + ") if piece.strip()]

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        return [piece.strip().strip("'\"") for piece in text.split(",") if piece.strip()]

    return [text]


def deduplicate_top_rows(df: pd.DataFrame, top_n: int) -> list[tuple[pd.Series, list[str]]]:
    """Conserve les premières lignes en retirant les doublons exacts de sets de features."""
    top_df = df.head(top_n).copy()
    kept: list[tuple[pd.Series, list[str]]] = []
    seen: set[tuple[str, ...]] = set()

    for _, row in top_df.iterrows():
        feature_list = parse_feature_list(row.get("Feature_List", row.get("Features", "")))
        key = tuple(sorted(feature_list))
        if key in seen:
            continue
        seen.add(key)
        kept.append((row, feature_list))

    return kept


def rmse(y_true, y_pred) -> float:
    """Calcule la racine de l'erreur quadratique moyenne."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def r2_score_manual(y_true, y_pred) -> float:
    """Calcule R² sans dépendre d'une fonction externe."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
    ss_tot = float(np.sum((y_true_arr - np.mean(y_true_arr)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Retourne N, RMSE et R² après filtrage des valeurs invalides."""
    y_true_arr = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy(dtype=float)
    y_pred_arr = pd.to_numeric(pd.Series(y_pred), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if mask.sum() == 0:
        return {"N": 0, "RMSE": np.nan, "R2": np.nan}
    return {
        "N": int(mask.sum()),
        "RMSE": rmse(y_true_arr[mask], y_pred_arr[mask]),
        "R2": r2_score_manual(y_true_arr[mask], y_pred_arr[mask]),
    }


def csds_tau(u: np.ndarray, a: float, b: float, c: float, d: float, e: float) -> np.ndarray:
    """Évalue la courbe CSDS pour un déplacement `u`."""
    return a + b * np.exp(-c * u) - d * np.exp(-e * u)


def make_u_grid(row: pd.Series, n_points: int = N_CURVE_POINTS) -> np.ndarray:
    """Construit la grille de déplacement commune pour la comparaison des courbes."""
    max_u = max(float(row["delta_peak_mm"]) * 2.0, float(row["u_r_mm"]) * 1.25, 1e-6)
    return np.linspace(0.0, max_u, n_points)


def compute_d_from_e_peak_equation(df: pd.DataFrame, e_col: str = "e_pred") -> pd.Series:
    """Reconstruit `d` à partir de `e` via l'équation du pic CSDS à `u = u_p`."""
    u_p = pd.to_numeric(df["delta_peak_mm"], errors="coerce")
    u_r = pd.to_numeric(df["u_r_mm"], errors="coerce")
    tau_p = pd.to_numeric(df["tau_peak_MPa_csds"], errors="coerce")
    tau_r = pd.to_numeric(df["tau_r_MPa"], errors="coerce")
    e_vals = pd.to_numeric(df[e_col], errors="coerce")

    exp_5up_ur = np.exp(-5.0 * u_p / u_r)
    exp_eup = np.exp(-e_vals * u_p)
    numerator = tau_p - tau_r * (1.0 - exp_5up_ur)
    denominator = exp_5up_ur - exp_eup

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d_pred = numerator / denominator

    d_pred = pd.Series(d_pred, index=df.index, dtype=float)
    d_pred = d_pred.where(np.isfinite(d_pred), np.nan)
    return d_pred


def compute_curve_metrics_for_direct_prediction(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calcule les métriques de courbe en utilisant `d_pred`, `b_pred` et `e_pred`."""
    work = df.copy()
    curve_rmse_values = []
    curve_mae_values = []
    curve_r2_values = []
    curve_valid_values = []
    tau_true_all = []
    tau_pred_all = []

    for _, row in work.iterrows():
        try:
            u = make_u_grid(row)
            tau_true = csds_tau(
                u=u,
                a=float(row["a_csds"]),
                b=float(row["b_csds"]),
                c=float(row["c_target"]),
                d=float(row["d_csds"]),
                e=float(row["e_csds"]),
            )
            tau_pred = csds_tau(
                u=u,
                a=float(row["a_csds"]),
                b=float(row["b_pred"]),
                c=float(row["c_target"]),
                d=float(row["d_pred"]),
                e=float(row["e_pred"]),
            )
            curve_rmse = rmse(tau_true, tau_pred)
            curve_mae = float(np.mean(np.abs(tau_true - tau_pred)))
            curve_r2 = r2_score_manual(tau_true, tau_pred)

            curve_rmse_values.append(curve_rmse)
            curve_mae_values.append(curve_mae)
            curve_r2_values.append(curve_r2)
            curve_valid_values.append(True)
            tau_true_all.append(tau_true)
            tau_pred_all.append(tau_pred)
        except Exception:
            curve_rmse_values.append(np.nan)
            curve_mae_values.append(np.nan)
            curve_r2_values.append(np.nan)
            curve_valid_values.append(False)

    work["curve_rmse_tau_u"] = curve_rmse_values
    work["curve_mae_tau_u"] = curve_mae_values
    work["curve_r2_tau_u"] = curve_r2_values
    work["curve_valid"] = curve_valid_values

    if not tau_true_all:
        pooled = {"rmse_tau_u": np.nan, "r2_tau_u": np.nan}
    else:
        tau_true_vec = np.concatenate(tau_true_all)
        tau_pred_vec = np.concatenate(tau_pred_all)
        pooled = {
            "rmse_tau_u": rmse(tau_true_vec, tau_pred_vec),
            "r2_tau_u": r2_score_manual(tau_true_vec, tau_pred_vec),
        }

    return work, pooled
