"""Utilitaires partagés pour l'interprétation FULL de la méthodologie 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from methodologie_2.common import (
    REGRESSION_DIR,
    build_direct_e_dataset,
    parse_feature_list,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INTERPRETATION_ROOT = PROJECT_ROOT / "results" / "interpretation" / "methodologie_2_full_only"
SCATTER_DIR = INTERPRETATION_ROOT / "scatter_log_e_minus_c"
PCA_DIR = INTERPRETATION_ROOT / "pca"
CORRELATION_DIR = INTERPRETATION_ROOT / "correlation"

FEATURE_IMPORTANCE_FILE = REGRESSION_DIR / "feature_importance_poly_log_e_gap_full.csv"
POLY_SELECTION_FILE = REGRESSION_DIR / "poly_log_e_gap_selection_full.csv"

TOP10_FEATURES_FILE = INTERPRETATION_ROOT / "top10_features_methodologie_2_full.csv"
TOP10_DATASET_FILE = INTERPRETATION_ROOT / "methodologie_2_full_top10_dataset.csv"

TARGET_COL = "log_e_minus_c_csds"
TARGET_LABEL = r"$\log(e - c)$"
DATASET_NAME = "FULL"
TOP_N_FEATURES = 10

DISPLAY_LABELS = {
    "sigma_n_MPa": r"$\sigma_n$ (MPa)",
    "u_r_mm": r"$u_r$ (mm)",
    "delta_peak_mm": r"$u_p$ (mm)",
    "tau_peak_MPa_csds": r"$\tau_p$ (MPa)",
    "tau_r_MPa": r"$\tau_r$ (MPa)",
    "u_p_div_u_r": r"$u_p/u_r$",
    "u_r_div_u_p": r"$u_r/u_p$",
    "tau_p_div_tau_r": r"$\tau_p/\tau_r$",
    "tau_r_div_tau_p": r"$\tau_r/\tau_p$",
    "u_r_div_tau_p": r"$u_r/\tau_p$",
    "u_r_div_tau_r": r"$u_r/\tau_r$",
    "u_p_div_tau_p": r"$u_p/\tau_p$",
    "u_p_div_tau_r": r"$u_p/\tau_r$",
    "tau_p_div_u_r": r"$\tau_p/u_r$",
    "tau_p_div_u_p": r"$\tau_p/u_p$",
    "tau_r_div_u_r": r"$\tau_r/u_r$",
    "tau_r_div_u_p": r"$\tau_r/u_p$",
    "tau_p_div_sigma_n": r"$\tau_p/\sigma_n$",
    "tau_r_div_sigma_n": r"$\tau_r/\sigma_n$",
    "sigma_n_div_tau_p": r"$\sigma_n/\tau_p$",
    "sigma_n_div_tau_r": r"$\sigma_n/\tau_r$",
    "sigma_n_div_u_r": r"$\sigma_n/u_r$",
    "sigma_n_div_u_p": r"$\sigma_n/u_p$",
    "sigma_n_x_u_r": r"$\sigma_n u_r$",
    "sigma_n_x_u_p": r"$\sigma_n u_p$",
    "sigma_n_x_tau_p": r"$\sigma_n \tau_p$",
    "sigma_n_x_tau_r": r"$\sigma_n \tau_r$",
    "u_r_x_u_p": r"$u_r u_p$",
    "u_r_x_tau_p": r"$u_r \tau_p$",
    "u_r_x_tau_r": r"$u_r \tau_r$",
    "u_p_x_tau_p": r"$u_p \tau_p$",
    "u_p_x_tau_r": r"$u_p \tau_r$",
    "tau_p_x_tau_r": r"$\tau_p \tau_r$",
    TARGET_COL: TARGET_LABEL,
}


def ensure_output_dirs() -> None:
    """Crée les dossiers utilisés par les analyses FULL-only."""
    for path in [INTERPRETATION_ROOT, SCATTER_DIR, PCA_DIR, CORRELATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    """Produit un nom de fichier stable."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def label(name: str) -> str:
    """Retourne une étiquette plus lisible pour les figures."""
    return DISPLAY_LABELS.get(name, name)


def load_full_polynomial_feature_frequency() -> pd.DataFrame:
    """Charge ou reconstitue la fréquence des variables des modèles polynomiaux FULL."""
    if FEATURE_IMPORTANCE_FILE.exists():
        freq = pd.read_csv(FEATURE_IMPORTANCE_FILE).rename(
            columns={"Feature": "feature", "Frequency": "frequency"}
        )
    elif POLY_SELECTION_FILE.exists():
        selection = pd.read_csv(POLY_SELECTION_FILE)
        counts: dict[str, int] = {}
        for value in selection.get("Feature_List", selection.get("Features", [])):
            for feature in parse_feature_list(value):
                counts[feature] = counts.get(feature, 0) + 1
        freq = pd.DataFrame(
            [{"feature": feature, "frequency": frequency} for feature, frequency in counts.items()]
        )
    else:
        raise FileNotFoundError(
            "Missing FULL polynomial methodology 2 files. "
            f"Expected {FEATURE_IMPORTANCE_FILE} or {POLY_SELECTION_FILE}."
        )

    if freq.empty:
        raise ValueError("No FULL methodology 2 polynomial features were found.")

    freq = freq.sort_values(["frequency", "feature"], ascending=[False, True]).reset_index(drop=True)
    freq["feature_label"] = freq["feature"].map(label)
    return freq


def load_top10_features(top_n: int = TOP_N_FEATURES) -> pd.DataFrame:
    """Retourne le top 10 des variables FULL pour la méthodologie 2 polynomiale."""
    freq = load_full_polynomial_feature_frequency().head(top_n).copy()
    freq.insert(0, "rank", range(1, len(freq) + 1))
    return freq


def load_full_dataset() -> pd.DataFrame:
    """Charge le dataset FULL complet de la méthodologie 2."""
    data = build_direct_e_dataset(DATASET_NAME)
    if TARGET_COL not in data.columns:
        raise KeyError(f"Target column missing from FULL dataset: {TARGET_COL}")
    return data


def build_top10_analysis_dataset(top_n: int = TOP_N_FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit le dataset FULL restreint au top 10 et à la cible log(e-c)."""
    top10 = load_top10_features(top_n=top_n)
    data = load_full_dataset()
    feature_cols = [feature for feature in top10["feature"] if feature in data.columns]
    missing = sorted(set(top10["feature"]) - set(feature_cols))
    if missing:
        raise KeyError(f"Top features missing from FULL dataset: {missing}")

    columns = ["sample_id", TARGET_COL, *feature_cols]
    subset = data[columns].dropna().reset_index(drop=True)
    return top10, subset
