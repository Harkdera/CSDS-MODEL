"""Outils génériques de recherche de combinaisons de variables pour la branche `methodologie_1`."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, tools
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.utils.common_methodologie_1 import RANDOM_SEED, compute_metrics


def has_exact_redundancy(feature_cols: list[str]) -> bool:
    """Écarte les combinaisons contenant une redondance algébrique exacte."""
    fs = set(feature_cols)
    exact_redundancies = [
        {"u_r_mm", "tau_peak_MPa_csds", "tau_p_div_u_r"},
        {"delta_peak_mm", "tau_peak_MPa_csds", "tau_p_div_u_p"},
        {"tau_r_MPa", "tau_peak_MPa_csds", "tau_p_div_tau_r"},
        {"tau_peak_MPa_csds", "tau_r_MPa", "tau_r_div_tau_p"},
        {"delta_peak_mm", "u_r_mm", "u_r_div_u_p"},
        {"u_r_mm", "delta_peak_mm", "u_p_div_u_r"},
        {"sigma_n_MPa", "tau_peak_MPa_csds", "tau_p_div_sigma_n"},
        {"sigma_n_MPa", "tau_r_MPa", "tau_r_div_sigma_n"},
        {"sigma_n_MPa", "u_r_mm", "sigma_n_div_u_r"},
        {"sigma_n_MPa", "delta_peak_mm", "sigma_n_div_u_p"},
    ]
    return any(pattern.issubset(fs) for pattern in exact_redundancies)


def has_exact_redundancy_degree2(feature_cols: list[str], degree: int) -> bool:
    """Version spécifique aux modèles polynomiaux degré 2."""
    if degree != 2:
        return False
    return has_exact_redundancy(feature_cols)


def build_random_individual(individual_cls, n_features: int, min_features: int, max_features: int):
    """Construit un individu binaire avec un nombre raisonnable de variables actives."""
    genes = [0] * n_features
    selected_count = random.randint(min_features, max_features)
    for idx in random.sample(range(n_features), selected_count):
        genes[idx] = 1
    return individual_cls(genes)


def jaccard_similarity(list_a: list[str], list_b: list[str]) -> float:
    """Mesure la proximité entre deux ensembles de variables."""
    set_a = set(list_a)
    set_b = set(list_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def is_diverse_enough(feature_list: list[str], selected_models: list[dict], threshold: float) -> bool:
    """Vérifie qu'un modèle est suffisamment différent des modèles déjà retenus."""
    for model in selected_models:
        if jaccard_similarity(feature_list, model["Feature_List"]) >= threshold:
            return False
    return True


def ensure_creator_classes(fitness_name: str, individual_name: str):
    """Crée les classes DEAP si elles n'existent pas encore dans le processus."""
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(1.0,))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
    return getattr(creator, individual_name)


def fit_and_evaluate_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: list[str],
    model_builder: Callable[[], object],
    target_col: str = "d_csds",
    metric_target_col: str = "d_csds",
    cv_folds: int = 5,
    transform_prediction: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, float]:
    """Ajuste un modèle et retourne les métriques train/validation/test/CV.

    Cette fonction garde la méthodologie 1 alignée avec la méthodologie 2 :
    le modèle est sélectionné sur validation + stabilité en validation croisée,
    puis la performance est aussi rapportée sur un jeu test externe non utilisé
    pendant l'ajustement ni pendant la sélection génétique.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    model = model_builder()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = None
    if test_df is not None:
        X_test = test_df[feature_cols]
        test_pred = model.predict(X_test)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)
    if test_df is not None and test_pred is not None:
        test_metrics = compute_metrics(test_df[target_col], test_pred)
    else:
        test_metrics = {"R2": np.nan, "RMSE": np.nan}

    result = {
        "n_features": len(feature_cols),
        "R2_train_target": train_metrics["R2"],
        "RMSE_train_target": train_metrics["RMSE"],
        "R2_val_target": val_metrics["R2"],
        "RMSE_val_target": val_metrics["RMSE"],
        "R2_test_target": test_metrics["R2"],
        "RMSE_test_target": test_metrics["RMSE"],
        "R2_cv_mean_target": float(np.mean(cv_scores)),
        "R2_cv_std_target": float(np.std(cv_scores)),
    }

    if transform_prediction is not None:
        train_metric_pred = transform_prediction(train_pred)
        val_metric_pred = transform_prediction(val_pred)
        train_metric_metrics = compute_metrics(train_df[metric_target_col], train_metric_pred)
        val_metric_metrics = compute_metrics(val_df[metric_target_col], val_metric_pred)
        if test_df is not None and test_pred is not None:
            test_metric_pred = transform_prediction(test_pred)
            test_metric_metrics = compute_metrics(test_df[metric_target_col], test_metric_pred)
        else:
            test_metric_metrics = {"R2": np.nan, "RMSE": np.nan}
        result.update({
            "R2_train_metric": train_metric_metrics["R2"],
            "RMSE_train_metric": train_metric_metrics["RMSE"],
            "R2_val_metric": val_metric_metrics["R2"],
            "RMSE_val_metric": val_metric_metrics["RMSE"],
            "R2_test_metric": test_metric_metrics["R2"],
            "RMSE_test_metric": test_metric_metrics["RMSE"],
        })

    return result
