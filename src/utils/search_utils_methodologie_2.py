"""Outils génériques de recherche de combinaisons de variables pour la cible `log(e-c)`."""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, tools
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from src.utils.common_methodologie_2 import (
    RANDOM_SEED,
    compute_curve_metrics_for_direct_prediction,
    compute_d_from_e_peak_equation,
    compute_metrics,
)


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
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_builder: Callable[[], object],
    target_col: str = "log_e_minus_c_csds",
    original_target_col: str = "e_csds",
    gap_target_col: str = "e_minus_c_csds",
    c_col: str = "c_target",
    cv_folds: int = 5,
) -> dict[str, float]:
    """Ajuste un modèle, puis retourne des métriques sur `z`, `e-c`, `e` et `tau(u)`."""
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = model_builder()
    model.fit(X_train, y_train)

    z_train_pred = model.predict(X_train)
    z_val_pred = model.predict(X_val)
    z_test_pred = model.predict(X_test)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

    gap_train_pred = np.exp(z_train_pred)
    gap_val_pred = np.exp(z_val_pred)
    gap_test_pred = np.exp(z_test_pred)
    e_train_pred = train_df[c_col].to_numpy(dtype=float) + gap_train_pred
    e_val_pred = val_df[c_col].to_numpy(dtype=float) + gap_val_pred
    e_test_pred = test_df[c_col].to_numpy(dtype=float) + gap_test_pred

    z_train_metrics = compute_metrics(train_df[target_col], z_train_pred)
    z_val_metrics = compute_metrics(val_df[target_col], z_val_pred)
    z_test_metrics = compute_metrics(test_df[target_col], z_test_pred)
    gap_train_metrics = compute_metrics(train_df[gap_target_col], gap_train_pred)
    gap_val_metrics = compute_metrics(val_df[gap_target_col], gap_val_pred)
    gap_test_metrics = compute_metrics(test_df[gap_target_col], gap_test_pred)
    e_train_metrics = compute_metrics(train_df[original_target_col], e_train_pred)
    e_val_metrics = compute_metrics(val_df[original_target_col], e_val_pred)
    e_test_metrics = compute_metrics(test_df[original_target_col], e_test_pred)

    train_curve_df = train_df.copy()
    train_curve_df["e_pred"] = e_train_pred
    train_curve_df["d_pred"] = compute_d_from_e_peak_equation(train_curve_df, e_col="e_pred")
    train_curve_df["b_pred"] = train_curve_df["d_pred"] - train_curve_df["a_csds"]
    train_curve_df, train_curve_metrics = compute_curve_metrics_for_direct_prediction(train_curve_df)

    val_curve_df = val_df.copy()
    val_curve_df["e_pred"] = e_val_pred
    val_curve_df["d_pred"] = compute_d_from_e_peak_equation(val_curve_df, e_col="e_pred")
    val_curve_df["b_pred"] = val_curve_df["d_pred"] - val_curve_df["a_csds"]
    val_curve_df, val_curve_metrics = compute_curve_metrics_for_direct_prediction(val_curve_df)

    test_curve_df = test_df.copy()
    test_curve_df["e_pred"] = e_test_pred
    test_curve_df["d_pred"] = compute_d_from_e_peak_equation(test_curve_df, e_col="e_pred")
    test_curve_df["b_pred"] = test_curve_df["d_pred"] - test_curve_df["a_csds"]
    test_curve_df, test_curve_metrics = compute_curve_metrics_for_direct_prediction(test_curve_df)

    return {
        "n_features": len(feature_cols),
        "target_mode": "direct_constrained_e_via_log_e_minus_c",
        "target_used_for_training": target_col,
        "reconstructed_gap_formula": "e_minus_c_pred = exp(z_pred)",
        "reconstructed_e_formula": "e_pred = c_target + exp(z_pred)",
        "R2_train_z": z_train_metrics["R2"],
        "RMSE_train_z": z_train_metrics["RMSE"],
        "R2_val_z": z_val_metrics["R2"],
        "RMSE_val_z": z_val_metrics["RMSE"],
        "R2_test_z": z_test_metrics["R2"],
        "RMSE_test_z": z_test_metrics["RMSE"],
        "R2_cv_mean_z": float(np.mean(cv_scores)),
        "R2_cv_std_z": float(np.std(cv_scores)),
        "R2_train_e_gap": gap_train_metrics["R2"],
        "RMSE_train_e_gap": gap_train_metrics["RMSE"],
        "R2_val_e_gap": gap_val_metrics["R2"],
        "RMSE_val_e_gap": gap_val_metrics["RMSE"],
        "R2_test_e_gap": gap_test_metrics["R2"],
        "RMSE_test_e_gap": gap_test_metrics["RMSE"],
        "R2_train_e": e_train_metrics["R2"],
        "RMSE_train_e": e_train_metrics["RMSE"],
        "R2_val_e": e_val_metrics["R2"],
        "RMSE_val_e": e_val_metrics["RMSE"],
        "R2_test_e": e_test_metrics["R2"],
        "RMSE_test_e": e_test_metrics["RMSE"],
        "RMSE_train_tau_u": train_curve_metrics["rmse_tau_u"],
        "R2_train_tau_u": train_curve_metrics["r2_tau_u"],
        "RMSE_val_tau_u": val_curve_metrics["rmse_tau_u"],
        "R2_val_tau_u": val_curve_metrics["r2_tau_u"],
        "RMSE_test_tau_u": test_curve_metrics["rmse_tau_u"],
        "R2_test_tau_u": test_curve_metrics["r2_tau_u"],
        "Valid_curve_count_train": int(train_curve_df["curve_valid"].sum()),
        "Valid_curve_count_val": int(val_curve_df["curve_valid"].sum()),
        "Valid_curve_count_test": int(test_curve_df["curve_valid"].sum()),
    }


def run_genetic_feature_search(
    data: pd.DataFrame,
    feature_names: list[str],
    dataset_name: str,
    model_builder: Callable[[], object],
    model_family: str,
    model_params: dict,
    genetic_params: dict,
    diversity_params: dict,
    cv_folds: int,
) -> tuple[list[dict], pd.DataFrame, object]:
    """Lance la recherche génétique et retourne les meilleurs modèles diversifiés."""
    train_val_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    individual_name = f"Individual_{model_family}_e_gap_{dataset_name}"
    fitness_name = f"Fitness_{model_family}_e_gap_{dataset_name}"
    individual_cls = ensure_creator_classes(fitness_name, individual_name)

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        build_random_individual,
        individual_cls,
        len(feature_names),
        genetic_params["min_features"],
        genetic_params["max_features"],
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_individual(individual):
        selected = [feature_names[idx] for idx, bit in enumerate(individual) if bit == 1]

        if len(selected) < genetic_params["min_features"] or len(selected) > genetic_params["max_features"]:
            return (-1e9,)

        if has_exact_redundancy(selected):
            return (-1e9,)

        try:
            result = fit_and_evaluate_model(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=selected,
                model_builder=model_builder,
                cv_folds=cv_folds,
            )
            score = (
                result["R2_val_z"]
                + 0.5 * result["R2_cv_mean_z"]
                - model_params["feature_penalty"] * len(selected)
                - model_params["cv_std_penalty"] * result["R2_cv_std_z"]
            )
            return (score,)
        except Exception:
            return (-1e9,)

    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.08)
    toolbox.register("select", tools.selTournament, tournsize=genetic_params["tournament_size"])

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    population = toolbox.population(n=genetic_params["population_size"])
    hall_of_fame = tools.HallOfFame(genetic_params["hall_of_fame_size"])

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=genetic_params["cx_prob"],
        mutpb=genetic_params["mut_prob"],
        ngen=genetic_params["generations"],
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True,
    )

    best_models: list[dict] = []
    feature_importance: dict[str, int] = {}

    for individual in hall_of_fame:
        feature_list = [feature_names[idx] for idx, bit in enumerate(individual) if bit == 1]

        if not feature_list:
            continue
        if has_exact_redundancy(feature_list):
            continue
        if not is_diverse_enough(feature_list, best_models, diversity_params["similarity_threshold"]):
            continue

        result = fit_and_evaluate_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_list,
            model_builder=model_builder,
            cv_folds=cv_folds,
        )
        selection_score = (
            result["R2_val_z"]
            + 0.5 * result["R2_cv_mean_z"]
            - model_params["feature_penalty"] * len(feature_list)
            - model_params["cv_std_penalty"] * result["R2_cv_std_z"]
        )

        best_models.append({
            "Dataset": dataset_name,
            "Model_Family": model_family,
            "Features": " + ".join(feature_list),
            "Feature_List": repr(feature_list),
            "N_Features": len(feature_list),
            "R2_train_z": result["R2_train_z"],
            "RMSE_train_z": result["RMSE_train_z"],
            "R2_val_z": result["R2_val_z"],
            "RMSE_val_z": result["RMSE_val_z"],
            "R2_cv_mean_z": result["R2_cv_mean_z"],
            "R2_cv_std_z": result["R2_cv_std_z"],
            "R2_train_e_gap": result["R2_train_e_gap"],
            "RMSE_train_e_gap": result["RMSE_train_e_gap"],
            "R2_val_e_gap": result["R2_val_e_gap"],
            "RMSE_val_e_gap": result["RMSE_val_e_gap"],
            "R2_test_e_gap": result["R2_test_e_gap"],
            "RMSE_test_e_gap": result["RMSE_test_e_gap"],
            "R2_train_e": result["R2_train_e"],
            "RMSE_train_e": result["RMSE_train_e"],
            "R2_val_e": result["R2_val_e"],
            "RMSE_val_e": result["RMSE_val_e"],
            "R2_test_z": result["R2_test_z"],
            "RMSE_test_z": result["RMSE_test_z"],
            "R2_test_e": result["R2_test_e"],
            "RMSE_test_e": result["RMSE_test_e"],
            "RMSE_train_tau_u": result["RMSE_train_tau_u"],
            "R2_train_tau_u": result["R2_train_tau_u"],
            "RMSE_val_tau_u": result["RMSE_val_tau_u"],
            "R2_val_tau_u": result["R2_val_tau_u"],
            "RMSE_test_tau_u": result["RMSE_test_tau_u"],
            "R2_test_tau_u": result["R2_test_tau_u"],
            "Valid_curve_count_train": result["Valid_curve_count_train"],
            "Valid_curve_count_val": result["Valid_curve_count_val"],
            "Valid_curve_count_test": result["Valid_curve_count_test"],
            "Target_Mode": result["target_mode"],
            "Target_Used_For_Training": result["target_used_for_training"],
            "Reconstructed_Gap_Formula": result["reconstructed_gap_formula"],
            "Reconstructed_E_Formula": result["reconstructed_e_formula"],
            "Selection_Score": selection_score,
        })

        for feature in feature_list:
            feature_importance[feature] = feature_importance.get(feature, 0) + 1

        if len(best_models) >= diversity_params["max_selected_models"]:
            break

    importance_df = pd.DataFrame(
        [{"Feature": feature, "Frequency": count} for feature, count in feature_importance.items()]
    ).sort_values(by="Frequency", ascending=False)

    return best_models, importance_df, logbook
