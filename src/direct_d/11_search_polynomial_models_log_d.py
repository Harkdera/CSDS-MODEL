"""Recherche des combinaisons polynomiales régularisées pour prédire `log(d_csds)`."""

import numpy as np
import pandas as pd
import random
import warnings

from deap import base, creator, tools, algorithms

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

try:
    from direct_d.common import CV_FOLDS_BY_DATASET, REGRESSION_DIR, SPLIT_FILES, RANDOM_SEED, build_d_dataset, get_candidate_feature_names
    from direct_d.search_utils import has_exact_redundancy_degree2, jaccard_similarity, is_diverse_enough, fit_and_evaluate_model
except ModuleNotFoundError:
    from common import CV_FOLDS_BY_DATASET, REGRESSION_DIR, SPLIT_FILES, RANDOM_SEED, build_d_dataset, get_candidate_feature_names
    from search_utils import has_exact_redundancy_degree2, jaccard_similarity, is_diverse_enough, fit_and_evaluate_model

warnings.filterwarnings("ignore")


# ============================================================
# 1) Chemins des données
# ============================================================
REGRESSION_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) Colonnes nécessaires
# ============================================================
required_cols = [
    "sigma_n_MPa",
    "u_r_mm",
    "delta_peak_mm",
    "tau_peak_MPa_csds",
    "tau_r_MPa",
    "d_csds",
]


# ============================================================
# 3) Paramètres globaux
# ============================================================
GENETIC_PARAMS = {
    "population_size": 120,
    "generations": 60,
    "cx_prob": 0.7,
    "mut_prob": 0.25,
    "tournament_size": 3,
    "min_features": 1,
    "max_features": 6,
    "hall_of_fame_size": 80,
    "random_seed": 42,
}

MODEL_PARAMS = {
    "degree": 2,
    "ridge_alpha": 1.0,
    "feature_penalty": 0.01,
    "cv_std_penalty": 0.10,
}

DIVERSITY_PARAMS = {
    "similarity_threshold": 0.80,
    "max_models_per_branch": 3,
    "max_selected_models": 30,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 4) Fonctions utilitaires
# ============================================================
def load_and_prepare(dataset_name):
    """Charge le dataset a partir du helper commun de la branche `direct_d`."""
    return build_d_dataset(dataset_name, include_targets=("d_csds",))


def build_model(degree=2, ridge_alpha=1.0):
    """Construit le pipeline polynomial régularisé utilisé pendant l'évaluation."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("reg", Ridge(alpha=ridge_alpha))
    ])


def fit_and_evaluate(train_df, val_df, feature_cols, degree=2, ridge_alpha=1.0, cv_folds=5):
    """Ajuste le modèle sur `log(d_csds)` puis retourne les métriques sur `log(d)` et `d`."""
    result = fit_and_evaluate_model(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        model_builder=lambda: build_model(degree=degree, ridge_alpha=ridge_alpha),
        target_col="log_d_csds",
        metric_target_col="d_csds",
        cv_folds=cv_folds,
        transform_prediction=np.exp,
    )
    return {
        "features": feature_cols,
        "degree": degree,
        "n_features": len(feature_cols),
        "R2_train_log": result["R2_train_target"],
        "RMSE_train_log": result["RMSE_train_target"],
        "R2_val_log": result["R2_val_target"],
        "RMSE_val_log": result["RMSE_val_target"],
        "R2_cv_mean_log": result["R2_cv_mean_target"],
        "R2_cv_std_log": result["R2_cv_std_target"],
        "R2_train_d": result["R2_train_metric"],
        "RMSE_train_d": result["RMSE_train_metric"],
        "R2_val_d": result["R2_val_metric"],
        "RMSE_val_d": result["RMSE_val_metric"],
    }


def get_branch_signature(feature_list):
    """Construit une signature de branche pour diversifier les modèles sélectionnés."""
    """
    Assign a branch signature so we do not keep only one family of models.
    """
    tags = []

    if "sigma_n_MPa" in feature_list:
        tags.append("base_sigma")
    if "u_r_mm" in feature_list:
        tags.append("base_ur")
    if "delta_peak_mm" in feature_list:
        tags.append("base_up")
    if "tau_peak_MPa_csds" in feature_list:
        tags.append("base_taup")
    if "tau_r_MPa" in feature_list:
        tags.append("base_taur")

    if any("div" in f for f in feature_list):
        tags.append("ratio")
    if any("x_" in f for f in feature_list):
        tags.append("product")

    if "tau_p_div_u_r" in feature_list:
        tags.append("core_taup_over_ur")
    if "u_p_x_tau_p" in feature_list:
        tags.append("core_up_xtaup")
    if "tau_p_div_sigma_n" in feature_list:
        tags.append("core_taup_over_sigma")
    if any("sigma_n" in f and f != "sigma_n_MPa" for f in feature_list):
        tags.append("core_sigma_combo")

    return tuple(sorted(tags))


# ============================================================
# 6) DEAP SETUP
# ============================================================
if not hasattr(creator, "FitnessMaxCSDSLog"):
    creator.create("FitnessMaxCSDSLog", base.Fitness, weights=(1.0,))

if not hasattr(creator, "IndividualCSDSLog"):
    creator.create("IndividualCSDSLog", list, fitness=creator.FitnessMaxCSDSLog)


def setup_genetic_algorithm(train_df, val_df, all_features, cv_folds):
    """Configure l'algorithme génétique qui explore les sous-ensembles de variables."""
    """
    Configure the genetic algorithm.
    """
    toolbox = base.Toolbox()

    def create_individual():
        n_features = random.randint(
            GENETIC_PARAMS["min_features"],
            GENETIC_PARAMS["max_features"]
        )
        return creator.IndividualCSDSLog(random.sample(all_features, n_features))

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_features(individual):
        feature_list = list(dict.fromkeys(individual))

        if len(feature_list) < GENETIC_PARAMS["min_features"]:
            return (-9999.0,)

        if len(feature_list) > GENETIC_PARAMS["max_features"]:
            return (-9999.0,)

        if has_exact_redundancy_degree2(feature_list, MODEL_PARAMS["degree"]):
            return (-9999.0,)

        try:
            result = fit_and_evaluate(
                train_df=train_df,
                val_df=val_df,
                feature_cols=feature_list,
                degree=MODEL_PARAMS["degree"],
                ridge_alpha=MODEL_PARAMS["ridge_alpha"],
                cv_folds=cv_folds
            )

            # Selection based on log(d)
            score = (
                result["R2_val_log"]
                + 0.5 * result["R2_cv_mean_log"]
                - MODEL_PARAMS["feature_penalty"] * len(feature_list)
                - MODEL_PARAMS["cv_std_penalty"] * result["R2_cv_std_log"]
            )

            if not np.isfinite(score):
                return (-9999.0,)

            return (score,)
        except Exception:
            return (-9999.0,)

    def mate_features(ind1, ind2):
        set1 = list(dict.fromkeys(ind1))
        set2 = list(dict.fromkeys(ind2))

        union = list(dict.fromkeys(set1 + set2))
        if len(union) == 0:
            return ind1, ind2

        size1 = random.randint(
            GENETIC_PARAMS["min_features"],
            min(GENETIC_PARAMS["max_features"], len(union))
        )
        size2 = random.randint(
            GENETIC_PARAMS["min_features"],
            min(GENETIC_PARAMS["max_features"], len(union))
        )

        child1 = random.sample(union, size1)
        child2 = random.sample(union, size2)

        ind1[:] = child1
        ind2[:] = child2
        return ind1, ind2

    def mutate_features(individual):
        current = list(dict.fromkeys(individual))
        current_set = set(current)
        all_set = set(all_features)

        if random.random() < 0.5 and len(current) < GENETIC_PARAMS["max_features"]:
            available = list(all_set - current_set)
            if available:
                current.append(random.choice(available))
        elif len(current) > GENETIC_PARAMS["min_features"]:
            idx = random.randrange(len(current))
            current.pop(idx)

        individual[:] = current
        return (individual,)

    toolbox.register("evaluate", eval_features)
    toolbox.register("mate", mate_features)
    toolbox.register("mutate", mutate_features)
    toolbox.register("select", tools.selTournament, tournsize=GENETIC_PARAMS["tournament_size"])

    return toolbox


def run_genetic_algorithm(toolbox):
    """Exécute l'algorithme génétique et renvoie la population finale et le hall of fame."""
    """
    Run the genetic algorithm.
    """
    pop = toolbox.population(n=GENETIC_PARAMS["population_size"])
    hof = tools.HallOfFame(GENETIC_PARAMS["hall_of_fame_size"])

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=GENETIC_PARAMS["cx_prob"],
        mutpb=GENETIC_PARAMS["mut_prob"],
        ngen=GENETIC_PARAMS["generations"],
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    return pop, logbook, hof


# ============================================================
# 7) MAIN LOOP
# ============================================================
for dataset_name in SPLIT_FILES:
    print("\n" + "=" * 100)
    print(f"DATASET: {dataset_name}")
    print("=" * 100)

    data = load_and_prepare(dataset_name)
    all_features = get_candidate_feature_names(data)
    cv_folds = CV_FOLDS_BY_DATASET[dataset_name]

    print(f"Nombre de lignes utilisables: {len(data)}")
    print(f"Validation croisee utilisee: {cv_folds} folds")
    print("Target utilisee pour la selection: log_d_csds")

    train_df, val_df = train_test_split(
        data,
        test_size=0.2,
        random_state=GENETIC_PARAMS["random_seed"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    toolbox = setup_genetic_algorithm(train_df, val_df, all_features, cv_folds)
    pop, logbook, hof = run_genetic_algorithm(toolbox)

    print("\n" + "=" * 100)
    print(f"TOP 30 MODELES DIVERSIFIES (recherche genetique sur log_d_csds) - {dataset_name}")
    print("=" * 100)

    best_models = []
    seen_exact_sets = set()
    branch_count = {}

    for indiv in hof:
        feature_list = list(dict.fromkeys(indiv))
        exact_key = tuple(sorted(feature_list))

        if exact_key in seen_exact_sets:
            continue
        seen_exact_sets.add(exact_key)

        if has_exact_redundancy_degree2(feature_list, MODEL_PARAMS["degree"]):
            continue

        branch_signature = get_branch_signature(feature_list)

        if branch_count.get(branch_signature, 0) >= DIVERSITY_PARAMS["max_models_per_branch"]:
            continue

        if not is_diverse_enough(
            feature_list,
            best_models,
            DIVERSITY_PARAMS["similarity_threshold"]
        ):
            continue

        result = fit_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_list,
            degree=MODEL_PARAMS["degree"],
            ridge_alpha=MODEL_PARAMS["ridge_alpha"],
            cv_folds=cv_folds
        )

        score_used = (
            result["R2_val_log"]
            + 0.5 * result["R2_cv_mean_log"]
            - MODEL_PARAMS["feature_penalty"] * len(feature_list)
            - MODEL_PARAMS["cv_std_penalty"] * result["R2_cv_std_log"]
        )

        best_models.append({
            "Dataset": dataset_name,
            "Features": " + ".join(feature_list),
            "Feature_List": feature_list,
            "Branch_Signature": " | ".join(branch_signature),
            "Degree": result["degree"],
            "N_Features": result["n_features"],

            # Metrics on log(d)
            "R2_train_log": result["R2_train_log"],
            "RMSE_train_log": result["RMSE_train_log"],
            "R2_val_log": result["R2_val_log"],
            "RMSE_val_log": result["RMSE_val_log"],
            "R2_cv_mean_log": result["R2_cv_mean_log"],
            "R2_cv_std_log": result["R2_cv_std_log"],

            # Metrics on original d
            "R2_train_d": result["R2_train_d"],
            "RMSE_train_d": result["RMSE_train_d"],
            "R2_val_d": result["R2_val_d"],
            "RMSE_val_d": result["RMSE_val_d"],

            "Selection_Score": score_used,
        })

        branch_count[branch_signature] = branch_count.get(branch_signature, 0) + 1

        print(f"\n--- Modele Genetique #{len(best_models)} ---")
        print(f"Features ({len(feature_list)}): {feature_list}")
        print(f"Branche           = {branch_signature}")
        print(f"R2_train_log      = {result['R2_train_log']:.6f}")
        print(f"R2_val_log        = {result['R2_val_log']:.6f}")
        print(f"R2_cv_mean_log    = {result['R2_cv_mean_log']:.6f}")
        print(f"R2_cv_std_log     = {result['R2_cv_std_log']:.6f}")
        print(f"RMSE_val_log      = {result['RMSE_val_log']:.6f}")
        print(f"R2_val_d          = {result['R2_val_d']:.6f}")
        print(f"RMSE_val_d        = {result['RMSE_val_d']:.6f}")
        print(f"Selection_Score   = {score_used:.6f}")

        if len(best_models) >= DIVERSITY_PARAMS["max_selected_models"]:
            break

    print("\n" + "=" * 100)
    print(f"STATISTIQUES DE CONVERGENCE - {dataset_name}")
    print("=" * 100)

    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    print(f"Meilleure fitness initiale: {max_fitness[0]:.6f}")
    print(f"Meilleure fitness finale:   {max_fitness[-1]:.6f}")
    print(f"Fitness moyenne finale:     {avg_fitness[-1]:.6f}")

    if best_models:
        summary_df = pd.DataFrame(best_models)
        summary_df = summary_df.sort_values(
            by=["Selection_Score", "R2_val_log", "R2_cv_mean_log"],
            ascending=False
        ).reset_index(drop=True)

        output_path = REGRESSION_DIR / f"log_d_csds_genetic_algorithm_diverse_{dataset_name.lower()}.csv"
        summary_df.to_csv(output_path, index=False)
        print(f"\nSauvegarde: {output_path}")

        feature_importance = {}
        for model in best_models:
            for f in model["Feature_List"]:
                feature_importance[f] = feature_importance.get(f, 0) + 1

        importance_df = pd.DataFrame([
            {"Feature": k, "Frequency": v}
            for k, v in feature_importance.items()
        ]).sort_values(by="Frequency", ascending=False)

        importance_path = REGRESSION_DIR / f"log_feature_importance_genetic_diverse_{dataset_name.lower()}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"Importance des features: {importance_path}")

        print("\n" + "=" * 100)
        print("FEATURES LES PLUS IMPORTANTES")
        print("=" * 100)
        print(importance_df.head(15).to_string(index=False))

    print("\n" + "=" * 100)
    print(f"TERMINE POUR {dataset_name}")
    print("=" * 100)

print("\n" + "=" * 100)
print("ANALYSE GENETIQUE DIVERSIFIEE COMPLETE SUR log_d_csds POUR TOUS LES DATASETS")
print("=" * 100)
