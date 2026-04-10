"""Recherche des meilleures combinaisons exponentielles pour prédire `log(e-c)`."""

from __future__ import annotations

from pathlib import Path
import sys
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    CV_FOLDS_BY_DATASET,
    REGRESSION_DIR,
    SPLIT_FILES,
    build_direct_e_dataset,
    dataset_slug,
    ensure_output_dirs,
    get_candidate_feature_names,
)
from direct_e.search_utils import run_genetic_feature_search  # noqa: E402


warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

GENETIC_PARAMS = {
    "population_size": 120,
    "generations": 60,
    "cx_prob": 0.7,
    "mut_prob": 0.25,
    "tournament_size": 3,
    "min_features": 1,
    "max_features": 6,
    "hall_of_fame_size": 80,
}

MODEL_PARAMS = {
    "ridge_alpha": 1.0,
    "feature_penalty": 0.01,
    "cv_std_penalty": 0.10,
}

DIVERSITY_PARAMS = {
    "similarity_threshold": 0.80,
    "max_selected_models": 30,
}


def build_exponential_model():
    """Construit le pipeline Ridge utilisé sur `z = log(e-c)`."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=MODEL_PARAMS["ridge_alpha"])),
    ])


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("RECHERCHE EXPONENTIELLE SUR log(e-c)")
    print("=" * 100)

    for dataset_name in SPLIT_FILES:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        data = build_direct_e_dataset(dataset_name)
        feature_names = get_candidate_feature_names(data)

        best_models, importance_df, logbook = run_genetic_feature_search(
            data=data,
            feature_names=feature_names,
            dataset_name=dataset_name,
            model_builder=build_exponential_model,
            model_family="exponential",
            model_params=MODEL_PARAMS,
            genetic_params=GENETIC_PARAMS,
            diversity_params=DIVERSITY_PARAMS,
            cv_folds=CV_FOLDS_BY_DATASET[dataset_name],
        )

        results_df = pd.DataFrame(best_models).sort_values(
            by=["Selection_Score", "R2_val_z", "R2_val_e"],
            ascending=False,
        ).reset_index(drop=True)

        results_file = REGRESSION_DIR / f"exp_log_e_gap_selection_{dataset_slug(dataset_name)}.csv"
        importance_file = REGRESSION_DIR / f"feature_importance_exp_log_e_gap_{dataset_slug(dataset_name)}.csv"
        results_df.to_csv(results_file, index=False)
        importance_df.to_csv(importance_file, index=False)

        print(f"Rows available: {len(data)}")
        print(f"Candidate features: {len(feature_names)}")
        print(f"Models kept: {len(results_df)}")
        print(f"Saved: {results_file}")
        print(f"Feature importance: {importance_file}")
        if not results_df.empty:
            best_row = results_df.iloc[0]
            print(
                "Best model metrics | "
                f"Val z: RMSE={best_row['RMSE_val_z']:.6f}, R2={best_row['R2_val_z']:.6f} | "
                f"Val e-c: RMSE={best_row['RMSE_val_e_gap']:.6f}, R2={best_row['R2_val_e_gap']:.6f} | "
                f"Val e: RMSE={best_row['RMSE_val_e']:.6f}, R2={best_row['R2_val_e']:.6f} | "
                f"Val tau(u): RMSE={best_row['RMSE_val_tau_u']:.6f}, R2={best_row['R2_val_tau_u']:.6f}"
            )
            print(
                "Held-out test | "
                f"z: RMSE={best_row['RMSE_test_z']:.6f}, R2={best_row['R2_test_z']:.6f} | "
                f"e-c: RMSE={best_row['RMSE_test_e_gap']:.6f}, R2={best_row['R2_test_e_gap']:.6f} | "
                f"e: RMSE={best_row['RMSE_test_e']:.6f}, R2={best_row['R2_test_e']:.6f} | "
                f"tau(u): RMSE={best_row['RMSE_test_tau_u']:.6f}, R2={best_row['R2_test_tau_u']:.6f}"
            )
            print(
                "CV stability on z | "
                f"mean R2={best_row['R2_cv_mean_z']:.6f}, std={best_row['R2_cv_std_z']:.6f}"
            )

        if len(logbook) > 0:
            max_fitness = logbook.select("max")
            avg_fitness = logbook.select("avg")
            print(f"Best fitness initiale: {max_fitness[0]:.6f}")
            print(f"Best fitness finale:   {max_fitness[-1]:.6f}")
            print(f"Fitness moyenne finale:{avg_fitness[-1]:.6f}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
