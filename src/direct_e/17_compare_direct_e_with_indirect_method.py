"""Compare l'estimation directe contrainte de `e` avec la méthode indirecte via `d`."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    COMPARISON_DIR,
    EVALUATION_DIR,
    ensure_output_dirs,
)
from direct_d.common import COMPARE_DIR as INDIRECT_COMPARE_DIR, E_FROM_D_DIR as INDIRECT_E_FROM_D_DIR


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DIRECT_SUMMARY_FILE = EVALUATION_DIR / "summary_all_b_d_from_e_models.csv"
INDIRECT_E_SUMMARY_FILE = INDIRECT_E_FROM_D_DIR / "summary_all_retained_models_e_from_d.csv"
INDIRECT_CURVE_SUMMARY_FILE = INDIRECT_COMPARE_DIR / "summary_all_models_compare_d_b_e_tau_u.csv"


def load_direct_summary() -> pd.DataFrame:
    """Charge et normalise le résumé des modèles directs."""
    df = pd.read_csv(DIRECT_SUMMARY_FILE)
    return df.assign(method="direct_log_e_minus_c").copy()


def load_indirect_summary() -> pd.DataFrame:
    """Charge et normalise le résumé des modèles indirects `d -> e`."""
    e_df = pd.read_csv(INDIRECT_E_SUMMARY_FILE)
    curve_df = pd.read_csv(INDIRECT_CURVE_SUMMARY_FILE)

    curve_df = curve_df.rename(columns={"dataset": "Dataset", "model_name": "Model_Label"})
    merged = e_df.merge(
        curve_df[["Dataset", "Model_Label", "rmse_tau_u", "r2_tau_u", "mean_curve_rmse_tau_u"]],
        on=["Dataset", "Model_Label"],
        how="left",
    )

    return pd.DataFrame({
        "dataset": merged["Dataset"],
        "model_name": merged["Model_Label"],
        "model_family": np.where(merged["Model_Label"].str.contains("_poly_"), "polynomial", "exponential"),
        "selection_mode": merged["Selection_Mode"],
        "rank_in_saved_results": merged["Rank_in_saved_results"],
        "rmse_e": merged["RMSE_e_pred_vs_e_csds"],
        "r2_e": merged["R2_e_pred_vs_e_csds"],
        "rmse_tau_u": merged["rmse_tau_u"],
        "r2_tau_u": merged["r2_tau_u"],
        "mean_curve_rmse_tau_u": merged["mean_curve_rmse_tau_u"],
        "valid_curve_count": merged["N_converged_and_c_lt_e"],
        "constraint_ok_count": merged["N_c_lt_e"],
        "method": "indirect_d_then_simon",
    })


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("COMPARAISON ENTRE METHODE DIRECTE ET METHODE INDIRECTE")
    print("=" * 100)

    direct_df = load_direct_summary()
    indirect_df = load_indirect_summary()

    comparison_df = pd.concat([direct_df, indirect_df], ignore_index=True, sort=False)
    comparison_df = comparison_df.sort_values(
        by=["dataset", "method", "rmse_tau_u", "rmse_e", "r2_e"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    overall_file = COMPARISON_DIR / "direct_vs_indirect_all_models.csv"
    comparison_df.to_csv(overall_file, index=False)

    best_by_method = comparison_df.groupby(["dataset", "method"], as_index=False).first()
    best_by_method_file = COMPARISON_DIR / "direct_vs_indirect_best_by_method.csv"
    best_by_method.to_csv(best_by_method_file, index=False)

    best_overall = comparison_df.groupby("dataset", as_index=False).first()
    best_overall_file = COMPARISON_DIR / "direct_vs_indirect_best_overall.csv"
    best_overall.to_csv(best_overall_file, index=False)

    print(f"All-model comparison saved: {overall_file}")
    print(f"Best-by-method summary saved: {best_by_method_file}")
    print(f"Best-overall summary saved: {best_overall_file}")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
