"""Valide l'implémentation CSDS sur le fichier test de calibration.

Le fichier `data/test/csds_calibration_test.csv` contient des valeurs de
référence pour les paramètres `d` et `e`. Les paramètres `a`, `b` et `c` sont
reconstruits avec les relations du modèle CSDS :

    a = tau_r
    c = 5 / u_r
    b = d - a

Le script applique ensuite l'implémentation de `04_fit_csds_model.py`, compare
les paramètres calculés aux valeurs de référence et sauvegarde les écarts.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

INPUT_CSV = BASE_DIR / "data" / "test" / "csds_calibration_test.csv"
OUTPUT_DIR = BASE_DIR / "results" / "csds_implementation_validation"
DETAILED_OUTPUT = OUTPUT_DIR / "csds_calibration_test_parameter_comparison.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "csds_calibration_test_parameter_summary.csv"

PARAMETERS = ["a", "b", "c", "d", "e"]


def rmse(values: pd.Series) -> float:
    """Calcule le RMSE d'une série d'erreurs."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def mae(values: pd.Series) -> float:
    """Calcule le MAE d'une série d'erreurs."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.mean(np.abs(values)))


def build_reference_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les paramètres CSDS de référence disponibles dans le CSV test."""
    reference = pd.DataFrame(index=df.index)
    reference["a_reference"] = pd.to_numeric(df["tau_r_MPa"], errors="coerce")
    reference["c_reference"] = 5.0 / pd.to_numeric(df["u_r_mm"], errors="coerce")
    reference["d_reference"] = pd.to_numeric(df["d"], errors="coerce")
    reference["e_reference"] = pd.to_numeric(df["e"], errors="coerce")
    reference["b_reference"] = reference["d_reference"] - reference["a_reference"]
    return reference


def main() -> None:
    """Exécute la validation et sauvegarde les fichiers de comparaison."""
    fit_module = import_module("src.04_fit_csds_model")

    df = pd.read_csv(INPUT_CSV)
    reference_df = build_reference_parameters(df)

    implemented_df = df.apply(
        fit_module.fit_csds_one_row,
        axis=1,
        result_type="expand",
    )
    implemented_df = implemented_df.rename(
        columns={
            "a_csds": "a_implemented",
            "b_csds": "b_implemented",
            "c_csds": "c_implemented",
            "d_csds": "d_implemented",
            "e_csds": "e_implemented",
        }
    )

    comparison_df = pd.concat([df, reference_df, implemented_df], axis=1)

    summary_rows = []
    for param in PARAMETERS:
        ref_col = f"{param}_reference"
        impl_col = f"{param}_implemented"
        diff_col = f"{param}_difference"
        abs_diff_col = f"{param}_abs_difference"
        rel_diff_col = f"{param}_relative_difference_percent"

        comparison_df[diff_col] = comparison_df[impl_col] - comparison_df[ref_col]
        comparison_df[abs_diff_col] = comparison_df[diff_col].abs()
        comparison_df[rel_diff_col] = np.where(
            comparison_df[ref_col].abs() > 0,
            100.0 * comparison_df[diff_col] / comparison_df[ref_col],
            np.nan,
        )

        summary_rows.append(
            {
                "parameter": param,
                "n": int(comparison_df[diff_col].notna().sum()),
                "mean_reference": comparison_df[ref_col].mean(),
                "mean_implemented": comparison_df[impl_col].mean(),
                "mean_difference": comparison_df[diff_col].mean(),
                "mae": mae(comparison_df[diff_col]),
                "rmse": rmse(comparison_df[diff_col]),
                "max_abs_difference": comparison_df[abs_diff_col].max(),
                "max_abs_relative_difference_percent": comparison_df[rel_diff_col].abs().max(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(DETAILED_OUTPUT, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 90)
    print("CSDS IMPLEMENTATION VALIDATION ON TEST CSV")
    print("=" * 90)
    print(f"Input file: {INPUT_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Detailed comparison saved: {DETAILED_OUTPUT}")
    print(f"Summary saved: {SUMMARY_OUTPUT}")
    print("\nParameter summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
