"""Extrait uniquement les cas où l'ajustement CSDS a convergé."""

from pathlib import Path
import pandas as pd


def main():
    """Filtre les lignes convergées et enregistre un CSV dédié."""
    # -------------------------------------------------------------
    # 1) Définir le fichier d'entrée
    # -------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_csv = BASE_DIR / "data" / "processed" / "csds_parameters_with_model.csv"

    # -------------------------------------------------------------
    # 2) Lire le fichier complet
    # -------------------------------------------------------------
    print(f"Reading file: {input_csv}")
    df = pd.read_csv(input_csv)

    # Vérifier que la colonne de convergence existe
    if "csds_converged" not in df.columns:
        raise KeyError(
            "The column 'csds_converged' is missing from the file. "
            "Make sure the CSDS fitting script added this column."
        )

    # -------------------------------------------------------------
    # 3) Filtrer : conserver uniquement les lignes convergées
    # -------------------------------------------------------------
    df_converged = df[df["csds_converged"] == True].copy()

    n_total = len(df)
    n_conv = len(df_converged)

    print(f"Total number of rows      : {n_total}")
    print(f"Number of converged rows  : {n_conv}")
    if n_total > 0:
        print(f"Proportion converged      : {n_conv / n_total:.2%}")

    # -------------------------------------------------------------
    # 4) Vérifier que `sigma_n_MPa` est toujours présent
    # -------------------------------------------------------------
    print("\nColumns in converged file:")
    print(df_converged.columns.tolist())

    if "sigma_n_MPa" in df_converged.columns:
        print("\nsigma_n_MPa is present in converged output.")
    else:
        print("\nsigma_n_MPa is missing in converged output.")

    # -------------------------------------------------------------
    # 5) Enregistrer le résultat dans un nouveau CSV
    # -------------------------------------------------------------
    output_csv = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_converged.to_csv(output_csv, index=False)

    print(f"\nSaved file: {output_csv}")
    print("This file contains only samples where csds_converged == True.")


if __name__ == "__main__":
    main()
