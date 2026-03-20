from pathlib import Path
import pandas as pd


def main():
    # -------------------------------------------------------------
    # 1) Input file
    # -------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_csv = BASE_DIR / "data" / "processed" / "csds_parameters_with_model.csv"

    # -------------------------------------------------------------
    # 2) Read full CSV
    # -------------------------------------------------------------
    print(f"Reading file: {input_csv}")
    df = pd.read_csv(input_csv)

    # Check that the convergence column exists
    if "csds_converged" not in df.columns:
        raise KeyError(
            "The column 'csds_converged' is missing from the file. "
            "Make sure the CSDS fitting script added this column."
        )

    # -------------------------------------------------------------
    # 3) Filter: keep only rows that converged
    # -------------------------------------------------------------
    df_converged = df[df["csds_converged"] == True].copy()

    n_total = len(df)
    n_conv = len(df_converged)

    print(f"Total number of rows      : {n_total}")
    print(f"Number of converged rows  : {n_conv}")
    if n_total > 0:
        print(f"Proportion converged      : {n_conv / n_total:.2%}")

    # -------------------------------------------------------------
    # 4) Save to a new CSV
    # -------------------------------------------------------------
    output_csv = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_converged.to_csv(output_csv, index=False)

    print(f"\nSaved file: {output_csv}")
    print("This file contains only samples where csds_converged == True.")


if __name__ == "__main__":
    main()