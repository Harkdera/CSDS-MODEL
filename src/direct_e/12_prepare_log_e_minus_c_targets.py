"""Prépare les datasets pour la cible contrainte `z = log(e-c)`."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    DATASET_DIR,
    GROUP_DIR,
    SPLIT_FILES,
    build_direct_e_dataset,
    dataset_slug,
    ensure_output_dirs,
)


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("PREPARATION DES TARGETS CONTRAINTS POUR e")
    print("=" * 100)

    summary_rows = []

    for dataset_name in SPLIT_FILES:
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        data = build_direct_e_dataset(dataset_name)
        group_dir = DATASET_DIR / GROUP_DIR[dataset_name]
        group_dir.mkdir(parents=True, exist_ok=True)

        output_file = group_dir / f"{dataset_slug(dataset_name)}_direct_e_dataset.csv"
        data.to_csv(output_file, index=False)

        summary_rows.append({
            "Dataset": dataset_name,
            "Rows": len(data),
            "Min_e_minus_c": data["e_minus_c_csds"].min(),
            "Max_e_minus_c": data["e_minus_c_csds"].max(),
            "Mean_log_e_minus_c": data["log_e_minus_c_csds"].mean(),
            "Output_File": str(output_file),
        })

        print(f"Rows kept: {len(data)}")
        print(f"Saved: {output_file}")

    summary_df = pd.DataFrame(summary_rows)
    summary_file = DATASET_DIR / "summary_prepared_direct_e_datasets.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Summary saved: {summary_file}")


if __name__ == "__main__":
    main()
