from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import re

from src.utils.common_methodologie_1 import B_FROM_D_DIR, E_FROM_D_DIR, GROUP_DIR


# ============================================================
# CONFIG
# ============================================================

INPUT_ROOT = E_FROM_D_DIR
OUTPUT_ROOT = B_FROM_D_DIR
DATASET_FOLDERS = dict(GROUP_DIR)


# ============================================================
# HELPERS
# ============================================================

def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true_num.notna() & y_pred_num.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true_num[mask], y_pred_num[mask])))


def r2_safe(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_num = pd.to_numeric(y_true, errors="coerce")
    y_pred_num = pd.to_numeric(y_pred, errors="coerce")
    mask = y_true_num.notna() & y_pred_num.notna()
    if mask.sum() < 2:
        return np.nan
    return float(r2_score(y_true_num[mask], y_pred_num[mask]))


def find_tau_r_column(df: pd.DataFrame) -> str:
    candidates = [
        "tau_r_MPa",
        "tau_r",
        "residual_shear_strength",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "No tau_r column found. Expected one of: "
        f"{candidates}"
    )


def find_estimated_d_column(df: pd.DataFrame) -> str:
    candidates = [
        "d_pred",
        "estimated_d",
        "d_estimated",
        "predicted_d",
        "d_predicted",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "No estimated d column found. Expected one of: "
        f"{candidates}"
    )


def find_true_d_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "d_csds",
        "true_d",
        "actual_d",
        "d",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_true_b_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "b_csds",
        "true_b",
        "actual_b",
        "b",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_model_metadata(file_name: str) -> dict:
    """Extract model family, selection mode, and rank from a prediction file name."""
    stem = file_name.replace("_e_predictions.csv", "")

    if "_exp_" in stem:
        match = re.match(r"^(?P<dataset>.+?)_exp_(?P<selection_mode>.+?)_rank_(?P<rank>\d+)$", stem)
        if match:
            return {
                "model_family": "exponential",
                "selection_mode": match.group("selection_mode"),
                "rank_in_saved_results": int(match.group("rank")),
            }

    if "_poly_rank_" in stem:
        match = re.match(r"^(?P<dataset>.+?)_poly_rank_(?P<rank>\d+)$", stem)
        if match:
            return {
                "model_family": "polynomial",
                "selection_mode": "log",
                "rank_in_saved_results": int(match.group("rank")),
            }

    return {
        "model_family": "",
        "selection_mode": "",
        "rank_in_saved_results": np.nan,
    }


def process_prediction_file(input_file: Path, output_file: Path) -> dict:
    df = pd.read_csv(input_file)

    tau_r_col = find_tau_r_column(df)
    estimated_d_col = find_estimated_d_column(df)
    true_d_col = find_true_d_column(df)
    true_b_col = find_true_b_column(df)
    model_meta = parse_model_metadata(input_file.name)

    # Simon:
    # a + b = d
    # with a = tau_r
    # so b = d - tau_r
    df["estimated_b"] = df[estimated_d_col] - df[tau_r_col]

    if true_b_col is not None:
        df["true_b"] = df[true_b_col]
    elif true_d_col is not None:
        df["true_b"] = df[true_d_col] - df[tau_r_col]
    else:
        df["true_b"] = np.nan

    valid_mask = pd.to_numeric(df["true_b"], errors="coerce").notna() & pd.to_numeric(df["estimated_b"], errors="coerce").notna()

    b_rmse = rmse(df["true_b"], df["estimated_b"])
    b_r2 = r2_safe(df["true_b"], df["estimated_b"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return {
        "file_name": input_file.name,
        "rows": len(df),
        "rows_used_for_metrics": int(valid_mask.sum()),
        "model_family": model_meta["model_family"],
        "selection_mode": model_meta["selection_mode"],
        "rank_in_saved_results": model_meta["rank_in_saved_results"],
        "tau_r_column": tau_r_col,
        "estimated_d_column": estimated_d_col,
        "true_d_column": true_d_col if true_d_col is not None else "",
        "true_b_column": true_b_col if true_b_col is not None else "",
        "rmse_b": b_rmse,
        "r2_b": b_r2,
        "output_file": str(output_file),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 90)
    print("COMPUTE ESTIMATED B FROM RETAINED D MODELS")
    print("=" * 90)

    all_summaries: list[dict] = []

    for dataset_name, folder_name in DATASET_FOLDERS.items():
        input_dir = INPUT_ROOT / folder_name
        output_dir = OUTPUT_ROOT / folder_name

        if not input_dir.exists():
            print(f"\nDataset {dataset_name}: input folder not found -> {input_dir}")
            continue

        print("\n" + "=" * 90)
        print(f"DATASET: {dataset_name}")
        print("=" * 90)

        prediction_files = sorted(input_dir.glob(f"{dataset_name}_*_e_predictions.csv"))

        if not prediction_files:
            print("No prediction files found.")
            continue

        dataset_summary: list[dict] = []

        for input_file in prediction_files:
            print(f"\nProcessing: {input_file.name}")

            output_file = output_dir / input_file.name.replace(
                "_e_predictions.csv",
                "_b_predictions.csv"
            )

            try:
                result = process_prediction_file(input_file, output_file)
                dataset_summary.append(result)
                all_summaries.append(
                    {
                        "dataset": dataset_name,
                        **result,
                    }
                )

                print(f"Saved: {output_file}")
                print(f"RMSE(b) = {result['rmse_b']:.6f}" if pd.notna(result["rmse_b"]) else "RMSE(b) = NaN")
                print(f"R2(b)   = {result['r2_b']:.6f}" if pd.notna(result["r2_b"]) else "R2(b)   = NaN")

            except Exception as exc:
                print(f"Error while processing {input_file.name}: {exc}")

        if dataset_summary:
            summary_df = pd.DataFrame(dataset_summary).sort_values(
                by=["model_family", "selection_mode", "r2_b", "rmse_b"],
                ascending=[True, True, False, True],
                na_position="last",
            )
            dataset_summary_file = output_dir / f"{dataset_name.lower()}_summary_b_from_d_models.csv"
            summary_df.to_csv(dataset_summary_file, index=False)
            print(f"\nDataset summary saved: {dataset_summary_file}")

    if all_summaries:
        global_summary_df = pd.DataFrame(all_summaries).sort_values(
            by=["dataset", "model_family", "selection_mode", "r2_b", "rmse_b"],
            ascending=[True, True, True, False, True],
            na_position="last",
        )
        global_summary_file = OUTPUT_ROOT / "summary_all_retained_models_b_from_d.csv"
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        global_summary_df.to_csv(global_summary_file, index=False)

        print("\n" + "=" * 90)
        print("DONE")
        print("=" * 90)
        print(f"Global summary saved: {global_summary_file}")


if __name__ == "__main__":
    main()
