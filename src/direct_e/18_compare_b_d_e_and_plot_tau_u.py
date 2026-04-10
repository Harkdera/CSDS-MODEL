"""Compare les parametres reconstruits de la branche directe `e` et trace tau(u) par echantillon."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from direct_e.common import (  # noqa: E402
    COMPARISON_DIR,
    CONVERGED_FILE,
    DIRECT_E_MODEL_FIGURES_DIR,
    EVALUATION_DIR,
    GROUP_DIR,
    csds_tau,
    ensure_output_dirs,
    find_sample_id_column,
)


INPUT_ROOT = EVALUATION_DIR
OUTPUT_ROOT = COMPARISON_DIR / "direct_e_models"
DATASET_FOLDERS = dict(GROUP_DIR)
MASTER_DATA_CANDIDATES = [CONVERGED_FILE]


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


def nan_stat(series: pd.Series, op: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    if op == "mean":
        return float(values.mean())
    if op == "median":
        return float(values.median())
    if op == "max":
        return float(values.max())
    raise ValueError(f"Unsupported op: {op}")


def find_existing_file(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find the master CSDS file. Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def sanitize_filename(text: str) -> str:
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def make_comparison_u_grid(row: pd.Series, n_points: int = 100) -> np.ndarray:
    max_u = max(float(row["u_peak"]) * 2.0, float(row["u_r"]) * 1.25, 1e-6)
    return np.linspace(0.0, max_u, n_points)


def ensure_unique_sample_ids(
    df: pd.DataFrame,
    sample_id_col: str,
    dataset_name: str,
    model_name: str,
    label: str,
) -> None:
    duplicated = df[df[sample_id_col].duplicated(keep=False)][sample_id_col].tolist()
    if duplicated:
        raise ValueError(
            f"[{dataset_name} - {model_name}] duplicate sample IDs in {label}: {duplicated[:10]}"
        )


def keep_only_valid_prediction_rows(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> pd.DataFrame:
    required_cols = [
        "e_constraint_respected",
        "d_positive",
        "b_positive",
        "curve_valid",
        "b_pred",
        "d_pred",
        "e_pred",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"[{dataset_name} - {model_name}] missing columns: {missing}")

    filtered = df[
        df["e_constraint_respected"].fillna(False).astype(bool) &
        df["d_positive"].fillna(False).astype(bool) &
        df["b_positive"].fillna(False).astype(bool) &
        df["curve_valid"].fillna(False).astype(bool)
    ].copy()

    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.dropna(
        subset=[
            "sample_id",
            "delta_peak_mm",
            "u_r_mm",
            "tau_peak_MPa_csds",
            "tau_r_MPa",
            "a_csds",
            "b_csds",
            "c_target",
            "d_csds",
            "e_csds",
            "b_pred",
            "d_pred",
            "e_pred",
        ]
    ).reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            f"[{dataset_name} - {model_name}] no valid direct-e prediction rows were found."
        )

    return filtered


def merge_predictions_with_master(
    master_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> pd.DataFrame:
    master_id_col = find_sample_id_column(master_df)
    pred_id_col = find_sample_id_column(pred_df)

    ensure_unique_sample_ids(master_df, master_id_col, dataset_name, model_name, "master file")
    ensure_unique_sample_ids(pred_df, pred_id_col, dataset_name, model_name, "prediction file")

    pred_small = pred_df.copy().rename(columns={pred_id_col: "sample_id"})
    master_small = master_df[[master_id_col]].copy().rename(columns={master_id_col: "sample_id"})

    merged = pred_small.merge(master_small, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(pred_small):
        raise ValueError(
            f"[{dataset_name} - {model_name}] sample_id mismatch detected. "
            f"merged={len(merged)}, predictions={len(pred_small)}"
        )
    return merged


def compute_row_comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    work = merged.copy()

    work["u_peak"] = work["delta_peak_mm"]
    work["u_r"] = work["u_r_mm"]
    work["tau_peak"] = work["tau_peak_MPa_csds"]
    work["tau_r"] = work["tau_r_MPa"]

    work["true_a"] = work["a_csds"]
    work["true_b"] = work["b_csds"]
    work["true_c"] = work["c_target"]
    work["true_d"] = work["d_csds"]
    work["true_e"] = work["e_csds"]

    work["estimated_a"] = work["a_csds"]
    work["estimated_b"] = work["b_pred"]
    work["estimated_c"] = work["c_target"]
    work["estimated_d"] = work["d_pred"]
    work["estimated_e"] = work["e_pred"]

    work["error_b"] = work["estimated_b"] - work["true_b"]
    work["abs_error_b"] = work["error_b"].abs()
    work["error_d"] = work["estimated_d"] - work["true_d"]
    work["abs_error_d"] = work["error_d"].abs()
    work["error_e"] = work["estimated_e"] - work["true_e"]
    work["abs_error_e"] = work["error_e"].abs()

    keep_cols = [
        "sample_id",
        "u_peak",
        "u_r",
        "tau_peak",
        "tau_r",
        "true_a",
        "true_b",
        "true_c",
        "true_d",
        "true_e",
        "estimated_a",
        "estimated_b",
        "estimated_c",
        "estimated_d",
        "estimated_e",
        "error_b",
        "abs_error_b",
        "error_d",
        "abs_error_d",
        "error_e",
        "abs_error_e",
        "curve_rmse_tau_u",
        "curve_mae_tau_u",
        "curve_r2_tau_u",
        "curve_valid",
        "e_constraint_respected",
        "d_positive",
        "b_positive",
    ]
    return work[keep_cols].copy()


def plot_sample_across_models(
    sample_rows: pd.DataFrame,
    dataset_name: str,
    sample_id: str | int,
    output_file: Path,
) -> None:
    if sample_rows.empty:
        return

    first_row = sample_rows.iloc[0]
    u = make_comparison_u_grid(first_row)

    true_tau = csds_tau(
        u=u,
        a=float(first_row["true_a"]),
        b=float(first_row["true_b"]),
        c=float(first_row["true_c"]),
        d=float(first_row["true_d"]),
        e=float(first_row["true_e"]),
    )

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.plot(u, true_tau, color="black", linewidth=2.5, label="True CSDS curve")

    u_peak = float(first_row["u_peak"])
    u_r = float(first_row["u_r"])
    tau_peak = float(first_row["tau_peak"])
    tau_r = float(first_row["tau_r"])

    ax.scatter([0.0], [0.0], marker="o", color="black", s=35, label="Origin")
    ax.scatter([u_peak], [tau_peak], marker="x", color="black", s=70, label="True peak point")
    ax.scatter([u_r], [tau_r], marker="s", color="black", s=55, label="True residual point")

    ranked_rows = sample_rows.sort_values(
        by=["curve_rmse_tau_u", "model_name"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    cmap = plt.get_cmap("tab20")
    for idx, (_, row) in enumerate(ranked_rows.iterrows()):
        tau_est = csds_tau(
            u=u,
            a=float(row["estimated_a"]),
            b=float(row["estimated_b"]),
            c=float(row["estimated_c"]),
            d=float(row["estimated_d"]),
            e=float(row["estimated_e"]),
        )
        label = (
            f"{row['model_name']} | "
            f"RMSE={row['curve_rmse_tau_u']:.4f} | "
            f"R2={row['curve_r2_tau_u']:.4f}"
        )
        ax.plot(
            u,
            tau_est,
            linestyle="--",
            linewidth=1.4,
            color=cmap(idx % 20),
            alpha=0.95,
            label=label,
        )

    best_row = ranked_rows.iloc[0]
    left_info = (
        f"sample_id={sample_id}\n"
        f"Best model: {best_row['model_name']}\n"
        f"Best curve RMSE (100 pts)={best_row['curve_rmse_tau_u']:.4f}\n"
        f"Best curve R2 (100 pts)={best_row['curve_r2_tau_u']:.4f}\n"
        f"Models shown={len(ranked_rows)}\n\n"
        f"True params\n"
        f"b={first_row['true_b']:.4f} | c={first_row['true_c']:.4f}\n"
        f"d={first_row['true_d']:.4f} | e={first_row['true_e']:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        left_info,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    param_lines = ["Estimated params by direct-e model"]
    for _, row in ranked_rows.iterrows():
        param_lines.append(
            f"{row['model_name']}: "
            f"b={row['estimated_b']:.4f}, "
            f"c={row['estimated_c']:.4f}, "
            f"d={row['estimated_d']:.4f}, "
            f"e={row['estimated_e']:.4f}"
        )

    ax.text(
        1.02,
        0.98,
        "\n".join(param_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"),
    )

    ax.set_title(f"{dataset_name} | sample_id={sample_id} | all direct-e models")
    ax.set_xlabel("u")
    ax.set_ylabel("tau")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.02), fontsize=8, frameon=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_combined_plots_for_dataset(
    dataset_plot_rows: list[pd.DataFrame],
    dataset_name: str,
    output_dir: Path,
) -> None:
    if not dataset_plot_rows:
        return

    combined_df = pd.concat(dataset_plot_rows, ignore_index=True)
    for sample_id, sample_rows in combined_df.groupby("sample_id", sort=True):
        output_file = output_dir / f"{sanitize_filename(sample_id)}.png"
        try:
            plot_sample_across_models(
                sample_rows=sample_rows,
                dataset_name=dataset_name,
                sample_id=sample_id,
                output_file=output_file,
            )
        except Exception as exc:
            print(f"Combined plot failed for {dataset_name} | sample_id={sample_id}: {exc}")


def compute_curve_metrics(comparison_df: pd.DataFrame) -> dict[str, float]:
    tau_true_all = []
    tau_est_all = []

    for _, row in comparison_df.iterrows():
        try:
            u = make_comparison_u_grid(row)
            tau_true = csds_tau(
                u=u,
                a=float(row["true_a"]),
                b=float(row["true_b"]),
                c=float(row["true_c"]),
                d=float(row["true_d"]),
                e=float(row["true_e"]),
            )
            tau_est = csds_tau(
                u=u,
                a=float(row["estimated_a"]),
                b=float(row["estimated_b"]),
                c=float(row["estimated_c"]),
                d=float(row["estimated_d"]),
                e=float(row["estimated_e"]),
            )
            tau_true_all.append(tau_true)
            tau_est_all.append(tau_est)
        except Exception:
            continue

    if not tau_true_all:
        return {"rmse_tau_u": np.nan, "r2_tau_u": np.nan}

    tau_true_vec = np.concatenate(tau_true_all)
    tau_est_vec = np.concatenate(tau_est_all)
    return {
        "rmse_tau_u": rmse(pd.Series(tau_true_vec), pd.Series(tau_est_vec)),
        "r2_tau_u": r2_safe(pd.Series(tau_true_vec), pd.Series(tau_est_vec)),
    }


def summarize_model(comparison_df: pd.DataFrame, dataset_name: str, model_name: str) -> dict:
    curve_metrics = compute_curve_metrics(comparison_df)
    return {
        "dataset": dataset_name,
        "model_name": model_name,
        "n_rows": len(comparison_df),
        "rmse_d": rmse(comparison_df["true_d"], comparison_df["estimated_d"]),
        "r2_d": r2_safe(comparison_df["true_d"], comparison_df["estimated_d"]),
        "rmse_b": rmse(comparison_df["true_b"], comparison_df["estimated_b"]),
        "r2_b": r2_safe(comparison_df["true_b"], comparison_df["estimated_b"]),
        "rmse_e": rmse(comparison_df["true_e"], comparison_df["estimated_e"]),
        "r2_e": r2_safe(comparison_df["true_e"], comparison_df["estimated_e"]),
        "rmse_tau_u": curve_metrics["rmse_tau_u"],
        "mean_curve_rmse_tau_u": nan_stat(comparison_df["curve_rmse_tau_u"], "mean"),
        "median_curve_rmse_tau_u": nan_stat(comparison_df["curve_rmse_tau_u"], "median"),
        "max_curve_rmse_tau_u": nan_stat(comparison_df["curve_rmse_tau_u"], "max"),
        "valid_curve_count": int(comparison_df["curve_valid"].sum()),
        "r2_tau_u": curve_metrics["r2_tau_u"],
        "constraint_ok_count": int(comparison_df["e_constraint_respected"].sum()),
        "d_positive_count": int(comparison_df["d_positive"].sum()),
        "b_positive_count": int(comparison_df["b_positive"].sum()),
    }


def extract_model_name_from_prediction_file(pred_file: Path, dataset_name: str) -> str:
    suffix = "_b_d_from_e_predictions.csv"
    name = pred_file.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected prediction file name: {name}")
    return name.replace(suffix, "")


def main() -> None:
    ensure_output_dirs()

    print("=" * 100)
    print("COMPARE DIRECT-E B D E AND PLOT TAU VS U FOR EACH SAMPLE")
    print("=" * 100)

    master_file = find_existing_file(MASTER_DATA_CANDIDATES)
    master_df = pd.read_csv(master_file)
    print(f"Master CSDS file: {master_file}")
    print(f"Master rows: {len(master_df)}")

    global_summary = []

    for dataset_name, folder_name in DATASET_FOLDERS.items():
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        input_dir = INPUT_ROOT / folder_name
        out_dir = OUTPUT_ROOT / folder_name

        if not input_dir.exists():
            print(f"Missing direct-e input folder: {input_dir}")
            continue

        prediction_files = sorted(input_dir.glob(f"{dataset_name}_*_b_d_from_e_predictions.csv"))
        if not prediction_files:
            print("No direct-e prediction files found.")
            continue

        dataset_summary = []
        dataset_plot_rows = []

        for pred_file in prediction_files:
            model_name = extract_model_name_from_prediction_file(pred_file, dataset_name)
            print(f"\nProcessing model: {model_name}")

            pred_df = pd.read_csv(pred_file)

            try:
                pred_df = keep_only_valid_prediction_rows(pred_df, dataset_name, model_name)
                merged = merge_predictions_with_master(
                    master_df=master_df,
                    pred_df=pred_df,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )

                comparison_df = compute_row_comparison_table(merged)
                comparison_df["dataset"] = dataset_name
                comparison_df["model_name"] = model_name
                dataset_plot_rows.append(comparison_df.copy())

                model_dir = out_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                comparison_file = model_dir / f"{model_name}_comparison_b_d_e_tau_u.csv"
                comparison_df.to_csv(comparison_file, index=False)

                summary_row = summarize_model(comparison_df, dataset_name, model_name)
                dataset_summary.append(summary_row)
                global_summary.append(summary_row)

                print(f"Comparison saved: {comparison_file}")
                print(f"RMSE(d) = {summary_row['rmse_d']:.6f}")
                print(f"RMSE(b) = {summary_row['rmse_b']:.6f}")
                print(f"RMSE(e) = {summary_row['rmse_e']:.6f}")
                print(f"Mean curve RMSE tau(u) = {summary_row['mean_curve_rmse_tau_u']:.6f}")
            except Exception as exc:
                print(f"Failed for {model_name}: {exc}")

        if dataset_summary:
            dataset_summary_df = pd.DataFrame(dataset_summary).sort_values(
                by=["mean_curve_rmse_tau_u", "r2_e", "r2_d", "r2_b"],
                ascending=[True, False, False, False],
            )

            dataset_summary_file = out_dir / f"{dataset_name.lower()}_summary_compare_b_d_e_tau_u.csv"
            dataset_summary_df.to_csv(dataset_summary_file, index=False)
            print(f"\nDataset summary saved: {dataset_summary_file}")

            combined_plots_dir = DIRECT_E_MODEL_FIGURES_DIR / folder_name / "all_models_by_sample"
            create_combined_plots_for_dataset(
                dataset_plot_rows=dataset_plot_rows,
                dataset_name=dataset_name,
                output_dir=combined_plots_dir,
            )
            print(f"Combined sample plots saved in: {combined_plots_dir}")

    if global_summary:
        global_summary_df = pd.DataFrame(global_summary).sort_values(
            by=["dataset", "mean_curve_rmse_tau_u", "r2_e", "r2_d", "r2_b"],
            ascending=[True, True, False, False, False],
        )
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        global_summary_file = OUTPUT_ROOT / "summary_all_models_compare_b_d_e_tau_u.csv"
        global_summary_df.to_csv(global_summary_file, index=False)

        print("\n" + "=" * 100)
        print("DONE")
        print("=" * 100)
        print(f"Global summary saved: {global_summary_file}")


if __name__ == "__main__":
    main()
