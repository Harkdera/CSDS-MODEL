from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

try:
    from methodologie_1.common import B_FROM_D_DIR, COMPARE_DIR, CONVERGED_FILE, METHOD_1_MODEL_FIGURES_DIR, E_FROM_D_DIR, GROUP_DIR, N_CURVE_POINTS, find_sample_id_column, csds_tau, make_u_grid
except ModuleNotFoundError:
    from common import B_FROM_D_DIR, COMPARE_DIR, CONVERGED_FILE, METHOD_1_MODEL_FIGURES_DIR, E_FROM_D_DIR, GROUP_DIR, N_CURVE_POINTS, find_sample_id_column, csds_tau, make_u_grid


# ============================================================
# CONFIG
# ============================================================

E_INPUT_ROOT = E_FROM_D_DIR
B_INPUT_ROOT = B_FROM_D_DIR
OUTPUT_ROOT = COMPARE_DIR
DATASET_FOLDERS = dict(GROUP_DIR)
MASTER_DATA_CANDIDATES = [CONVERGED_FILE]


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


def nan_stat(series: pd.Series, op: str) -> float:
    """Compute a NaN-safe summary statistic without runtime warnings."""
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


def find_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find column for {label}. Tried: {candidates}")


def sanitize_filename(text: str) -> str:
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def get_required_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {}

    cols["sample_id"] = find_sample_id_column(df)

    cols["u_peak"] = find_first_existing_column(
        df,
        ["delta_peak_mm", "u_peak_mm", "u_peak", "delta_peak"],
        "u_peak",
    )

    cols["u_r"] = find_first_existing_column(
        df,
        ["u_r_mm", "u_r", "ur_mm", "ur"],
        "u_r",
    )

    cols["tau_peak"] = find_first_existing_column(
        df,
        ["tau_peak_MPa_csds", "tau_peak_MPa", "tau_peak", "peak_shear_strength"],
        "tau_peak",
    )

    cols["tau_r"] = find_first_existing_column(
        df,
        ["tau_r_MPa", "tau_r", "residual_shear_strength"],
        "tau_r",
    )

    cols["d_true"] = find_first_existing_column(
        df,
        ["d_csds", "d"],
        "true d",
    )

    cols["e_true"] = find_first_existing_column(
        df,
        ["e_csds", "e"],
        "true e",
    )

    if "c_csds" in df.columns:
        cols["c_true"] = "c_csds"
    elif "c" in df.columns:
        cols["c_true"] = "c"
    else:
        cols["c_true"] = ""

    if "b_csds" in df.columns:
        cols["b_true"] = "b_csds"
    elif "b" in df.columns:
        cols["b_true"] = "b"
    else:
        cols["b_true"] = ""

    return cols


def compute_c_from_row(row: pd.Series, u_r_col: str, c_true_col: str) -> float:
    if c_true_col and pd.notna(row[c_true_col]):
        return float(row[c_true_col])

    u_r = row[u_r_col]
    if pd.isna(u_r) or float(u_r) == 0.0:
        return np.nan

    return 5.0 / float(u_r)


def keep_only_converged_rows(df: pd.DataFrame, dataset_name: str, model_name: str) -> pd.DataFrame:
    required_flags = ["e_converged", "condition_c_lt_e"]
    missing_flags = [col for col in required_flags if col not in df.columns]
    if missing_flags:
        raise KeyError(
            f"[{dataset_name} - {model_name}] missing convergence columns: {missing_flags}"
        )

    filtered = df[
        df["e_converged"].fillna(False).astype(bool) &
        df["condition_c_lt_e"].fillna(False).astype(bool)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"[{dataset_name} - {model_name}] no converged rows with c < e were found."
        )

    return filtered


def ensure_unique_sample_ids(df: pd.DataFrame, sample_id_col: str, dataset_name: str, model_name: str, label: str) -> None:
    duplicated = df[df[sample_id_col].duplicated(keep=False)][sample_id_col].tolist()
    if duplicated:
        raise ValueError(
            f"[{dataset_name} - {model_name}] duplicate sample IDs in {label}: "
            f"{duplicated[:10]}"
        )


def merge_predictions_with_master(
    master_df: pd.DataFrame,
    e_df: pd.DataFrame,
    b_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
) -> pd.DataFrame:
    master_id_col = find_sample_id_column(master_df)
    e_id_col = find_sample_id_column(e_df)
    b_id_col = find_sample_id_column(b_df)
    ensure_unique_sample_ids(master_df, master_id_col, dataset_name, model_name, "master file")
    ensure_unique_sample_ids(e_df, e_id_col, dataset_name, model_name, "e file")
    ensure_unique_sample_ids(b_df, b_id_col, dataset_name, model_name, "b file")

    e_needed = [e_id_col]
    b_needed = [b_id_col]

    e_estimated_d_col = find_first_existing_column(
        e_df,
        ["d_pred", "estimated_d", "d_estimated", "predicted_d", "d_predicted"],
        "estimated d in e file",
    )
    e_estimated_e_col = find_first_existing_column(
        e_df,
        ["e_pred", "estimated_e", "e_estimated", "predicted_e", "e_predicted"],
        "estimated e in e file",
    )

    b_estimated_b_col = find_first_existing_column(
        b_df,
        ["estimated_b", "b_estimated", "predicted_b", "b_predicted"],
        "estimated b in b file",
    )

    # Reuse the true fitted parameters already present in the e-file so we are
    # comparing directly against the CSDS fitted model values.
    e_true_cols = get_required_columns(e_df)
    e_needed.extend([
        e_estimated_d_col,
        e_estimated_e_col,
        e_true_cols["u_peak"],
        e_true_cols["u_r"],
        e_true_cols["tau_peak"],
        e_true_cols["tau_r"],
        e_true_cols["d_true"],
        e_true_cols["e_true"],
    ])

    if e_true_cols["b_true"]:
        e_needed.append(e_true_cols["b_true"])
    if e_true_cols["c_true"]:
        e_needed.append(e_true_cols["c_true"])

    b_needed.extend([b_estimated_b_col])

    e_small = e_df[e_needed].copy()
    b_small = b_df[b_needed].copy()

    e_small = e_small.rename(
        columns={
            e_id_col: "sample_id",
            e_estimated_d_col: "estimated_d",
            e_estimated_e_col: "estimated_e",
            e_true_cols["u_peak"]: "u_peak",
            e_true_cols["u_r"]: "u_r",
            e_true_cols["tau_peak"]: "tau_peak",
            e_true_cols["tau_r"]: "tau_r",
            e_true_cols["d_true"]: "true_d",
            e_true_cols["e_true"]: "true_e",
        }
    )

    if e_true_cols["b_true"]:
        e_small = e_small.rename(columns={e_true_cols["b_true"]: "true_b"})
    if e_true_cols["c_true"]:
        e_small = e_small.rename(columns={e_true_cols["c_true"]: "true_c"})

    b_small = b_small.rename(
        columns={
            b_id_col: "sample_id",
            b_estimated_b_col: "estimated_b",
        }
    )

    if "tau_r" in e_small.columns and "estimated_b" in b_small.columns and "estimated_d" in e_small.columns:
        e_check = e_small[["sample_id", "tau_r", "estimated_d"]].copy()
        b_check = b_small[["sample_id", "estimated_b"]].copy()
        check_df = e_check.merge(b_check, on="sample_id", how="inner", validate="one_to_one")
        reconstructed_b = pd.to_numeric(check_df["estimated_d"], errors="coerce") - pd.to_numeric(check_df["tau_r"], errors="coerce")
        mismatch_mask = ~np.isclose(
            reconstructed_b.to_numpy(dtype=float),
            pd.to_numeric(check_df["estimated_b"], errors="coerce").to_numpy(dtype=float),
            rtol=1e-8,
            atol=1e-8,
            equal_nan=False,
        )
        if mismatch_mask.any():
            bad_ids = check_df.loc[mismatch_mask, "sample_id"].tolist()[:10]
            raise ValueError(
                f"[{dataset_name} - {model_name}] b file does not match e file for sample IDs: {bad_ids}"
            )

    merged = e_small.merge(b_small, on="sample_id", how="inner", validate="one_to_one")

    # The master file is only used to validate that the sample IDs exist in the
    # reference CSDS dataset.
    master_small = master_df[[master_id_col]].copy().rename(columns={master_id_col: "sample_id"})
    merged = merged.merge(master_small, on="sample_id", how="inner", validate="one_to_one")

    if len(merged) != len(e_small) or len(merged) != len(b_small):
        raise ValueError(
            f"[{dataset_name} - {model_name}] sample_id mismatch detected. "
            f"master/e/b row counts do not align after merge. "
            f"merged={len(merged)}, e={len(e_small)}, b={len(b_small)}"
        )

    return merged


def compute_row_comparison_table(merged: pd.DataFrame) -> pd.DataFrame:
    work = merged.copy()
    cols = {
        "u_peak": "u_peak",
        "u_r": "u_r",
        "tau_peak": "tau_peak",
        "tau_r": "tau_r",
    }

    work["true_a"] = work["tau_r"]

    if "true_b" not in work.columns:
        work["true_b"] = work["true_d"] - work["tau_r"]

    if "true_c" not in work.columns:
        work["true_c"] = work.apply(
            lambda row: compute_c_from_row(row, "u_r", ""),
            axis=1,
        )

    work["estimated_a"] = work["tau_r"]
    work["estimated_c"] = work["true_c"]

    work["error_d"] = work["estimated_d"] - work["true_d"]
    work["abs_error_d"] = work["error_d"].abs()

    work["error_b"] = work["estimated_b"] - work["true_b"]
    work["abs_error_b"] = work["error_b"].abs()

    work["error_e"] = work["estimated_e"] - work["true_e"]
    work["abs_error_e"] = work["error_e"].abs()

    curve_rmse_list = []
    curve_mae_list = []
    curve_r2_list = []
    valid_curve_list = []

    for _, row in work.iterrows():
        try:
            u = make_u_grid(row, cols)

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

            curve_rmse = float(np.sqrt(np.mean((tau_true - tau_est) ** 2)))
            curve_mae = float(np.mean(np.abs(tau_true - tau_est)))
            ss_res = float(np.sum((tau_true - tau_est) ** 2))
            ss_tot = float(np.sum((tau_true - np.mean(tau_true)) ** 2))
            curve_r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)

            curve_rmse_list.append(curve_rmse)
            curve_mae_list.append(curve_mae)
            curve_r2_list.append(curve_r2)
            valid_curve_list.append(True)

        except Exception:
            curve_rmse_list.append(np.nan)
            curve_mae_list.append(np.nan)
            curve_r2_list.append(np.nan)
            valid_curve_list.append(False)

    work["curve_rmse_tau_u"] = curve_rmse_list
    work["curve_mae_tau_u"] = curve_mae_list
    work["curve_r2_tau_u"] = curve_r2_list
    work["curve_valid"] = valid_curve_list

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
    ]

    return work[keep_cols].copy()


def plot_row_curve(
    row: pd.Series,
    dataset_name: str,
    model_name: str,
    output_file: Path,
) -> None:
    cols = {
        "u_peak": "u_peak",
        "u_r": "u_r",
        "tau_peak": "tau_peak",
        "tau_r": "tau_r",
    }
    u = make_u_grid(row, cols)

    true_a = float(row["true_a"])
    true_b = float(row["true_b"])
    true_c = float(row["true_c"])
    true_d = float(row["true_d"])
    true_e = float(row["true_e"])

    estimated_a = float(row["estimated_a"])
    estimated_b = float(row["estimated_b"])
    estimated_c = float(row["estimated_c"])
    estimated_d = float(row["estimated_d"])
    estimated_e = float(row["estimated_e"])

    tau_true = csds_tau(u, true_a, true_b, true_c, true_d, true_e)
    tau_est = csds_tau(u, estimated_a, estimated_b, estimated_c, estimated_d, estimated_e)

    u_peak = float(row["u_peak"])
    u_r = float(row["u_r"])
    tau_peak = float(row["tau_peak"])
    tau_r = float(row["tau_r"])

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(u, tau_true, label="True CSDS curve")
    ax.plot(u, tau_est, linestyle="--", label="Estimated CSDS curve")

    ax.scatter([0.0], [0.0], marker="o", label="Origin")
    ax.scatter([u_peak], [tau_peak], marker="x", s=70, label="True peak point")
    ax.scatter([u_r], [tau_r], marker="s", s=60, label="True residual point")
    ax.scatter([u_peak], [csds_tau(np.array([u_peak]), estimated_a, estimated_b, estimated_c, estimated_d, estimated_e)[0]],
               marker="^", s=60, label="Estimated at u_peak")
    ax.scatter([u_r], [csds_tau(np.array([u_r]), estimated_a, estimated_b, estimated_c, estimated_d, estimated_e)[0]],
               marker="D", s=50, label="Estimated at u_r")

    info_text = (
        f"True: b={true_b:.4f}, c={true_c:.4f}, d={true_d:.4f}, e={true_e:.4f}\n"
        f"Est.: b={estimated_b:.4f}, c={estimated_c:.4f}, d={estimated_d:.4f}, e={estimated_e:.4f}\n"
        f"Curve RMSE (100 pts)={row['curve_rmse_tau_u']:.4f} | R2={row['curve_r2_tau_u']:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    ax.set_title(f"{dataset_name} | {model_name} | sample_id={row['sample_id']}")
    ax.set_xlabel("u")
    ax.set_ylabel("tau")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def create_plots_for_model(
    comparison_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    plots_dir: Path,
) -> None:
    for _, row in comparison_df.iterrows():
        sample_id = sanitize_filename(row["sample_id"])
        output_file = plots_dir / f"{sample_id}.png"
        try:
            plot_row_curve(
                row=row,
                dataset_name=dataset_name,
                model_name=model_name,
                output_file=output_file,
            )
        except Exception as exc:
            print(f"Plot failed for {dataset_name} | {model_name} | {row['sample_id']}: {exc}")


def plot_sample_across_models(
    sample_rows: pd.DataFrame,
    dataset_name: str,
    sample_id: str | int,
    output_file: Path,
) -> None:
    if sample_rows.empty:
        return

    first_row = sample_rows.iloc[0]
    cols = {
        "u_peak": "u_peak",
        "u_r": "u_r",
        "tau_peak": "tau_peak",
        "tau_r": "tau_r",
    }
    u = make_u_grid(first_row, cols)

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
    true_params_text = (
        f"True params\n"
        f"b={first_row['true_b']:.4f} | c={first_row['true_c']:.4f}\n"
        f"d={first_row['true_d']:.4f} | e={first_row['true_e']:.4f}"
    )
    info_text = (
        f"sample_id={sample_id}\n"
        f"Best model: {best_row['model_name']}\n"
        f"Best curve RMSE (100 pts)={best_row['curve_rmse_tau_u']:.4f}\n"
        f"Best curve R2 (100 pts)={best_row['curve_r2_tau_u']:.4f}\n"
        f"Models shown={len(ranked_rows)}"
    )
    ax.text(
        0.02,
        0.98,
        info_text + "\n\n" + true_params_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black"),
    )

    param_lines = ["Estimated params by model"]
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

    ax.set_title(f"{dataset_name} | sample_id={sample_id} | all models")
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
    filename_suffix: str = "d",
) -> None:
    if not dataset_plot_rows:
        return

    combined_df = pd.concat(dataset_plot_rows, ignore_index=True)
    for sample_id, sample_rows in combined_df.groupby("sample_id", sort=True):
        output_file = output_dir / f"{sanitize_filename(sample_id)}_{filename_suffix}.png"
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
    """Compute pooled RMSE/R² on tau(u) over all valid sample curves."""
    tau_true_all = []
    tau_est_all = []

    cols = {
        "u_peak": "u_peak",
        "u_r": "u_r",
        "tau_peak": "tau_peak",
        "tau_r": "tau_r",
    }

    for _, row in comparison_df.iterrows():
        try:
            u = make_u_grid(row, cols)
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

    summary = {
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
    }
    return summary


def extract_model_name_from_e_file(e_file: Path, dataset_name: str) -> str:
    suffix = "_e_predictions.csv"
    name = e_file.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected e file name: {name}")
    return name.replace(suffix, "")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 100)
    print("COMPARE D B E AND PLOT TAU VS U FOR EACH SAMPLE")
    print("=" * 100)

    master_file = find_existing_file(MASTER_DATA_CANDIDATES)
    master_df = pd.read_csv(master_file)
    master_cols = get_required_columns(master_df)

    print(f"Master CSDS file: {master_file}")
    print(f"Master rows: {len(master_df)}")
    print(f"Sample ID column: {master_cols['sample_id']}")
    print("Using only converged rows with e_converged=True and c<e.")

    global_summary = []

    for dataset_name, folder_name in DATASET_FOLDERS.items():
        print("\n" + "=" * 100)
        print(f"DATASET: {dataset_name}")
        print("=" * 100)

        e_dir = E_INPUT_ROOT / folder_name
        b_dir = B_INPUT_ROOT / folder_name
        out_dir = OUTPUT_ROOT / folder_name

        if not e_dir.exists():
            print(f"Missing e input folder: {e_dir}")
            continue

        if not b_dir.exists():
            print(f"Missing b input folder: {b_dir}")
            continue

        e_files = sorted(e_dir.glob(f"{dataset_name}_*_e_predictions.csv"))

        if not e_files:
            print("No e prediction files found.")
            continue

        dataset_summary = []
        dataset_plot_rows = []

        for e_file in e_files:
            model_name = extract_model_name_from_e_file(e_file, dataset_name)
            b_file = b_dir / e_file.name.replace("_e_predictions.csv", "_b_predictions.csv")

            if not b_file.exists():
                print(f"Skipping {model_name}: matching b file not found -> {b_file}")
                continue

            print(f"\nProcessing model: {model_name}")

            e_df = pd.read_csv(e_file)
            b_df = pd.read_csv(b_file)

            try:
                e_df = keep_only_converged_rows(e_df, dataset_name, model_name)
                b_sample_id_col = find_sample_id_column(b_df)
                e_sample_id_col = find_sample_id_column(e_df)
                b_df = b_df[
                    b_df[b_sample_id_col].isin(e_df[e_sample_id_col])
                ].copy()

                merged = merge_predictions_with_master(
                    master_df=master_df,
                    e_df=e_df,
                    b_df=b_df,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )

                comparison_df = compute_row_comparison_table(merged)
                comparison_df["dataset"] = dataset_name
                comparison_df["model_name"] = model_name
                dataset_plot_rows.append(comparison_df.copy())

                model_dir = out_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                comparison_file = model_dir / f"{model_name}_comparison_d_b_e_tau_u.csv"
                comparison_df.to_csv(comparison_file, index=False)

                plots_dir = METHOD_1_MODEL_FIGURES_DIR / folder_name / "per_model_curves" / model_name
                create_plots_for_model(
                    comparison_df=comparison_df,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    plots_dir=plots_dir,
                )

                summary_row = summarize_model(
                    comparison_df=comparison_df,
                    dataset_name=dataset_name,
                    model_name=model_name,
                )
                dataset_summary.append(summary_row)
                global_summary.append(summary_row)

                print(f"Comparison saved: {comparison_file}")
                print(f"Plots saved in: {plots_dir}")
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

            dataset_summary_file = out_dir / f"{dataset_name.lower()}_summary_compare_d_b_e_tau_u.csv"
            dataset_summary_df.to_csv(dataset_summary_file, index=False)
            print(f"\nDataset summary saved: {dataset_summary_file}")

            combined_plots_dir = METHOD_1_MODEL_FIGURES_DIR / folder_name / "all_models_by_sample"
            create_combined_plots_for_dataset(
                dataset_plot_rows=dataset_plot_rows,
                dataset_name=dataset_name,
                output_dir=combined_plots_dir,
                filename_suffix="d",
            )
            print(f"Combined sample plots saved in: {combined_plots_dir}")

    if global_summary:
        global_summary_df = pd.DataFrame(global_summary).sort_values(
            by=["dataset", "mean_curve_rmse_tau_u", "r2_e", "r2_d", "r2_b"],
            ascending=[True, True, False, False, False],
        )

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        global_summary_file = OUTPUT_ROOT / "summary_all_models_compare_d_b_e_tau_u.csv"
        global_summary_df.to_csv(global_summary_file, index=False)

        print("\n" + "=" * 100)
        print("DONE")
        print("=" * 100)
        print(f"Global summary saved: {global_summary_file}")


if __name__ == "__main__":
    main()
