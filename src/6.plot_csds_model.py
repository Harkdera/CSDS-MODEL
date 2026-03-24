from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# CSDS model from Simon (1999) - exact formulation used:
#
# tau(u) = tau_r * [1 - exp(-5u / u_r)]
#        + d * [exp(-5u / u_r) - exp(-e * u)]
#
# with:
#   - u in mm
#   - tau in MPa
#   - tau_r : residual shear stress
#   - u_r   : residual displacement
#   - d, e  : fitted model parameters
#
# Input file:
#   data/processed/csds_parameters_converged_only.csv
#
# Output folder:
#   figures/tau_vs_u/
# =====================================================================


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"
OUTPUT_DIR = BASE_DIR / "figures" / "tau_vs_u"


def csds_tau_simon(u, tau_r, u_r, d, e):
    """
    CSDS model:
    tau(u) = tau_r * [1 - exp(-5u / u_r)]
           + d * [exp(-5u / u_r) - exp(-e * u)]
    """
    u = np.asarray(u, dtype=float)
    exp_5u_ur = np.exp(-5.0 * u / u_r)
    return tau_r * (1.0 - exp_5u_ur) + d * (exp_5u_ur - np.exp(-e * u))


def main():
    # -------------------------------------------------------------
    # 1) General parameter
    # -------------------------------------------------------------
    displacement_factor = 1.5
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 2) Read input file
    # -------------------------------------------------------------
    df = pd.read_csv(INPUT_CSV)

    # -------------------------------------------------------------
    # 3) Loop over all samples
    # -------------------------------------------------------------
    for row_index, row in df.iterrows():
        try:
            tau_r = float(row["tau_r_MPa"])
            u_r = float(row["u_r_mm"])
            d = float(row["d_csds"])
            e = float(row["e_csds"])
            u_p = float(row["delta_peak_mm"])
            tau_p = float(row["tau_peak_MPa_csds"])

            if any(np.isnan([tau_r, u_r, d, e, u_p, tau_p])) or u_r <= 0 or u_p <= 0:
                print(f"Skipping row {row_index}: invalid values")
                continue

            # ---------------------------------------------------------
            # 4) Displacement range
            # ---------------------------------------------------------
            u_max = displacement_factor * max(u_p, u_r)
            u_vals = np.linspace(0.0, u_max, 300)

            tau_vals = csds_tau_simon(u_vals, tau_r, u_r, d, e)
            tau_r_model = csds_tau_simon(u_r, tau_r, u_r, d, e)

            # ---------------------------------------------------------
            # 5) Plot
            # ---------------------------------------------------------
            plt.figure(figsize=(8, 6))

            plt.plot(u_vals, tau_vals, label="CSDS model")
            plt.scatter([u_p], [tau_p], marker="o", label="Peak (u_p, tau_p)")
            plt.scatter([u_r], [tau_r], marker="s", label="Residual data")
            plt.scatter([u_r], [tau_r_model], marker="x", label="Residual model")

            # Parameter text box
            param_text = (
                f"tau_r = {tau_r:.3f} MPa\n"
                f"u_r = {u_r:.3f} mm\n"
                f"d = {d:.3f}\n"
                f"e = {e:.3f}\n"
                f"u_p = {u_p:.3f} mm\n"
                f"tau_p = {tau_p:.3f} MPa"
            )

            plt.text(
                0.65, 0.97,
                param_text,
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
            )

            plt.xlabel("Shear displacement u (mm)")
            plt.ylabel("Shear stress tau (MPa)")
            plt.title(f"CSDS curve (row {row_index})")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # ---------------------------------------------------------
            # 6) Save figure
            # ---------------------------------------------------------
            filename = f"tau_vs_u_row_{row_index}.png"
            filepath = OUTPUT_DIR / filename
            plt.savefig(filepath, dpi=300)
            plt.close()

            print(f"Saved figure for row {row_index} -> {filepath}")

        except Exception as exc:
            print(f"Error on row {row_index}: {exc}")
            plt.close()


if __name__ == "__main__":
    main()