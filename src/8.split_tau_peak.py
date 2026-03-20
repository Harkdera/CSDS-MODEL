from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ================================
# 1) Load dataset
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "csds_parameters_converged_only.csv"
OUTPUT_LOW = BASE_DIR / "data" / "interim" / "csds_tau_peak_low.csv"
OUTPUT_HIGH = BASE_DIR / "data" / "interim" / "csds_tau_peak_high.csv"
OUTPUT_FIG = BASE_DIR / "figures" / "tau_peak_split_histogram.png"

df = pd.read_csv(INPUT_FILE)

# Column check
if "tau_peak_MPa_csds" not in df.columns:
    raise ValueError("Column tau_peak_MPa_csds not found in file.")

tau = df["tau_peak_MPa_csds"].dropna()

# ================================
# 2) Choose split threshold
# ================================
SPLIT = 5  # MPa

# Group definitions
df_low = df[df["tau_peak_MPa_csds"] < SPLIT]
df_high = df[df["tau_peak_MPa_csds"] >= SPLIT]

# Group sizes
count_low = len(df_low)
count_high = len(df_high)

print(f"Group LOW (< {SPLIT} MPa): {count_low}")
print(f"Group HIGH (>= {SPLIT} MPa): {count_high}")

# ================================
# 3) Save the two split datasets
# ================================
OUTPUT_LOW.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_HIGH.parent.mkdir(parents=True, exist_ok=True)
df_low.to_csv(OUTPUT_LOW, index=False)
df_high.to_csv(OUTPUT_HIGH, index=False)

# ================================
# 4) Plot histogram with threshold + group counts
# ================================
OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))
sns.histplot(tau, kde=True, bins=30, color="skyblue")

# Threshold line
plt.axvline(
    SPLIT,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Split threshold = {SPLIT} MPa"
)

# Annotate counts
plt.text(
    0.02, 0.95,
    f"LOW group (< {SPLIT}): {count_low}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

plt.text(
    0.02, 0.85,
    f"HIGH group (>= {SPLIT}): {count_high}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor="white", alpha=0.8)
)

# Axis labels
plt.xlabel("tau_peak_MPa_csds (MPa)")
plt.ylabel("Frequency")
plt.title("Histogram of tau_peak_MPa_csds with Split Threshold")

plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print("\nDone!")
print("Saved files:")
print(f"- {OUTPUT_LOW}")
print(f"- {OUTPUT_HIGH}")
print(f"- {OUTPUT_FIG}")