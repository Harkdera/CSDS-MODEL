import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# 1) Load dataset
# ================================
FILE = "CSDS_parameters_CONVERGED_ONLY.csv"
df = pd.read_csv(FILE)

# Column check
if "tau_peak_MPa_csds" not in df.columns:
    raise ValueError("Column tau_peak_MPa_csds not found in file.")

tau = df["tau_peak_MPa_csds"].dropna()

# ================================
# 2) Choose split threshold (from observed bimodal distribution)
# ================================
SPLIT = 5  # MPa  <-- threshold we identified earlier

# Group definitions
df_low  = df[df["tau_peak_MPa_csds"] < SPLIT]
df_high = df[df["tau_peak_MPa_csds"] >= SPLIT]

# Group sizes
count_low = len(df_low)
count_high = len(df_high)

print("Group LOW (< 7 MPa):", count_low)
print("Group HIGH (>= 7 MPa):", count_high)

# ================================
# 3) Save the two split datasets
# ================================
df_low.to_csv("CSDS_tauPeak_LOW.csv", index=False)
df_high.to_csv("CSDS_tauPeak_HIGH.csv", index=False)

# ================================
# 4) Plot histogram with threshold + group counts
# ================================
plt.figure(figsize=(9, 5))
sns.histplot(tau, kde=True, bins=30, color="skyblue")

# Threshold line
plt.axvline(SPLIT, color="red", linestyle="--", linewidth=2,
            label=f"Split threshold = {SPLIT} MPa")

# Annotate counts
plt.text(0.02, 0.95,
         f"LOW group (<{SPLIT}): {count_low}",
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor="white", alpha=0.8))

plt.text(0.02, 0.85,
         f"HIGH group (≥{SPLIT}): {count_high}",
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor="white", alpha=0.8))

# Axis labels
plt.xlabel("tau_peak_MPa_csds (MPa)")
plt.ylabel("Frequency")
plt.title("Histogram of tau_peak_MPa_csds with Split Threshold")

plt.legend()
plt.tight_layout()
plt.savefig("tau_peak_split_histogram.png", dpi=300)
plt.close()

print("\nDone!")
print("Saved files:")
print("- CSDS_tauPeak_LOW.csv")
print("- CSDS_tauPeak_HIGH.csv")
print("- tau_peak_split_histogram.png")
