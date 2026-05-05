import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# MAD
def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))

# set up / value
means = []
uncertainties = []
all_data = []

true_value = 3.25

# overlay
plt.figure()

for i in range(1, 11):
    filename = f'grain_data_BnW{i}.csv'
    df = pd.read_csv(filename)
    
    y = df['avg_diameter_mm'].dropna().values
    x = np.arange(len(y))
    
    plt.plot(x, y, alpha=0.4)

    # Collect all data
    all_data.extend(y)

    # Robust stats
    median = np.median(y)
    mad_val = mad(y)
    robust_std = 1.4826 * mad_val

    means.append(median)
    uncertainties.append(robust_std)

plt.xlabel('Sample Index')
plt.ylabel('Average Diameter (mm)')
plt.title('Raw Data Overlay')
plt.savefig("overlay.png", dpi=300)
plt.close()

# error + stats
means = np.array(means)
uncertainties = np.array(uncertainties)

errors = means - true_value
abs_errors = np.abs(errors)
rmse = np.sqrt(np.mean(errors**2))
covs = np.array([
    u/m if m != 0 else 0 for m, u in zip(means, uncertainties)
])

x = np.arange(1, 11)

# errorbar plot
plt.figure()
plt.errorbar(x, means, yerr=uncertainties, fmt='o', capsize=5)
plt.axhline(y=true_value, linestyle='--')
plt.xlabel('File Index')
plt.ylabel('Diameter (mm)')
plt.title('Mean Diameter with Uncertainty')
plt.savefig("errorbar.png", dpi=300)
plt.close()

# bias plot
plt.figure()
plt.bar(x, errors)
plt.axhline(0, linestyle='--')
plt.xlabel('File Index')
plt.ylabel('Error (mm)')
plt.title('Measurement Bias per File')
plt.savefig("bias.png", dpi=300)
plt.close()

# histogram
plt.figure()
plt.hist(all_data, bins=30)
plt.axvline(true_value, linestyle='--')
plt.xlabel('Diameter (mm)')
plt.ylabel('Frequency')
plt.title('Overall Distribution')
plt.savefig("histogram.png", dpi=300)
plt.close()

# convergence plot
plt.figure()

for i in range(1, 11):
    df = pd.read_csv(f'grain_data_BnW{i}.csv')
    y = df['avg_diameter_mm'].dropna().values
    
    running_mean = np.cumsum(y) / np.arange(1, len(y)+1)
    plt.plot(running_mean, label=f'File {i}')

plt.axhline(true_value, linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Running Mean (mm)')
plt.title('Convergence Behavior')
plt.legend()
plt.savefig("convergence.png", dpi=300)
plt.close()

# processed csv
results_df = pd.DataFrame({
    "file": x,
    "mean_mm": means,
    "uncertainty_mm": uncertainties,
    "error_mm": errors,
    "abs_error_mm": abs_errors,
    "cov": covs
})

results_df.to_csv("processed_results.csv", index=False)

# summary text file for reference
with open("summary.txt", "w") as f:
    f.write("Grain Size Analysis Summary\n")
    f.write("===========================\n\n")

    f.write(f"True Value: {true_value:.4f} mm\n\n")

    f.write("Per-file statistics:\n")
    for i in range(len(means)):
        f.write(
            f"File {i+1}: mean = {means[i]:.4f} mm, "
            f"uncertainty = {uncertainties[i]:.4f} mm\n"
        )

    f.write("\nErrors:\n")
    for i, e in enumerate(errors, 1):
        f.write(f"File {i}: error = {e:.4f} mm\n")

    f.write(f"\nRMSE: {rmse:.4f} mm\n")

    f.write("\nCoefficient of Variation:\n")
    for i, c in enumerate(covs, 1):
        f.write(f"File {i}: {c:.4f}\n")

print("Analysis complete. Files saved.")