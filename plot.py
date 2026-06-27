'''
Plot the cumulative trajectory of the Visual Odometry data.
Can be swapped with the UWB data sets to plot and compare trajectories.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "vo_data_scaled.csv"

dx_vals = []
dy_vals = []

with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        dx_vals.append(float(row["dx"]))
        dy_vals.append(float(row["dy"]))

x_cum = np.cumsum(dx_vals)
y_cum = np.cumsum(dy_vals)

plt.figure(figsize=(6, 6))
plt.plot(x_cum, y_cum, "-o", markersize=2, linewidth=1)
plt.scatter([x_cum[0]], [y_cum[0]], color="green", s=60, label="start", zorder=5)
plt.scatter([x_cum[-1]], [y_cum[-1]], color="red", s=60, label="end", zorder=5)
plt.xlabel("cumulative x")
plt.ylabel("cumulative y")
plt.title("VO Cumulative Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()