import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kf import KF

# Settings
dt = 0.1
kf = KF(dt)

# base R (UWB variance)
R_base = np.diag([0.05, 0.05])

# Load CSV
data = pd.read_csv("fake_uwb_data.csv")  # must have columns: x, y

# storage
meas_x, meas_y = [], []
est_x, est_y = [], []
sigma_x, sigma_y = [], []
time_steps = []

# Loop through CSV
for i in range(len(data)):
    t = i*dt

    px = data.iloc[i]["x"]
    py = data.iloc[i]["y"]
    z = np.array([px, py])

    # predict and update
    kf.predict()
    kf.update(z, R_base)

    # log
    meas_x.append(px)
    meas_y.append(py)
    est_x.append(kf.x[0,0])
    est_y.append(kf.x[1,0])
    sigma_x.append(np.sqrt(kf.P[0,0]))
    sigma_y.append(np.sqrt(kf.P[1,1]))
    time_steps.append(t)

# Plot trajectory
plt.figure(figsize=(8,8))
plt.plot(meas_x, meas_y, label="UWB Measurement", alpha=0.5)
plt.plot(est_x, est_y, label="KF Estimate", linewidth=2)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("UWB vs Kalman Filter Estimate")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plot uncertainty
plt.figure(figsize=(8,5))
plt.plot(time_steps, sigma_x, label="σ_x (X Uncertainty)")
plt.plot(time_steps, sigma_y, label="σ_y (Y Uncertainty)")
plt.xlabel("Time [s]")
plt.ylabel("Standard Deviation [m]")
plt.title("Kalman Filter Position Uncertainty Over Time")
plt.legend()
plt.grid(True)
plt.show()