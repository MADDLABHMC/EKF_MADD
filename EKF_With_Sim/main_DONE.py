from kf import EKF
from sim import elliptical_path, linear_path, Rover, Soil, UWB_sensor, brekkerWong_accel_model
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
time_per_lap = 625  # seconds per lap
num_laps = 75        # number of repeated laps

# Rover and soil
rover = Rover(mass=200, wheel_radius=0.3, wheel_width=0.2, num_wheels=4)
soil = Soil(kc=1500, kphi=1500, n=1.1, c=300, phi=35, k=0.03)

# Initialize EKF
ekf = EKF(dt)

# Sensors
sensors = [
    UWB_sensor("Anchor_1", 0.04, (0,0)),
    UWB_sensor("Anchor_2", 0.06, (10,0)),
    UWB_sensor("Anchor_3", 0.02, (0,10)),
    UWB_sensor("Anchor_4", 0.05, (10,10))
]
for s in sensors:
    ekf.addSensor(s)

# -----------------------------
# Data storage
true_x_history, true_y_history = [], []
pred_x_history, pred_y_history = [], []
uncertainty_history = []
time_steps = []

# -----------------------------
# Run multiple laps
for lap in range(num_laps):
    print(f"Starting lap {lap+1}")

    # Optionally decay P slightly to retain some memory but like not accurate 
    ekf.P *= 0.90  # keeps it from going back to original uncertainty fully

    for step in range(time_per_lap):
        t = step * dt

        # move true rover
        true_px, true_py = elliptical_path(t, center=[3,3], a=3, b=2, omega=0.1)

        # get EKF acceleration (optional)
        ax, ay = brekkerWong_accel_model(rover, soil, ekf.x)
        u = np.array([ax, ay])

        # step EKF
        ekf.step(true_px, true_py, u)

        # log positions
        true_x_history.append(true_px)
        true_y_history.append(true_py)
        pred_x_history.append(ekf.x[0,0])
        pred_y_history.append(ekf.x[1,0])

        # log uncertainty
        sigma_x = np.sqrt(ekf.P[0,0])
        sigma_y = np.sqrt(ekf.P[1,1])
        uncertainty_history.append((sigma_x, sigma_y))

        time_steps.append(t + lap*time_per_lap*dt)  # accumulate time across laps
        

# -----------------------------
# Plotting Trajectory
plt.figure(figsize=(8,8))
plt.plot(true_x_history, true_y_history, label="True Position", linewidth=2)
plt.plot(pred_x_history, pred_y_history, label="EKF Predicted", linestyle="--")
for sensor in ekf.sensors:
    plt.scatter(sensor.anchor_pos[0], sensor.anchor_pos[1], marker='X', s=100, label=sensor.name)
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Bot Trajectory: True vs EKF Predicted with Anchors & Uncertainty")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# Plotting uncertainty over time
plt.figure(figsize=(8,5))
sigma_x_vals = [s[0] for s in uncertainty_history]
sigma_y_vals = [s[1] for s in uncertainty_history]
plt.plot(time_steps, sigma_x_vals, label="σ_x (X Uncertainty)")
plt.plot(time_steps, sigma_y_vals, label="σ_y (Y Uncertainty)")
plt.xlabel("Time (s)")
plt.ylabel("Standard Deviation (m)")
plt.title("EKF Position Uncertainty Over Time")
plt.legend()
plt.grid(True)
plt.show()