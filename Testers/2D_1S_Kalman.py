# 2D_1S_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 1/30/2026

# Next iteration of a Kalman Filter for a 2D constant velocity model
# A car with set velocity and 1 distance sensor approaches the wall in both x, y plane

import numpy as np
# Initial state
x = np.array([10.0, 5.0]) # starting distances (m) in 2D
P = np.array([[1.0, 0.0], # starting uncertainty (covariance)
              [0.0, 1.0]])

# Velocity
v = np.array([1.0, -0.5]) # m/s in 2D
dt = 1.0                  # time step (s)

# Process noise
Q = np.array([[0.1, 0.0], # process noise covariance
              [0.0, 0.1]])

# Sensor noise
R = np.array([[0.25, 0.0], # sensor noise covariance
              [0.0, 0.25]])

# Sensor readings
sensor_readings = [np.array([8.3, 4.0]),   # simulated sensor readings in 2D
                   np.array([7.5, 3.5]),
                   np.array([6.8, 3.0]),
                   np.array([5.9, 2.5])]

# Filter Loop
for k, z in enumerate(sensor_readings, start=1):
    # 1. Predict
    x_pred = x + v * dt
    P_pred = P + Q

    # 2. Compute Kalman Gain
    K = P_pred @ np.linalg.inv(P_pred + R)

    # 3. Update
    x = x_pred + K @ (z - x_pred)

    # 4. Update uncertainty
    P = (np.eye(2) - K) @ P_pred

    print(f"Step {k}:")
    print(f"  Predicted: {x_pred}")
    print(f"  Sensor: {z}")
    print(f"  Updated: {x}")
    print(f"  Covariance:\n{P}\n")