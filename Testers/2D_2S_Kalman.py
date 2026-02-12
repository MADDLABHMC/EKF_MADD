# 1D_1S_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 1/30/2026

# combination kalman filter for 2D constant velocity model
# A car with set velocity and 2 distance sensors approaches the wall in both x, y plane

import numpy as np

# State
x = np.array([10.0, 5.0])        # [x, y] initial position (m)
P = np.array([[1.0, 0.0],
              [0.0, 1.0]])       # initial uncertainty (covariance)

v = np.array([1.0, -0.5])        # velocity (m/s) in 2D
dt = 1.0                      # time step (s)
Q = np.array([[0.1, 0.0],       # process noise covariance
              [0.0, 0.1]])

# Sensor readings
z = np.array([8.3, 4.0, 8.6, 4.2])  # stacked x,y for two sensors

# Measurement noise covariance
R = np.diag([0.25, 0.25, 0.36, 0.36])  # 4x4

# --- Measurement matrix H ---
H = np.array([[1, 0],
              [0, 1],
              [1, 0],
              [0, 1]])  # 4x2

# --- Prediction ---
x_pred = x + v*dt
P_pred = P + Q

# --- Kalman Gain ---
S = H @ P_pred @ H.T + R         # 4x4 innovation covariance
K = P_pred @ H.T @ np.linalg.inv(S)  # 2x4 Kalman gain

# --- Update ---
innovation = z - H @ x_pred      # 4x1
x_upd = x_pred + K @ innovation
P_upd = (np.eye(2) - K @ H) @ P_pred

print("Updated state:", x_upd)
print("Updated covariance:\n", P_upd)
