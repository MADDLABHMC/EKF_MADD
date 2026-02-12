# 1D_1S_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 1/30/2026

# Another iteration of a Kalman Filter for a 1D constant velocity model
# A car with set velocity and 2 distance sensors approaches the wall

import numpy as np

# State
x = 10.0         # initial position (m)
P = 1.0          # initial uncertainty (variance)

v = 1.0          # velocity (toward wall) (m/s)
dt = 1.0
Q = 0.1          # process noise (variance)

# --- Sensor readings ---
z = np.array([8.3, 8.6])  # two sensors (measurements) (becomes input vector when generalized)
R = np.array([[0.25, 0.0],
              [0.0, 0.36]])  # sensor noise covariance

# --- Measurement matrix H ---> becomes jacobian if nonlinear for EKF
# 2 sensors measuring x
H = np.array([[1],
              [1]])  

# --- Prediction ---
x_pred = x - v*dt
P_pred = P + Q

# --- Kalman gain ---
S = H @ P_pred @ H.T + R    # 2x2 innovation covariance
K = P_pred * H.T @ np.linalg.inv(S)  # 1x2 Kalman gain

# --- Update ---
innovation = z - (H @ x_pred)
x_upd = x_pred + K @ innovation
P_upd = (1 - K @ H) * P_pred