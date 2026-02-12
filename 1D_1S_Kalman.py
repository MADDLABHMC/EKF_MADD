# 1D_1S_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 1/30/2026

# First iteration of a Kalman Filter for a 1D constant velocity model
# A car with set velocity and 1 distance sensor approaches the wall

import numpy as np

x = 10.0        # starting distance (m)
P = 1.0         # starting uncertainty (variance)

velocity = 1.0  # m/s
dt = 1.0        # time step (s)

Q = 0.1         # process noise (variance)
R = 0.25        # sensor noise uncertainty (variance)

# Fake sensor readings (from simulation for rn)
sensor_readings = [8.3, 7.5, 6.8, 5.9, 5.1]

# Filter Loop
for k, z in enumerate(sensor_readings, start=1):
    # Predict
    x_pred = x - velocity * dt
    P_pred = P + Q

    # Kalman Gain
    K = P_pred / (P_pred + R)

    # Update estimate
    x = x_pred + K * (z - x_pred)

    # Update uncertainty
    P = (1 - K) * P_pred

    # Results
    print(f"Step {k}: Sensor={z:.2f} | Predicted={x_pred:.2f} | Updated={x:.2f} | Uncertainty={P:.2f}")
