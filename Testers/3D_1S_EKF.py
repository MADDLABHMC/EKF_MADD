# 3D_1S_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 02/07/2026

# This code implements an Extended Kalman Filter (EKF) for estimating the position of an object in 3D space using a single sensor that measures the distance to the object. The EKF is used to handle the non-linear measurement model, which is based on the distance from the sensor to the object. The function takes in the current position estimate, the current estimate covariance, the sensor measurement, and the sensor measurement noise covariance, and returns the updated position estimate and covariance after incorporating the new measurement.

import numpy as np

def EKF(pos_current, P_current, z_sensor, R_sensor):
    # EKF for 3D position estimation with a single sensor
    # pos_current: current position estimate (3D vector)
    # P_current: current estimate covariance (3x3 matrix)
    # z_sensor: sensor measurement (distance to the sensor)
    # R_sensor: sensor measurement noise covariance (scalar)
    # th_sensor: sensor standard deviation (scalar)

    # Prediction step (no control input for now)
    
    # Measurement update step
    # Compute the expected measurement based on the current position estimate
    z_expected = np.sqrt(pos_current[0]**2 + pos_current[1]**2 + pos_current[2]**2)

    # Compute the measurement residual
    meas_residual = z_sensor - z_expected

    # Compute the Jacobian of the measurement function
    H = np.array([[pos_current[0] / z_expected, pos_current[1] / z_expected, pos_current[2] / z_expected]])

    # Compute the Covariance and Kalman gain
    S = H @ P_current @ H.T + R_sensor
    K = P_current @ H.T @ np.linalg.inv(S)

    # Update the position estimate
    pos_current = pos_current.reshape(3, 1)
    meas_residual = np.array([[meas_residual]])

    pos_updated = pos_current + K @ meas_residual
    
    # Update the estimate covariance using the Joseph form for numerical stability
    I = np.eye(3)
    P_updated = (I - K @ H) @ P_current @ (I - K @ H).T + K @ R_sensor @ K.T

    return pos_updated, P_updated
        