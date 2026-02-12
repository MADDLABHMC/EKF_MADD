# 2D_XS_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated 2/12/2026

# Generalizing Kalman Filter for a 2D constant velocity model
# A car with set acceleration and X # of distance sensors moves

import numpy as np
import matplotlib.pyplot as plt


# initializes a sensor with a name, uncertainty and its own faux measurements at each time step
class UWB_sensor:
    def __init__(self, name, uncertainty):
        self.name = name
        self.uncertainty = uncertainty  # variance
        self.value = None
        self.true_distance = 0.0 
        print(f"Sensor {self.name} initialized.")

    def setUncertainty(self, new_uncertainty):
        self.uncertainty = new_uncertainty

    def getUncertainty(self):
        return self.uncertainty
    
    # actually used by kalman filter 
    def getR(self):
        return np.array([[self.uncertainty]])

    # For testing ONLY: simulate a measurement (0 m --> 10 m) with Gaussian noise
    def stepFakeMeasurement(self):
        if self.true_distance > 10:
            self.true_distance = 0.0

        sigma = np.sqrt(self.uncertainty)
        noisy_measurement = self.true_distance + np.random.normal(0, sigma)

        self.value = noisy_measurement
        self.true_distance += 1.0

    def getMeasurement(self):
        if self.value is None:
            return None
        return np.array([[self.value]])

# Actual EKF for 2D constant velocity model with distance measurements to anchors
# note: measurements read in as z = distance to anchor, not (x,y) position
# note: prediction done with x,y values making it nonlinear...ugh 

class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0 
        self.sensors = []
        dt2 = 0.5 * dt**2

        # State transition (for constant acceleration)
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise (COME BACK TO THIS LATER)
        q = 0.1
        self.Q = np.eye(6) * q

    # allows EKF to talk to the sensors and anchors
    def addSensor(self, sensor, anchor_pos):
        self.sensors.append((sensor, np.array(anchor_pos)))

    # PREDICTION -------------------------------------

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    # Nonlinear distance to anchor **Pythagorean thm
    def h(self, anchor):
        px, py = self.x[0, 0], self.x[1, 0]
        ax, ay = anchor
        return np.sqrt((px - ax)**2 + (py - ay)**2)

    # Jacobian of h(x)...ew
    def computeJacobian(self, anchor):
        px, py = self.x[0, 0], self.x[1, 0]
        ax, ay = anchor

        dx = px - ax
        dy = py - ay
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 1e-6:
            dist = 1e-6

        H = np.array([[dx/dist, dy/dist, 0, 0, 0, 0]])
        return H

    # UPDATE -------------------------------------
    # note: performed per sensor for now

    def updateWithSensor(self, sensor, anchor):
        z = sensor.getMeasurement()
        if z is None:
            return

        R = sensor.getR()
        z_pred = np.array([[self.h(anchor)]])

        # Linearization
        H = self.computeJacobian(anchor)

        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    # STEP -------------------------------------

    def step(self):
        self.predict()
        for sensor, anchor in self.sensors:
            self.updateWithSensor(sensor, anchor)


if __name__ == "__main__":
    dt = 0.1
    ekf = EKF(dt)

    # Create sensors with variance = 0.04 (std ≈ 0.2 m)
    s1 = UWB_sensor("Anchor_1", uncertainty=0.04)
    s2 = UWB_sensor("Anchor_2", uncertainty=0.04)

    # Anchor positions
    ekf.addSensor(s1, anchor_pos=(0, 0))
    ekf.addSensor(s2, anchor_pos=(10, 0))

    # Run simulation
    for step in range(20):
        s1.stepFakeMeasurement()
        s2.stepFakeMeasurement()

        ekf.step()

        state = ekf.x.flatten()
        print(f"Step {step}: x={state[0]:.2f}, y={state[1]:.2f}, "
              f"vx={state[2]:.2f}, vy={state[3]:.2f}")