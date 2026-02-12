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
    def __init__(self, name, uncertainty, anchor_pos):
        """
        name: string, sensor name
        uncertainty: variance of measurement noise (sigma^2)
        anchor_pos: (x, y) tuple, location of the sensor anchor
        """
        self.name = name
        self.uncertainty = uncertainty  
        self.anchor_pos = np.array(anchor_pos)
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
    def stepFakeMeasurement(self, true_px, true_py):
        dx = true_px - self.anchor_pos[0]
        dy = true_py - self.anchor_pos[1]
        true_distance = np.sqrt(dx**2 + dy**2)

        # Add Gaussian noise
        sigma = np.sqrt(self.uncertainty)
        noisy_measurement = true_distance + np.random.normal(0, sigma)

        self.value = noisy_measurement

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
    def addSensor(self, sensor):
        self.sensors.append((sensor, sensor.anchor_pos))

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



# Actual simulation to see if it is working (1D motion with 2 anchors for now, but can easily be extended to 2D with more anchors)
# note: lots of changable variables to mess with woo
# goal to amke it more general later this is to see if it works at all ugh
if __name__ == "__main__":
    dt = 0.1
    ekf = EKF(dt)

    # Create sensors here:
    s1 = UWB_sensor("Anchor_1", uncertainty=0.04, anchor_pos=(0, 0))
    s2 = UWB_sensor("Anchor_2", uncertainty=0.04, anchor_pos=(10, 0))

    # add sensors to EKF IMPORTANT
    ekf.addSensor(s1)
    ekf.addSensor(s2)

    # Ground-truth state for testing: [px, py, vx, vy, ax, ay]
    true_state = np.zeros((6, 1))
    true_state[2, 0] = 1.0   # slow starting velocity
    true_state[4, 0] = 0.2   # gave it baby accel for now

    # for plotting later 
    true_x_history = []
    pred_x_history = []
    uncertainty_history = []
    time_steps = []

    # Run simulation
    for step in range(50):
        # fake real motion without noise ***for testing only***
        true_state = ekf.F @ true_state
        true_px, true_py = true_state[0, 0], true_state[1, 0]

        # sensors "measure" distance to true position w/noise
        s1.stepFakeMeasurement(true_px, true_py)
        s2.stepFakeMeasurement(true_px, true_py)

        # EKF steps
        ekf.step()
        state = ekf.x.flatten()

        # Logging for plotting
        true_x_history.append(true_px)
        pred_x_history.append(state[0])
        uncertainty_history.append(np.sqrt(ekf.P[0, 0]))
        time_steps.append(step)

    # Plot
    plt.figure(figsize=(10, 5))

    plt.plot(time_steps, true_x_history, label="True Position (x)", linewidth=2)
    plt.plot(time_steps, pred_x_history, label="Kalman Predicted (x)", linestyle="--")

    pred = np.array(pred_x_history)
    unc = np.array(uncertainty_history)

    plt.fill_between(
        time_steps,
        pred - unc,
        pred + unc,
        alpha=0.2,
        label="Uncertainty (±1σ)"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Position (x)")
    plt.title("EKF: True vs Predicted Position with Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.show()