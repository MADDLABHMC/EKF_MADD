import numpy as np

def make_Q(dt, sigma_a=1.0):
    dt2 = dt**2
    dt3 = dt**3
    dt4 = dt**4
    dt5 = dt**5

    q = np.array([
        [dt5/20, 0,      dt4/8, 0,      dt3/6, 0],
        [0,      dt5/20, 0,      dt4/8, 0,      dt3/6],
        [dt4/8,  0,      dt3/3, 0,      dt2/2, 0],
        [0,      dt4/8,  0,      dt3/3, 0,      dt2/2],
        [dt3/6,  0,      dt2/2, 0,      dt,    0],
        [0,      dt3/6,  0,      dt2/2, 0,      dt]
    ])
    return q * sigma_a**2

class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1.0 
        self.sensors = []
        dt2 = 0.5 * dt**2

        # Process noise (working on nonlinear here)
        self.Q = make_Q(dt, sigma_a=2.0)  

    # allows EKF to talk to the sensors and anchors
    def addSensor(self, sensor):
        self.sensors.append((sensor))

    # PREDICTION -------------------------------------
    # adding in nonlinear accel plus brekker-wong model
    
    def predict(self, u=None):
        dt = self.dt
        dt2 = 0.5*dt**2

        ax, ay = 0, 0
        if u is not None:
            ax, ay = u

        # recompute F each time
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Predict next state
        self.x = self.F @ self.x + np.array([[0],[0],[0],[0],[ax*dt],[ay*dt]])
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

    def updateWithSensor(self, sensor):
        z = sensor.getMeasurement()
        if z is None:
            return

        R = sensor.getR()
        anchor = sensor.anchor_pos  # get anchor from sensor itself
        z_pred = np.array([[self.h(anchor)]])
        H = self.computeJacobian(anchor)

        y = z - z_pred
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    # STEP -------------------------------------

    def step(self, true_px, true_py, u = None):
        """
        Perform one EKF step:
        - Predict
        - Have each sensor generate a noisy measurement from true position
        - Update EKF with each sensor
        """
        self.predict(u)
        for sensor in self.sensors:
            sensor.stepFakeMeasurement(true_px, true_py)  # generate measurement
            self.updateWithSensor(sensor)                  # EKF update 