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


class KF:
    def __init__(self, dt, sigma_ax=1.5, sigma_ay=2.0):
        self.dt = dt
        # state: [px, py, vx, vy, ax, ay]
        self.x = np.zeros((6,1))
        self.P = np.eye(6) * 1.0

        # base Q
        self.Q_base = make_Q(dt, sigma_a=1.0)
        self.sigma_ax = sigma_ax
        self.sigma_ay = sigma_ay

        # measurement matrix
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0]])

        # innovation tracking
        self.innovation_ema = 0.0
        self.alpha = 0.05
        self.target_innovation = 0.5

        # state transition
        dt2 = 0.5*dt**2
        self.F = np.array([
            [1,0,dt,0,dt2,0],
            [0,1,0,dt,0,dt2],
            [0,0,1,0,dt,0],
            [0,0,0,1,0,dt],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])

    # Predict
    def predict(self):
        # adaptive Q based on current velocity magnitude
        vx, vy = self.x[2,0], self.x[3,0]
        v = np.sqrt(vx**2 + vy**2)
        Q = self.Q_base.copy()
        Q[4,4] = self.sigma_ax**2
        Q[5,5] = self.sigma_ay**2
        Q *= np.clip(v/1.0, 1.0, 3.0)

        # predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    # Update
    def update(self, z, R):
        z = z.reshape(2,1)
        y = z - self.H @ self.x  # residual

        # innovation EMA
        innovation_mag = np.linalg.norm(y)
        self.innovation_ema = (1 - self.alpha)*self.innovation_ema + self.alpha*innovation_mag

        # adaptive R
        R_scale = np.clip(self.innovation_ema/self.target_innovation, 0.5, 5.0)
        R = R * R_scale

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P