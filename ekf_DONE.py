import numpy as np

class EKF:
    def __init__(self, dt, rover, soil):
        self.dt = dt
        self.rover = rover
        self.soil = soil

        # [px, py, vx, vy]
        self.x = np.zeros((4, 1))
        self.P = np.eye(4)

        self.Q = np.eye(4) * 0.01

        self.sensors = []

    def addSensor(self, sensor):
        self.sensors.append(sensor)

    # Bekker-W based dynamics / makin the accel
    def dynamics(self, x, wheel_omega):
        dt = self.dt

        px, py, vx, vy = x.flatten()

        velocity = np.hypot(vx, vy)

        if velocity < 1e-6:
            direction_x, direction_y = 1.0, 0.0
        else:
            direction_x = vx / velocity
            direction_y = vy / velocity

        r = self.rover.wheel_radius

        omega_eff = wheel_omega + 1e-6
        slip = (r * wheel_omega - velocity) / omega_eff
        slip = np.clip(slip, 0.0, 1.0)

        coeff = (self.soil.kc / self.rover.wheel_width) + self.soil.kphi

        j = slip * r

        z = ((self.rover.mass * 9.81 / self.rover.num_wheels) / coeff) ** (1 / self.soil.n)

        sigma = coeff * (z ** self.soil.n)

        tau = (
            self.soil.c
            + sigma * np.tan(np.radians(self.soil.phi))
        ) * (1 - np.exp(-j / (self.soil.k + 1e-6)))

        A = self.rover.wheel_width * 2 * np.sqrt(r * z + 1e-6)

        Ft = tau * A * self.rover.num_wheels

        resistance = 0.1 * self.rover.mass * 9.81

        F_net = Ft - resistance

        a = F_net / self.rover.mass

        ax = a * direction_x
        ay = a * direction_y

        px_new = px + vx * dt + 0.5 * ax * dt**2
        py_new = py + vy * dt + 0.5 * ay * dt**2

        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        return np.array([[px_new], [py_new], [vx_new], [vy_new]])

    # prediction
    def predict(self, wheel_omega):
        self.x = self.dynamics(self.x, wheel_omega)
        F = self.compute_F(wheel_omega)
        self.P = F @ self.P @ F.T + self.Q
        
    
    #### THIS IS NEW AND PULLED FROM VO_MAIN
    def update_vo(self, dx, dy, inlier_ratio=1.0, sv_ratio=1.0):        
        z = np.array([[dx], [dy]])

        z_pred = np.array([[self.x[2, 0]], 
                        [self.x[3, 0]]]) 

        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        y = z - z_pred

        vo_quality = inlier_ratio * sv_ratio  # both between 0 and 1
        R_vo = np.eye(2) * (0.3 / max(vo_quality, 0.1))

        S = H @ self.P @ H.T + R_vo
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    # jacobian from dynamics
    def compute_F(self, wheel_omega, eps=1e-5):
        n = self.x.shape[0]
        F = np.zeros((n, n))

        fx = self.dynamics(self.x, wheel_omega)

        for i in range(n):
            x_perturbed = self.x.copy()
            x_perturbed[i, 0] += eps

            fx_p = self.dynamics(x_perturbed, wheel_omega)

            F[:, i] = ((fx_p - fx) / eps).flatten()

        return F

    # measurement model
    def h(self, anchor):
        px, py = self.x[0, 0], self.x[1, 0]
        ax, ay = anchor
        return np.sqrt((px - ax)**2 + (py - ay)**2)

    def H_jacobian(self, anchor):
        px, py = self.x[0, 0], self.x[1, 0]
        ax, ay = anchor

        dx = px - ax
        dy = py - ay
        dist = np.hypot(dx, dy) + 1e-6

        H = np.array([[dx/dist, dy/dist, 0, 0]])
        return H
    
    def camera_soil_parameters(features):
        D10, D50, D90, Cu = features

        # placeholder mapping (you will replace later)
        kc = 1000 + 2.0 * D50
        kphi = 800 + 1.5 * Cu
        n = 1.0 + 0.1 * (D90 / (D10 + 1e-6))
        c = 200 + 0.5 * D10
        phi = 30 + 0.05 * Cu
        k = 0.02 + 0.001 * D50

        return kc, kphi, n, c, phi, k

    # update
    def update_uwb(self, z):
        px, py = self.x[0,0], self.x[1,0]

        z_pred = np.array([[px],
                        [py]])

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        y = z.reshape(2,1) - z_pred

        R = np.eye(2) * 0.05  # tune this laterrr with data

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    # step ADDED VO HERE
    def step(self, wheel_omega, z_uwb, vo_result=None):
        self.predict(wheel_omega)
        self.update_uwb(z_uwb)
        if vo_result is not None:
            self.update_vo(vo_result["dx"], vo_result["dy"])