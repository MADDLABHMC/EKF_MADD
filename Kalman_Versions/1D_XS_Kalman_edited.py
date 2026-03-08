# 2D_XS_Kalman.py
# Eleanor Champlin-Wilson
# echamplinwilson@g.hmc.edu
# Updated on 2/12/2026

# Generalizing Kalman Filter for a 2D constant velocity model
# A car with set acceleration and X # of distance sensors moves

from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt

# utility functions
def plot_covariance_ellipse(px, py, cov, n_std=1.0, **kwargs):
    """
    px, py : center of the ellipse
    cov    : 2x2 covariance matrix
    n_std  : number of standard deviations (1σ, 2σ, etc)
    kwargs : additional plotting kwargs for Ellipse
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = n_std * np.sqrt(eigvals)
    ellipse = Ellipse((px, py), width, height, angle=angle, **kwargs)    
    plt.gca().add_patch(ellipse)

def linear_path(t, start, velocity):
    '''
    Docstring for linear_path
    
    :param t: time
    :param start: starting location
    :param velocity: velocity at a time
    '''
    x = start[0] + velocity[0]*t
    y = start[1] + velocity[1]*t
    return x, y

def circular_path(t, center, radius, omega):
    '''
    Docstring for circular_path
    
    :param t: time
    :param center: center coords
    :param radius: in m
    :param omega: rad/sec
    '''
    x = center[0] + radius * np.cos(omega * t)
    y = center[1] + radius * np.sin(omega * t)
    return x, y

def elliptical_path(t, center, a, b, omega):
    '''
    Docstring for elliptical_path
    
    :param t: time
    :param center: center coords
    :param a: small
    :param b: big
    :param omega: rad/sec
    '''
    # a = x semi-axis, b = y semi-axis
    x = center[0] + a * np.cos(omega * t)
    y = center[1] + b * np.sin(omega * t)
    return x, y

# parameters = [posn_sigma, vel_sigma, env_sigma, meas_sigma]

# initializes a sensor with a name, uncertainty and its own faux measurements at each time step
class UWB_sensor:
    def __init__(self, name, uncertainty, anchor_pos, parameters):
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

        # State transition (for constant acceleration) propagated non-linearly.... what are some commonly used non-linear tire models/robot models for martian terrain..# fiala tire model.. #
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Process noise (COME BACK TO THIS LATER) make this a generic param.. 
        self.Q = np.diag([0.01, 0.01, 0.05, 0.05, 0.01, 0.01])

    # allows EKF to talk to the sensors and anchors
    def addSensor(self, sensor):
        self.sensors.append((sensor))

    # PREDICTION -------------------------------------

    def predict(self):
        self.x = self.F @ self.x   #true for a kalman .. propagate directly non-linearly..
        self.P = self.F @ self.P @ self.F.T + self.Q

    # Nonlinear distance to anchor **Pythagorean thm
    def h(self, anchor):
        px, py = self.x[0, 0], self.x[1, 0]
        ax, ay = anchor
        return np.sqrt((px - ax)**2 + (py - ay)**2)

    # Jacobian of h(x)...ew
    # call jacobian library or numerical differentiation.. 
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

        # monitor the innovation residual over time... 
        y = z - z_pred

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    # STEP -------------------------------------

    def step(self, true_px, true_py):
        """
        Perform one EKF step:
        - Predict
        - Have each sensor generate a noisy measurement from true position
        - Update EKF with each sensor
        """
        self.predict()
        for sensor in self.sensors:
            sensor.stepFakeMeasurement(true_px, true_py)  # generate measurement
            self.updateWithSensor(sensor)                  # EKF update             


# Actual simulation to see if it is working (1D motion with 2 anchors for now, but can easily be extended to 2D with more anchors)
# note: lots of changable variables to mess with woo
# goal to make it more general later this is to see if it works at all ugh
if __name__ == "__main__":
    
    # STUFF TO MESS WITH
    dt = 0.1
    time = 625 # seconds sim runs
    x_pos = 1.0
    y_pos = 1.1
    x_velocity = 1.0  # slow starting velocity
    y_velocity = 0.5  # no initial y velocity
    x_acceleration = 0.1  # gave it baby accel for now
    y_acceleration = 0.15  # no y accel...for now

    ekf = EKF(dt)
    
    # THIS SECTION COULD / SHOULD BE CONDENSED --------------------------------------------------
    # Create sensors here:
    s1 = UWB_sensor("Anchor_1", uncertainty=0.04, anchor_pos=(0, 0))
    s2 = UWB_sensor("Anchor_2", uncertainty=0.06, anchor_pos=(10, 0))
    s3 = UWB_sensor("Anchor_3", uncertainty=0.02, anchor_pos=(0, 10))
    s4 = UWB_sensor("Anchor_4", uncertainty=0.05, anchor_pos=(10, 10))  # UWB SLAM..look into that... vision, map-based system..

    # add sensors to EKF **IMPORTANT**
    ekf.addSensor(s1)
    ekf.addSensor(s2)
    ekf.addSensor(s3)
    ekf.addSensor(s4)
    #------------------------------------------------------------------------------------
    
    # Ground-truth state for testing: [px, py, vx, vy, ax, ay]
    ts = [x_pos, y_pos,x_velocity, y_velocity, x_acceleration, y_acceleration]
    true_state = np.array(ts)[:, np.newaxis]

    # for plotting later 
    true_x_history = []
    true_y_history = []
    pred_x_history = []
    pred_y_history = []
    uncertainty_history = []
    time_steps = []

    # Run simulation
    for step in range(time):
        # fake real motion without noise ***for testing only***
        
        # version for constant accel or velocity
        #true_state = ekf.F @ true_state
        #true_px, true_py = true_state[0, 0], true_state[1, 0]
        
        # version for ellipse, line, circle etc
        t = step * dt
        true_px, true_py = circular_path(t, [5,5], 3, .1)

        ekf.step(true_px, true_py)
        state = ekf.x.flatten()

        # True position
        true_x, true_y = true_px, true_py
        true_x_history.append(true_x)
        true_y_history.append(true_y)

        # Predicted position
        pred_x, pred_y = ekf.x[0,0], ekf.x[1,0]
        pred_x_history.append(pred_x)
        pred_y_history.append(pred_y)
        
        cov_xy = ekf.P[0:2, 0:2].copy()
        uncertainty_history.append(cov_xy)

    # Plotting
    plt.figure(figsize=(8,8))

    # True trajectory
    plt.plot(true_x_history, true_y_history, label="True Position", linewidth=2)

    # EKF predicted trajectory
    plt.plot(pred_x_history, pred_y_history, label="EKF Predicted", linestyle="--")

    # Anchors
    for sensor in ekf.sensors:
        plt.scatter(sensor.anchor_pos[0], sensor.anchor_pos[1],
                    marker='X', s=100, label=sensor.name)

    # Uncertainty  
    # p covariance  


    posn_x_cov = ekf.P[0,0]
    posn_y_cov = ekf.P[1,1]

    tol = 1e-2

    for i in range(timesteps - 20, timesteps -1):

        cov_res = posn_x_cov[i+1] - posn_x_cov[i]
        if cov_res> tol:
            raise 
        







    #for px, py in zip(pred_x_history, pred_y_history):
    #    cov = ekf.P[0:2, 0:2]  # x-y covariance at current time step
    #    plot_covariance_ellipse(px, py, cov, n_std=1.0, alpha=0.2, color='blue')

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Bot Trajectory: True vs EKF Predicted with Anchors & Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()