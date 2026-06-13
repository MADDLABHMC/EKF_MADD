from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt

def plot_covariance_ellipse(px, py, cov, n_std=1.0, **kwargs):
    """
    px, py : center of the ellipse
    cov    : 2x2 covariance matrix
    n_std  : number of standard deviations (1σ, 2σ, etc)
    kwargs : additional plotting kwargs for Ellipse
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
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

class Rover:
    def __init__(self, mass, wheel_radius, wheel_width, num_wheels):
        self.mass = mass
        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width
        self.num_wheels = num_wheels
        
class Soil:
    def __init__(self, kc, kphi, n, c, phi, k):
        self.kc = kc
        self.kphi = kphi
        self.n = n
        self.c = c
        self.phi = phi
        self.k = k
        
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

