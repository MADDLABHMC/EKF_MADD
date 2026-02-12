import numpy as np

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
