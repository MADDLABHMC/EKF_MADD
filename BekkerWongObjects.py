# A file to define the classes for the Rover and Soil
# Meant to keep track of the properties and behaviors of the Rover and Soil
# just so that there are not lots of variable floating around that update frequently

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

