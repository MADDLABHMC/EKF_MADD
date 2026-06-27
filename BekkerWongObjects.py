'''
A file to define the classes for the Rover and Soil
Meant to keep track of the properties and behaviors of the Rover and Soil
just so that there are not lots of variable floating around that update frequently
'''

class Rover:
    def __init__(self, mass, wheel_radius, wheel_width, num_wheels):
        '''
        Initialize the Rover with physical properties. 
        mass: The mass of the rover (kg)
        wheel_radius: The radius of the wheels (m)
        wheel_width: The width of the wheels (m)
        num_wheels: The number of wheels
        '''
        self.mass = mass
        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width
        self.num_wheels = num_wheels
        
class Soil:
    def __init__(self, kc, kphi, n, c, phi, k):
        '''
        Initialize the Soil with physical properties.
        kc: The cohesion of the soil (Pa)
        kphi: The internal friction angle of the soil (degrees)
        n: The exponent for the soil model
        c: The adhesion of the soil (Pa)
        phi: The angle of internal friction (degrees)
        k: The coefficient of rolling resistance
        '''
        self.kc = kc
        self.kphi = kphi
        self.n = n
        self.c = c
        self.phi = phi
        self.k = k

