from ekf_DONE import EKF
from sim_DONE import Rover, Soil, elliptical_path
from camera import GrainCamera
import numpy as np

dt = 0.1
wheel_omega = 8.0

rover = Rover(200, 0.3, 0.2, 4)
soil = Soil(1500, 1500, 1.1, 300, 35, 0.03)

ekf = EKF(dt, rover, soil)

camera = GrainCamera()

def soil_from_camera(features, prev_soil):
    if features is None:
        return prev_soil

    D10, D50, D90, Cu = features

    kc = 1000 + 2*D50
    kphi = 800 + 1.5*Cu
    n = 1.0 + 0.05*(D90/(D10+1e-6))
    c = 200 + 0.2*D10
    phi = 30 + 0.05*Cu
    k = 0.02 + 0.001*D50

    # optional smoothing (VERY important later)
    alpha = 0.2

    prev_soil.kc = (1-alpha)*prev_soil.kc + alpha*kc
    prev_soil.kphi = (1-alpha)*prev_soil.kphi + alpha*kphi
    prev_soil.n = (1-alpha)*prev_soil.n + alpha*n
    prev_soil.c = (1-alpha)*prev_soil.c + alpha*c
    prev_soil.phi = (1-alpha)*prev_soil.phi + alpha*phi
    prev_soil.k = (1-alpha)*prev_soil.k + alpha*k

    return prev_soil


for step in range(10000):

    # camera
    features = camera.step()

    # update soil model (single source of truth)
    if step % 5 == 0:
        ekf.soil = soil_from_camera(features, ekf.soil)

    # truth (temporary sim)
    true_px, true_py = elliptical_path(step*dt, [3,3], 3,2,0.1)

    z_uwb = np.array([[true_px], [true_py]])

    # EKF
    ekf.step(wheel_omega, z_uwb)