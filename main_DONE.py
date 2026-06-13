from ekf_DONE import EKF
from sim_DONE import Rover, Soil, elliptical_path
from camera import GrainCamera
import numpy as np
import cv2
import numpy as np
import pyrealsense2 as rs
from VO import VO

dt = 0.1
wheel_omega = 8.0

rover = Rover(200, 0.3, 0.2, 4)
soil = Soil(1500, 1500, 1.1, 300, 35, 0.03)

ekf = EKF(dt, rover, soil)

camera = GrainCamera()

# this chunk was pulled from VO_main #########################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

intrinsics = profile.get_stream(rs.stream.color) \
    .as_video_stream_profile().get_intrinsics()

K = np.array([
    [intrinsics.fx, 0,             intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1             ]
])

print("K=\n", K)

vo = VO(K)

###########################################################################

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


GRAIN_EVERY = 50
vo_result = None

try:
    for step in range(10000):

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        if step % GRAIN_EVERY == 0:
            features = camera.analyze(frame)
            ekf.soil = soil_from_camera(features, ekf.soil)
        else:
            vo_result = vo.process(frame)

        true_px, true_py = elliptical_path(step * dt, [3, 3], 3, 2, 0.1)
        z_uwb = np.array([[true_px], [true_py]])

        ekf.step(wheel_omega, z_uwb, vo_result)

        if vo_result:
            print(
                f"dx={vo_result['dx']:.3f} dy={vo_result['dy']:.3f} "
                f"dyaw={vo_result['dyaw']:.3f} inliers={vo_result['inliers']} "
                f"sv={vo_result['sv_ratio']:.3f}"
            )

        vo_result = None

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()