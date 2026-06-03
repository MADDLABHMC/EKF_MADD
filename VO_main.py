import cv2
import numpy as np
import pyrealsense2 as rs
from VO import VO

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

stream = profile.get_stream(
    rs.stream.color
)

intrinsics = stream \
    .as_video_stream_profile() \
    .get_intrinsics()

# construct the K matrix from the intrinsics from the calibration code output
K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

# calibration code to get the E matrix used for the actual odometry estimation after
vo = VO(K)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        e = vo.getE(frame)

        # once the E matrix is frozen set E and look at it
        if vo.is_calibrated() and E is None:
            E = vo.getFrozenE()
            print("calibration complete\nE matrix:\n", E)

        # use E for motion esitamtion, same E going forwards 
        if E is not None:
            motion = VO.getMotion(frame, E)
            if motion:
                print(
                    f"dx={motion['dx']:.3f} "
                    f"dy={motion['dy']:.3f} "
                    f"dyaw={motion['dyaw']:.3f} "
                    f"inliers={motion['inliers']} "
                    f"matches={motion['matches']}")
finally:
    pipeline.stop()