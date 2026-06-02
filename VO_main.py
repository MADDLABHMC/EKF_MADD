import cv2
import numpy as np
import pyrealsense2 as rs
from VO_progress import VO

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

K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

vo = VO(K)

E = vo.getE()
    