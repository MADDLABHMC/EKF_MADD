import cv2
import numpy as np
import pyrealsense2 as rs
from VO import VO

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

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        result = vo.VO_process(frame, display=True)

        if result:
            print(
                f"dx={result['dx']:.3f} dy={result['dy']:.3f} dz={result['dz']:.3f} "
                f"dyaw={result['dyaw']:.3f} inliers={result['inliers']} "
                f"sv={result['sv_ratio']:.3f}"
            )

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()