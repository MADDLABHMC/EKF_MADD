# obnoxiously commented for my understanding rn
import cv2
import numpy as np
import pyrealsense2 as rs
from VO import VO

# camera setup using pyrealsense2 to initialze the camera 
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

# create a vo object with the new K matrix
freeze_duration = 5.0  # seconds to wait for E to freeze
vo = VO(K, freeze_duration) 

# big try catch block to ensure the camera stops on exit 
try:
    # initialize E as none to clear any possible past values and ensure new calibration
    E = None
    
    print("Starting calibration to compute E matrix...")
    print(f"This will take about {freeze_duration} seconds.")
    print("Please keep the camera still during this time for best results.")
    
    # keep calibrating until the essential matrix is frozen for the configured duration
    while not vo.is_calibrated():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        _ = vo.getE(frame)

    E = vo.getFrozenE()
    print("Calibration complete\nE matrix:\n", E)
    print("Entering motion estimation. First valid motion output may take a few frames.")
    print("Press ESC in the OpenCV window to exit.")

    # run until exit on "esc" key press
    while True:
        # fetch frames and process
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        
        # ask for motion computation with the E and print the results for now till decide what to do with them
        motion = vo.getMotion(frame, E, True)
        if motion:
            print(
                f"dx={motion['dx']:.3f} "
                f"dy={motion['dy']:.3f} "
                f"dyaw={motion['dyaw']:.3f} "
                f"inliers={motion['inliers']} "
                f"matches={motion['matches']}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("ESC pressed, exiting motion estimation loop.")
            break

# STOP THE CAMERA ON EXIT
finally:
    pipeline.stop()
    cv2.destroyAllWindows()