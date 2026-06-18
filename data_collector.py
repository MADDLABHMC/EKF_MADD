import time
import csv
import serial
import numpy as np
import cv2
import pyrealsense2 as rs
from camera import GrainCamera
from VO import VO


UWB_PORT = "COM5"
UWB_BAUD = 115200
GRAIN_EVERY = 50
OUTPUT_DIR = "."

uwb_file = open(f"{OUTPUT_DIR}/uwb_data.csv", "w", newline="")
uwb_writer = csv.writer(uwb_file)
uwb_writer.writerow(["timestamp", "x", "y", "quality"])

grain_file = open(f"{OUTPUT_DIR}/grain_data.csv", "w", newline="")
grain_writer = csv.writer(grain_file)
grain_writer.writerow(["timestamp", "D10", "D50", "D90", "Cu"])

vo_file = open(f"{OUTPUT_DIR}/vo_data.csv", "w", newline="")
vo_writer = csv.writer(vo_file)
vo_writer.writerow([
    "timestamp", "dx", "dy", "dz", "dyaw",
    "inliers", "matches", "inlier_ratio", "sv_ratio", "scale"
])


def parse_uwb_line(line):
    parts = line.strip().split(",")
    if len(parts) < 7 or parts[0] != "POS":
        return None
    try:
        x = float(parts[3])
        y = float(parts[4])
        quality = float(parts[6])
        return x, y, quality
    except ValueError:
        return None


uwb_ser = None
try:
    uwb_ser = serial.Serial(UWB_PORT, UWB_BAUD, timeout=0.01)
    print(f"[UWB] connected on {UWB_PORT}")

    time.sleep(0.5)
    uwb_ser.write(b"\r\n")
    time.sleep(0.2)
    uwb_ser.write(b"\r\n")
    time.sleep(0.2)

    uwb_ser.write(b"lec\r\n")
    time.sleep(0.2)
    print("[UWB] sent init sequence + 'lec' to start streaming")

except Exception as e:
    print(f"[UWB] could not open {UWB_PORT}: {e}")
    print("[UWB] continuing without UWB data")


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)

intrinsics = profile.get_stream(rs.stream.color) \
    .as_video_stream_profile().get_intrinsics()

K = np.array([
    [intrinsics.fx, 0,             intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1             ]
])
print("K=\n", K)

vo = VO(K)
grain_camera = GrainCamera(intrinsics.fx, intrinsics.fy)


step = 0

try:
    print("Logging started. Press ESC in the VO window to stop.")

    t_start = time.time()

    while True:
        now = time.time() - t_start

        if uwb_ser is not None:
            try:
                while uwb_ser.in_waiting:
                    raw = uwb_ser.readline().decode(errors="ignore")
                    if raw:
                        parsed = parse_uwb_line(raw)
                        if parsed is not None:
                            x, y, quality = parsed
                            uwb_writer.writerow([time.time() - t_start, x, y, quality])
            except Exception as e:
                print(f"[UWB] read error: {e}")

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        if step % GRAIN_EVERY == 0:
            features = grain_camera.step(frame, depth_frame)
            if features is not None:
                D10, D50, D90, Cu = features
                grain_writer.writerow([now, D10, D50, D90, Cu])
        else:
            result = vo.VO_process(frame, depth_frame=depth_frame, display=True)
            if result:
                vo_writer.writerow([
                    now,
                    result["dx"], result["dy"], result["dz"], result["dyaw"],
                    result["inliers"], result["matches"],
                    result["inlier_ratio"], result["sv_ratio"], result["scale"]
                ])

        step += 1

        if cv2.waitKey(1) & 0xFF == 27:
            print("ESC pressed, stopping.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    uwb_file.close()
    grain_file.close()
    vo_file.close()
    if uwb_ser is not None:
        try:
            uwb_ser.write(b"\r\n")
            time.sleep(0.2)
        except Exception:
            pass
        uwb_ser.close()
    print("Files saved: uwb_data.csv, grain_data.csv, vo_data.csv")