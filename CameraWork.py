import pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv

# Camera Setup
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

profile = pipeline.get_active_profile()
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

fx = intrinsics.fx
fy = intrinsics.fy

print("fx:", fx, "fy:", fy)

# CSV
csv_file = open("grain_data_movement.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "avg_diameter_mm",
    "num_grains",
    "D10_mm",
    "D50_mm",
    "D90_mm",
    "Cu",
    "variance",
])

# Parameters (interval is a guess for rn)
interval = 0.01
last_log_time = time.time()
logging_enabled = False

rolling_buffer = []
max_buffer_size = 1500

alpha = 0.2  # smoothing factor (adapt this its a guess)

prev_D50 = None
prev_Cu = None

warmup_samples = 200
total_samples = 0

# Loop
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        output = frame.copy()

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        seen_centers = []
        new_areas = []

        # Detect Grains (oval and contour based)
        for cnt in contours:
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (x, y), (major_axis, minor_axis), _ = ellipse

            if major_axis < 5 or minor_axis < 5:
                continue

            cx, cy = int(x), int(y)

            # Prevent duplicate detections (slow rover = hopefully less overlapping data)
            if any(np.linalg.norm(np.array([cx, cy]) - np.array(prev)) < 10 for prev in seen_centers):
                continue
            seen_centers.append((cx, cy))

            # Depth averaging
            depths = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px = cx + dx
                    py = cy + dy

                    if 0 <= px < 640 and 0 <= py < 480:
                        d = depth_frame.get_distance(px, py)
                        if d > 0:
                            depths.append(d)

            if not depths:
                continue

            # Reject noisy depth
            if np.std(depths) > 0.01:
                continue

            depth = np.mean(depths)

            # Convert to real-world size
            scale_x = depth / fx
            scale_y = depth / fy

            major_m = major_axis * scale_x
            minor_m = minor_axis * scale_y

            area_m2 = np.pi * (major_m / 2) * (minor_m / 2)
            area_mm2 = area_m2 * 1e6

            # Shape sanity check
            contour_area = cv2.contourArea(cnt)
            ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)

            if contour_area / ellipse_area < 0.6:
                continue

            if 0.001 < area_mm2 < 10000:
                new_areas.append(area_mm2)
                cv2.ellipse(output, ellipse, (0, 255, 0), 2)

        # Update the rolling buffer
        rolling_buffer.extend(new_areas)
        total_samples += len(new_areas)

        if len(rolling_buffer) > max_buffer_size:
            rolling_buffer = rolling_buffer[-max_buffer_size:]
            
        if not logging_enabled and total_samples > warmup_samples:
            logging_enabled = True
            print("=== LOGGING STARTED ===")

        # Logging 
        current_time = time.time()

        if current_time - last_log_time >= interval and logging_enabled:

            if len(rolling_buffer) > 0:
                areas = np.array(rolling_buffer)
                diameters = np.sqrt(4 * areas / np.pi)

                # MAD filtering
                if total_samples > warmup_samples:
                    median = np.median(diameters)
                    mad = np.median(np.abs(diameters - median))
                    robust_std = 1.4826 * mad

                    lower = median - 1.5 * robust_std
                    upper = median + 1.5 * robust_std

                    diameters = diameters[(diameters > lower) & (diameters < upper)]

                # Upper tail clipping
                if len(diameters) > 10:
                    upper_clip = np.percentile(diameters, 85)
                    diameters = diameters[diameters < upper_clip]

                if len(diameters) > 0:
                    avg_diameter = np.mean(diameters)
                    num_grains = len(diameters)

                    D10 = np.percentile(diameters, 10)
                    D50 = np.percentile(diameters, 50)
                    D90 = np.percentile(diameters, 90)

                    Cu = D90 / D10 if D10 > 0 else 0
                    variance = np.var(diameters)

                    # temporal smoothing
                    if prev_D50 is not None:
                        D50 = alpha * D50 + (1 - alpha) * prev_D50
                        Cu = alpha * Cu + (1 - alpha) * prev_Cu

                    prev_D50 = D50
                    prev_Cu = Cu

                else:
                    avg_diameter = num_grains = D10 = D50 = D90 = Cu = variance = 0

            else:
                avg_diameter = num_grains = D10 = D50 = D90 = Cu = variance = 0

            csv_writer.writerow([
                avg_diameter,
                num_grains,
                D10,
                D50,
                D90,
                Cu,
                variance
            ])
            csv_file.flush()

            print(f"[LOG] D50:{D50:.3f} mm | Cu:{Cu:.2f} | Var:{variance:.4f}")

            last_log_time = current_time

        # display
        cv2.putText(output,
                    "Press 'q' to quit",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2)

        cv2.imshow("Grain Detection", output)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    csv_file.close()
    cv2.destroyAllWindows()