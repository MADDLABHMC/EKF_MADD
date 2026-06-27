'''
A class for analyzing grain size in images.
'''

import numpy as np
import cv2


class GrainCamera:
    def __init__(self, fx, fy):
        '''
        Initialize the GrainCamera with the camera's intrinsic parameters.
        fx: The focal length in the x-direction (pixels)
        fy: The focal length in the y-direction (pixels)
        '''
        self.fx = fx
        self.fy = fy

        self.alpha = 0.2
        self.prev_D50 = None
        self.prev_Cu = None

        self.buffer = []
        self.max_buffer = 1500

        self.latest = None

    def step(self, frame, depth_frame):
        '''
        Process a single frame and extract grain size information.
        frame: The input image frame
        depth_frame: The depth frame corresponding to the image
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        areas = []
        seen = []

        for cnt in contours:
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), _ = ellipse

            if MA < 5 or ma < 5:
                continue

            cx, cy = int(x), int(y)

            if any(np.linalg.norm(np.array([cx, cy]) - np.array(p)) < 10 for p in seen):
                continue

            seen.append((cx, cy))

            depths = []

            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px = cx + dx
                    py = cy + dy

                    if 0 <= px < 640 and 0 <= py < 480:
                        d = depth_frame.get_distance(px, py)
                        if d > 0:
                            depths.append(d)

            if len(depths) == 0:
                continue

            if np.std(depths) > 0.01:
                continue

            depth_val = np.mean(depths)

            sx = depth_val / self.fx
            sy = depth_val / self.fy

            major_m = MA * sx
            minor_m = ma * sy

            area_m2 = np.pi * (major_m / 2) * (minor_m / 2)
            area_mm2 = area_m2 * 1e6

            if 0.001 < area_mm2 < 10000:
                areas.append(area_mm2)

        self.buffer.extend(areas)

        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]

        if len(self.buffer) < 20:
            return None

        diameters = np.sqrt(4 * np.array(self.buffer) / np.pi)

        D10 = np.percentile(diameters, 10)
        D50 = np.percentile(diameters, 50)
        D90 = np.percentile(diameters, 90)

        Cu = D90 / (D10 + 1e-6)

        if self.prev_D50 is not None:
            D50 = 0.2 * D50 + 0.8 * self.prev_D50
            Cu = 0.2 * Cu + 0.8 * self.prev_Cu

        self.prev_D50 = D50
        self.prev_Cu = Cu

        self.latest = np.array([D10, D50, D90, Cu])

        return self.latest

    def get_features(self):
        '''
        Get the latest extracted grain size features.
        Returns:
            The latest extracted grain size features as a numpy array.
        '''
        return self.latest