import cv2
import numpy as np
import pyrealsense2 as rs
import time


class VO:

    # constructor needing a K matrix
    def __init__(self, K, freeze_duration=10.0):

        self.K = K
        self.freeze_duration = freeze_duration

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8
        )

        self.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

        self.clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        self.alpha = 0.7

        self.dx_filt = 0.0
        self.dy_filt = 0.0
        self.dyaw_filt = 0.0

        self.start_time = None
        self.E_candidates = []
        self.E_frozen = None
        self.E_last = None

    # localized image contrast enhancement using CLAHE
    def preprocess(self, gray):

        return self.clahe.apply(gray)

    # feature detection and description using ORB
    def detect(self, gray):

        return self.orb.detectAndCompute(
            gray,
            None
        )

    def _normalize_E(self, E):
        E = E.astype(np.float64)
        norm = np.linalg.norm(E)
        return E if norm == 0 else E / norm

    def _align_sign(self, E, reference):
        if reference is None:
            return E
        if np.trace(E @ reference.T) < 0:
            return -E
        return E

    def _average_E_candidates(self):
        if not self.E_candidates:
            return None

        reference = self._normalize_E(self.E_candidates[0][0])
        weighted_sum = np.zeros_like(reference)

        for E, inliers in self.E_candidates:
            E_norm = self._normalize_E(E)
            E_norm = self._align_sign(E_norm, reference)
            weighted_sum += E_norm * max(inliers, 1)

        U, _, Vt = np.linalg.svd(weighted_sum)
        return U @ np.diag([1.0, 1.0, 0.0]) @ Vt

    # main processing function to compute the essential matrix from the current frame 
    def getE(self, frame):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time

        if self.E_frozen is not None:
            return self.E_frozen

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.preprocess(gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = self.detect(gray)
            return None

        kp, des = self.detect(gray)

        if des is None or self.prev_des is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = [m for m in matches if m.distance < 5]
        matches = matches[:200]

        if len(matches) < 8:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return self.E_last

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is not None:
            inliers = int(mask.sum()) if mask is not None else len(matches)
            self.E_candidates.append((E, inliers))
            self.E_last = E

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        elapsed = current_time - self.start_time
        if elapsed >= self.freeze_duration and self.E_candidates:
            self.E_frozen = self._average_E_candidates()
            return self.E_frozen

        return None

    def getFrozenE(self):
        return self.E_frozen

    def is_calibrated(self):
        return self.E_frozen is not None
    
    def getMotion(self, frame, E):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.preprocess(gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp, self.prev_des = self.detect(gray)
            return None

        kp, des = self.detect(gray)

        if des is None or self.prev_des is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # Match features between previous and current frame
        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = [m for m in matches if m.distance < 5]
        matches = matches[:200]

        if len(matches) < 8:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # Get point coordinates
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        # Use frozen E to recover pose (R, t)
        _, R, t, mask = cv2.recoverPose(E, pts_curr, pts_prev, self.K)

        inlier_mask = mask.ravel().astype(bool)
        inliers = np.count_nonzero(inlier_mask)

        if inliers < 15:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # Extract yaw from rotation matrix
        yaw = np.arctan2(R[1, 0], R[0, 0])

        # Extract translation components
        tx = float(t[0])
        tz = float(t[2])

        # Map to motion (you can tune these mappings)
        dx = tz
        dy = 0.15 * tx

        # Clip to reasonable ranges
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        yaw = np.clip(yaw, -0.5, 0.5)

        # Apply exponential smoothing filter
        self.dx_filt = self.alpha * self.dx_filt + (1 - self.alpha) * dx
        self.dy_filt = self.alpha * self.dy_filt + (1 - self.alpha) * dy
        self.dyaw_filt = self.alpha * self.dyaw_filt + (1 - self.alpha) * yaw

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return {
            "dx": float(self.dx_filt),
            "dy": float(self.dy_filt),
            "dyaw": float(self.dyaw_filt),
            "inliers": int(inliers),
            "matches": int(len(matches))
        }