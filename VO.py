import cv2
import numpy as np
import pyrealsense2 as rs
import time


class VO:

    # constructor needing a K matrix
    def __init__(self, K, freeze_duration):

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

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = 0.75
        self.max_matches = 200
        self.min_inliers = 15
        self.min_inlier_ratio = 0.3

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

        # cumulative camera motion in the vehicle plane
        self.pose = np.zeros(2, dtype=np.float32)  # [x, y]
        self.pose_yaw = 0.0
        self.trajectory = [self.pose.copy()]

    # localized image contrast enhancement using CLAHE
    def preprocess(self, gray):

        return self.clahe.apply(gray)

    # feature detection and description using ORB
    def detect(self, gray):

        return self.orb.detectAndCompute(
            gray,
            None
        )

    def _match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.match_ratio * n.distance and m.distance < 60:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        return good_matches[: self.max_matches]

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

        matches = self._match_features(self.prev_des, des)

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
            threshold=3.5
        )

        if E is not None:
            inliers = int(mask.sum()) if mask is not None else len(matches)
            self.E_candidates.append((E, inliers))
            self.E_last = E
            print(f"[VO] found E candidate #{len(self.E_candidates)} with {inliers} inliers")

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        elapsed = current_time - self.start_time
        print(f"[VO] calibration elapsed={elapsed:.2f}s, valid E candidates={len(self.E_candidates)}")

        if elapsed >= self.freeze_duration and self.E_candidates:
            self.E_frozen = self._average_E_candidates()
            print("[VO] calibration complete: E frozen from", len(self.E_candidates), "candidates")
            return self.E_frozen

        return None

    def getFrozenE(self):
        return self.E_frozen

    def getTrajectory(self):
        return [tuple(p) for p in self.trajectory]

    def drawTrajectory(self, size=600, scale=150):
        traj_img = np.zeros((size, size, 3), dtype=np.uint8)
        center = (size // 2, size // 2)
        trajectory = self.getTrajectory()

        for i in range(1, len(trajectory)):
            x0, y0 = trajectory[i - 1]
            x1, y1 = trajectory[i]
            pt0 = (int(center[0] + x0 * scale), int(center[1] - y0 * scale))
            pt1 = (int(center[0] + x1 * scale), int(center[1] - y1 * scale))
            cv2.line(traj_img, pt0, pt1, (0, 255, 0), 2)

        if trajectory:
            x, y = trajectory[-1]
            pt = (int(center[0] + x * scale), int(center[1] - y * scale))
            cv2.circle(traj_img, pt, 5, (0, 0, 255), -1)

        cv2.circle(traj_img, center, 3, (255, 255, 255), -1)
        cv2.putText(traj_img, f"x={trajectory[-1][0]:.2f} y={trajectory[-1][1]:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(traj_img, "origin", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return traj_img

    def is_calibrated(self):
        return self.E_frozen is not None
    
    def getMotion(self, frame, E, display=False):
        
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
        matches = self._match_features(self.prev_des, des)

        if len(matches) < 8:
            if display:
                vis = frame.copy()
                cv2.putText(
                    vis,
                    f"waiting for matches: {len(matches)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("mvo", vis)
                cv2.waitKey(1)
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # Get point coordinates
        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        # Use frozen E to recover pose (R, t)
        _, R, t, mask = cv2.recoverPose(E, pts_prev, pts_curr, self.K)

        inlier_mask = mask.ravel().astype(bool)
        inliers = np.count_nonzero(inlier_mask)
        inlier_ratio = float(inliers) / max(1, len(matches))

        if inliers < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            if display:
                vis = frame.copy()
                pts_prev_in = pts_prev[inlier_mask]
                pts_curr_in = pts_curr[inlier_mask]
                for p1, p2 in zip(pts_prev_in, pts_curr_in):
                    x1, y1 = map(int, p1)
                    x2, y2 = map(int, p2)
                    cv2.circle(vis, (x2, y2), 2, (0, 255, 255), -1)
                    cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(
                    vis,
                    f"waiting for inliers: {inliers}/{len(matches)}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("mvo", vis)
                cv2.waitKey(1)
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # Extract yaw from rotation matrix around the camera's vertical axis
        yaw = np.arctan2(R[0, 2], R[2, 2])

        # Extract translation components in camera-local coordinates
        tx = float(t[0])
        tz = float(t[2])

        # Map to local camera motion (forward, right)
        dx = -tz
        dy = tx

        # Clip to reasonable ranges
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        yaw = np.clip(yaw, -0.5, 0.5)

        # Apply exponential smoothing filter
        self.dx_filt = self.alpha * self.dx_filt + (1 - self.alpha) * dx
        self.dy_filt = self.alpha * self.dy_filt + (1 - self.alpha) * dy
        self.dyaw_filt = self.alpha * self.dyaw_filt + (1 - self.alpha) * yaw

        # update global orientation and pose
        self.pose_yaw += self.dyaw_filt
        cos_yaw = np.cos(self.pose_yaw)
        sin_yaw = np.sin(self.pose_yaw)
        global_step = np.array([
            cos_yaw * self.dx_filt - sin_yaw * self.dy_filt,
            sin_yaw * self.dx_filt + cos_yaw * self.dy_filt
        ], dtype=np.float32)

        self.pose += global_step
        self.trajectory.append(self.pose.copy())
        if display:
            vis = frame.copy()
            pts_prev_in = pts_prev[inlier_mask]
            pts_curr_in = pts_curr[inlier_mask]
            for p1, p2 in zip(pts_prev_in, pts_curr_in):
                x1, y1 = map(int, p1)
                x2, y2 = map(int, p2)
                cv2.circle(vis, (x2, y2), 2, (0, 255, 0), -1)
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.putText(
                vis,
                f"inliers: {inliers}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("mvo", vis)
            cv2.waitKey(1)

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return {
            "dx": float(self.dx_filt),
            "dy": float(self.dy_filt),
            "x": float(self.pose[0]),
            "y": float(self.pose[1]),
            "dyaw": float(self.dyaw_filt),
            "inliers": int(inliers),
            "matches": int(len(matches))
        }