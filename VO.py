import cv2
import numpy as np


class VO:

    def __init__(self, K):
        self.K = K

        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.prev_depth = None

        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8
        )

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = 0.75
        self.max_matches = 100
        self.min_inliers = 15

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _match_features(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []

        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.match_ratio * n.distance and m.distance < 40:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda x: x.distance)
        return good_matches[:self.max_matches]

    def _process_frame(self, frame, depth_frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth_frame
            return None

        matches = self._match_features(self.prev_des, des)

        if len(matches) < self.min_inliers:
            print(f"[VO] not enough matches: {len(matches)}")
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            self.prev_depth = depth_frame
            return None

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        prev_depth = self.prev_depth

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des
        self.prev_depth = depth_frame

        return {
            "kp": kp,
            "des": des,
            "pts_prev": pts_prev,
            "pts_curr": pts_curr,
            "matches": matches,
            "prev_depth": prev_depth,
            "curr_depth": depth_frame
        }

    def _computeE(self, pts_prev, pts_curr):
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=2.0
        )

        if E is None or mask is None:
            print("[VO] findEssentialMat returned None")
            return None

        inliers = int(mask.sum())

        if inliers < self.min_inliers:
            print(f"[VO] too few inliers: {inliers}/{len(pts_prev)}")
            return None

        U, S, Vt = np.linalg.svd(E)
        sv_ratio = float(S[1] / S[0]) if S[0] > 0 else 0.0

        return {
            "E": E,
            "mask": mask,
            "inliers": inliers,
            "sv_ratio": sv_ratio,
            "inlier_ratio": inliers / len(pts_prev)
        }

    def _backproject(self, pt, depth_frame):
        u, v = pt
        d = depth_frame.get_distance(int(round(u)), int(round(v)))
        if d <= 0:
            return None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

        return np.array([X, Y, Z])

    def _recover_scale(self, pts_prev, pts_curr, R, t_unit, prev_depth, curr_depth, inlier_mask):
        if prev_depth is None or curr_depth is None:
            return None

        t_unit = t_unit.flatten()
        scales = []

        for i in range(len(pts_prev)):
            if not inlier_mask[i]:
                continue

            P1 = self._backproject(pts_prev[i], prev_depth)
            P2 = self._backproject(pts_curr[i], curr_depth)

            if P1 is None or P2 is None:
                continue

            residual = P2 - R @ P1
            s = np.dot(residual, t_unit)
            scales.append(s)

        if len(scales) < 5:
            return None

        scales = np.array(scales)
        return float(np.median(scales))

    def _computeMotion(self, pts_prev, pts_curr, E, mask, prev_depth=None, curr_depth=None):
        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=mask)

        print(f"R=\n{np.round(R, 4)}\nt_unit={t.ravel().round(4)}")

        t_unit = t.flatten()

        if abs(abs(t_unit[0]) - 0.5774) < 0.01 and abs(abs(t_unit[1]) - 0.5774) < 0.01:
            print("[VO] degenerate t rejected")
            return None

        dyaw = float(np.arctan2(R[0, 2], R[2, 2]))

        if abs(dyaw) > 0.5:
            print("[VO] dyaw too large, rejected")
            return None

        inlier_mask = mask.ravel().astype(bool)
        scale = self._recover_scale(
            pts_prev, pts_curr, R, t_unit, prev_depth, curr_depth, inlier_mask
        )

        if scale is None:
            print("[VO] scale recovery failed, falling back to unit-scale t")
            scale = 1.0

        t_scaled = t_unit * scale

        dx = float(t_scaled[0])
        dy = float(t_scaled[1])
        dz = float(t_scaled[2])

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dyaw": dyaw,
            "scale": float(scale)
        }

    def VO_process(self, frame, depth_frame=None, display=False):
        frame_data = self._process_frame(frame, depth_frame)
        if frame_data is None:
            return None

        pts_prev = frame_data["pts_prev"]
        pts_curr = frame_data["pts_curr"]

        e_data = self._computeE(pts_prev, pts_curr)
        if e_data is None:
            return None

        motion = self._computeMotion(
            pts_prev, pts_curr, e_data["E"], e_data["mask"],
            prev_depth=frame_data["prev_depth"],
            curr_depth=frame_data["curr_depth"]
        )
        if motion is None:
            return None

        if display:
            vis = frame.copy()
            inlier_mask = e_data["mask"].ravel().astype(bool)
            for i, (p1, p2) in enumerate(zip(pts_prev, pts_curr)):
                color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                cv2.line(vis, tuple(map(int, p1)), tuple(map(int, p2)), color, 1)
                cv2.circle(vis, tuple(map(int, p2)), 2, color, -1)
            cv2.putText(vis,
                f"matches={len(frame_data['matches'])} inliers={e_data['inliers']} sv={e_data['sv_ratio']:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis,
                f"dx={motion['dx']:.3f} dy={motion['dy']:.3f} dyaw={motion['dyaw']:.3f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("VO", vis)
            cv2.waitKey(1)

        return {
            "E": e_data["E"],
            "sv_ratio": e_data["sv_ratio"],
            "inliers": e_data["inliers"],
            "inlier_ratio": e_data["inlier_ratio"],
            "matches": len(frame_data["matches"]),
            **motion
        }