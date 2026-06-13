import cv2
import numpy as np


class VO:

    def __init__(self, K):
        self.K = K

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

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        kp, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        matches = self._match_features(self.prev_des, des)

        if len(matches) < self.min_inliers:
            print(f"[VO] not enough matches: {len(matches)}")
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        return {
            "kp": kp,
            "des": des,
            "pts_prev": pts_prev,
            "pts_curr": pts_curr,
            "matches": matches
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

    def _computeMotion(self, pts_prev, pts_curr, E, mask):
        _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=mask)

        print(f"R=\n{np.round(R, 4)}\nt={t.ravel().round(4)}")

        dx = float(t[0, 0])
        dy = float(t[1, 0])
        dz = float(t[2, 0])
        dyaw = float(np.arctan2(R[0, 2], R[2, 2]))

        if abs(abs(dx) - 0.5774) < 0.01 and abs(abs(dy) - 0.5774) < 0.01:
            print("[VO] degenerate t rejected")
            return None

        if abs(dyaw) > 0.5:
            print("[VO] dyaw too large, rejected")
            return None

        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "dyaw": dyaw
        }

    def VO_process(self, frame, display=False):
        frame_data = self._process_frame(frame)
        if frame_data is None:
            return None

        pts_prev = frame_data["pts_prev"]
        pts_curr = frame_data["pts_curr"]

        e_data = self._computeE(pts_prev, pts_curr)
        if e_data is None:
            return None

        motion = self._computeMotion(pts_prev, pts_curr, e_data["E"], e_data["mask"])
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