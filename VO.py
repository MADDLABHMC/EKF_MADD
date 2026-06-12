import cv2
import numpy as np
import time


class VO:

    # constructor requiring camera intrinsic matrix K
    def __init__(self, K):
        self.K  = K

        # camera parameters to help image processing
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None

        # cv2 object ORB for feature detection
        # COME CHECK THESE PARAMETERS
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8
        )

        # cv2 object for feature matching
        # COME CHECK THESE PARAMETERS
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = 0.75  # the amount the pair must be better than the second-best match (25%)
        self.max_matches = 200  # most allowable matches after the LOWE's ratio test 
        self.min_inliers = 15

        # cv2 object for image enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # helper function to do the matching features process (statistically inconsistent)
    def _match_features(self, des1, des2):
        # error handling for the start of process
        if des1 is None or des2 is None:
            return []
        
        # perform k-nearest neighbor matching
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        
        # filter matches based on distance and ratio and assigns a quality score
        # this is a LOWE's ratio test where we keep only the matches that are significantly better than the second-best match
        for m_n in knn_matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.match_ratio * n.distance and m.distance < 60:
                good_matches.append(m)
                
        # sort the best matches by quality score (distance)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        
        # return the top matches
        return good_matches[:self.max_matches]

    def getE(self, frame, display=False):
        # convert to grayscale and apply CLAHE for local contrast enhancement
        # CLAHE helps ORB find stable features in low-contrast textures like gravel        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        # detect features in the clahe - preprocessed image 
        # uses ORB object and its ability to do detect and compute (thanks cv2)
        kp, des = self.orb.detectAndCompute(gray, None)

        # First frame — just store and return
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # match features between the current and previous frames
        matches = self._match_features(self.prev_des, des)

        # error handling for feature matching if there isnt enough to compute E
        if len(matches) < 8:
            print(f"[VO] not enough matches: {len(matches)}")
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        pts_prev = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

        # pass and return 
        # note keep mask in future / return it eventully so can pass to find motion
        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0  
        )

        # error handling for the essential matrix calculation
        if E is None or mask is None:
            print("[VO] findEssentialMat returned None")
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        inliers = int(mask.sum())

        # if the number of inliers is less than the minimum required, return None
        if inliers < self.min_inliers:
            print(f"[VO] too few inliers: {inliers}/{len(matches)}")
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des
            return None

        # window to help see what its picking up / what matches it finds
        if display:
            vis = frame.copy()
            inlier_mask = mask.ravel().astype(bool)
            for i, (p1, p2) in enumerate(zip(pts_prev, pts_curr)):
                color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                cv2.line(vis, tuple(map(int, p1)), tuple(map(int, p2)), color, 1)
                cv2.circle(vis, tuple(map(int, p2)), 2, color, -1)
            cv2.putText(vis, f"matches={len(matches)} inliers={inliers}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("VO", vis)
            cv2.waitKey(1)

        # update the previous frame's data to be used in next iteration
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_des = des

        # helper printout for debugging
        # note: if the E matrix is ill-conditioned then the ratio will be <0.7
        U, S, Vt = np.linalg.svd(E)
        sv_ratio = float(S[1] / S[0]) if S[0] > 0 else 0.0
        
        return {
            "E": E,
            "SV_ratio": sv_ratio,
            "inliers": inliers,
            "matches": len(matches),
            "inlier_ratio": inliers / len(matches)
        }