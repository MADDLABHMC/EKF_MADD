import cv2
import numpy as np
import pyrealsense2 as rs


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

    def preprocess(self, gray):

        return self.clahe.apply(gray)

    def detect(self, gray):

        return self.orb.detectAndCompute(
            gray,
            None
        )
        
    def getE(self, frame):
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

        matches = [
            m for m in matches
            if m.distance < 5
        ]

        matches = matches[:200] # come back and print for debug 

        if len(matches) < 8:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des

            return None

        pts_prev = np.float32([
            self.prev_kp[m.queryIdx].pt
            for m in matches
        ])

        pts_curr = np.float32([
            kp[m.trainIdx].pt
            for m in matches
        ])

        E, _ = cv2.findEssentialMat(pts_curr, pts_prev, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des

            return None
        else:
            return E

    def printout(self, inliers, matches):
        
        return {
            "dx": float(self.dx_filt),
            "dy": float(self.dy_filt),
            "dyaw": float(self.dyaw_filt),
            "inliers": int(inliers),
            "matches": int(len(matches))
        }
        

