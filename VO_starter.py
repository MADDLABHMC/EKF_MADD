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

    def process(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

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

        matches = self.matcher.match(
            self.prev_des,
            des
        )

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

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

        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:

            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des

            return None

        inlier_mask = mask.ravel().astype(bool)

        inliers = np.count_nonzero(
            inlier_mask
        )

        if inliers < 15:

            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_des = des

            return None

        pts_prev_in = pts_prev[inlier_mask]
        pts_curr_in = pts_curr[inlier_mask]

        _, R, t, _ = cv2.recoverPose(
            E,
            pts_curr_in,
            pts_prev_in,
            self.K
        )

        yaw = np.arctan2(
            R[1, 0],
            R[0, 0]
        )

        tx = float(t[0])
        tz = float(t[2])

        dx = tz
        dy = 0.15 * tx

        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)
        yaw = np.clip(yaw, -0.5, 0.5)

        self.dx_filt = (
            self.alpha * self.dx_filt
            + (1 - self.alpha) * dx
        )

        self.dy_filt = (
            self.alpha * self.dy_filt
            + (1 - self.alpha) * dy
        )

        self.dyaw_filt = (
            self.alpha * self.dyaw_filt
            + (1 - self.alpha) * yaw
        )

        vis = frame.copy()

        for p1, p2 in zip(
            pts_prev_in,
            pts_curr_in
        ):

            x1, y1 = map(int, p1)
            x2, y2 = map(int, p2)

            cv2.circle(
                vis,
                (x2, y2),
                2,
                (0, 255, 0),
                -1
            )

            cv2.line(
                vis,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                1
            )

        cv2.putText(
            vis,
            f"inliers: {inliers}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "mvo",
            vis
        )

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

K = np.array([
    [intrinsics.fx, 0, intrinsics.ppx],
    [0, intrinsics.fy, intrinsics.ppy],
    [0, 0, 1]
])

vo = VO(K)

while True:

    frames = pipeline.wait_for_frames()

    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    frame = np.asanyarray(
        color_frame.get_data()
    )

    result = vo.process(frame)

    if result is not None:

        print(
            f"dx={result['dx']:.3f} "
            f"dy={result['dy']:.3f} "
            f"dyaw={result['dyaw']:.3f} "
            f"inliers={result['inliers']} "
            f"matches={result['matches']}"
        )

    cv2.imshow(
        "camera",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

pipeline.stop()

cv2.destroyAllWindows()