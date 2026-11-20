import numpy as np
import cv2
from src.bundle_adjustment import SlidingWindowBA

class StereoVisualOdometry:
    """
    Production Stereo Visual Odometry (Stereo VO) Engine with Local Bundle Adjustment.
    
    Why Stereo?
    1. Scale is physically observable from baseline b: Z = (f * b) / disparity.
    2. Zero scale drift: Every 3D landmark is in physical meters on frame 1.
    3. Works while stationary (no minimum parallax motion required).
    4. Sliding-window Local Bundle Adjustment simultaneously optimizes poses & 3D points.
    """
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, 
                 baseline=0.54, min_features=1200, enable_ba=True, window_size=5):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        self.min_features = min_features
        self.enable_ba = enable_ba
        self.window_size = window_size

        # Stereo Block Matcher
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

        # Sliding Window Bundle Adjuster
        self.ba = SlidingWindowBA(fx, fy, cx, cy, window_size=window_size) if enable_ba else None

        # State tracking
        self.prev_left_gray = None
        self.prev_pts_left = None
        self.prev_pts_3d = None

        # Global World Pose (X: right, Y: down, Z: forward)
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)

        self.trajectory = []
        self.euler_angles = [0.0, 0.0, 0.0]
        self.total_distance = 0.0
        self.num_tracked = 0
        self.inlier_ratio = 1.0
        self.status = "INITIALIZING"
        self.ba_correction_m = 0.0

    def compute_disparity(self, left_gray, right_gray):
        """Computes stereo disparity map."""
        disp = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp[disp <= 0.0] = 0.1
        return disp

    def triangulate_stereo_points(self, left_pts, disparity_map):
        """
        Direct physical 3D triangulation:
        Z = (fx * baseline) / disparity
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        """
        h, w = disparity_map.shape
        pts3d_cam = []
        valid_indices = []

        for idx, pt in enumerate(left_pts):
            u, v = int(round(pt[0])), int(round(pt[1]))
            if 0 <= u < w and 0 <= v < h:
                d = disparity_map[v, u]
                if d > 0.5:
                    z = (self.fx * self.baseline) / d
                    if 1.0 < z < 80.0:  # Valid metric range
                        x = (u - self.cx) * z / self.fx
                        y = (v - self.cy) * z / self.fy
                        pts3d_cam.append([x, y, z])
                        valid_indices.append(idx)

        return np.array(pts3d_cam, dtype=np.float64), np.array(valid_indices, dtype=int)

    def detect_features(self, gray_frame):
        """Corner detection with sub-pixel precision."""
        corners = cv2.goodFeaturesToTrack(
            gray_frame, maxCorners=self.min_features, qualityLevel=0.01, minDistance=8, blockSize=7
        )
        if corners is not None:
            pts = corners.reshape(-1, 2)
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            pts = cv2.cornerSubPix(gray_frame, pts, (3, 3), (-1, -1), term)
            return pts
        return np.empty((0, 2), dtype=np.float32)

    def rotation_matrix_to_euler(self, R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees([x, y, z])

    def process_frame(self, left_frame, right_frame):
        """
        Processes a synchronized stereo pair.
        Returns detailed telemetry and pose updates.
        """
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY) if len(left_frame.shape) == 3 else left_frame
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY) if len(right_frame.shape) == 3 else right_frame

        # Compute Disparity Map
        disparity_map = self.compute_disparity(left_gray, right_gray)

        # First Frame Initialization
        if self.prev_left_gray is None:
            self.prev_left_gray = left_gray
            self.prev_pts_left = self.detect_features(left_gray)
            pts3d_cam, valid_idx = self.triangulate_stereo_points(self.prev_pts_left, disparity_map)
            self.prev_pts_left = self.prev_pts_left[valid_idx]
            self.prev_pts_3d = pts3d_cam
            self.status = "STEREO_INITIALIZED"

            pos = [float(self.cur_t[0, 0]), float(self.cur_t[1, 0]), float(self.cur_t[2, 0])]
            self.trajectory.append(pos)
            return self._build_return_dict(np.empty((0,2)), np.empty((0,2)), np.empty(0, dtype=bool), disparity_map)

        # KLT Optical Flow Tracking between consecutive left frames
        curr_pts_left, status_lk, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_left_gray, left_gray, self.prev_pts_left, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        valid_mask = (status_lk.ravel() == 1)
        valid_pts2d = curr_pts_left[valid_mask]
        valid_pts3d = self.prev_pts_3d[valid_mask]

        inlier_mask = np.zeros(len(valid_pts2d), dtype=bool)

        if len(valid_pts3d) >= 6:
            # 3D-to-2D Motion Estimation via Perspective-n-Point RANSAC (EPnP / P3P)
            # Solves metric R, t without scale ambiguity!
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                valid_pts3d, valid_pts2d, self.K, None,
                flags=cv2.SOLVEPNP_EPNP, reprojectionError=2.0, confidence=0.999
            )

            if success and inliers is not None and len(inliers) >= 12:
                R_rel, _ = cv2.Rodrigues(rvec)
                t_rel = tvec.reshape(3, 1)

                # Kinematic Angular Velocity Guard
                rel_rot_angle = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
                if rel_rot_angle > 6.0:
                    self.status = "ROT_SPIKE_REJECTED"
                    return self._build_return_dict(valid_pts2d, valid_pts2d, inlier_mask)

                inlier_mask[inliers.ravel()] = True
                self.inlier_ratio = len(inliers) / max(len(valid_pts2d), 1)
                self.num_tracked = len(inliers)

                # World Pose Integration
                # P_c = R_rel * P_prev + t_rel  ==>  T_w_curr = T_w_prev * T_prev_curr
                # R_w_curr = R_w_prev * R_rel^T,  t_w_curr = t_w_prev - R_w_curr * t_rel
                R_step = R_rel.T
                t_step = -R_step @ t_rel

                old_t = self.cur_t.copy()
                self.cur_t = self.cur_t + self.cur_R @ t_step
                self.cur_R = self.cur_R @ R_step

                step_norm = float(np.linalg.norm(t_step))
                self.total_distance += step_norm
                self.euler_angles = self.rotation_matrix_to_euler(self.cur_R).tolist()
                self.status = "STEREO_TRACKING_OK"

                # Triangulate 3D points in current left frame
                curr_pts3d_cam, valid_curr_idx = self.triangulate_stereo_points(valid_pts2d, disparity_map)
                
                # Bundle Adjustment Step (Keyframe-based optimization every 5th frame)
                if self.enable_ba and self.ba is not None and len(curr_pts3d_cam) > 10 and (len(self.trajectory) % 5 == 0):
                    # Convert camera points to global world coordinates for BA
                    pts3d_world = (self.cur_R @ curr_pts3d_cam.T + self.cur_t).T
                    self.ba.add_keyframe(self.cur_R, self.cur_t, valid_pts2d[valid_curr_idx], pts3d_world)
                    opt_R, opt_t = self.ba.optimize(max_nfev=6)
                    # Update with BA-smoothed pose
                    self.cur_R = opt_R[-1]
                    ba_t = opt_t[-1]
                    self.ba_correction_m = float(np.linalg.norm(self.cur_t - ba_t))
                    self.cur_t = ba_t

            else:
                self.status = "PNP_DEGENERATE"
        else:
            self.status = "LOST_STEREO_POINTS"

        # Update previous frame points
        new_features = self.detect_features(left_gray)
        new_pts3d, new_valid = self.triangulate_stereo_points(new_features, disparity_map)
        self.prev_pts_left = new_features[new_valid]
        self.prev_pts_3d = new_pts3d
        self.prev_left_gray = left_gray

        pos = [float(self.cur_t[0, 0]), float(self.cur_t[1, 0]), float(self.cur_t[2, 0])]
        self.trajectory.append(pos)

        return self._build_return_dict(valid_pts3d, valid_pts2d, inlier_mask, disparity_map)

    def _build_return_dict(self, pts3d, pts2d, inliers, disparity_map):
        pos = [float(self.cur_t[0, 0]), float(self.cur_t[1, 0]), float(self.cur_t[2, 0])]
        return {
            "position": pos,
            "trajectory": self.trajectory,
            "euler": self.euler_angles,
            "cur_R": self.cur_R,
            "cur_t": self.cur_t,
            "tracked_pts1": np.empty((0, 2)),
            "tracked_pts2": pts2d,
            "inliers": inliers,
            "num_tracked": self.num_tracked,
            "inlier_ratio": self.inlier_ratio,
            "distance": self.total_distance,
            "status": self.status,
            "ba_correction_m": self.ba_correction_m,
            "disparity_map": disparity_map
        }
