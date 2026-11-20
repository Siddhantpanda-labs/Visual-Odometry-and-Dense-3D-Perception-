import numpy as np
import cv2
from src.robustness import RobustnessEngine

class VisualOdometry:
    """
    Production-grade Monocular Visual Odometry (VO) Engine.
    Features:
    - Lucas-Kanade Pyramidal Optical Flow with Sub-Pixel Refinement
    - 5-Point Epipolar RANSAC (Essential Matrix E = [t]x R)
    - Cheirality Pose Recovery
    - Motion Blur Rejection Gating (Laplacian Variance)
    - Parallax-Based Adaptive Keyframing (Displacement Threshold)
    - Degeneracy Detection (Pure Rotation vs Planar Guard)
    - Dynamic Metric Scale Recovery via Ground Plane Fitting
    """
    def __init__(self, fx=718.856, fy=718.856, cx=607.1928, cy=185.2157, 
                 min_features=1200, 
                 enable_blur_rejection=True,
                 blur_threshold=15.0,
                 enable_parallax_gating=True,
                 enable_degeneracy_guard=True,
                 enable_ground_plane_scale=True,
                 scale_fixed=1.0):
        # Camera intrinsics
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        self.min_features = min_features
        self.scale_fixed = scale_fixed

        # Robustness switches
        self.enable_blur_rejection = enable_blur_rejection
        self.enable_parallax_gating = enable_parallax_gating
        self.enable_degeneracy_guard = enable_degeneracy_guard
        self.enable_ground_plane_scale = enable_ground_plane_scale

        self.robustness = RobustnessEngine(
            blur_threshold=blur_threshold,
            min_parallax_px=1.2,
            rotation_degeneracy_ratio=0.85,
            camera_height_prior=1.65
        )

        # State tracking
        self.ref_frame = None      # Keyframe reference image
        self.ref_pts = None        # Keyframe reference features
        self.prev_frame = None     # Immediate previous image
        self.prev_pts = None       # Immediate previous features
        
        # 6-DoF World Pose: X: right, Y: down, Z: forward
        self.cur_R = np.eye(3, dtype=np.float64)
        self.cur_t = np.zeros((3, 1), dtype=np.float64)
        
        # Diagnostics & Trajectory History
        self.trajectory = []
        self.euler_angles = [0.0, 0.0, 0.0]
        self.inlier_ratio = 1.0
        self.num_tracked = 0
        self.total_distance = 0.0
        self.status = "INITIALIZING"
        self.current_scale = self.scale_fixed
        self.is_keyframe = True

    def detect_features(self, gray_frame):
        """Shi-Tomasi / GFTT corner detection with sub-pixel refinement."""
        corners = cv2.goodFeaturesToTrack(
            gray_frame, 
            maxCorners=self.min_features, 
            qualityLevel=0.01, 
            minDistance=8, 
            blockSize=7
        )
        if corners is not None:
            pts = corners.reshape(-1, 2)
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            pts = cv2.cornerSubPix(gray_frame, pts, (3, 3), (-1, -1), term)
            return pts
        return np.empty((0, 2), dtype=np.float32)

    def rotation_matrix_to_euler(self, R):
        """Convert 3x3 rotation matrix to Euler angles (pitch, yaw, roll in degrees)."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees([x, y, z])

    def process_frame(self, frame):
        """
        Processes frame with full robustness pipeline.
        Returns detailed telemetry dict.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Step 1: Motion Blur Rejection Gate
        is_sharp, blur_score = self.robustness.check_blur(gray)
        if self.enable_blur_rejection and not is_sharp and self.prev_frame is not None:
            self.status = "BLUR_REJECTED"
            return self._build_return_dict(valid_prev=np.empty((0,2)), valid_curr=np.empty((0,2)), inlier_mask=np.empty(0, dtype=bool))

        # Initialization
        if self.prev_frame is None:
            self.prev_frame = gray
            self.ref_frame = gray
            self.prev_pts = self.detect_features(gray)
            self.ref_pts = np.copy(self.prev_pts)
            self.status = "INITIALIZED"
            return self._build_return_dict(valid_prev=np.empty((0,2)), valid_curr=np.empty((0,2)), inlier_mask=np.empty(0, dtype=bool))

        # Step 2: Optical Flow Tracking (KLT)
        curr_pts, status_lk, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        valid_prev = self.prev_pts[status_lk.ravel() == 1]
        valid_curr = curr_pts[status_lk.ravel() == 1]

        inlier_mask = np.zeros(len(valid_curr), dtype=bool)

        if len(valid_curr) < 12:
            self.status = "LOST_POINTS"
            self.prev_pts = self.detect_features(gray)
            self.prev_frame = gray
            return self._build_return_dict(valid_prev, valid_curr, inlier_mask)

        # Step 3: Parallax-Based Adaptive Keyframe Selection
        has_parallax, mean_disp = self.robustness.check_parallax(valid_prev, valid_curr)
        if self.enable_parallax_gating and not has_parallax:
            # Camera is stationary or moving with sub-pixel parallax; hold pose to avoid noise integration
            self.status = "PARALLAX_GATED"
            self.prev_frame = gray
            self.prev_pts = valid_curr
            self.is_keyframe = False
            return self._build_return_dict(valid_prev, valid_curr, inlier_mask)

        # Step 4: Degeneracy Guard (Pure Rotation / Planar Scene Check)
        is_degenerate, ratio = self.robustness.detect_motion_degeneracy(valid_curr, valid_prev, self.K)
        
        # Step 5: Epipolar Geometry & Pose Recovery
        E, mask = cv2.findEssentialMat(
            valid_curr, valid_prev, self.K,
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E is not None and E.shape == (3, 3):
            inliers_count, R, t, pose_mask = cv2.recoverPose(
                E, valid_curr, valid_prev, self.K, mask=mask
            )
            inlier_mask = (pose_mask.ravel() > 0)
            self.inlier_ratio = float(np.sum(inlier_mask)) / max(len(valid_curr), 1)
            self.num_tracked = int(np.sum(inlier_mask))

            # Kinematic & Inlier Verification Gates
            # Gate A: Minimum Inlier Count (prevent zero/few inlier mathematical hallucinations)
            if inliers_count < 15:
                self.status = "LOW_INLIERS"
                self.prev_pts = self.detect_features(gray)
                self.prev_frame = gray
                return self._build_return_dict(valid_prev, valid_curr, inlier_mask)

            # Gate B: Kinematic Angular Velocity Guard
            # In automotive driving, a vehicle cannot rotate > 6.0 deg per frame (150 deg/sec at 25 FPS)
            rel_rot_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            if rel_rot_angle > 6.0:
                self.status = "ROT_SPIKE_REJECTED"
                self.prev_pts = self.detect_features(gray)
                self.prev_frame = gray
                return self._build_return_dict(valid_prev, valid_curr, inlier_mask)

            # Gate C: Epipolar Hemisphere Alignment (Forward Motion Prior)
            # Optical axis Z is forward in vehicle camera frame; if t_z < 0, resolve sign ambiguity
            if t[2, 0] < 0:
                t = -t

            # Step 6: Dynamic Scale Estimation (Ground Plane RANSAC)
            if self.enable_ground_plane_scale and inliers_count >= 20:
                # Triangulate points to 3D for plane estimation
                proj1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                proj2 = self.K @ np.hstack([R, t])
                pts4d = cv2.triangulatePoints(proj1, proj2, valid_prev[inlier_mask].T, valid_curr[inlier_mask].T)
                pts3d = (pts4d[:3] / np.maximum(pts4d[3:], 1e-6)).T
                step_scale = self.robustness.estimate_ground_plane_scale(pts3d)
            else:
                step_scale = self.scale_fixed

            self.current_scale = step_scale

            # Update Pose
            if self.enable_degeneracy_guard and is_degenerate:
                # Pure rotation: update orientation only, zero translation to avoid false drift
                self.cur_R = self.cur_R @ R
                self.status = "ROTATION_ONLY"
            else:
                delta_t = step_scale * (self.cur_R @ t)
                self.cur_t = self.cur_t + delta_t
                self.cur_R = self.cur_R @ R
                self.total_distance += float(np.linalg.norm(delta_t))
                self.status = "TRACKING_OK"

            self.euler_angles = self.rotation_matrix_to_euler(self.cur_R).tolist()
            self.is_keyframe = True
        else:
            self.status = "E_DEGENERATE"

        # Keypoint replenishment
        if len(valid_curr) < self.min_features * 0.65:
            self.prev_pts = self.detect_features(gray)
        else:
            self.prev_pts = valid_curr[inlier_mask] if np.sum(inlier_mask) > 50 else valid_curr
        
        self.prev_frame = gray
        return self._build_return_dict(valid_prev, valid_curr, inlier_mask)

    def _build_return_dict(self, valid_prev, valid_curr, inlier_mask):
        pos = [float(self.cur_t[0, 0]), float(self.cur_t[1, 0]), float(self.cur_t[2, 0])]
        # Always synchronize trajectory with incoming frame index
        self.trajectory.append(pos)
        return {
            "position": pos,
            "trajectory": self.trajectory,
            "cur_R": self.cur_R,
            "cur_t": self.cur_t,
            "euler": self.euler_angles,
            "tracked_pts1": valid_prev,
            "tracked_pts2": valid_curr,
            "inliers": inlier_mask,
            "num_tracked": self.num_tracked,
            "inlier_ratio": self.inlier_ratio,
            "distance": self.total_distance,
            "status": self.status,
            "scale": self.current_scale,
            "blur_score": self.robustness.last_blur_score,
            "parallax_px": self.robustness.last_mean_parallax,
            "is_keyframe": self.is_keyframe,
            "blur_rejected_count": self.robustness.blur_rejected_count,
            "parallax_gated_count": self.robustness.parallax_gated_count,
            "degeneracy_events_count": self.robustness.degeneracy_events_count
        }
