import numpy as np
import cv2

class RobustnessEngine:
    """
    Robustness and Quality Control Gating for Visual Odometry:
    1. Motion Blur Detection & Quality Filtering (Laplacian Variance)
    2. Parallax-Based Adaptive Keyframe Selection (Displacement gating)
    3. Degeneracy Detection (Pure Rotation vs Translation Guard)
    4. Ground Plane Metric Scale Recovery (Road Plane RANSAC)
    """
    def __init__(self, blur_threshold=15.0, min_parallax_px=1.2, 
                 rotation_degeneracy_ratio=0.85, camera_height_prior=1.65):
        self.blur_threshold = blur_threshold
        self.min_parallax_px = min_parallax_px
        self.rotation_degeneracy_ratio = rotation_degeneracy_ratio
        self.camera_height_prior = camera_height_prior

        # Diagnostics counters
        self.total_frames = 0
        self.blur_rejected_count = 0
        self.parallax_gated_count = 0
        self.degeneracy_events_count = 0
        self.last_blur_score = 0.0
        self.last_mean_parallax = 0.0
        self.is_pure_rotation = False
        self.last_estimated_scale = 1.0

    def check_blur(self, gray_frame):
        """
        Evaluates frame sharpness via Variance of the Laplacian.
        Returns:
            is_sharp (bool): True if frame has sufficient high-frequency texture.
            score (float): Sharpness metric value.
        """
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        score = float(laplacian.var())
        self.last_blur_score = score
        
        is_sharp = score >= self.blur_threshold
        if not is_sharp:
            self.blur_rejected_count += 1
        return is_sharp, score

    def check_parallax(self, pts1, pts2):
        """
        Checks if feature displacement from reference keyframe is sufficient
        to reliably triangulate 3D points and avoid degenerate epipolar geometry.
        Returns:
            has_sufficient_parallax (bool): True if motion baseline is large enough.
            mean_displacement (float): Average pixel displacement.
        """
        if len(pts1) == 0 or len(pts2) == 0:
            return False, 0.0

        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        mean_disp = float(np.mean(displacements))
        self.last_mean_parallax = mean_disp

        has_parallax = mean_disp >= self.min_parallax_px
        if not has_parallax:
            self.parallax_gated_count += 1
        return has_parallax, mean_disp

    def detect_motion_degeneracy(self, valid_curr, valid_prev, K):
        """
        Detects whether motion is degenerate (pure rotation or planar scene).
        Compares Homography residual error against Essential Matrix residual error.
        If Homography explains >= 85% of inliers without translation, flags degeneracy.
        """
        if len(valid_curr) < 12:
            return False, 1.0

        # Fit Homography (Pure Rotation / Planar)
        _, h_mask = cv2.findHomography(valid_prev, valid_curr, cv2.RANSAC, 2.0)
        h_inliers = int(np.sum(h_mask)) if h_mask is not None else 0

        # Fit Essential Matrix (Translation + Rotation)
        _, e_mask = cv2.findEssentialMat(valid_curr, valid_prev, K, cv2.RANSAC, 0.999, 1.0)
        e_inliers = int(np.sum(e_mask)) if e_mask is not None else 1

        ratio = h_inliers / max(e_inliers, 1)
        is_degenerate = (ratio >= self.rotation_degeneracy_ratio) and (ratio > 0.6)
        
        if is_degenerate:
            self.degeneracy_events_count += 1
            self.is_pure_rotation = True
        else:
            self.is_pure_rotation = False

        return is_degenerate, ratio

    def estimate_ground_plane_scale(self, pts3d):
        """
        Fits a 3D ground plane to triangulated points using RANSAC.
        Recovers physical metric scale from known camera mounting height.
        pts3d: (N, 3) points in camera frame.
        Returns:
            scale (float): Scale multiplier in meters per unit baseline.
        """
        if len(pts3d) < 30:
            return self.last_estimated_scale

        # Filter points in lower half of visual frustum (likely road plane)
        road_candidates = pts3d[(pts3d[:, 1] > 0.2) & (pts3d[:, 2] > 2.0) & (pts3d[:, 2] < 35.0)]
        if len(road_candidates) < 15:
            return self.last_estimated_scale

        # RANSAC plane fitting: n_x * X + n_y * Y + n_z * Z + d = 0
        best_inliers = 0
        best_height = None
        num_iters = 40
        thresh = 0.15

        for _ in range(num_iters):
            sample = road_candidates[np.random.choice(len(road_candidates), 3, replace=False)]
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm

            # Check if normal points roughly upward in camera frame (Y is down, so normal_Y ~ -1)
            if abs(normal[1]) < 0.6:
                continue

            d = -np.dot(normal, sample[0])
            dists = np.abs(np.dot(road_candidates, normal) + d)
            inliers = np.sum(dists < thresh)
            if inliers > best_inliers:
                best_inliers = inliers
                # Camera height is distance from camera origin (0,0,0) to plane: |d| / ||normal||
                best_height = abs(d)

        if best_height is not None and 0.2 < best_height < 5.0:
            # Scale factor: target physical height / estimated unscaled height
            scale = float(self.camera_height_prior / best_height)
            # Smooth scale update with exponential moving average
            self.last_estimated_scale = float(0.85 * self.last_estimated_scale + 0.15 * scale)
            return self.last_estimated_scale

        return self.last_estimated_scale
