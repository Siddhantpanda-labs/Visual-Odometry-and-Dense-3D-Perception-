import numpy as np
import cv2
from scipy.optimize import least_squares

class SlidingWindowBA:
    """
    Sliding-Window Local Bundle Adjustment (BA) Engine.
    Optimizes the last W camera poses (R, t) and 3D landmarks simultaneously
    to minimize Huber-weighted reprojection error:
        min_{r_i, t_i, P_j} sum_{i,j} rho( || p_ij - pi(K, R(r_i) P_j + t_i) ||^2 )
    
    Guarantees gauge freedom by fixing the first pose in the sliding window.
    """
    def __init__(self, fx, fy, cx, cy, window_size=5, max_landmarks=30):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.window_size = window_size
        self.max_landmarks = max_landmarks

        # Sliding window buffer of keyframes:
        # Each item: {'R': 3x3, 't': 3x1, 'obs': dict(landmark_id -> 2D pixel)}
        self.keyframes = []
        self.landmarks = {}  # landmark_id -> 3D point in world coordinates
        self.next_lm_id = 0

    def add_keyframe(self, R, t, pts2d, pts3d_world=None):
        """
        Adds a keyframe to the sliding window with its 2D-3D point associations.
        """
        obs = {}
        if pts3d_world is not None and len(pts3d_world) > 0:
            # Associate up to max_landmarks
            step = max(1, len(pts3d_world) // self.max_landmarks)
            for k in range(0, len(pts3d_world), step):
                lm_id = self.next_lm_id
                self.next_lm_id += 1
                self.landmarks[lm_id] = pts3d_world[k].copy()
                obs[lm_id] = pts2d[k].copy()
        
        self.keyframes.append({
            'R': R.copy(),
            't': t.copy(),
            'obs': obs
        })

        # Evict old keyframes outside sliding window
        if len(self.keyframes) > self.window_size:
            evicted = self.keyframes.pop(0)
            # Clean up unobserved landmarks
            active_ids = set()
            for kf in self.keyframes:
                active_ids.update(kf['obs'].keys())
            self.landmarks = {k: v for k, v in self.landmarks.items() if k in active_ids}

    def optimize(self, max_nfev=8):
        """
        Performs non-linear Levenberg-Marquardt optimization on the sliding window.
        Returns optimized camera poses for the keyframes.
        """
        num_kf = len(self.keyframes)
        if num_kf < 3 or len(self.landmarks) < 10:
            return [kf['R'] for kf in self.keyframes], [kf['t'] for kf in self.keyframes]

        # Parameter vector layout:
        # Poses (excluding keyframe 0 which is fixed as gauge):
        # [r_1 (3), t_1 (3), r_2 (3), t_2 (3), ..., r_{N-1} (3), t_{N-1} (3)]
        # followed by 3D landmark points:
        # [P_0 (3), P_1 (3), ..., P_{M-1} (3)]
        lm_keys = list(self.landmarks.keys())
        lm_index_map = {k: idx for idx, k in enumerate(lm_keys)}

        num_opt_poses = num_kf - 1
        x0_poses = []
        for i in range(1, num_kf):
            rvec, _ = cv2.Rodrigues(self.keyframes[i]['R'])
            tvec = self.keyframes[i]['t'].ravel()
            x0_poses.extend(rvec.ravel().tolist())
            x0_poses.extend(tvec.ravel().tolist())

        x0_landmarks = []
        for k in lm_keys:
            x0_landmarks.extend(self.landmarks[k].ravel().tolist())

        x0 = np.array(x0_poses + x0_landmarks, dtype=np.float64)

        # Precompute observations
        obs_list = []
        for kf_idx, kf in enumerate(self.keyframes):
            for lm_id, p2d in kf['obs'].items():
                if lm_id in lm_index_map:
                    obs_list.append((kf_idx, lm_index_map[lm_id], p2d[0], p2d[1]))

        if len(obs_list) < 15:
            return [kf['R'] for kf in self.keyframes], [kf['t'] for kf in self.keyframes]

        # Fixed Keyframe 0 pose
        R0 = self.keyframes[0]['R']
        t0 = self.keyframes[0]['t'].ravel()

        def residual_fn(params):
            residuals = []
            # Decode poses
            poses_R = [R0]
            poses_t = [t0]
            for i in range(num_opt_poses):
                idx = i * 6
                rvec = params[idx:idx+3]
                tvec = params[idx+3:idx+6]
                R_mat, _ = cv2.Rodrigues(rvec)
                poses_R.append(R_mat)
                poses_t.append(tvec)

            # Decode landmarks
            lm_start = num_opt_poses * 6
            landmarks_arr = params[lm_start:].reshape(-1, 3)

            for kf_idx, lm_idx, u_obs, v_obs in obs_list:
                Pw = landmarks_arr[lm_idx]
                R_curr = poses_R[kf_idx]
                t_curr = poses_t[kf_idx]

                # Transform to camera frame: P_c = R * P_w + t
                Pc = R_curr @ Pw + t_curr
                z = Pc[2]
                if z < 0.1:
                    residuals.extend([100.0, 100.0])
                    continue

                u_proj = (self.fx * Pc[0]) / z + self.cx
                v_proj = (self.fy * Pc[1]) / z + self.cy

                residuals.append(u_proj - u_obs)
                residuals.append(v_proj - v_obs)

            return np.array(residuals, dtype=np.float64)

        # Fast non-linear robust optimization with Huber loss
        try:
            res = least_squares(
                residual_fn, x0, method='trf', loss='soft_l1', f_scale=1.5,
                max_nfev=max_nfev, ftol=1e-3, xtol=1e-3
            )
            opt_params = res.x

            # Update keyframe poses
            for i in range(1, num_kf):
                idx = (i - 1) * 6
                rvec = opt_params[idx:idx+3]
                tvec = opt_params[idx+3:idx+6]
                R_opt, _ = cv2.Rodrigues(rvec)
                self.keyframes[i]['R'] = R_opt
                self.keyframes[i]['t'] = tvec.reshape(3, 1)

            # Update landmarks
            lm_start = num_opt_poses * 6
            landmarks_arr = opt_params[lm_start:].reshape(-1, 3)
            for idx, k in enumerate(lm_keys):
                self.landmarks[k] = landmarks_arr[idx]

        except Exception as e:
            pass  # Fall back to unoptimized PnP poses if singular

        return [kf['R'] for kf in self.keyframes], [kf['t'] for kf in self.keyframes]
