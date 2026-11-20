import numpy as np

def umeyama_alignment(X, Y):
    """
    Computes optimal Sim(3) alignment (scale s, rotation R, translation t)
    between estimated trajectory X and ground truth Y:
    min_{s, R, t} || Y - (s * R * X + t) ||^2
    This is the standard evaluation method for monocular VO benchmarks (TUM, KITTI, evo).
    """
    n, m = X.shape
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    sig_x = np.mean(np.sum((X - mu_x)**2, axis=1))
    if sig_x < 1e-9:
        return 1.0, np.eye(3), np.zeros(3)

    Sigma = (Y - mu_y).T @ (X - mu_x) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(m)
    if np.linalg.det(Sigma) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = (1.0 / sig_x) * np.trace(np.diag(D) @ S)
    t = mu_y - s * (R @ mu_x)
    return s, R, t

def compute_trajectory_metrics(est_traj, gt_traj):
    """
    Computes standard Visual Odometry & SLAM benchmark evaluation metrics:
    - Absolute Trajectory Error (ATE RMSE, Mean, Max in meters) via Sim(3) Umeyama alignment
    - Relative Pose Error (RPE): Translational drift percentage
    """
    est = np.asarray(est_traj, dtype=np.float64)
    gt = np.asarray(gt_traj, dtype=np.float64)

    min_len = min(len(est), len(gt))
    if min_len < 5:
        return {
            "ate_rmse": 0.0, "ate_mean": 0.0, "ate_max": 0.0,
            "rpe_drift_percent": 0.0, "total_distance_m": 0.0,
            "optimal_scale": 1.0
        }

    est = est[:min_len]
    gt = gt[:min_len]

    # Standard Sim(3) Alignment (Umeyama algorithm)
    s, R, t = umeyama_alignment(est, gt)
    est_aligned = (s * (R @ est.T)).T + t

    # Euclidean distance errors per frame
    errors = np.linalg.norm(est_aligned - gt, axis=1)

    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))
    ate_median = float(np.median(errors))
    ate_max = float(np.max(errors))

    gt_deltas = np.linalg.norm(np.diff(gt, axis=0), axis=1)
    total_distance = float(np.sum(gt_deltas))
    rpe_drift_percent = (ate_rmse / max(total_distance, 1e-3)) * 100.0

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "ate_median": ate_median,
        "ate_max": ate_max,
        "total_distance_m": total_distance,
        "rpe_drift_percent": rpe_drift_percent,
        "optimal_scale": float(s)
    }
