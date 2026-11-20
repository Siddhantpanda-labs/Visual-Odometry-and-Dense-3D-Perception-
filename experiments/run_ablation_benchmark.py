import os
import sys
import time
import csv
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vo_engine import VisualOdometry
from src.stereo_engine import StereoVisualOdometry
from src.eval_metrics import compute_trajectory_metrics
from data.sample_generator import generate_benchmark_sequence

def run_ablation_study():
    """
    Executes a structured scientific benchmark across the perception evolution:
    1. Monocular VO (Baseline Naive)
    2. Monocular VO + Robustness Gating (Blur + Parallax)
    3. Stereo VO (Physical Baseline b=0.54m, No Bundle Adjustment)
    4. Stereo VO + Sliding-Window Local Bundle Adjustment (Full Production Stack)
    """
    left_path = "data/demo_driving_left.mp4"
    right_path = "data/demo_driving_right.mp4"
    gt_path = "data/ground_truth.csv"

    if not os.path.exists(left_path) or not os.path.exists(right_path) or not os.path.exists(gt_path):
        generate_benchmark_sequence(left_path, right_path, gt_path, num_frames=300)

    # Load Ground Truth
    gt_positions = []
    with open(gt_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_positions.append([float(row["x"]), float(row["y"]), float(row["z"])])

    results = []

    print("=" * 85)
    print(" EXECUTING COMPREHENSIVE VISUAL ODOMETRY BENCHMARK")
    print(" Comparing Monocular Gating vs Stereo Triangulation vs Local Bundle Adjustment")
    print("=" * 85)

    # Test 1: Monocular Naive
    print("\n[Evaluating] 1. Monocular VO (Baseline Naive)...")
    cap = cv2.VideoCapture(left_path)
    vo_mono_naive = VisualOdometry(640*0.9, 640*0.9, 320, 180, enable_blur_rejection=False, enable_parallax_gating=False, scale_fixed=0.6)
    inliers_1 = []
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret: break
        d = vo_mono_naive.process_frame(frame)
        if d["num_tracked"] > 0: inliers_1.append(d["inlier_ratio"])
    cap.release()
    fps_1 = len(vo_mono_naive.trajectory) / max(time.perf_counter() - t0, 1e-4)
    m1 = compute_trajectory_metrics(vo_mono_naive.trajectory, gt_positions)
    results.append({
        "Configuration": "1. Monocular Baseline",
        "Sensor": "Monocular", "BA": "No",
        "ATE_RMSE_m": m1["ate_rmse"], "Drift_%": m1["rpe_drift_percent"],
        "Inliers_%": float(np.mean(inliers_1))*100, "FPS": fps_1
    })

    # Test 2: Monocular + Robustness Gating
    print("[Evaluating] 2. Monocular VO + Robustness Gating (Blur + Parallax)...")
    cap = cv2.VideoCapture(left_path)
    vo_mono_robust = VisualOdometry(640*0.9, 640*0.9, 320, 180, enable_blur_rejection=True, enable_parallax_gating=True, scale_fixed=0.6)
    inliers_2 = []
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret: break
        d = vo_mono_robust.process_frame(frame)
        if d["num_tracked"] > 0: inliers_2.append(d["inlier_ratio"])
    cap.release()
    fps_2 = len(vo_mono_robust.trajectory) / max(time.perf_counter() - t0, 1e-4)
    m2 = compute_trajectory_metrics(vo_mono_robust.trajectory, gt_positions)
    results.append({
        "Configuration": "2. Monocular + Gating",
        "Sensor": "Monocular", "BA": "No",
        "ATE_RMSE_m": m2["ate_rmse"], "Drift_%": m2["rpe_drift_percent"],
        "Inliers_%": float(np.mean(inliers_2))*100, "FPS": fps_2
    })

    # Test 3: Stereo VO (No BA)
    print("[Evaluating] 3. Stereo VO (Baseline b=0.54m, No BA)...")
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    vo_stereo_noba = StereoVisualOdometry(640*0.9, 640*0.9, 320, 180, baseline=0.54, enable_ba=False)
    inliers_3 = []
    t0 = time.perf_counter()
    while True:
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r: break
        d = vo_stereo_noba.process_frame(f_l, f_r)
        if d["num_tracked"] > 0: inliers_3.append(d["inlier_ratio"])
    cap_l.release()
    cap_r.release()
    fps_3 = len(vo_stereo_noba.trajectory) / max(time.perf_counter() - t0, 1e-4)
    m3 = compute_trajectory_metrics(vo_stereo_noba.trajectory, gt_positions)
    results.append({
        "Configuration": "3. Stereo VO (No BA)",
        "Sensor": "Stereo (b=0.54m)", "BA": "No",
        "ATE_RMSE_m": m3["ate_rmse"], "Drift_%": m3["rpe_drift_percent"],
        "Inliers_%": float(np.mean(inliers_3))*100, "FPS": fps_3
    })

    # Test 4: Stereo VO + Sliding-Window Local Bundle Adjustment
    print("[Evaluating] 4. Stereo VO + Sliding-Window Local Bundle Adjustment...")
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    vo_stereo_ba = StereoVisualOdometry(640*0.9, 640*0.9, 320, 180, baseline=0.54, enable_ba=True, window_size=5)
    inliers_4 = []
    t0 = time.perf_counter()
    while True:
        ret_l, f_l = cap_l.read()
        ret_r, f_r = cap_r.read()
        if not ret_l or not ret_r: break
        d = vo_stereo_ba.process_frame(f_l, f_r)
        if d["num_tracked"] > 0: inliers_4.append(d["inlier_ratio"])
    cap_l.release()
    cap_r.release()
    fps_4 = len(vo_stereo_ba.trajectory) / max(time.perf_counter() - t0, 1e-4)
    m4 = compute_trajectory_metrics(vo_stereo_ba.trajectory, gt_positions)
    results.append({
        "Configuration": "4. Stereo VO + Local BA",
        "Sensor": "Stereo (b=0.54m)", "BA": "Yes (W=5)",
        "ATE_RMSE_m": m4["ate_rmse"], "Drift_%": m4["rpe_drift_percent"],
        "Inliers_%": float(np.mean(inliers_4))*100, "FPS": fps_4
    })

    # Print Formatted Comparison Table
    print("\n" + "=" * 90)
    print(" COMPREHENSIVE BENCHMARK RESULTS (DEFENSE & PUBLICATION TABLE)")
    print("=" * 90)
    header = f"{'Configuration':<28} | {'Sensor':<17} | {'BA':<9} | {'ATE RMSE':<10} | {'Drift %':<8} | {'FPS'}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['Configuration']:<28} | {r['Sensor']:<17} | {r['BA']:<9} | {r['ATE_RMSE_m']:<9.2f}m | {r['Drift_%']:<7.2f}% | {r['FPS']:.1f}")
    print("=" * 90)

    # Save CSV
    out_csv = "outputs/ablation_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Benchmark] Results saved to: {os.path.abspath(out_csv)}")
    return results

if __name__ == "__main__":
    run_ablation_study()
