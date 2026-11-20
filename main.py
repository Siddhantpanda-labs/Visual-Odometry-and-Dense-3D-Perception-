import argparse
import os
import sys
import time
import csv
import cv2
import numpy as np

from src.vo_engine import VisualOdometry
from src.stereo_engine import StereoVisualOdometry
from src.depth_engine import DepthEstimator, CleanStereoDepthEngine
from src.visualizer import PresentationDashboard
from src.point_cloud import PointCloudExporter
from data.sample_generator import generate_benchmark_sequence

def main():
    parser = argparse.ArgumentParser(description="Visual Odometry & 3D Reconstruction Perception Framework")
    parser.add_argument("--source", type=str, default="demo", 
                        help="Video file path, camera index (e.g. '0'), or 'demo'")
    parser.add_argument("--source_right", type=str, default=None,
                        help="Right camera video path for stereo mode (optional)")
    parser.add_argument("--sensor", type=str, default="auto", choices=["auto", "mono", "stereo"],
                        help="Sensor mode: 'mono', 'stereo', or 'auto'")
    parser.add_argument("--enable_ba", action="store_true", default=True,
                        help="Enable Sliding-Window Local Bundle Adjustment (for stereo)")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "benchmark"],
                        help="Operating mode: 'demo' for live 4-pane visualizer, 'benchmark' for comprehensive evaluation")
    parser.add_argument("--export_ply", type=str, default="outputs/scene_pointcloud.ply",
                        help="Path to export 3D point cloud file (.ply) for MeshLab / 3D Viewer")
    parser.add_argument("--export_html", type=str, default="outputs/reconstruction_3d.html",
                        help="Path to export interactive 3D WebGL viewer (.html)")
    parser.add_argument("--save", type=str, default="outputs/presentation_demo.mp4",
                        help="Output video path for recorded 4-pane presentation dashboard")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI window")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum frames to process (0 = full video duration)")
    args = parser.parse_args()

    if args.mode == "benchmark":
        from experiments.run_ablation_benchmark import run_ablation_study
        run_ablation_study()
        return

    gt_positions = []

    # Configure input sources
    if args.source == "demo":
        left_path = "data/demo_driving_left.mp4"
        right_path = "data/demo_driving_right.mp4"
        gt_path = "data/ground_truth.csv"

        if not os.path.exists(left_path) or not os.path.exists(right_path) or not os.path.exists(gt_path):
            generate_benchmark_sequence(left_path, right_path, gt_path, num_frames=450)

        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gt_positions.append([float(row["x"]), float(row["y"]), float(row["z"])])
        sensor_mode = "stereo" if args.sensor in ["auto", "stereo"] else "mono"
    else:
        left_path = int(args.source) if args.source.isdigit() else args.source
        right_path = args.source_right
        if args.sensor == "auto":
            sensor_mode = "stereo" if right_path is not None else "mono"
        else:
            sensor_mode = args.sensor

    cap_left = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path) if (sensor_mode == "stereo" and right_path is not None) else None

    if not cap_left.isOpened():
        print(f"[Error] Failed to open primary input stream: {left_path}")
        return

    width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    total_input_frames = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps): fps = 25.0

    print("=" * 75)
    print(" VISUAL ODOMETRY & 3D RECONSTRUCTION PERCEPTION FRAMEWORK")
    print(f" Sensor Mode:       {sensor_mode.upper()} VISUAL ODOMETRY")
    print(f" Bundle Adjustment: {'ENABLED (Sliding-Window W=5)' if (sensor_mode=='stereo' and args.enable_ba) else 'DISABLED'}")
    print(f" Video Stream:      {left_path} ({width}x{height} @ {fps:.1f} FPS, {total_input_frames} frames)")
    print(f" Output Video:      {args.save} (Processing 100% full video length)")
    print(f" 3D Point Cloud:    {args.export_ply}")
    print(f" 3D WebGL Viewer:   {args.export_html}")
    print("=" * 75)

    fx = fy = width * 0.95
    cx, cy = width / 2.0, height * 0.46

    if sensor_mode == "stereo":
        vo = StereoVisualOdometry(fx, fy, cx, cy, baseline=0.54, enable_ba=args.enable_ba, window_size=5)
        stereo_depth_engine = CleanStereoDepthEngine(fx=fx, baseline=0.54)
    else:
        vo = VisualOdometry(fx, fy, cx, cy, enable_blur_rejection=True, enable_parallax_gating=True, scale_fixed=0.65)
        mono_depth_engine = DepthEstimator(model_type="fast")

    dashboard = PresentationDashboard(out_width=1280, out_height=720)
    pc_exporter = PointCloudExporter(fx, fy, cx, cy, sample_step=6) if (args.export_ply or args.export_html) else None

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save, fourcc, fps, (1280, 720))

    frame_idx = 0
    t_start = time.perf_counter()
    fps_smooth = 30.0

    try:
        while True:
            ret_l, frame_l = cap_left.read()
            if not ret_l:
                print("\n[Stream] Full input video duration processed.")
                break
            if cap_right is not None:
                ret_r, frame_r = cap_right.read()
                if not ret_r: break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            t0 = time.perf_counter()

            # Process frame through VO & Depth Engines
            if sensor_mode == "stereo":
                vo_data = vo.process_frame(frame_l, frame_r)
                # Compute clean, noise-free filtered stereo depth
                depth_norm, depth_color, infer_ms, obstacle_alert = stereo_depth_engine.compute_depth(frame_l, frame_r)
            else:
                vo_data = vo.process_frame(frame_l)
                depth_norm, depth_color, infer_ms, obstacle_alert = mono_depth_engine.estimate(frame_l)

            # 3D Point Cloud accumulation (adaptive keyframe interval)
            pc_interval = 6 if total_input_frames > 500 else 3
            if pc_exporter and frame_idx % pc_interval == 0:
                pc_exporter.add_frame(frame_l, depth_norm, vo_data["cur_R"], vo_data["cur_t"])

            t1 = time.perf_counter()
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / max(t1 - t0, 1e-4))

            # Render 4-Pane Dashboard
            gt_slice = gt_positions[:frame_idx] if gt_positions else None
            composite = dashboard.render(
                frame_l, depth_color, vo_data, infer_ms, fps_smooth,
                obstacle_alert=obstacle_alert, gt_trajectory=gt_slice,
                is_stereo=(sensor_mode == "stereo")
            )

            if writer:
                writer.write(composite)

            if not args.headless:
                cv2.imshow("Visual Odometry & 3D Reconstruction Demo", composite)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("\n[User] Terminated early by user.")
                    break
                elif key == 32:
                    cv2.waitKey(0)

            pos = vo_data["position"]
            progress_pct = (frame_idx / max(total_input_frames, 1)) * 100.0
            sys.stdout.write(
                f"\rFrame {frame_idx:04d}/{total_input_frames:04d} ({progress_pct:4.1f}%) | "
                f"Pos: [{pos[0]:+5.1f}, {pos[1]:+5.1f}, {pos[2]:+5.1f}]m | "
                f"Dist: {vo_data['distance']:5.1f}m | "
                f"Inliers: {vo_data['num_tracked']:3d} | "
                f"FPS: {fps_smooth:4.1f}"
            )
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[User] Interrupted by user.")
    finally:
        cap_left.release()
        if cap_right: cap_right.release()
        if writer: writer.release()
        if not args.headless: cv2.destroyAllWindows()

    if pc_exporter:
        if args.export_ply:
            pc_exporter.export_ply(args.export_ply)
        if args.export_html:
            pc_exporter.export_interactive_html(args.export_html)
        pc_exporter.export_trajectory_plot("outputs/trajectory_3d_plot.png")

    t_total = time.perf_counter() - t_start
    avg_fps = frame_idx / max(t_total, 1e-4)

    print("\n" + "=" * 75)
    print(" EXECUTION COMPLETE - RECONSTRUCTION & PRESENTATION REPORT")
    print("=" * 75)
    print(f" Mode:                     {sensor_mode.upper()} VO")
    print(f" Total Frames Processed:   {frame_idx} (100% of input video duration)")
    print(f" Total Trajectory Length:  {vo.total_distance:.2f} meters")
    print(f" Final Position (X, Y, Z): [{vo.cur_t[0,0]:+.2f}, {vo.cur_t[1,0]:+.2f}, {vo.cur_t[2,0]:+.2f}] m")
    print(f" Average Framerate:        {avg_fps:.1f} FPS")
    if args.save:
        print(f" Presentation Video:       {os.path.abspath(args.save)}")
    if args.export_ply:
        print(f" 3D Point Cloud (.ply):    {os.path.abspath(args.export_ply)}")
    if args.export_html:
        print(f" Interactive 3D Web Viewer:{os.path.abspath(args.export_html)}")
    print(f" 3D Trajectory Plot:       {os.path.abspath('outputs/trajectory_3d_plot.png')}")
    print("=" * 75)

if __name__ == "__main__":
    main()
