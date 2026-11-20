import cv2
import numpy as np

class PresentationDashboard:
    """
    Synchronized 4-Pane Presentation Dashboard:
    - [Top-Left] RGB Feed with KLT Feature Flow & Inliers
    - [Top-Right] Stereo Disparity & Dense Metric Depth Map (Turbo/Inferno)
    - [Bottom-Left] Real-Time Top-Down Bird's-Eye-View (BEV) Trajectory vs Ground Truth
    - [Bottom-Right] Telemetry Cockpit & Metric Tracking HUD
    """
    def __init__(self, out_width=1280, out_height=720):
        self.out_width = out_width
        self.out_height = out_height
        self.pane_w = out_width // 2
        self.pane_h = out_height // 2

        self.bev_scale = 1.8  # pixels per meter (auto-scaling)
        self.bev_origin = (self.pane_w // 2, int(self.pane_h * 0.82))

    def _draw_rgb_pane(self, frame_bgr, vo_data, is_stereo=True):
        pane = cv2.resize(frame_bgr, (self.pane_w, self.pane_h))
        sx = self.pane_w / frame_bgr.shape[1]
        sy = self.pane_h / frame_bgr.shape[0]

        pts1 = vo_data.get("tracked_pts1", np.empty((0, 2)))
        pts2 = vo_data.get("tracked_pts2", np.empty((0, 2)))
        inliers = vo_data.get("inliers", np.empty(0, dtype=bool))

        if len(pts2) > 0:
            for i in range(min(len(pts2), 260)):
                p2 = (int(pts2[i][0] * sx), int(pts2[i][1] * sy))
                is_inlier = inliers[i] if i < len(inliers) else True
                
                if is_inlier:
                    color = (0, 255, 128)
                    if len(pts1) > i and len(pts1[i]) == 2:
                        p1 = (int(pts1[i][0] * sx), int(pts1[i][1] * sy))
                        cv2.line(pane, p1, p2, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.circle(pane, p2, 3, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(pane, p2, 2, (70, 70, 220), -1, cv2.LINE_AA)

        # Header Badge
        cv2.rectangle(pane, (10, 10), (330, 42), (0, 0, 0), -1)
        badge_title = "1. STEREO LEFT CAM & KLT FLOW" if is_stereo else "1. RGB VIDEO & KLT FLOW"
        cv2.rectangle(pane, (10, 10), (330, 42), (0, 255, 128), 1)
        cv2.putText(pane, badge_title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Inlier HUD
        count_str = f"Inliers: {vo_data.get('num_tracked', 0)} pts ({vo_data.get('inlier_ratio', 1.0)*100:.1f}%)"
        cv2.putText(pane, count_str, (15, self.pane_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        return pane

    def _draw_depth_pane(self, depth_color, infer_ms, obstacle_alert, is_stereo=True):
        pane = cv2.resize(depth_color, (self.pane_w, self.pane_h))

        # Header Badge
        cv2.rectangle(pane, (10, 10), (350, 42), (0, 0, 0), -1)
        badge_title = "2. STEREO DISPARITY & METRIC DEPTH" if is_stereo else "2. MONOCULAR DENSE DEPTH"
        badge_col = (0, 200, 255) if is_stereo else (255, 100, 0)
        cv2.rectangle(pane, (10, 10), (350, 42), badge_col, 1)
        cv2.putText(pane, badge_title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # Obstacle Hazard Indicator
        if obstacle_alert:
            cv2.rectangle(pane, (self.pane_w//2 - 95, 12), (self.pane_w//2 + 95, 40), (0, 0, 200), -1)
            cv2.putText(pane, "! PROXIMITY HAZARD !", (self.pane_w//2 - 85, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # Distance Scale Legend Bar
        bar_x, bar_y, bar_w, bar_h = self.pane_w - 200, self.pane_h - 30, 160, 12
        gradient = np.linspace(0, 255, bar_w, dtype=np.uint8)
        grad_bar = np.tile(gradient, (bar_h, 1))
        cmap = cv2.COLORMAP_TURBO if is_stereo else cv2.COLORMAP_INFERNO
        grad_color = cv2.applyColorMap(grad_bar, cmap)
        pane[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = grad_color
        cv2.rectangle(pane, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (255, 255, 255), 1)
        cv2.putText(pane, "FAR", (bar_x - 30, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(pane, "NEAR (1.5m)", (bar_x + bar_w + 5, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        depth_tag = "Stereo Triangulation (b=0.54m)" if is_stereo else f"Latency: {infer_ms:.1f} ms"
        cv2.putText(pane, depth_tag, (15, self.pane_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return pane

    def _draw_bev_pane(self, trajectory, cur_euler, gt_trajectory=None):
        pane = np.full((self.pane_h, self.pane_w, 3), 20, dtype=np.uint8)

        # Metric Grid
        grid_color = (38, 42, 45)
        for gx in range(0, self.pane_w, 40):
            cv2.line(pane, (gx, 0), (gx, self.pane_h), grid_color, 1)
        for gy in range(0, self.pane_h, 40):
            cv2.line(pane, (0, gy), (self.pane_w, gy), grid_color, 1)

        ox, oz = self.bev_origin
        cv2.circle(pane, (ox, oz), 4, (0, 255, 0), -1)
        cv2.putText(pane, "START (0,0)", (ox + 8, oz + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # Ground Truth Trajectory (Gray line)
        if gt_trajectory is not None and len(gt_trajectory) > 1:
            gt_pts = []
            for gp in gt_trajectory:
                px = int(ox + gp[0] * self.bev_scale)
                py = int(oz - gp[2] * self.bev_scale)
                gt_pts.append((px, py))
            for i in range(1, len(gt_pts), 2):
                cv2.line(pane, gt_pts[i-1], gt_pts[i], (120, 120, 120), 1, cv2.LINE_AA)

        # Estimated Trajectory (Vivid Yellow line)
        if len(trajectory) > 1:
            pts_bev = []
            for p in trajectory:
                px = int(ox + p[0] * self.bev_scale)
                py = int(oz - p[2] * self.bev_scale)
                pts_bev.append((px, py))

            # Auto-zoom scale
            last_px, last_py = pts_bev[-1]
            if last_px < 35 or last_px > self.pane_w - 35 or last_py < 35 or last_py > self.pane_h - 35:
                self.bev_scale = max(0.35, self.bev_scale * 0.96)

            for i in range(1, len(pts_bev)):
                cv2.line(pane, pts_bev[i-1], pts_bev[i], (0, 215, 255), 2, cv2.LINE_AA)

            # Camera Frustum Heading
            yaw = cur_euler[1]
            rad = np.radians(yaw)
            tip_x = int(last_px + 16 * np.sin(rad))
            tip_y = int(last_py - 16 * np.cos(rad))
            cv2.circle(pane, (last_px, last_py), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.line(pane, (last_px, last_py), (tip_x, tip_y), (0, 255, 255), 2, cv2.LINE_AA)

        # Header Badge
        cv2.rectangle(pane, (10, 10), (330, 42), (0, 0, 0), -1)
        cv2.rectangle(pane, (10, 10), (330, 42), (255, 200, 0), 1)
        cv2.putText(pane, "3. BIRD'S-EYE TRAJECTORY (BEV)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(pane, f"Scale: {self.bev_scale:.1f} px/m", (self.pane_w - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
        return pane

    def _draw_telemetry_pane(self, vo_data, total_fps, is_stereo=True):
        pane = np.full((self.pane_h, self.pane_w, 3), 16, dtype=np.uint8)

        # Header Badge
        cv2.rectangle(pane, (10, 10), (330, 42), (0, 0, 0), -1)
        cv2.rectangle(pane, (10, 10), (330, 42), (0, 200, 255), 1)
        cv2.putText(pane, "4. TELEMETRY & 6-DOF HUD", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        pos = vo_data.get("position", [0.0, 0.0, 0.0])
        euler = vo_data.get("euler", [0.0, 0.0, 0.0])
        status = vo_data.get("status", "ACTIVE")
        dist = vo_data.get("distance", 0.0)

        status_color = (0, 255, 128) if "OK" in status or status == "INITIALIZED" else (0, 165, 255)
        cv2.circle(pane, (self.pane_w - 180, 28), 5, status_color, -1)
        cv2.putText(pane, status, (self.pane_w - 165, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1, cv2.LINE_AA)

        # Card 1: 3D World Coordinates
        cv2.rectangle(pane, (20, 55), (self.pane_w - 20, 125), (28, 30, 36), -1)
        cv2.rectangle(pane, (20, 55), (self.pane_w - 20, 125), (45, 50, 60), 1)
        cv2.putText(pane, "CAMERA POSITION (WORLD FRAME, METERS)", (32, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        pos_str = f"X: {pos[0]:+6.2f}m   |   Y: {pos[1]:+6.2f}m   |   Z: {pos[2]:+6.2f}m"
        cv2.putText(pane, pos_str, (32, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        # Card 2: Attitude & Heading
        cv2.rectangle(pane, (20, 135), (self.pane_w - 20, 205), (28, 30, 36), -1)
        cv2.rectangle(pane, (20, 135), (self.pane_w - 20, 205), (45, 50, 60), 1)
        cv2.putText(pane, "ORIENTATION & ATTITUDE (DEGREES)", (32, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        att_str = f"PITCH: {euler[0]:+5.1f} deg  |  YAW: {euler[1]:+5.1f} deg  |  ROLL: {euler[2]:+5.1f} deg"
        cv2.putText(pane, att_str, (32, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 50), 1, cv2.LINE_AA)

        # Card 3: Performance & Scale
        cv2.rectangle(pane, (20, 215), (self.pane_w - 20, 340), (28, 30, 36), -1)
        cv2.rectangle(pane, (20, 215), (self.pane_w - 20, 340), (45, 50, 60), 1)
        cv2.putText(pane, "PERFORMANCE & MOTION DIAGNOSTICS", (32, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        cv2.putText(pane, f"Distance Traveled: {dist:.1f} m", (32, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        scale_label = "Baseline b: 0.54 m (Metric)" if is_stereo else f"Scale: {vo_data.get('scale', 1.0):.2f}x"
        cv2.putText(pane, scale_label, (32, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
        ba_correction = vo_data.get("ba_correction_m", 0.0)
        ba_label = f"Local BA Smoothing: {ba_correction:.3f}m" if is_stereo else f"Parallax: {vo_data.get('parallax_px', 0.0):.1f} px"
        cv2.putText(pane, ba_label, (32, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.putText(pane, f"Framerate: {total_fps:.1f} FPS", (self.pane_w // 2 + 15, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)
        cv2.putText(pane, f"Motion Model: 3D-to-2D PnP" if is_stereo else "5-Pt Epipolar RANSAC", (self.pane_w // 2 + 15, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(pane, f"Track Status: NOMINAL", (self.pane_w // 2 + 15, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1)

        return pane

    def render(self, frame_bgr, depth_color, vo_data, infer_ms, total_fps, obstacle_alert=False, gt_trajectory=None, is_stereo=True):
        p1 = self._draw_rgb_pane(frame_bgr, vo_data, is_stereo)
        p2 = self._draw_depth_pane(depth_color, infer_ms, obstacle_alert, is_stereo)
        p3 = self._draw_bev_pane(vo_data.get("trajectory", []), vo_data.get("euler", [0, 0, 0]), gt_trajectory)
        p4 = self._draw_telemetry_pane(vo_data, total_fps, is_stereo)

        top_row = np.hstack([p1, p2])
        bot_row = np.hstack([p3, p4])
        canvas = np.vstack([top_row, bot_row])

        cv2.line(canvas, (self.pane_w, 0), (self.pane_w, self.out_height), (50, 55, 60), 2)
        cv2.line(canvas, (0, self.pane_h), (self.out_width, self.pane_h), (50, 55, 60), 2)
        return canvas
