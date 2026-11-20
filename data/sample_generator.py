import numpy as np
import cv2
import math
import csv
import os

def generate_benchmark_sequence(output_video_left="data/demo_driving_left.mp4",
                                output_video_right="data/demo_driving_right.mp4",
                                output_gt="data/ground_truth.csv",
                                num_frames=450, width=640, height=360, fps=30,
                                baseline=0.54, inject_blur=True, inject_stops=True):
    """
    Generates a high-detail 3D perspective STEREO driving sequence:
    - Left and Right camera streams offset by physical baseline b = 0.54m
    - High-detail asphalt road texture, curbs, dashed & solid lane markings, crosswalks
    - Multi-story buildings with windows, street lamps, traffic signs, lush foliage
    - Distant multi-layer mountain terrain and gradient sky
    - Realistic vehicle suspension pitch/roll dynamics and smooth curves
    - 6-DoF Ground Truth trajectory output
    """
    os.makedirs(os.path.dirname(output_video_left), exist_ok=True)
    os.makedirs(os.path.dirname(output_video_right), exist_ok=True)
    os.makedirs(os.path.dirname(output_gt), exist_ok=True)

    print(f"[SampleGenerator] Rendering high-detail STEREO driving sequence ({num_frames} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_left = cv2.VideoWriter(output_video_left, fourcc, fps, (width, height))
    out_right = cv2.VideoWriter(output_video_right, fourcc, fps, (width, height))

    cx, cy = width / 2.0, height * 0.46
    fx, fy = width * 0.95, width * 0.95
    cam_h = 1.65  # mounting height in meters

    # Generate 3D landmarks (Buildings, Trees, Signs, Lamps)
    landmarks = []
    # Continuous city corridor (Z: 4m to 600m)
    for z in range(6, 650, 6):
        # Roadside Streetlamps with overhang
        if z % 18 == 0:
            landmarks.append((-4.2, 0.0, float(z), 'lamp_left'))
            landmarks.append((4.2, 0.0, float(z), 'lamp_right'))
        # Traffic / Warning signs
        if z % 36 == 0:
            landmarks.append((4.0, 0.0, float(z), 'sign'))
        # Trees & Greenery
        landmarks.append((-6.2, 0.0, float(z), 'tree_large'))
        landmarks.append((6.2, 0.0, float(z), 'tree_large'))
        landmarks.append((-8.0, 0.0, float(z) + 3, 'tree_small'))
        landmarks.append((8.0, 0.0, float(z) + 3, 'tree_small'))
        # Architecture / Buildings
        if z % 18 == 0:
            b_height = 8.0 + (z % 5) * 2.5
            landmarks.append((-13.5, 0.0, float(z), 'building', b_height, (90, 85, 95)))
            landmarks.append((13.5, 0.0, float(z), 'building', b_height + 2.0, (85, 90, 100)))

    ego_x = 0.0
    ego_z = 0.0
    ego_yaw = 0.0
    nominal_speed = 0.65  # ~70 km/h

    gt_records = []

    for f in range(num_frames):
        # Controlled driving profile:
        # Start smooth -> Gentle left curve -> Red-light pause -> Gentle right curve -> Accelerate
        if inject_stops and 180 <= f <= 215:
            current_speed = 0.0
        else:
            current_speed = nominal_speed

        if 50 < f < 150:
            ego_yaw += 0.0028
        elif 240 < f < 350:
            ego_yaw -= 0.0032

        # Small suspension pitch & roll from engine/road dynamics
        pitch_wobble = 0.004 * math.sin(f * 0.15) if current_speed > 0 else 0.0
        roll_wobble = 0.002 * math.cos(f * 0.18) if current_speed > 0 else 0.0

        ego_x += current_speed * math.sin(ego_yaw)
        ego_z += current_speed * math.cos(ego_yaw)

        gt_records.append([f, ego_x, 0.0, ego_z, math.degrees(ego_yaw)])

        # Right camera rigid baseline offset
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)
        right_cam_x = ego_x + baseline * cos_y
        right_cam_z = ego_z - baseline * sin_y

        def render_frame_view(cam_x, cam_z):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Sky gradient
            horizon = int(cy + pitch_wobble * fy)
            for y in range(horizon):
                t = y / max(horizon, 1)
                b = int(245 * (1 - t*0.45))
                g = int(190 * (1 - t*0.35))
                r = int(130 * (1 - t*0.25))
                frame[y, :] = (b, g, r)

            # Layered Mountain Backdrop
            for mx in range(0, width, 2):
                mh1 = int(18 * math.sin(mx * 0.015) + 9 * math.cos(mx * 0.04 + 0.8))
                mh2 = int(10 * math.sin(mx * 0.03 + 1.2) + 5 * math.cos(mx * 0.07))
                frame[max(0, horizon - mh1):horizon, mx:mx+2] = (110, 95, 90)
                frame[max(0, horizon - mh2):horizon, mx:mx+2] = (85, 75, 75)

            # Ground & Road
            frame[horizon:, :] = (50, 52, 55)  # dark asphalt

            def project_point(X, Y, Z):
                dx = X - cam_x
                dz = Z - cam_z
                # Apply yaw and small roll
                c_rot = math.cos(-ego_yaw)
                s_rot = math.sin(-ego_yaw)
                c_x = dx * c_rot - dz * s_rot
                c_z = dx * s_rot + dz * c_rot
                c_y = (Y + cam_h) + roll_wobble * c_x

                if c_z <= 0.6:
                    return None
                u = int(cx + (c_x * fx) / c_z)
                v = int(cy + (c_y * fy) / c_z)
                return (u, v, c_z)

            # Sidewalk Curbs
            for z_side in range(int(cam_z) - 5, int(cam_z) + 130, 2):
                if z_side <= cam_z: continue
                p_cleft1 = project_point(-4.1, 0.15, z_side)
                p_cleft2 = project_point(-4.1, 0.15, z_side + 1.8)
                if p_cleft1 and p_cleft2:
                    cv2.line(frame, (p_cleft1[0], p_cleft1[1]), (p_cleft2[0], p_cleft2[1]), (130, 130, 135), 3)

                p_cright1 = project_point(4.1, 0.15, z_side)
                p_cright2 = project_point(4.1, 0.15, z_side + 1.8)
                if p_cright1 and p_cright2:
                    cv2.line(frame, (p_cright1[0], p_cright1[1]), (p_cright2[0], p_cright2[1]), (130, 130, 135), 3)

            # Road Markings (Solid White Borders, Yellow Dashed Centerline)
            for z_m in range(int(cam_z) - 5, int(cam_z) + 130, 2):
                if z_m <= cam_z: continue
                # Outer White Lines
                p_l1 = project_point(-3.5, 0.0, z_m)
                p_l2 = project_point(-3.5, 0.0, z_m + 1.8)
                if p_l1 and p_l2:
                    cv2.line(frame, (p_l1[0], p_l1[1]), (p_l2[0], p_l2[1]), (230, 230, 230), 2)

                p_r1 = project_point(3.5, 0.0, z_m)
                p_r2 = project_point(3.5, 0.0, z_m + 1.8)
                if p_r1 and p_r2:
                    cv2.line(frame, (p_r1[0], p_r1[1]), (p_r2[0], p_r2[1]), (230, 230, 230), 2)

                # Dashed Center Lane (every 4 meters)
                if z_m % 4 == 0:
                    p_c1 = project_point(0.0, 0.0, z_m)
                    p_c2 = project_point(0.0, 0.0, z_m + 2.2)
                    if p_c1 and p_c2:
                        cv2.line(frame, (p_c1[0], p_c1[1]), (p_c2[0], p_c2[1]), (0, 215, 255), 2)

            # Crosswalk at frame 170 (before the red light stop)
            for xw_z in [112.0, 113.2, 114.4]:
                if cam_z - 2 < xw_z < cam_z + 80:
                    for xw_x in np.arange(-3.0, 3.2, 0.8):
                        p1 = project_point(xw_x, 0.0, xw_z)
                        p2 = project_point(xw_x, 0.0, xw_z + 0.7)
                        if p1 and p2:
                            cv2.line(frame, (p1[0], p1[1]), (p2[0], p2[1]), (240, 240, 240), 4)

            # Project Landmarks
            vis = []
            for item in landmarks:
                lx, ly, lz, ltype = item[0], item[1], item[2], item[3]
                if cam_z - 2 < lz < cam_z + 115:
                    p = project_point(lx, ly, lz)
                    if p and 0 <= p[0] < width and 0 <= p[1] < height:
                        vis.append((p[2], p[0], p[1], ltype, item))

            vis.sort(key=lambda x: x[0], reverse=True)

            for depth, u, v, ltype, item in vis:
                s = min(max(fx / (depth * 25.0), 1.5), 65.0)

                if ltype == 'lamp_left' or ltype == 'lamp_right':
                    h_lamp = int(s * 4.2)
                    cv2.line(frame, (u, v), (u, max(0, v - h_lamp)), (175, 175, 180), max(1, int(s * 0.16)))
                    # Curved arm
                    arm_dir = 1 if ltype == 'lamp_left' else -1
                    cv2.line(frame, (u, max(0, v - h_lamp)), (u + arm_dir * int(s * 1.2), max(0, v - h_lamp)), (175, 175, 180), 2)
                    cv2.circle(frame, (u + arm_dir * int(s * 1.2), max(0, v - h_lamp)), int(s * 0.3), (0, 255, 255), -1)

                elif ltype == 'sign':
                    h_post = int(s * 2.8)
                    cv2.line(frame, (u, v), (u, max(0, v - h_post)), (160, 160, 160), max(1, int(s * 0.12)))
                    # Yellow warning diamond sign
                    r_sign = int(s * 0.7)
                    pts_d = np.array([[u, v - h_post - r_sign], [u + r_sign, v - h_post], [u, v - h_post + r_sign], [u - r_sign, v - h_post]])
                    cv2.fillPoly(frame, [pts_d], (0, 215, 255))
                    cv2.polylines(frame, [pts_d], True, (0, 0, 0), 1)

                elif ltype == 'tree_large':
                    h_tree = int(s * 4.5)
                    r_f = int(s * 2.0)
                    cv2.line(frame, (u, v), (u, max(0, v - int(h_tree * 0.6))), (35, 55, 75), max(2, int(s * 0.22)))
                    cv2.circle(frame, (u, max(0, v - h_tree)), r_f, (30, 110, 40), -1)
                    cv2.circle(frame, (u - int(r_f*0.3), max(0, v - h_tree - int(r_f*0.2))), int(r_f * 0.7), (40, 140, 50), -1)

                elif ltype == 'tree_small':
                    h_tree = int(s * 3.0)
                    r_f = int(s * 1.4)
                    cv2.line(frame, (u, v), (u, max(0, v - int(h_tree * 0.5))), (40, 60, 80), max(1, int(s * 0.18)))
                    cv2.circle(frame, (u, max(0, v - h_tree)), r_f, (35, 125, 45), -1)

                elif ltype == 'building':
                    b_height_m = item[4]
                    b_col = item[5]
                    bh = int(s * b_height_m * 0.65)
                    bw = int(s * 4.8)
                    cv2.rectangle(frame, (u - bw//2, max(0, v - bh)), (u + bw//2, v), b_col, -1)
                    cv2.rectangle(frame, (u - bw//2, max(0, v - bh)), (u + bw//2, v), (40, 40, 45), 1)
                    # Window grid
                    if s > 8.0:
                        for wy in range(max(0, v - bh + 10), v - 10, int(s * 1.2)):
                            for wx in range(u - bw//2 + 8, u + bw//2 - 8, int(s * 1.0)):
                                cv2.rectangle(frame, (wx, wy), (wx + max(2, int(s*0.4)), wy + max(2, int(s*0.5))), (220, 240, 255), -1)

            # Asphalt road micro-texture
            noise = np.random.normal(0, 1.8, (height, width, 3)).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            return frame

        frame_l = render_frame_view(ego_x, ego_z)
        frame_r = render_frame_view(right_cam_x, right_cam_z)

        # Inject brief motion blur (camera bump at frame 85-90 and 280-285)
        if inject_blur and (85 <= f <= 90 or 280 <= f <= 285):
            k = 21
            kv = np.zeros((k, k))
            kv[k//2, :] = np.ones(k) / k
            frame_l = cv2.filter2D(frame_l, -1, kv)
            frame_r = cv2.filter2D(frame_r, -1, kv)

        out_left.write(frame_l)
        out_right.write(frame_r)

    out_left.release()
    out_right.release()

    import shutil
    shutil.copyfile(output_video_left, "data/demo_driving.mp4")

    with open(output_gt, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "z", "yaw"])
        writer.writerows(gt_records)

    print(f"[SampleGenerator] High-detail 450-frame STEREO sequence rendered successfully.")
    return output_video_left, output_video_right, output_gt

if __name__ == "__main__":
    generate_benchmark_sequence()
