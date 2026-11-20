import numpy as np
import cv2
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PointCloudExporter:
    """
    3D Reprojection, Point Cloud Fusion, and Multi-Format 3D Exporter:
    1. Standard ASCII .ply file (for Windows 3D Viewer, MeshLab, CloudCompare, Blender)
    2. Interactive 3D WebGL HTML Viewer (Three.js orbit controls, zero dependencies)
    3. 3D Trajectory & Point Cloud Matplotlib Plot (.png)
    """
    def __init__(self, fx, fy, cx, cy, max_depth_m=45.0, min_depth_m=1.0, sample_step=6):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.max_depth_m = max_depth_m
        self.min_depth_m = min_depth_m
        self.sample_step = sample_step

        self.points_world = []
        self.colors_rgb = []
        self.camera_path = []

    def add_frame(self, frame_bgr, depth_norm, cur_R, cur_t):
        """
        Unprojects depth into 3D metric world coordinates and accumulates colors and camera path.
        """
        h, w = depth_norm.shape
        step = self.sample_step

        u_coords = np.arange(0, w, step)
        v_coords = np.arange(0, h, step)
        uu, vv = np.meshgrid(u_coords, v_coords)

        sampled_d = depth_norm[vv, uu]
        z_metric = self.min_depth_m + (1.0 - sampled_d) * (self.max_depth_m - self.min_depth_m)

        valid = (z_metric > self.min_depth_m) & (z_metric < self.max_depth_m * 0.85)
        uu_valid = uu[valid]
        vv_valid = vv[valid]
        z_valid = z_metric[valid]

        x_cam = (uu_valid - self.cx) * z_valid / self.fx
        y_cam = (vv_valid - self.cy) * z_valid / self.fy
        pts_cam = np.vstack([x_cam, y_cam, z_valid])

        # World transform: P_w = R * P_c + t
        pts_world = (cur_R @ pts_cam) + cur_t

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sampled_colors = rgb_frame[vv_valid, uu_valid]

        self.points_world.append(pts_world.T)
        self.colors_rgb.append(sampled_colors)
        self.camera_path.append([float(cur_t[0, 0]), float(cur_t[1, 0]), float(cur_t[2, 0])])

    def export_ply(self, filepath):
        """Exports accumulated 3D vertices to standard ASCII PLY file."""
        if not self.points_world:
            return False

        all_points = np.vstack(self.points_world)
        all_colors = np.vstack(self.colors_rgb)
        num_vertices = len(all_points)

        print(f"[PointCloud] Saving {num_vertices} 3D vertices to {filepath}...")
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {num_vertices}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with open(filepath, "w") as f:
            f.write(header)
            for i in range(num_vertices):
                p = all_points[i]
                c = all_colors[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        return True

    def export_interactive_html(self, filepath="outputs/reconstruction_3d.html"):
        """
        Exports an interactive 3D WebGL viewer using Three.js:
        Users can open this in Chrome/Edge, rotate, zoom, and inspect the 3D road and camera path.
        """
        if not self.points_world:
            return False

        # Subsample for smooth 60 FPS WebGL rendering (max 15,000 points)
        all_points = np.vstack(self.points_world)
        all_colors = np.vstack(self.colors_rgb) / 255.0

        total_pts = len(all_points)
        sub_step = max(1, total_pts // 15000)
        pts_sub = all_points[::sub_step].tolist()
        cols_sub = all_colors[::sub_step].tolist()
        path_sub = self.camera_path

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3D Scene Reconstruction & Camera Trajectory</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #0e1117; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #hud {{ position: absolute; top: 16px; left: 16px; color: #fff; background: rgba(14, 17, 23, 0.88); backdrop-filter: blur(8px); padding: 16px 20px; border-radius: 10px; border: 1px solid #30363d; box-shadow: 0 8px 24px rgba(0,0,0,0.6); max-width: 360px; }}
        #hud h3 {{ margin: 0 0 8px 0; color: #58a6ff; font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: #238636; color: #fff; font-size: 10px; padding: 2px 7px; border-radius: 10px; text-transform: uppercase; font-weight: bold; }}
        .control-row {{ display: flex; align-items: center; gap: 6px; margin: 5px 0; font-size: 12px; color: #c9d1d9; }}
        kbd {{ background: #21262d; border: 1px solid #30363d; border-radius: 4px; padding: 2px 6px; font-size: 11px; font-family: monospace; color: #58a6ff; font-weight: bold; }}
        .stat {{ font-size: 11px; color: #8b949e; margin-top: 10px; border-top: 1px solid #21262d; padding-top: 8px; }}
        #speed-tag {{ position: absolute; bottom: 20px; right: 20px; color: #8b949e; font-size: 12px; background: rgba(0,0,0,0.6); padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <h3>3D SCENE RECONSTRUCTION <span class="badge">ACTIVE</span></h3>
        <div class="control-row"><kbd>Left Click + Drag</kbd> Rotate Camera</div>
        <div class="control-row"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> Move Around</div>
        <div class="control-row"><kbd>Q</kbd> / <kbd>E</kbd> Move Up / Down</div>
        <div class="control-row"><kbd>Shift</kbd> Sprint / Fast Move</div>
        <div class="control-row"><kbd>Scroll</kbd> Zoom | <kbd>R</kbd> Reset View</div>
        <div class="stat">
            • <b>Trajectory:</b> {len(path_sub)} keyframes (Green Path)<br>
            • <b>Point Cloud:</b> {len(pts_sub):,} metric 3D vertices
        </div>
    </div>
    <div id="speed-tag">Mode: Fly-Through Orbit | Speed: <span id="cur-speed" style="color:#58a6ff;">Normal</span></div>
    <script>
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0e1117);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 2000);
        
        const initialPos = new THREE.Vector3(0, -18, -25);
        const initialTarget = new THREE.Vector3(0, 0, 50);
        camera.position.copy(initialPos);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.copy(initialTarget);
        controls.update();

        // Ground Grid
        const grid = new THREE.GridHelper(400, 80, 0x58a6ff, 0x21262d);
        grid.position.y = 1.65;
        scene.add(grid);

        // 3D Point Cloud Geometry
        const points = {json.dumps(pts_sub)};
        const colors = {json.dumps(cols_sub)};
        const geom = new THREE.BufferGeometry();
        const positions = [];
        const col_arr = [];

        for(let i = 0; i < points.length; i++) {{
            positions.push(points[i][0], -points[i][1], points[i][2]);
            col_arr.push(colors[i][0], colors[i][1], colors[i][2]);
        }}

        geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geom.setAttribute('color', new THREE.Float32BufferAttribute(col_arr, 3));
        const mat = new THREE.PointsMaterial({{ size: 0.22, vertexColors: true }});
        const cloud = new THREE.Points(geom, mat);
        scene.add(cloud);

        // Camera Trajectory Line
        const path = {json.dumps(path_sub)};
        const pathGeom = new THREE.BufferGeometry();
        const pathPos = [];
        for(let i = 0; i < path.length; i++) {{
            pathPos.push(path[i][0], -path[i][1], path[i][2]);
        }}
        pathGeom.setAttribute('position', new THREE.Float32BufferAttribute(pathPos, 3));
        const pathMat = new THREE.LineBasicMaterial({{ color: 0x3fb950, linewidth: 3 }});
        const pathLine = new THREE.Line(pathGeom, pathMat);
        scene.add(pathLine);

        // Camera Frustum Indicator at Current Pose
        if (path.length > 0) {{
            const lastPos = pathPos.slice(-3);
            const coneGeom = new THREE.ConeGeometry(0.8, 1.8, 4);
            coneGeom.rotateX(Math.PI / 2);
            const coneMat = new THREE.MeshBasicMaterial({{ color: 0xf85149, wireframe: true }});
            const cone = new THREE.Mesh(coneGeom, coneMat);
            cone.position.set(lastPos[0], lastPos[1], lastPos[2]);
            scene.add(cone);
        }}

        // Keyboard Controls State
        const keys = {{}};
        window.addEventListener('keydown', (e) => {{
            keys[e.key.toLowerCase()] = true;
            if (e.key === 'Shift') keys['shift'] = true;
            if (e.key === 'r' || e.key === 'R') resetCamera();
            updateSpeedTag();
        }});

        window.addEventListener('keyup', (e) => {{
            keys[e.key.toLowerCase()] = false;
            if (e.key === 'Shift') keys['shift'] = false;
            updateSpeedTag();
        }});

        function updateSpeedTag() {{
            const tag = document.getElementById('cur-speed');
            if (tag) tag.textContent = keys['shift'] ? 'Sprint (Fast)' : 'Normal';
            if (tag) tag.style.color = keys['shift'] ? '#f0883e' : '#58a6ff';
        }}

        function resetCamera() {{
            camera.position.copy(initialPos);
            controls.target.copy(initialTarget);
            controls.update();
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        const clock = new THREE.Clock();

        function animate() {{
            requestAnimationFrame(animate);
            const dt = Math.min(clock.getDelta(), 0.1);

            // WASD Free-Fly Translation
            const baseSpeed = keys['shift'] ? 45.0 : 16.0;
            const moveDist = baseSpeed * dt;

            const forward = new THREE.Vector3();
            camera.getWorldDirection(forward);
            forward.normalize();

            const right = new THREE.Vector3();
            right.crossVectors(forward, camera.up).normalize();

            const delta = new THREE.Vector3();
            if (keys['w']) delta.addScaledVector(forward, moveDist);
            if (keys['s']) delta.addScaledVector(forward, -moveDist);
            if (keys['d']) delta.addScaledVector(right, moveDist);
            if (keys['a']) delta.addScaledVector(right, -moveDist);
            if (keys['q'] || keys[' ']) delta.y += moveDist;
            if (keys['e'] || keys['c']) delta.y -= moveDist;

            if (delta.lengthSq() > 0) {{
                camera.position.add(delta);
                controls.target.add(delta);
            }}

            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[PointCloud] Interactive 3D WebGL viewer saved to: {filepath}")
        return True

    def export_trajectory_plot(self, filepath="outputs/trajectory_3d_plot.png"):
        """Saves a static publication-quality 3D trajectory plot."""
        if not self.camera_path:
            return False

        path = np.array(self.camera_path)
        fig = plt.figure(figsize=(10, 7), facecolor='#121316')
        ax = fig.add_subplot(111, projection='3d', facecolor='#121316')

        # Plot trajectory line
        ax.plot(path[:, 0], path[:, 2], -path[:, 1], color='#00e5ff', linewidth=2.5, label='Estimated Camera Path')
        ax.scatter(path[0, 0], path[0, 2], -path[0, 1], color='#00ff88', s=60, label='Start (0,0,0)')
        ax.scatter(path[-1, 0], path[-1, 2], -path[-1, 1], color='#ff3366', s=60, label='Current Pose')

        ax.set_title("3D Reconstructed Camera Trajectory", color='white', fontsize=14, pad=15)
        ax.set_xlabel("X (Lateral, m)", color='white', labelpad=10)
        ax.set_ylabel("Z (Forward, m)", color='white', labelpad=10)
        ax.set_zlabel("Y (Height, m)", color='white', labelpad=10)
        ax.tick_params(colors='white')
        ax.grid(True, color='#2a2d34')
        ax.legend(facecolor='#1e2025', edgecolor='#333', labelcolor='white')

        plt.tight_layout()
        plt.savefig(filepath, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[PointCloud] 3D Trajectory plot saved to: {filepath}")
        return True
