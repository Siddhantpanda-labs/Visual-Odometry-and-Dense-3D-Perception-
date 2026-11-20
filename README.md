# Robust Multi-Sensor Visual Odometry and Dense 3D Perception
### For Autonomous Ground Navigation Under Geometric Degeneracy

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Paper](https://img.shields.io/badge/IEEE%20Format-ICRA%2FIROS-red.svg)](docs/IEEE_RESEARCH_REPORT_VISUAL_ODOMETRY.docx)
[![Interactive Dashboard](https://img.shields.io/badge/Web%20Dashboard-Streamlit-FF4B4B.svg)](app.py)

An end-to-end, automotive-grade Computer Vision and Visual Odometry (VO) perception platform. The system couples classical projective geometry with real-time failure-mode gating (motion blur, sub-pixel parallax, planar degeneracy, kinematic angular rate limits) and calibrated stereo triangulation ($b = 0.54\,\text{m}$) paired with sliding-window bundle adjustment.

## 🚀 Key Empirical Benchmark Findings

Evaluated across the 450-frame controlled autonomous driving course via closed-form Sim(3) Umeyama alignment:

| Architecture Configuration | Sensor Modality | Bundle Adjustment | ATE RMSE (m) ↓ | Relative Drift ↓ | Throughput ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Monocular Baseline** | Single Camera | None | $26.31\,\text{m}$ | $16.07\%$ | $57.4\,\text{FPS}$ |
| **2. Monocular + Robust Gates** | Single Camera | None | $23.93\,\text{m}$ | $14.61\%$ | **$66.3\,\text{FPS}$** |
| **3. Stereo VO (Calibrated $b=0.54\text{m}$)** | Dual Cameras | None | **$3.06\,\text{m}$** | **$1.87\%$** | $60.9\,\text{FPS}$ |
| **4. Stereo VO + Local BA ($W=5$)** | Dual Cameras | Huber LM ($W=5$) | $13.99\,\text{m}$ | $8.54\%$ | $8.5\,\text{FPS}$ |

### 💡 Core Scientific Breakthroughs:
1. **Scale Ambiguity Elimination:** Moving from monocular to calibrated stereo cuts Absolute Trajectory Error (ATE RMSE) by **$8.60\times$** ($26.31\,\text{m} \to 3.06\,\text{m}$) and reduces relative drift from $16.07\%$ to **$1.87\%$** at real-time speeds ($60.9\,\text{FPS}$).
2. **Robustness Gating Speedup:** Intercepting blurred frames ($\text{Var}(\nabla^2 I) < 100$) and sub-pixel stops ($\mu_{\text{flow}} < 1.5\,\text{px}$) avoids ill-conditioned matrix decompositions, boosting monocular throughput by **$+15.5\%$** ($57.4 \to 66.3\,\text{FPS}$).
3. **Real-World Highway Curvature Tracking:** Evaluated across 1,260 frames ($1.06\,\text{km}$) of authentic California interstate driving at 65 mph. The kinematic angular rate limiter ($|\Delta\theta| \le 6.0^\circ/\text{frame}$) and forward hemisphere prior ($t_z > 0$) resolved the classic unconstrained epipolar small-baseline ray ambiguity ($80^\circ$ drift), tracking the authentic AASHTO highway curve arc of $10^\circ\text{--}20^\circ$ ($25.5^\circ$ net envelope).

---

## 🖥️ Live Interactive Web Dashboard

Launch the full-featured automotive perception web dashboard:
```bash
streamlit run app.py
```

### Dashboard Features:
* **Upload Custom Video:** Drag-and-drop any driving video (`.mp4`, `.avi`, `.mov`) or select sample sequences.
* **3D WebGL Scene Reconstruction:** Full Three.js free-fly viewer supporting **WASD** flight + mouse orbit + Shift sprint.
* **Trajectory & BEV Graphs:** 2D lateral displacement, AASHTO highway reference, and frame-by-frame heading (Yaw) dynamics.
* **Benchmark Ablation:** Live comparative error bar charts and throughput gauges.
* **Synchronized 4-Pane Video:** High-definition video player showing live KLT flow, disparity, BEV, and 6-DoF HUD telemetry.
* **Direct Downloads:** One-click downloads for IEEE Report (`.docx`), 3D Point Cloud (`.ply`), and presentation deck (`.html`).

---

## 📁 Repository Structure

```
Visual-Odometry-and-Dense-3D-Perception/
├── app.py                      # Modern full-featured Streamlit interactive web dashboard
├── main.py                     # Unified CLI execution engine
├── requirements.txt           # Python environment dependencies
├── .gitignore                  # Git ignore rules
│
├── docs/                       # Research documentation & publications
│   ├── IEEE_RESEARCH_REPORT_VISUAL_ODOMETRY.docx # Official 2-column IEEE Conference Paper
│   ├── IEEE_RESEARCH_REPORT_VISUAL_ODOMETRY_FINAL.docx # Master report with university cover
│   ├── defense_presentation.html                # 8-slide interactive defense slide deck
│   ├── ieee_conference_paper.tex                # Raw IEEEtran LaTeX document
│   └── figures/                                 # 8 publication-grade high-res figures
│
├── src/                        # Core perception modules
│   ├── vo_engine.py            # Monocular VO (KLT Flow, 5-pt RANSAC, Gating)
│   ├── stereo_engine.py        # Stereo VO (StereoSGBM Disparity, EPnP RANSAC)
│   ├── bundle_adjustment.py    # Sliding-Window Local Bundle Adjustment (Huber LM)
│   ├── robustness.py           # Blur Laplacian & Parallax Keyframe Gating
│   ├── depth_engine.py         # Dense Depth Refinement & Bilateral Smoothing
│   ├── point_cloud.py          # 3D Metric Reprojection & WebGL Viewer Exporter
│   ├── eval_metrics.py         # Sim(3) Umeyama ATE & RPE Drift Calculations
│   └── visualizer.py           # Synchronized 4-Pane Presentation Dashboard
│
├── data/                       # Visual input sequences
│   ├── demo_driving_left.mp4   # Calibrated left stereo camera feed
│   ├── demo_driving_right.mp4  # Calibrated right stereo camera feed (b = 0.54m)
│   ├── ground_truth.csv        # 6-DoF millimeter-accurate ground truth
│   └── sample_generator.py     # Deterministic benchmark sequence generator
│
├── outputs/                    # Perception deliverables
│   ├── real_reconstruction_3d.html # Interactive 3D WebGL Scene (WASD Controls)
│   ├── reconstruction_3d.html      # Benchmark 3D WebGL Scene
│   └── ablation_results.csv        # Quantitative experimental benchmark results
│
└── experiments/
    └── run_ablation_benchmark.py # Automated 4-configuration benchmark script
```

---

## ⚡ Quick Start CLI Commands

### 1. Run Live Interactive Web Application
```bash
streamlit run app.py
```

### 2. Run Full Highway Driving Video (Monocular Gated VO)
```bash
python main.py --source data/real_driving.mp4 --headless --save outputs/real_presentation_demo.mp4 --export_ply outputs/real_scene_pointcloud.ply --export_html outputs/real_reconstruction_3d.html
```

### 3. Run Calibrated Stereo Presentation Demo
```bash
python main.py --source demo --sensor stereo
```

### 4. Run the Full 4-Configuration Scientific Benchmark
```bash
python main.py --mode benchmark
```

---

## 📄 IEEE Report Citations & License
Licensed under the [MIT License](LICENSE). Built for academic research and presentation defense.
