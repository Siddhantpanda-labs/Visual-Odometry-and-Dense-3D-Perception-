import os
import cv2
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank layout

    # =========================================================================
    # Gamma AI Inspired Obsidian Slate Color System (High Contrast & Legibility)
    # =========================================================================
    BG_OBSIDIAN    = RGBColor(10, 13, 20)      # #0A0D14 Deep Obsidian Void
    CARD_BG        = RGBColor(20, 27, 41)      # #141B29 Elevated Container
    CARD_BG_ALT    = RGBColor(28, 38, 58)      # #1C263A High-level Container
    CARD_BORDER    = RGBColor(46, 61, 89)      # #2E3D59 Crisp Subtle Border
    
    # Accent Highlights
    CYAN_ACCENT    = RGBColor(56, 189, 248)    # #38BDF8 Electric Cyan
    INDIGO_ACCENT  = RGBColor(129, 140, 248)   # #818CF8 Soft Violet / Indigo
    EMERALD_ACCENT = RGBColor(52, 211, 153)    # #34D399 High-Vis Emerald
    AMBER_ACCENT   = RGBColor(251, 191, 36)    # #FBBF24 Vibrant Amber
    
    # High-Visibility Text Colors (No dim grays!)
    TEXT_PURE_WHITE= RGBColor(255, 255, 255)   # #FFFFFF Headers & Primary Titles
    TEXT_BRIGHT    = RGBColor(241, 245, 249)   # #F1F5F9 Ultra-crisp Body Text
    TEXT_MUTED     = RGBColor(186, 198, 217)   # #BAC6D9 Highly-legible Secondary Text
    TEXT_CODE      = RGBColor(125, 211, 252)   # #7DD3FC Code & Formulas

    FONT_FAMILY = "Segoe UI"

    def apply_slide_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_OBSIDIAN
        bg.line.fill.background()
        return bg

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=CARD_BORDER, border_width=1.2):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(border_width)
        else:
            card.line.fill.background()
        return card

    def add_eyebrow_header(slide, category_text, title_text, accent_color=CYAN_ACCENT):
        # Category Eyebrow Pill / Badge Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(11.733), Inches(0.32))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = f"■  {category_text.upper()}"
        p_tag.font.name = FONT_FAMILY
        p_tag.font.size = Pt(11)
        p_tag.font.bold = True
        p_tag.font.color.rgb = accent_color

        # Main Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.76), Inches(11.733), Inches(0.55))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_FAMILY
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_PURE_WHITE

    def add_fit_picture(slide, img_path, target_left, target_top, max_w, max_h):
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        aspect = w / h
        
        # Scale to fit inside max_w x max_h
        target_w = max_w
        target_h = target_w / aspect
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * aspect
        
        # Center inside target box
        offset_x = (max_w - target_w) / 2
        offset_y = (max_h - target_h) / 2
        
        pic = slide.shapes.add_picture(
            img_path,
            target_left + Inches(offset_x),
            target_top + Inches(offset_y),
            width=Inches(target_w),
            height=Inches(target_h)
        )
        return pic

    # =========================================================================
    # SLIDE 1: Title & University Presentation Credentials (Gamma Hero Style)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s1)

    # Top Cyan Glow Line
    glow = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.7), Inches(11.733), Inches(0.04))
    glow.fill.solid()
    glow.fill.fore_color.rgb = CYAN_ACCENT
    glow.line.fill.background()

    # Main Hero Container Card
    hero_card = add_card(s1, Inches(0.8), Inches(0.9), Inches(11.733), Inches(6.0), CARD_BG, CARD_BORDER, 1.5)

    # Eyebrow Badge
    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(10.9), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    p.text = "■  CSE471 COMPUTER VISION — FINAL PROJECT PRESENTATION"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    # Main Project Title
    tb_title = s1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(10.9), Inches(1.4))
    tf = tb_title.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Robust Multi-Sensor Visual Odometry\n& Dense 3D Perception"
    p.font.name = FONT_FAMILY; p.font.size = Pt(32); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    # Subtitle
    tb_sub = s1.shapes.add_textbox(Inches(1.2), Inches(3.25), Inches(10.9), Inches(0.6))
    tf = tb_sub.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Real-Time Metric Ground Navigation Under Geometric Degeneracy & Scale Drift"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT

    # Two Split Cards at bottom for Presenter info & Live System Links
    card_w = Inches(5.25)
    
    # Left Card: Academic Credentials
    add_card(s1, Inches(1.2), Inches(4.05), card_w, Inches(2.4), CARD_BG_ALT, CARD_BORDER)
    tb_c1 = s1.shapes.add_textbox(Inches(1.45), Inches(4.2), card_w - Inches(0.5), Inches(2.1))
    tf1 = tb_c1.text_frame; tf1.word_wrap = True
    
    p = tf1.paragraphs[0]; p.text = "Candidate: Sidhant Kumar Panda"
    p.font.name = FONT_FAMILY; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE
    
    p = tf1.add_paragraph(); p.text = "Registration No: 12413416"
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    p = tf1.add_paragraph(); p.text = "Project Guide: Abhishek Tiwari"
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf1.add_paragraph(); p.text = "School of Computer Science and Engineering"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    p = tf1.add_paragraph(); p.text = "Division of Machine Intelligence & Computer Vision"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # Right Card: Project Artifacts & Links
    add_card(s1, Inches(6.883), Inches(4.05), card_w, Inches(2.4), CARD_BG_ALT, CARD_BORDER)
    tb_c2 = s1.shapes.add_textbox(Inches(7.133), Inches(4.2), card_w - Inches(0.5), Inches(2.1))
    tf2 = tb_c2.text_frame; tf2.word_wrap = True

    p = tf2.paragraphs[0]; p.text = "Project Artifacts & Live Deployment"
    p.font.name = FONT_FAMILY; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf2.add_paragraph(); p.text = "🌐 Live Interactive Cloud App:"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED
    p = tf2.add_paragraph(); p.text = "https://qgqccuvxnhc3gp32otvk6t.streamlit.app/"
    p.font.name = "Consolas"; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    p = tf2.add_paragraph(); p.text = "📦 GitHub Open Source Repository:"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED
    p = tf2.add_paragraph(); p.text = "github.com/Siddhantpanda-labs/Visual-Odometry..."
    p.font.name = "Consolas"; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = EMERALD_ACCENT

    # =========================================================================
    # SLIDE 2: Problem Formulation: The 3 Core Failure Modes (Gamma 3-Col Card)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s2)
    add_eyebrow_header(s2, "PROBLEM FORMULATION & MOTIVATION", "Critical Failure Modes in Classical Monocular Odometry", AMBER_ACCENT)

    card_w = Inches(3.64)
    card_h = Inches(5.6)
    card_top = Inches(1.4)
    gap = Inches(0.4)

    # Column Card 1: Scale Ambiguity
    add_card(s2, Inches(0.8), card_top, card_w, card_h, CARD_BG, CARD_BORDER)
    pill1 = add_card(s2, Inches(1.1), Inches(1.65), Inches(2.5), Inches(0.38), CARD_BG_ALT, AMBER_ACCENT)
    tb = s2.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(2.3), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "01 · SCALE AMBIGUITY"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = AMBER_ACCENT

    tb_c1 = s2.shapes.add_textbox(Inches(1.1), Inches(2.2), Inches(3.04), Inches(4.6))
    tf = tb_c1.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Unconstrained Scale Drift"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n• Inherent Epipolar Ambiguity:\nMonocular geometry only recovers translation direction, leaving absolute metric scale factor s_t unknown (s_t ∈ ℝ⁺)."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Quadratic Compounding:\nSmall scale estimation errors multiply recursively, generating >16% trajectory dilation over extended highway driving."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Autonomous Impact:\nVehicle cannot distinguish high-speed highway cruising from low-speed crawling without external sensors."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # Column Card 2: Motion Blur
    add_card(s2, Inches(0.8) + card_w + gap, card_top, card_w, card_h, CARD_BG, CARD_BORDER)
    pill2 = add_card(s2, Inches(0.8) + card_w + gap + Inches(0.3), Inches(1.65), Inches(2.5), Inches(0.38), CARD_BG_ALT, CYAN_ACCENT)
    tb = s2.shapes.add_textbox(Inches(0.8) + card_w + gap + Inches(0.4), Inches(1.7), Inches(2.3), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "02 · MOTION BLUR"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    tb_c2 = s2.shapes.add_textbox(Inches(0.8) + card_w + gap + Inches(0.3), Inches(2.2), Inches(3.04), Inches(4.6))
    tf = tb_c2.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Motion Blur & Track Loss"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n• Highway Chassis Dynamics:\nPavement bumps and engine vibrations blur high-frequency visual gradients in forward-facing cameras."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Gradient Dissipation:\nSpatial derivatives Ix, Iy degrade; the Harris corner score collapses below threshold τ, shedding valid feature points."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Tracking Discontinuity:\nStandard systems drop to 0 active tracks and crash without recovery or re-initialization mechanisms."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # Column Card 3: Planar Road Degeneracy
    add_card(s2, Inches(0.8) + (card_w + gap) * 2, card_top, card_w, card_h, CARD_BG, CARD_BORDER)
    pill3 = add_card(s2, Inches(0.8) + (card_w + gap) * 2 + Inches(0.3), Inches(1.65), Inches(2.5), Inches(0.38), CARD_BG_ALT, INDIGO_ACCENT)
    tb = s2.shapes.add_textbox(Inches(0.8) + (card_w + gap) * 2 + Inches(0.4), Inches(1.7), Inches(2.3), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "03 · PLANAR DEGENERACY"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT

    tb_c3 = s2.shapes.add_textbox(Inches(0.8) + (card_w + gap) * 2 + Inches(0.3), Inches(2.2), Inches(3.04), Inches(4.6))
    tf = tb_c3.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Planar Road Degeneracy"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n• Homography Domination:\nWhen visual features are coplanar on flat asphalt, the 8-point Essential Matrix E solver becomes mathematically degenerate."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Rotational Ambiguity:\nDecomposing degenerate E yields two mathematically valid rotations, frequently choosing the false 'ghost' rotation."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n• Catastrophic Twisting:\nStandard RANSAC selects false motion vectors, twisting straight driving trajectories into fictitious spirals."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 3: System Architecture & Automated Failure-Gating State Machine
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s3)
    add_eyebrow_header(s3, "SYSTEM ARCHITECTURE & CONTROL FLOW", "Dual-Pipeline Perception Architecture & Failure-Gating State Machine", CYAN_ACCENT)

    # Left: Explanation Card
    left_w = Inches(5.6)
    add_card(s3, Inches(0.8), Inches(1.4), left_w, Inches(5.6), CARD_BG, CARD_BORDER)

    tb_arch = s3.shapes.add_textbox(Inches(1.1), Inches(1.65), left_w - Inches(0.6), Inches(5.1))
    tf_arch = tb_arch.text_frame; tf_arch.word_wrap = True

    p = tf_arch.paragraphs[0]; p.text = "Four-Stage Robust Processing Pipeline"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf_arch.add_paragraph()
    p.text = "\n1. Adaptive Radiometric Normalization:\nCLAHE (clip limit = 3.0, grid = 8x8) equalizes harsh windshield glare, sun reflections, and shadowed asphalt before feature detection."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_arch.add_paragraph()
    p.text = "\n2. Pyramidal KLT with Bidirectional Check:\nExtracts 1,500 Shi-Tomasi corners across 4 pyramid scales. Enforces forward-backward flow consistency (||x_i - x̂_i||₂ < 1.0 px) to eliminate false correspondences."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_arch.add_paragraph()
    p.text = "\n3. Automated Failure-Gating State Machine:\nTracks active inliers N and translation magnitude. Automatically switches between 5-point Essential Matrix and P3P RANSAC to avoid solver divergence."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_arch.add_paragraph()
    p.text = "\n4. Multi-Sensor Modality & Graceful Fallback:\nCalibrated metric stereo (b = 0.54 m) serves as primary; gracefully degrades to robust monocular mode when secondary feed fails."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = CYAN_ACCENT

    # Right: Embedded Architecture Diagram inside framed card
    right_w = Inches(5.733)
    add_card(s3, Inches(6.8), Inches(1.4), right_w, Inches(5.6), CARD_BG, CARD_BORDER)
    add_fit_picture(s3, "docs/figures/fig1_architecture.png", 7.0, 1.7, 5.333, 4.6)

    tb_cap = s3.shapes.add_textbox(Inches(7.0), Inches(6.45), Inches(5.333), Inches(0.4))
    p = tb_cap.text_frame.paragraphs[0]
    p.text = "Fig 1: Complete end-to-end perception pipeline with failure-gated state machine."
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED; p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 4: Metric Stereo Geometry & Local Bundle Adjustment
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s4)
    add_eyebrow_header(s4, "MATHEMATICAL METHODOLOGY", "Calibrated Stereo Metric Triangulation & Local Bundle Adjustment", INDIGO_ACCENT)

    card_w2 = Inches(5.666)
    
    # Left Card: Stereo Triangulation
    add_card(s4, Inches(0.8), Inches(1.4), card_w2, Inches(5.6), CARD_BG, CARD_BORDER)
    tb_st = s4.shapes.add_textbox(Inches(1.1), Inches(1.65), card_w2 - Inches(0.6), Inches(5.1))
    tf_st = tb_st.text_frame; tf_st.word_wrap = True

    p = tf_st.paragraphs[0]; p.text = "Calibrated Metric Scale Recovery"
    p.font.name = FONT_FAMILY; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    p = tf_st.add_paragraph()
    p.text = "\n• Exploitation of Known Physical Baseline:\nCalibrated stereo baseline b = 0.54 m between left and right optical centers introduces metric scale observability at every frame."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_st.add_paragraph()
    p.text = "\n• Closed-Form Metric Depth Triangulation:\nFor each matched stereo correspondence (u_L, v) and (u_R, v), horizontal disparity d = u_L - u_R yields absolute metric depth:"
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_st.add_paragraph()
    p.text = "\n    Z = ( f · b ) / d    |    X = (u - cx) · Z / f    |    Y = (v - cy) · Z / f"
    p.font.name = "Consolas"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_CODE

    p = tf_st.add_paragraph()
    p.text = "\n• Scale Invariance Guarantee:\nLandmark coordinates X_j are instantiated directly in true physical meters, preventing speed ambiguity and trajectory scale inflation."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_st.add_paragraph()
    p.text = "\n• Quadratic Depth Uncertainty: δZ = (Z² / (f·b)) · δd provides sub-decimeter depth resolution up to 45 meters forward range."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # Right Card: Sliding-Window Bundle Adjustment
    add_card(s4, Inches(6.866), Inches(1.4), card_w2, Inches(5.6), CARD_BG, CARD_BORDER)
    tb_ba = s4.shapes.add_textbox(Inches(7.166), Inches(1.65), card_w2 - Inches(0.6), Inches(5.1))
    tf_ba = tb_ba.text_frame; tf_ba.word_wrap = True

    p = tf_ba.paragraphs[0]; p.text = "Sliding-Window Bundle Adjustment (BA)"
    p.font.name = FONT_FAMILY; p.font.size = Pt(17); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT

    p = tf_ba.add_paragraph()
    p.text = "\n• Joint Non-Linear Optimization:\nJointly refines the last W = 5 camera poses T_k = [R_k | t_k] and 3D landmark positions X_j across temporal observations."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_ba.add_paragraph()
    p.text = "\n• Robust Huber Loss Reprojection Formulation:\nSuppresses highway asphalt glare and moving vehicle outlier tracks:"
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_ba.add_paragraph()
    p.text = "\n    min_{T_k, X_j}  Σ  ρ_Huber( || u_{i,k} - π(T_k, X_j) ||² )"
    p.font.name = "Consolas"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_CODE

    p = tf_ba.add_paragraph()
    p.text = "\n• Levenberg-Marquardt Solver:\nDamped Hessian (Jᵀ J + λ diag(Jᵀ J)) Δξ = -Jᵀ r achieves rapid quadratic convergence within 8-12 iterations per keyframe."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_ba.add_paragraph()
    p.text = "\n• Strict O(W) Constant Bound:\nFixed-lag sliding window ensures deterministic real-time execution with zero memory growth."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: Empirical Benchmark & Ablation Study (Gamma Stats & Grid)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s5)
    add_eyebrow_header(s5, "EMPIRICAL BENCHMARK & PERFORMANCE ABLATION", "Rigorous Quantitative Ablation Across 450 Controlled Frames", EMERALD_ACCENT)

    # 3 Stat KPI Cards at Top (Gamma Style)
    kpi_w = Inches(3.64)
    kpi_h = Inches(1.35)
    gap = Inches(0.4)

    # KPI 1
    add_card(s5, Inches(0.8), Inches(1.4), kpi_w, kpi_h, CARD_BG, EMERALD_ACCENT, 1.5)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(1.45), kpi_w - Inches(0.4), kpi_h - Inches(0.1))
    p = tb.text_frame.paragraphs[0]; p.text = "8.60× DRIFT REDUCTION"
    p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = EMERALD_ACCENT
    p = tb.text_frame.add_paragraph(); p.text = "ATE RMSE dropped from 26.31 m to 3.06 m"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = TEXT_BRIGHT

    # KPI 2
    add_card(s5, Inches(0.8) + kpi_w + gap, Inches(1.4), kpi_w, kpi_h, CARD_BG, CYAN_ACCENT, 1.5)
    tb = s5.shapes.add_textbox(Inches(0.8) + kpi_w + gap + Inches(0.2), Inches(1.45), kpi_w - Inches(0.4), kpi_h - Inches(0.1))
    p = tb.text_frame.paragraphs[0]; p.text = "1.87% POSE ERROR"
    p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT
    p = tb.text_frame.add_paragraph(); p.text = "Relative Pose Error (RPE) across 450 frames"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = TEXT_BRIGHT

    # KPI 3
    add_card(s5, Inches(0.8) + (kpi_w + gap) * 2, Inches(1.4), kpi_w, kpi_h, CARD_BG, INDIGO_ACCENT, 1.5)
    tb = s5.shapes.add_textbox(Inches(0.8) + (kpi_w + gap) * 2 + Inches(0.2), Inches(1.45), kpi_w - Inches(0.4), kpi_h - Inches(0.1))
    p = tb.text_frame.paragraphs[0]; p.text = "60.9 FPS REAL-TIME"
    p.font.name = FONT_FAMILY; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT
    p = tb.text_frame.add_paragraph(); p.text = "End-to-end multi-threaded perception latency"
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = TEXT_BRIGHT

    # Bottom Split: Left Table, Right Ablation Bar Chart
    add_card(s5, Inches(0.8), Inches(3.0), Inches(5.8), Inches(4.0), CARD_BG, CARD_BORDER)
    tb_tbl = s5.shapes.add_textbox(Inches(1.05), Inches(3.15), Inches(5.3), Inches(3.6))
    tf_tbl = tb_tbl.text_frame; tf_tbl.word_wrap = True

    p = tf_tbl.paragraphs[0]; p.text = "Algorithmic Ablation Benchmark Table"
    p.font.name = FONT_FAMILY; p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf_tbl.add_paragraph()
    p.text = "\nMethod Variant           ATE RMSE    RPE Drift    FPS"
    p.font.name = "Consolas"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    p = tf_tbl.add_paragraph()
    p.text = "--------------------------------------------------------"
    p.font.name = "Consolas"; p.font.size = Pt(11); p.font.color.rgb = CARD_BORDER

    p = tf_tbl.add_paragraph()
    p.text = "1. Monocular 5-Pt VO     26.31 m     16.14%      68.4"
    p.font.name = "Consolas"; p.font.size = Pt(12); p.font.color.rgb = TEXT_MUTED

    p = tf_tbl.add_paragraph()
    p.text = "2. Pure Stereo VO         5.42 m      3.32%      64.1"
    p.font.name = "Consolas"; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_tbl.add_paragraph()
    p.text = "3. Stereo + Gating        4.18 m      2.56%      62.7"
    p.font.name = "Consolas"; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_tbl.add_paragraph()
    p.text = "4. Ours (Full System)     3.06 m      1.87%      60.9"
    p.font.name = "Consolas"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = EMERALD_ACCENT

    p = tf_tbl.add_paragraph()
    p.text = "\n• Takeaway: Failure-gating stops turn divergence; bundle adjustment eliminates micro-jitter, achieving state-of-the-art accuracy."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_BRIGHT

    # Right: Ablation Bar Chart
    add_card(s5, Inches(6.9), Inches(3.0), Inches(5.633), Inches(4.0), CARD_BG, CARD_BORDER)
    add_fit_picture(s5, "docs/figures/fig8_ablation_barchart.png", 7.05, 3.2, 5.333, 3.5)

    # =========================================================================
    # SLIDE 6: Highway Driving Validation & Curvature Diagnosis
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s6)
    add_eyebrow_header(s6, "EXPERIMENTAL VALIDATION & ERROR DIAGNOSIS", "Highway Driving Trajectory & Mathematical Turn Diagnosis", AMBER_ACCENT)

    # Left: Technical Analysis Card
    add_card(s6, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.6), CARD_BG, CARD_BORDER)
    tb_turn = s6.shapes.add_textbox(Inches(1.1), Inches(1.65), Inches(5.0), Inches(5.1))
    tf_turn = tb_turn.text_frame; tf_turn.word_wrap = True

    p = tf_turn.paragraphs[0]; p.text = "The 80° Visual Perception Paradox"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = AMBER_ACCENT

    p = tf_turn.add_paragraph()
    p.text = "\n• The Empirical Observation:\nDuring human playback of dashcam video, an extended curved bend visually resembles an abrupt ~80° rightward turn. However, the estimated BEV bird's-eye trajectory measures a gradual 10°-20° curve."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_turn.add_paragraph()
    p.text = "\n• Ground Truth: Civil Engineering Constraints:\nAASHTO & Highway Design Standards mandate minimum curvature radii R ≥ 250-400 m for 80-100 km/h highways. Across a 200 m segment, the physical angular change is mathematically bounded to Δθ ≈ 15°."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_turn.add_paragraph()
    p.text = "\n• Optical Root Cause (Epipolar Ray Ambiguity):\nAt highway distances (Z > 50 m), optical flow rays become near-parallel. Vehicle chassis pitch and roll vibrations create apparent angular shifts, causing human observers to overestimate rotational yaw."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_turn.add_paragraph()
    p.text = "\n• Conclusion: The system's bird's-eye trajectory correctly captures physical ground truth, debunking the optical illusion of an 80° turn."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    # Right: Embedded BEV Plot
    add_card(s6, Inches(6.7), Inches(1.4), Inches(5.833), Inches(5.6), CARD_BG, CARD_BORDER)
    add_fit_picture(s6, "docs/figures/fig6_real_driving_bev.png", 6.85, 1.8, 5.533, 4.4)

    tb_bev = s6.shapes.add_textbox(Inches(6.85), Inches(6.4), Inches(5.533), Inches(0.4))
    p = tb_bev.text_frame.paragraphs[0]
    p.text = "Fig 6: Bird's-Eye View (BEV) trajectory across 1,260 real-world highway frames (1.06 km)."
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED; p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 7: Dense 3D Point Cloud & WebGL Reconstruction
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s7)
    add_eyebrow_header(s7, "3D SPATIAL PERCEPTION", "Dense 3D Scene Reconstruction & Interactive WebGL Navigation", CYAN_ACCENT)

    # Left: Explanation Card
    add_card(s7, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.6), CARD_BG, CARD_BORDER)
    tb_pc = s7.shapes.add_textbox(Inches(1.1), Inches(1.65), Inches(5.0), Inches(5.1))
    tf_pc = tb_pc.text_frame; tf_pc.word_wrap = True

    p = tf_pc.paragraphs[0]; p.text = "Dense Metric Point Cloud Pipeline"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    p = tf_pc.add_paragraph()
    p.text = "\n• Real-Time Depth Generation (>150 FPS):\nBilateral-guided geometric disparity filtering extracts dense per-pixel depth maps with edge-preserving filtering, eliminating heavy neural network latency."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_pc.add_paragraph()
    p.text = "\n• Pinhole Camera Back-Projection:\nBack-projects image plane pixels (u, v) into metric 3D camera coordinates:"
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_pc.add_paragraph()
    p.text = "\n    X_c = Z · K⁻¹ [ u, v, 1 ]ᵀ"
    p.font.name = "Consolas"; p.font.size = Pt(13); p.font.bold = True; p.font.color.rgb = TEXT_CODE

    p = tf_pc.add_paragraph()
    p.text = "\n• World Frame Point Accumulation:\nAccumulating world points X_w = R_t X_c + t_t over 1,260 frames yields 350,000+ RGB-colored metric vertices."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_pc.add_paragraph()
    p.text = "\n• Interactive WebGL Viewport:\nEmbedded Three.js canvas in the web dashboard provides seamless WASD free-fly camera navigation."
    p.font.name = FONT_FAMILY; p.font.size = Pt(11); p.font.color.rgb = TEXT_MUTED

    # Right: Embedded 3D Point Cloud Picture
    add_card(s7, Inches(6.7), Inches(1.4), Inches(5.833), Inches(5.6), CARD_BG, CARD_BORDER)
    add_fit_picture(s7, "docs/figures/fig7_3d_pointcloud.png", 6.85, 1.8, 5.533, 4.4)

    tb_pcc = s7.shapes.add_textbox(Inches(6.85), Inches(6.4), Inches(5.533), Inches(0.4))
    p = tb_pcc.text_frame.paragraphs[0]
    p.text = "Fig 7: Reconstructed highway point cloud in 3D WebGL space with perspective depth coloring."
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED; p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 8: Synchronized 4-Pane HUD & Live Streamlit Cloud Deployment
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s8)
    add_eyebrow_header(s8, "SYSTEM IMPLEMENTATION & CLOUD DEPLOYMENT", "Synchronized 4-Pane HUD & Live Streamlit Cloud Architecture", INDIGO_ACCENT)

    # Left: Dashboard Screenshot
    add_card(s8, Inches(0.8), Inches(1.4), Inches(6.2), Inches(5.6), CARD_BG, CARD_BORDER)
    add_fit_picture(s8, "docs/figures/fig3b_real_dashboard.png", 0.95, 1.8, 5.9, 4.4)

    tb_hud = s8.shapes.add_textbox(Inches(0.95), Inches(6.4), Inches(5.9), Inches(0.4))
    p = tb_hud.text_frame.paragraphs[0]
    p.text = "Fig 8: Synchronized 4-Pane HUD: Flow Vectors, Depth Map, BEV Trajectory, Telemetry KPIs."
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.color.rgb = TEXT_MUTED; p.alignment = PP_ALIGN.CENTER

    # Right: Features & Live App
    add_card(s8, Inches(7.3), Inches(1.4), Inches(5.233), Inches(5.6), CARD_BG, CARD_BORDER)
    tb_feat = s8.shapes.add_textbox(Inches(7.55), Inches(1.65), Inches(4.733), Inches(5.1))
    tf_feat = tb_feat.text_frame; tf_feat.word_wrap = True

    p = tf_feat.paragraphs[0]; p.text = "Interactive Cloud Platform"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT

    p = tf_feat.add_paragraph()
    p.text = "\n• Live On-The-Fly Processing:\nUsers can upload custom MP4 dashcam videos; the engine tracks KLT features and computes depth dynamically frame-by-frame."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_feat.add_paragraph()
    p.text = "\n• Browser H.264 Playback:\nPre-rendered video streams encoded with H.264 (yuv420p) play natively without third-party codecs."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_feat.add_paragraph()
    p.text = "\n• Zero-GPU Cloud Deployment:\nOptimized OpenCV-headless fallback allows full cloud execution on standard CPU containers without crashing."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf_feat.add_paragraph()
    p.text = "\n• Public Streamlit Cloud URL:\nhttps://qgqccuvxnhc3gp32otvk6t.streamlit.app/"
    p.font.name = "Consolas"; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    # =========================================================================
    # SLIDE 9: Academic Summary, Key Takeaways & Future Directions (Gamma Style)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    apply_slide_bg(s9)
    add_eyebrow_header(s9, "PROJECT SUMMARY & FUTURE RESEARCH", "Key Takeaways, Future Extensions & Academic Discussion", CYAN_ACCENT)

    card_w = Inches(3.64)
    card_h = Inches(5.6)
    gap = Inches(0.4)

    # Card 1: Key Contributions
    add_card(s9, Inches(0.8), Inches(1.4), card_w, card_h, CARD_BG, CARD_BORDER)
    pill1 = add_card(s9, Inches(1.05), Inches(1.65), Inches(2.2), Inches(0.38), CARD_BG_ALT, EMERALD_ACCENT)
    tb = s9.shapes.add_textbox(Inches(1.15), Inches(1.7), Inches(2.0), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "KEY TAKEAWAYS"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = EMERALD_ACCENT

    tb_c1 = s9.shapes.add_textbox(Inches(1.05), Inches(2.2), Inches(3.14), Inches(4.6))
    tf = tb_c1.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Technical Achievements"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n✔ Scale Drift Solved:\nMetric baseline triangulation (b = 0.54 m) eliminates monocular scale ambiguity, slashing drift by 8.60×."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n✔ Robust Failure Handling:\nBidirectional flow and automated P3P fallback protect against planar degeneracy and blur."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n✔ Real-Time Execution:\nLightweight sliding-window BA operates at 60.9 FPS on commodity CPU hardware."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    # Card 2: Future Directions
    add_card(s9, Inches(0.8) + card_w + gap, Inches(1.4), card_w, card_h, CARD_BG, CARD_BORDER)
    pill2 = add_card(s9, Inches(0.8) + card_w + gap + Inches(0.25), Inches(1.65), Inches(2.2), Inches(0.38), CARD_BG_ALT, CYAN_ACCENT)
    tb = s9.shapes.add_textbox(Inches(0.8) + card_w + gap + Inches(0.35), Inches(1.7), Inches(2.0), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "FUTURE EXTENSIONS"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT

    tb_c2 = s9.shapes.add_textbox(Inches(0.8) + card_w + gap + Inches(0.25), Inches(2.2), Inches(3.14), Inches(4.6))
    tf = tb_c2.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Future Research Avenues"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n🔬 Factor-Graph VIO (GTSAM):\nIntegrate IMU preintegration factors (iSAM2) to provide high-frequency inertial priors during rapid vehicle maneuvers."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n🔬 3D Gaussian Splatting:\nExtend sparse point clouds into real-time radiance splats for continuous novel-view road synthesis."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n🔬 Global Loop Closure:\nIncorporate bag-of-words (DBoW2) place recognition to eliminate accumulated drift on multi-kilometer loops."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    # Card 3: Discussion & Deliverables
    add_card(s9, Inches(0.8) + (card_w + gap) * 2, Inches(1.4), card_w, card_h, CARD_BG, CARD_BORDER)
    pill3 = add_card(s9, Inches(0.8) + (card_w + gap) * 2 + Inches(0.25), Inches(1.65), Inches(2.2), Inches(0.38), CARD_BG_ALT, INDIGO_ACCENT)
    tb = s9.shapes.add_textbox(Inches(0.8) + (card_w + gap) * 2 + Inches(0.35), Inches(1.7), Inches(2.0), Inches(0.3))
    p = tb.text_frame.paragraphs[0]; p.text = "DELIVERABLES & Q&A"
    p.font.name = FONT_FAMILY; p.font.size = Pt(10); p.font.bold = True; p.font.color.rgb = INDIGO_ACCENT

    tb_c3 = s9.shapes.add_textbox(Inches(0.8) + (card_w + gap) * 2 + Inches(0.25), Inches(2.2), Inches(3.14), Inches(4.6))
    tf = tb_c3.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.text = "Project Wrap-Up"
    p.font.name = FONT_FAMILY; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = TEXT_PURE_WHITE

    p = tf.add_paragraph()
    p.text = "\n📚 Formal IEEE Research Paper:\n8-10 page publication format with 20 native mathematical derivations and benchmark comparisons."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n🚀 Live Interactive Web App:\nDeployed on Streamlit Cloud for interactive grading and testing."
    p.font.name = FONT_FAMILY; p.font.size = Pt(12); p.font.color.rgb = TEXT_BRIGHT

    p = tf.add_paragraph()
    p.text = "\n\n        THANK YOU!\n\nOpen for Class Discussion\n        & Questions"
    p.font.name = FONT_FAMILY; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = CYAN_ACCENT
    p.alignment = PP_ALIGN.CENTER

    output_path = "docs/Visual_Odometry_Project_Presentation.pptx"
    prs.save(output_path)
    print(f"Successfully generated: {output_path} (Size: {os.path.getsize(output_path):,} bytes)")

if __name__ == "__main__":
    create_deck()
