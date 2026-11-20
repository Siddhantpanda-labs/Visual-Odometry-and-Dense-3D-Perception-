import os
import cv2
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# =============================================================================
# DESIGN SYSTEM: "Quantum" Clean Academic — White + Forest Green + Mint
# Inspired by: Clean research methodology decks, white backgrounds,
# dark-green accent, geometric shapes, generous whitespace
# =============================================================================

# Colors
BG_WHITE       = RGBColor(255, 255, 255)
BG_OFFWHITE    = RGBColor(248, 250, 252)   # #F8FAFC very subtle cool-gray bg
SURFACE_MINT   = RGBColor(220, 240, 230)   # #DCF0E6 light mint panels
SURFACE_LIGHT  = RGBColor(236, 253, 243)   # #ECFDF3 ultra-light green tint

GREEN_DARK     = RGBColor(27, 67, 50)      # #1B4332 deep forest green
GREEN_PRIMARY  = RGBColor(45, 106, 79)     # #2D6A4F primary green
GREEN_MID      = RGBColor(64, 145, 108)    # #40916C medium green
GREEN_LIGHT    = RGBColor(82, 183, 136)    # #52B788 light green
GREEN_PALE     = RGBColor(183, 228, 199)   # #B7E4C7 pale green

TEXT_BLACK     = RGBColor(17, 24, 39)      # #111827 near-black
TEXT_DARK      = RGBColor(31, 41, 55)      # #1F2937 dark gray
TEXT_BODY      = RGBColor(55, 65, 81)      # #374151 body text
TEXT_MUTED     = RGBColor(107, 114, 128)   # #6B7280 muted gray
TEXT_LIGHT_MUT = RGBColor(156, 163, 175)   # #9CA3AF light muted

BORDER_LIGHT   = RGBColor(229, 231, 235)   # #E5E7EB subtle border
BORDER_GREEN   = RGBColor(167, 215, 189)   # #A7D7BD green-tinted border

FONT_TITLE     = "Segoe UI"
FONT_BODY      = "Segoe UI"
FONT_CODE      = "Consolas"

SW = Inches(13.333)
SH = Inches(7.5)


def create_presentation():
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH
    blank = prs.slide_layouts[6]

    # ── Helper Functions ─────────────────────────────────────────────

    def white_bg(slide):
        """Fill slide background white."""
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
        bg.fill.solid(); bg.fill.fore_color.rgb = BG_WHITE
        bg.line.fill.background()

    def top_bar(slide, slide_num=None):
        """Green accent stripe along top edge, like the Quantum nav bar."""
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.06))
        bar.fill.solid(); bar.fill.fore_color.rgb = GREEN_PRIMARY
        bar.line.fill.background()
        # Slide number badge in top-right
        if slide_num:
            badge = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(12.1), Inches(0.25), Inches(0.9), Inches(0.4)
            )
            badge.fill.solid(); badge.fill.fore_color.rgb = GREEN_PRIMARY
            badge.line.fill.background()
            tf = badge.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            run = p.add_run(); run.text = f"0{slide_num} / 09"
            run.font.name = FONT_BODY; run.font.size = Pt(9)
            run.font.bold = True; run.font.color.rgb = BG_WHITE

    def section_title(slide, text, top=Inches(0.55), left=Inches(0.9)):
        """Small green-accented section title with arrow chevron."""
        # Green chevron arrow
        chev = slide.shapes.add_shape(
            MSO_SHAPE.CHEVRON, left, top + Inches(0.03), Inches(0.3), Inches(0.3)
        )
        chev.fill.solid(); chev.fill.fore_color.rgb = GREEN_PRIMARY
        chev.line.fill.background()
        # Title text
        tb = slide.shapes.add_textbox(left + Inches(0.4), top, Inches(10), Inches(0.42))
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = text
        run.font.name = FONT_TITLE; run.font.size = Pt(22)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK

    def card_rect(slide, left, top, w, h, fill=SURFACE_MINT, border=None):
        """Soft rounded card."""
        c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, w, h)
        c.fill.solid(); c.fill.fore_color.rgb = fill
        if border:
            c.line.color.rgb = border; c.line.width = Pt(1)
        else:
            c.line.fill.background()
        return c

    def green_circle(slide, left, top, size, label_text="", icon_text=""):
        """Green circle icon badge, like the Quantum template circles."""
        circ = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, size, size)
        circ.fill.solid(); circ.fill.fore_color.rgb = GREEN_PRIMARY
        circ.line.fill.background()
        if icon_text:
            tf = circ.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            run = p.add_run(); run.text = icon_text
            run.font.name = FONT_BODY; run.font.size = Pt(int(size / Inches(1) * 11))
            run.font.bold = True; run.font.color.rgb = BG_WHITE

    def text_box(slide, left, top, w, h):
        tb = slide.shapes.add_textbox(left, top, w, h)
        tf = tb.text_frame; tf.word_wrap = True
        tf.margin_left = Pt(4); tf.margin_top = Pt(2)
        tf.margin_right = Pt(4); tf.margin_bottom = Pt(2)
        return tf

    def add_para(tf, text, size=11, color=TEXT_BODY, bold=False, font=FONT_BODY, align=None, spacing_before=0):
        p = tf.add_paragraph()
        run = p.add_run(); run.text = text
        run.font.name = font; run.font.size = Pt(size)
        run.font.bold = bold; run.font.color.rgb = color
        if align: p.alignment = align
        if spacing_before: p.space_before = Pt(spacing_before)
        return p

    def fit_picture(slide, path, left, top, max_w, max_h):
        if not os.path.exists(path): return None
        img = cv2.imread(path)
        if img is None: return None
        ih, iw = img.shape[:2]
        ratio = iw / ih
        tw = max_w; th = tw / ratio
        if th > max_h:
            th = max_h; tw = th * ratio
        ox = (max_w - tw) / 2; oy = (max_h - th) / 2
        return slide.shapes.add_picture(
            path,
            Inches(left + ox), Inches(top + oy),
            width=Inches(tw), height=Inches(th)
        )

    def bottom_divider(slide, top):
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0.9), Inches(top), Inches(11.533), Inches(0.015)
        )
        line.fill.solid(); line.fill.fore_color.rgb = BORDER_LIGHT
        line.line.fill.background()

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 1 — TITLE SLIDE (Hero)
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s)
    top_bar(s)

    # Left green accent sidebar
    sidebar = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.12), SH
    )
    sidebar.fill.solid(); sidebar.fill.fore_color.rgb = GREEN_PRIMARY
    sidebar.line.fill.background()

    # Course Badge Pill
    pill = card_rect(s, Inches(1.0), Inches(1.0), Inches(5.5), Inches(0.45), GREEN_PRIMARY)
    tf = pill.text_frame
    tf.margin_left = Pt(12); tf.margin_top = Pt(2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    run = p.add_run(); run.text = "CSE471  ·  Computer Vision  ·  Final Project Presentation"
    run.font.name = FONT_BODY; run.font.size = Pt(11)
    run.font.bold = True; run.font.color.rgb = BG_WHITE

    # Main Title
    tf = text_box(s, Inches(1.0), Inches(1.75), Inches(7.5), Inches(1.6))
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Robust Multi-Sensor\nVisual Odometry &\nDense 3D Perception"
    run.font.name = FONT_TITLE; run.font.size = Pt(36)
    run.font.bold = True; run.font.color.rgb = TEXT_BLACK

    # Subtitle
    tf_sub = text_box(s, Inches(1.0), Inches(3.55), Inches(7.5), Inches(0.5))
    p = tf_sub.paragraphs[0]
    run = p.add_run()
    run.text = "Real-Time Metric Navigation Under Geometric Degeneracy & Highway Scale Drift"
    run.font.name = FONT_BODY; run.font.size = Pt(14)
    run.font.color.rgb = GREEN_PRIMARY; run.font.bold = True

    # Thin green separator line
    sep = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(4.2), Inches(3.5), Inches(0.04)
    )
    sep.fill.solid(); sep.fill.fore_color.rgb = GREEN_PRIMARY
    sep.line.fill.background()

    # Candidate Info Rows
    tf_info = text_box(s, Inches(1.0), Inches(4.45), Inches(6.0), Inches(2.0))
    p = tf_info.paragraphs[0]
    run = p.add_run(); run.text = "Sidhant Kumar Panda"
    run.font.name = FONT_TITLE; run.font.size = Pt(16)
    run.font.bold = True; run.font.color.rgb = TEXT_BLACK

    add_para(tf_info, "Registration No: 12413416", 12, GREEN_PRIMARY, True, spacing_before=4)
    add_para(tf_info, "Project Guide: Abhishek Tiwari", 12, TEXT_BODY, spacing_before=4)
    add_para(tf_info, "School of Computer Science & Engineering · Division of Machine Intelligence", 11, TEXT_MUTED, spacing_before=4)

    # Stat Cards Row at Bottom
    stat_data = [("8.60×", "Drift Reduction"), ("1.87%", "Pose Error"), ("60.9", "FPS Real-Time"), ("350K+", "3D Points")]
    for i, (val, label) in enumerate(stat_data):
        cx = Inches(1.0 + i * 1.85)
        card_rect(s, cx, Inches(6.0), Inches(1.65), Inches(1.0), SURFACE_MINT, BORDER_GREEN)
        tf = text_box(s, cx + Inches(0.1), Inches(6.05), Inches(1.45), Inches(0.9))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = val
        run.font.name = FONT_TITLE; run.font.size = Pt(18)
        run.font.bold = True; run.font.color.rgb = GREEN_PRIMARY
        add_para(tf, label, 9, TEXT_MUTED)

    # Right side: Architecture image in a mint-tinted card (BIGGER)
    card_rect(s, Inches(8.3), Inches(1.0), Inches(4.633), Inches(5.8), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig1_architecture.png", 8.4, 1.15, 4.433, 5.5)

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 2 — PROBLEM STATEMENT: 3 Failure Modes
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 2)
    section_title(s, "Research Problem & Motivation")

    # Three column cards
    col_w = Inches(3.64); col_h = Inches(5.6)
    gap = Inches(0.4); col_top = Inches(1.3)

    for i, (num, title, color, bullets) in enumerate([
        ("01", "Unconstrained\nScale Drift", GREEN_DARK, [
            "Monocular geometry recovers only relative translation direction — absolute metric scale factor remains unobservable.",
            "Small per-frame scale errors compound quadratically, inflating trajectory length by >16% over extended routes.",
            "Without external sensors, the system cannot distinguish high-speed highway driving from low-speed crawling.",
        ]),
        ("02", "Motion Blur &\nTrack Loss", GREEN_PRIMARY, [
            "Highway chassis vibrations, pavement bumps, and engine oscillations smear spatial gradients across the image.",
            "Harris corner scores drop below threshold τ, causing KLT tracker to shed features and collapse to zero active tracks.",
            "Standard pipelines abort without recovery, causing complete odometry blackout during aggressive maneuvers.",
        ]),
        ("03", "Planar Road\nDegeneracy", GREEN_MID, [
            "When all features lie on the flat road plane, the Essential Matrix E becomes mathematically degenerate.",
            "Degenerate decomposition yields two valid rotations — the solver frequently selects the false 'ghost' solution.",
            "RANSAC converges on erroneous motion vectors, twisting straight trajectories into fictitious spiral paths.",
        ]),
    ]):
        cx = Inches(0.9) + (col_w + gap) * i

        # Card container
        card_rect(s, cx, col_top, col_w, col_h, BG_OFFWHITE, BORDER_LIGHT)

        # Number circle
        green_circle(s, cx + Inches(0.3), col_top + Inches(0.3), Inches(0.55), icon_text=num)

        # Card Title
        tf = text_box(s, cx + Inches(1.0), col_top + Inches(0.25), col_w - Inches(1.3), Inches(0.7))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(16)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK

        # Green divider inside card
        div = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cx + Inches(0.3), col_top + Inches(1.15),
            col_w - Inches(0.6), Inches(0.025)
        )
        div.fill.solid(); div.fill.fore_color.rgb = color
        div.line.fill.background()

        # Bullet points
        for j, bullet in enumerate(bullets):
            # Green dot
            dot = s.shapes.add_shape(
                MSO_SHAPE.OVAL,
                cx + Inches(0.35), col_top + Inches(1.45 + j * 1.4),
                Inches(0.12), Inches(0.12)
            )
            dot.fill.solid(); dot.fill.fore_color.rgb = color
            dot.line.fill.background()

            tf = text_box(s,
                cx + Inches(0.55), col_top + Inches(1.35 + j * 1.4),
                col_w - Inches(0.9), Inches(1.3)
            )
            p = tf.paragraphs[0]
            run = p.add_run(); run.text = bullet
            run.font.name = FONT_BODY; run.font.size = Pt(11)
            run.font.color.rgb = TEXT_BODY

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 3 — SYSTEM ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 3)
    section_title(s, "System Architecture & Processing Pipeline")

    # Left: 4-step pipeline cards (stacked vertically, narrower to give more room to image)
    steps = [
        ("01", "Adaptive Radiometric Normalization",
         "CLAHE (clip limit 3.0, 8×8 grid) equalizes harsh windshield shadows and asphalt glare before feature detection."),
        ("02", "Pyramidal KLT Optical Flow",
         "Tracks 1,500 Shi-Tomasi corners across 4 pyramid levels with bidirectional consistency check (‖xᵢ − x̂ᵢ‖₂ < 1.0 px)."),
        ("03", "Failure-Gating State Machine",
         "Monitors inlier count N and baseline magnitude. Automatically shifts between 5-point Essential Matrix and P3P RANSAC."),
        ("04", "Multi-Sensor Fallback",
         "Operates in calibrated stereo (b = 0.54 m) as primary mode; gracefully degrades to monocular when second feed drops."),
    ]
    for i, (num, title, desc) in enumerate(steps):
        y = Inches(1.25 + i * 1.45)
        card_rect(s, Inches(0.9), y, Inches(4.8), Inches(1.25), SURFACE_LIGHT if i % 2 == 0 else BG_OFFWHITE, BORDER_GREEN if i % 2 == 0 else BORDER_LIGHT)
        green_circle(s, Inches(1.1), y + Inches(0.17), Inches(0.45), icon_text=num)
        tf = text_box(s, Inches(1.7), y + Inches(0.08), Inches(3.7), Inches(0.35))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(13)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK
        tf = text_box(s, Inches(1.7), y + Inches(0.45), Inches(3.7), Inches(0.7))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = desc
        run.font.name = FONT_BODY; run.font.size = Pt(10.5)
        run.font.color.rgb = TEXT_BODY

    # Right: Architecture Diagram Image (BIGGER — takes more space)
    card_rect(s, Inches(6.0), Inches(1.25), Inches(6.733), Inches(5.8), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig1_architecture.png", 6.1, 1.4, 6.533, 5.3)
    tf = text_box(s, Inches(6.1), Inches(6.65), Inches(6.533), Inches(0.3))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Fig 1: End-to-end perception pipeline with failure-gated state machine"
    run.font.name = FONT_BODY; run.font.size = Pt(9); run.font.color.rgb = TEXT_MUTED

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 4 — MATHEMATICAL METHODOLOGY
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 4)
    section_title(s, "Mathematical Methodology")

    # Two large side-by-side panels with PROPER MATH
    pw = Inches(5.666)
    for i, (title, accent, items) in enumerate([
        ("Calibrated Stereo Metric Triangulation", GREEN_PRIMARY, [
            ("Physical Baseline", "Calibrated stereo baseline  b = 0.54 m  between left and right optical centers provides direct metric scale observability at every frame.", False),
            ("Depth Triangulation", "Z  =  f · b  /  d", True),
            ("3D Reconstruction", "X  =  (u − cₓ) · Z / f        Y  =  (v − c_y) · Z / f", True),
            ("Scale Guarantee", "Every 3D landmark Xⱼ is instantiated in physical meters (SI), eliminating speed ambiguity and trajectory scale inflation.", False),
            ("Precision Bound", "δZ  =  ( Z² / f·b ) · δd       →  sub-decimeter resolution up to 45 m", True),
        ]),
        ("Sliding-Window Bundle Adjustment", GREEN_MID, [
            ("Joint Optimization", "Jointly refines last W = 5 camera poses  Tₖ = [Rₖ | tₖ]  and 3D landmark positions  Xⱼ  across temporal views.", False),
            ("Robust Huber Cost", "min      Σ  ρ_Huber(  ‖ uᵢ,ₖ  −  π( Tₖ , Xⱼ ) ‖²  )", True),
            ("Levenberg-Marquardt", "( Jᵀ J  +  λ · diag(Jᵀ J) )  Δξ  =  − Jᵀ r", True),
            ("Convergence", "Achieves quadratic convergence within 8–12 iterations per keyframe window.", False),
            ("Real-Time Bound", "Fixed-lag sliding window ensures strict  O(W)  constant-time computation with zero memory growth.", False),
        ]),
    ]):
        cx = Inches(0.9) + (pw + Inches(0.4)) * i
        card_rect(s, cx, Inches(1.3), pw, Inches(5.8), BG_OFFWHITE, BORDER_LIGHT)

        # Panel title bar
        bar = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, cx, Inches(1.3), pw, Inches(0.5)
        )
        bar.fill.solid(); bar.fill.fore_color.rgb = accent
        bar.line.fill.background()
        tf = bar.text_frame
        tf.margin_left = Pt(16); tf.margin_top = Pt(4)
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(14)
        run.font.bold = True; run.font.color.rgb = BG_WHITE

        # Items
        for j, (sub_title, sub_desc, is_formula) in enumerate(items):
            y = Inches(2.0 + j * 1.0)
            # Green dot
            dot = s.shapes.add_shape(MSO_SHAPE.OVAL, cx + Inches(0.25), y + Inches(0.08), Inches(0.12), Inches(0.12))
            dot.fill.solid(); dot.fill.fore_color.rgb = accent; dot.line.fill.background()
            # Sub-title
            tf = text_box(s, cx + Inches(0.45), y - Inches(0.03), pw - Inches(0.7), Inches(0.3))
            p = tf.paragraphs[0]
            run = p.add_run(); run.text = sub_title
            run.font.name = FONT_TITLE; run.font.size = Pt(12)
            run.font.bold = True; run.font.color.rgb = TEXT_BLACK
            
            if is_formula:
                # Math formula in centered highlight container
                formula_bg = card_rect(s, cx + Inches(0.4), y + Inches(0.25),
                    pw - Inches(0.6), Inches(0.42), SURFACE_LIGHT, BORDER_GREEN)
                tf2 = text_box(s, cx + Inches(0.45), y + Inches(0.27), pw - Inches(0.7), Inches(0.36))
                p2 = tf2.paragraphs[0]; p2.alignment = PP_ALIGN.CENTER
                run2 = p2.add_run(); run2.text = sub_desc
                run2.font.name = "Cambria Math"; run2.font.size = Pt(12.5)
                run2.font.bold = True; run2.font.color.rgb = accent
                run2.font.italic = True
            else:
                tf = text_box(s, cx + Inches(0.45), y + Inches(0.28), pw - Inches(0.7), Inches(0.65))
                p = tf.paragraphs[0]
                run = p.add_run(); run.text = sub_desc
                run.font.name = FONT_BODY; run.font.size = Pt(10.5)
                run.font.color.rgb = TEXT_BODY

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 5 — BENCHMARK & ABLATION
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 5)
    section_title(s, "Benchmark Results & Ablation Study")

    # 3 large KPI stat circles
    kpis = [
        ("8.60×", "Drift Reduction", "ATE RMSE: 26.31 m → 3.06 m"),
        ("1.87%", "Relative Pose Error", "Across 450 controlled frames"),
        ("60.9 FPS", "Real-Time Throughput", "Multi-threaded perception pipeline"),
    ]
    for i, (val, label, sub) in enumerate(kpis):
        cx = Inches(1.3 + i * 2.0)
        # Outer green ring (using a bigger circle behind)
        ring = s.shapes.add_shape(MSO_SHAPE.OVAL, cx, Inches(1.3), Inches(1.6), Inches(1.6))
        ring.fill.solid(); ring.fill.fore_color.rgb = GREEN_PRIMARY
        ring.line.fill.background()
        inner = s.shapes.add_shape(MSO_SHAPE.OVAL, cx + Inches(0.08), Inches(1.38), Inches(1.44), Inches(1.44))
        inner.fill.solid(); inner.fill.fore_color.rgb = BG_WHITE
        inner.line.fill.background()
        # Value text centered in circle
        tf = text_box(s, cx + Inches(0.1), Inches(1.65), Inches(1.4), Inches(0.8))
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        run = p.add_run(); run.text = val
        run.font.name = FONT_TITLE; run.font.size = Pt(17)
        run.font.bold = True; run.font.color.rgb = GREEN_PRIMARY
        # Label below circle
        tf = text_box(s, cx - Inches(0.2), Inches(3.0), Inches(2.0), Inches(0.6))
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        run = p.add_run(); run.text = label
        run.font.name = FONT_TITLE; run.font.size = Pt(11)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK
        add_para(tf, sub, 9, TEXT_MUTED, align=PP_ALIGN.CENTER)

    # Benchmark Table (left bottom)
    card_rect(s, Inches(0.9), Inches(3.65), Inches(6.0), Inches(3.5), BG_OFFWHITE, BORDER_LIGHT)
    tf = text_box(s, Inches(1.15), Inches(3.75), Inches(5.5), Inches(3.2))
    p = tf.paragraphs[0]
    run = p.add_run(); run.text = "Ablation Benchmark Comparison"
    run.font.name = FONT_TITLE; run.font.size = Pt(14)
    run.font.bold = True; run.font.color.rgb = TEXT_BLACK

    rows = [
        ("Method Variant", "ATE RMSE", "RPE Drift", "FPS", True, GREEN_PRIMARY),
        ("1.  Monocular 5-Pt VO", "26.31 m", "16.14%", "68.4", False, TEXT_MUTED),
        ("2.  Pure Stereo VO", "5.42 m", "3.32%", "64.1", False, TEXT_BODY),
        ("3.  Stereo + Failure Gating", "4.18 m", "2.56%", "62.7", False, TEXT_BODY),
        ("4.  Full System (Ours)", "3.06 m", "1.87%", "60.9", True, GREEN_PRIMARY),
    ]
    for j, (meth, ate, rpe, fps, bold, color) in enumerate(rows):
        txt = f"{meth:<30s} {ate:<12s} {rpe:<12s} {fps}"
        add_para(tf, txt, 11, color, bold, FONT_CODE, spacing_before=6 if j == 0 else 3)

    # Ablation Chart (right — FULL HEIGHT for bigger image)
    card_rect(s, Inches(7.2), Inches(1.3), Inches(5.533), Inches(5.85), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig8_ablation_barchart.png", 7.3, 1.4, 5.333, 5.55)

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 6 — REAL-WORLD HIGHWAY VALIDATION
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 6)
    section_title(s, "Real-World Highway Validation & Error Diagnosis")

    # Left: Analysis panel (narrower — give more space to image)
    card_rect(s, Inches(0.9), Inches(1.3), Inches(5.0), Inches(5.8), BG_OFFWHITE, BORDER_LIGHT)
    tf = text_box(s, Inches(1.15), Inches(1.5), Inches(4.5), Inches(5.5))

    p = tf.paragraphs[0]
    run = p.add_run(); run.text = "The 80° Visual Perception Paradox"
    run.font.name = FONT_TITLE; run.font.size = Pt(17)
    run.font.bold = True; run.font.color.rgb = TEXT_BLACK

    analysis_items = [
        ("Empirical Observation", "Human viewers perceive an abrupt ~80° right turn in dashcam footage. However, the BEV trajectory measures a gentle 10°–20° curve.", GREEN_PRIMARY),
        ("AASHTO Ground Truth", "Highway design standards mandate curvature radii R ≥ 250–400 m at 80–100 km/h. Over a 200 m arc segment, the physical angular change is bounded: Δθ ≈ 15°.", GREEN_MID),
        ("Optical Root Cause", "At highway distances (Z > 50 m), flow vectors become near-parallel. Chassis pitch-roll vibrations create apparent angular shifts, causing observers to overestimate yaw rotation.", GREEN_LIGHT),
        ("Conclusion", "The system's bird's-eye trajectory correctly captures physical ground truth, debunking the optical illusion of an 80° turn.", GREEN_DARK),
    ]
    for label, desc, accent in analysis_items:
        add_para(tf, "", 4, TEXT_MUTED, spacing_before=10)
        # Label
        p = tf.add_paragraph()
        run = p.add_run(); run.text = f"  ● {label}"
        run.font.name = FONT_TITLE; run.font.size = Pt(12)
        run.font.bold = True; run.font.color.rgb = accent
        # Description
        p = tf.add_paragraph()
        run = p.add_run(); run.text = f"    {desc}"
        run.font.name = FONT_BODY; run.font.size = Pt(10.5)
        run.font.color.rgb = TEXT_BODY

    # Right: BEV trajectory image (BIGGER)
    card_rect(s, Inches(6.2), Inches(1.3), Inches(6.433), Inches(5.8), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig6_real_driving_bev.png", 6.3, 1.45, 6.233, 5.2)
    tf = text_box(s, Inches(6.3), Inches(6.55), Inches(6.233), Inches(0.4))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Fig 6: BEV trajectory — 1,260 highway frames (1.06 km)"
    run.font.name = FONT_BODY; run.font.size = Pt(9); run.font.color.rgb = TEXT_MUTED

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 7 — DENSE 3D RECONSTRUCTION
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 7)
    section_title(s, "Dense 3D Scene Reconstruction")

    # Left: explanation (narrower for bigger image)
    card_rect(s, Inches(0.9), Inches(1.3), Inches(5.0), Inches(5.8), BG_OFFWHITE, BORDER_LIGHT)
    
    recon_steps = [
        ("Real-Time Depth (>150 FPS)", "Bilateral-guided geometric disparity filtering extracts dense per-pixel depth maps with edge-preserving filtering — no neural network required.", False),
        ("Pinhole Back-Projection", "X_c  =  Z · K⁻¹ [ u,  v,  1 ]ᵀ", True),
        ("World Frame Accumulation", "X_w  =  Rₜ · X_c  +  tₜ", True),
        ("350,000+ Metric Vertices", "RGB-colored point cloud with true physical scale, exportable as standard .PLY format for downstream analysis.", False),
        ("Interactive WebGL Viewer", "Embedded Three.js canvas in the web dashboard with WASD free-fly camera controls for real-time 3D exploration.", False),
    ]
    for i, (title, desc, is_formula) in enumerate(recon_steps):
        y = Inches(1.5 + i * 1.08)
        green_circle(s, Inches(1.1), y + Inches(0.05), Inches(0.35), icon_text=str(i+1))
        tf = text_box(s, Inches(1.6), y - Inches(0.05), Inches(4.0), Inches(0.3))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(12)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK
        if is_formula:
            # Formula in highlight box
            formula_bg = card_rect(s, Inches(1.55), y + Inches(0.24), Inches(4.0), Inches(0.42), SURFACE_LIGHT, BORDER_GREEN)
            tf = text_box(s, Inches(1.65), y + Inches(0.27), Inches(3.8), Inches(0.35))
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            run = p.add_run(); run.text = desc
            run.font.name = "Cambria Math"; run.font.size = Pt(13)
            run.font.bold = True; run.font.italic = True; run.font.color.rgb = GREEN_PRIMARY
        else:
            tf = text_box(s, Inches(1.6), y + Inches(0.26), Inches(4.0), Inches(0.75))
            p = tf.paragraphs[0]
            run = p.add_run(); run.text = desc
            run.font.name = FONT_BODY; run.font.size = Pt(10)
            run.font.color.rgb = TEXT_BODY

    # Right: 3D pointcloud image (BIGGER)
    card_rect(s, Inches(6.2), Inches(1.3), Inches(6.433), Inches(5.8), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig7_3d_pointcloud.png", 6.3, 1.45, 6.233, 5.2)
    tf = text_box(s, Inches(6.3), Inches(6.55), Inches(6.233), Inches(0.4))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Fig 7: Reconstructed highway point cloud with perspective depth coloring"
    run.font.name = FONT_BODY; run.font.size = Pt(9); run.font.color.rgb = TEXT_MUTED

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 8 — LIVE DASHBOARD & CLOUD DEPLOYMENT
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 8)
    section_title(s, "Implementation & Live Cloud Deployment")

    # Left: Dashboard Screenshot (BIGGER — wider)
    card_rect(s, Inches(0.9), Inches(1.3), Inches(7.0), Inches(5.8), SURFACE_MINT, BORDER_GREEN)
    fit_picture(s, "docs/figures/fig3b_real_dashboard.png", 1.0, 1.5, 6.8, 5.1)
    tf = text_box(s, Inches(1.0), Inches(6.55), Inches(6.8), Inches(0.4))
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Fig 8: Synchronized 4-Pane HUD — Flow Vectors, Depth Map, BEV Trajectory, Telemetry"
    run.font.name = FONT_BODY; run.font.size = Pt(9); run.font.color.rgb = TEXT_MUTED

    # Right: Feature cards stacked (narrower)
    features = [
        ("🎥", "On-The-Fly Video Processing", "Users upload custom MP4 dashcam footage; the engine tracks KLT features and depth dynamically frame-by-frame."),
        ("🌐", "Zero-GPU Cloud Deployment", "Optimized OpenCV-headless pipeline runs on standard CPU containers — no GPU or PyTorch required."),
        ("📹", "Browser H.264 Playback", "Pre-rendered output videos encoded with H.264 (yuv420p) play natively in all modern browsers."),
        ("☁️", "Public Streamlit Cloud URL", "https://qgqccuvxnhc3gp32otvk6t.streamlit.app/"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        y = Inches(1.3 + i * 1.4)
        card_rect(s, Inches(8.2), y, Inches(4.233), Inches(1.2),
                  SURFACE_LIGHT if i % 2 == 0 else BG_OFFWHITE,
                  BORDER_GREEN if i == 3 else BORDER_LIGHT)
        # Icon circle
        green_circle(s, Inches(8.35), y + Inches(0.2), Inches(0.45), icon_text=icon)
        # Title
        tf = text_box(s, Inches(8.95), y + Inches(0.1), Inches(3.38), Inches(0.3))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(12)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK
        # Desc
        tf = text_box(s, Inches(8.95), y + Inches(0.42), Inches(3.38), Inches(0.65))
        p = tf.paragraphs[0]
        is_url = desc.startswith("http")
        run = p.add_run(); run.text = desc
        run.font.name = FONT_CODE if is_url else FONT_BODY
        run.font.size = Pt(10.5 if is_url else 10)
        run.font.bold = is_url
        run.font.color.rgb = GREEN_PRIMARY if is_url else TEXT_BODY

    # ══════════════════════════════════════════════════════════════════
    #  SLIDE 9 — CONCLUSION, FUTURE SCOPE & THANK YOU
    # ══════════════════════════════════════════════════════════════════
    s = prs.slides.add_slide(blank)
    white_bg(s); top_bar(s, 9)
    section_title(s, "Conclusion & Future Scope")

    # Three column layout
    col_w = Inches(3.64); gap = Inches(0.4); col_top = Inches(1.3)

    # Column 1: Key Takeaways
    card_rect(s, Inches(0.9), col_top, col_w, Inches(4.2), BG_OFFWHITE, BORDER_LIGHT)
    # Header bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), col_top, col_w, Inches(0.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN_DARK; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = Pt(16); tf.margin_top = Pt(4)
    p = tf.paragraphs[0]; run = p.add_run(); run.text = "Key Contributions"
    run.font.name = FONT_TITLE; run.font.size = Pt(13); run.font.bold = True; run.font.color.rgb = BG_WHITE

    takeaways = [
        "Scale drift eliminated — metric stereo baseline (b = 0.54 m) achieves 8.60× ATE improvement.",
        "Automated failure-gating handles planar degeneracy and motion blur without manual intervention.",
        "Real-time 60.9 FPS execution on commodity CPU hardware with sliding-window BA optimization.",
        "Dense 3D reconstruction generates 350K+ metric RGB point cloud vertices from monocular + stereo input.",
    ]
    for i, item in enumerate(takeaways):
        y = col_top + Inches(0.65 + i * 0.85)
        green_circle(s, Inches(1.1), y + Inches(0.02), Inches(0.25), icon_text="✓")
        tf = text_box(s, Inches(1.5), y - Inches(0.05), Inches(2.8), Inches(0.8))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = item
        run.font.name = FONT_BODY; run.font.size = Pt(10.5)
        run.font.color.rgb = TEXT_BODY

    # Column 2: Future Scope
    cx2 = Inches(0.9) + col_w + gap
    card_rect(s, cx2, col_top, col_w, Inches(4.2), BG_OFFWHITE, BORDER_LIGHT)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx2, col_top, col_w, Inches(0.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN_PRIMARY; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = Pt(16); tf.margin_top = Pt(4)
    p = tf.paragraphs[0]; run = p.add_run(); run.text = "Future Research Directions"
    run.font.name = FONT_TITLE; run.font.size = Pt(13); run.font.bold = True; run.font.color.rgb = BG_WHITE

    futures = [
        ("Factor-Graph VIO", "Integrate GTSAM iSAM2 with IMU preintegration for high-frequency inertial priors during rapid maneuvers."),
        ("3D Gaussian Splatting", "Extend sparse point clouds into real-time radiance splats for continuous novel-view road synthesis."),
        ("Loop Closure (DBoW2)", "Bag-of-words place recognition to detect and correct accumulated drift on multi-kilometer routes."),
    ]
    for i, (title, desc) in enumerate(futures):
        y = col_top + Inches(0.7 + i * 1.15)
        green_circle(s, cx2 + Inches(0.2), y + Inches(0.02), Inches(0.25), icon_text="→")
        tf = text_box(s, cx2 + Inches(0.55), y - Inches(0.05), Inches(2.85), Inches(0.3))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(11)
        run.font.bold = True; run.font.color.rgb = GREEN_PRIMARY
        tf = text_box(s, cx2 + Inches(0.55), y + Inches(0.25), Inches(2.85), Inches(0.8))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = desc
        run.font.name = FONT_BODY; run.font.size = Pt(10)
        run.font.color.rgb = TEXT_BODY

    # Column 3: Thank You / Deliverables
    cx3 = Inches(0.9) + (col_w + gap) * 2
    card_rect(s, cx3, col_top, col_w, Inches(4.2), SURFACE_MINT, BORDER_GREEN)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, cx3, col_top, col_w, Inches(0.5))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN_MID; bar.line.fill.background()
    tf = bar.text_frame; tf.margin_left = Pt(16); tf.margin_top = Pt(4)
    p = tf.paragraphs[0]; run = p.add_run(); run.text = "Project Deliverables"
    run.font.name = FONT_TITLE; run.font.size = Pt(13); run.font.bold = True; run.font.color.rgb = BG_WHITE

    deliverables = [
        ("📄 IEEE Research Paper", "8–10 page report with 20 mathematical derivations and benchmark comparisons."),
        ("🌐 Live Interactive Web App", "Deployed on Streamlit Cloud for live grading and peer testing."),
        ("📦 Open-Source GitHub Repo", "Complete source code, test suites, trained assets, and documentation."),
    ]
    for i, (title, desc) in enumerate(deliverables):
        y = col_top + Inches(0.7 + i * 1.15)
        tf = text_box(s, cx3 + Inches(0.25), y - Inches(0.05), Inches(3.1), Inches(0.3))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = title
        run.font.name = FONT_TITLE; run.font.size = Pt(11)
        run.font.bold = True; run.font.color.rgb = TEXT_BLACK
        tf = text_box(s, cx3 + Inches(0.25), y + Inches(0.25), Inches(3.1), Inches(0.7))
        p = tf.paragraphs[0]
        run = p.add_run(); run.text = desc
        run.font.name = FONT_BODY; run.font.size = Pt(10)
        run.font.color.rgb = TEXT_BODY

    # Thank You Banner at bottom
    banner = card_rect(s, Inches(0.9), Inches(5.8), Inches(11.533), Inches(1.2), GREEN_PRIMARY)
    tf = banner.text_frame
    tf.margin_left = Pt(20); tf.margin_top = Pt(8)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Thank You"
    run.font.name = FONT_TITLE; run.font.size = Pt(28)
    run.font.bold = True; run.font.color.rgb = BG_WHITE
    p = tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
    run = p.add_run(); run.text = "Open for Class Discussion & Questions"
    run.font.name = FONT_BODY; run.font.size = Pt(13)
    run.font.color.rgb = GREEN_PALE

    # ── Save ─────────────────────────────────────────────────────────
    out = "docs/Visual_Odometry_Final_Presentation.pptx"
    prs.save(out)
    sz = os.path.getsize(out)
    print(f"✓ Generated: {out} ({sz:,} bytes, {len(prs.slides)} slides)")
    
    for alt_name in ["docs/Visual_Odometry_Presentation_v2.pptx", "docs/Visual_Odometry_Project_Presentation.pptx"]:
        try:
            prs.save(alt_name)
            print(f"✓ Synchronized: {alt_name}")
        except Exception:
            pass


if __name__ == "__main__":
    create_presentation()
