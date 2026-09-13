
import io
import math
from datetime import datetime

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw

import streamlit as st

# PDF generation
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)


# ---------------------------------------------------------------------
# Page + Theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MedVision AI | Knee Assessment",
    page_icon="MV",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#183B63"
NAVY_2 = "#214D7D"
CHARCOAL = "#20262D"
STEEL = "#39434D"
STEEL_2 = "#59636E"
LIGHT_GREY = "#AEB7C0"
MID_GREY = "#77828D"
SURFACE = "#2B333B"
SURFACE_2 = "#313B45"
BG = "#171C21"
BORDER = "#4A555F"
ACCENT = "#49B7B0"
ACCENT_2 = "#76C8C3"
TEXT = "#E7EBEF"
MUTED = "#B8C0C8"
DANGER = "#D47C7C"
AMBER = "#D0A85B"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stHeader"] {{
        background: rgba(23, 28, 33, 0.96);
    }}

    [data-testid="stSidebar"] {{
        background: {CHARCOAL};
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        letter-spacing: -0.02em;
    }}

    h1 {{ font-size: 2.25rem !important; }}
    h2 {{ font-size: 1.55rem !important; }}
    h3 {{ font-size: 1.15rem !important; }}

    p, li, label, .stMarkdown {{
        color: {MUTED};
        font-size: 1rem;
    }}

    .mv-topbar {{
        background: linear-gradient(110deg, {CHARCOAL}, {SURFACE});
        border: 1px solid {BORDER};
        border-left: 5px solid {ACCENT};
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }}

    .brand {{
        color: {TEXT};
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
    }}

    .brand span {{
        color: {ACCENT};
    }}

    .subtitle {{
        color: {MUTED};
        font-size: .98rem;
        margin-top: 4px;
    }}

    .section-title {{
        color: {TEXT};
        font-size: 1.2rem;
        font-weight: 800;
        margin: 6px 0 12px 0;
    }}

    .card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px;
        min-height: 100%;
        box-shadow: 0 8px 24px rgba(0,0,0,.14);
    }}

    .metric-card {{
        background: linear-gradient(145deg, {SURFACE}, {SURFACE_2});
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 16px;
        min-height: 112px;
    }}

    .metric-label {{
        color: {MUTED};
        font-size: .88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 1.72rem;
        font-weight: 800;
        margin-top: 6px;
    }}

    .metric-note {{
        color: {ACCENT_2};
        font-size: .82rem;
        margin-top: 2px;
    }}

    .status {{
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .02em;
        background: rgba(73,183,176,.13);
        color: {ACCENT_2};
        border: 1px solid rgba(73,183,176,.35);
    }}

    .status-warn {{
        background: rgba(208,168,91,.12);
        color: #E2C47E;
        border-color: rgba(208,168,91,.35);
    }}

    .status-danger {{
        background: rgba(212,124,124,.12);
        color: #E49A9A;
        border-color: rgba(212,124,124,.35);
    }}

    .workflow {{
        display: flex;
        gap: 7px;
        flex-wrap: wrap;
        align-items: center;
        margin: 6px 0 18px 0;
    }}

    .workflow-step {{
        background: {STEEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 8px 11px;
        color: {TEXT};
        font-size: .78rem;
        font-weight: 600;
    }}

    .workflow-step.active {{
        background: {NAVY};
        border-color: {NAVY_2};
    }}

    .arrow {{
        color: {ACCENT};
        font-weight: 800;
    }}

    .callout {{
        background: rgba(24,59,99,.33);
        border: 1px solid rgba(73,183,176,.28);
        border-radius: 12px;
        padding: 13px 15px;
        color: {TEXT};
    }}

    .finding {{
        background: {SURFACE_2};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 11px 13px;
        margin: 7px 0;
    }}

    .finding b {{
        color: {TEXT};
    }}

    .small {{
        color: {MUTED};
        font-size: .82rem;
    }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid {BORDER};
        background: {STEEL};
        color: {TEXT};
        font-weight: 700;
        min-height: 42px;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {ACCENT};
        color: {TEXT};
        background: {NAVY};
    }}

    [data-testid="stFileUploader"] {{
        background: {SURFACE};
        border: 1px dashed {LIGHT_GREY};
        border-radius: 14px;
        padding: 6px;
    }}

    [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}

    [data-baseweb="tab"] {{
        background: {SURFACE};
        border-radius: 10px 10px 0 0;
        color: {MUTED};
        padding: 9px 16px;
    }}

    [data-baseweb="tab"][aria-selected="true"] {{
        color: {TEXT};
        background: {NAVY};
    }}

    div[data-testid="stExpander"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def normalize_image(image: Image.Image, max_size=1800) -> Image.Image:
    """Normalize uploaded images into a displayable RGB PIL image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        image = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image


def grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.grayscale(image), dtype=np.float32)


def percentile_contrast(gray: np.ndarray) -> float:
    p2, p98 = np.percentile(gray, [2, 98])
    return float(np.clip((p98 - p2) / 255.0, 0, 1))


def estimate_quality(image: Image.Image) -> dict:
    """Prototype image-quality agent using transparent image statistics."""
    gray = grayscale_array(image)
    contrast = percentile_contrast(gray)
    brightness = float(gray.mean() / 255.0)

    # Laplacian-like sharpness proxy without OpenCV.
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    sharpness = float(np.var(gx) + np.var(gy))
    sharpness_norm = float(np.clip(sharpness / 1800.0, 0, 1))

    score = 100 * (0.55 * contrast + 0.30 * sharpness_norm + 0.15 * (1 - abs(brightness - .5)))
    score = int(np.clip(score, 0, 100))

    if score >= 75:
        status = "Good"
    elif score >= 55:
        status = "Review"
    else:
        status = "Low"

    return {
        "score": score,
        "status": status,
        "contrast": round(contrast * 100, 1),
        "brightness": round(brightness * 100, 1),
        "sharpness": round(sharpness_norm * 100, 1),
    }


def create_overlay(image: Image.Image, quality_score: int) -> Image.Image:
    """
    Creates an explainability-style overlay for the prototype.
    It is deliberately labeled as an illustrative overlay rather than a
    clinical segmentation result.
    """
    base = ImageOps.grayscale(image).convert("RGB")
    base = ImageEnhance.Contrast(base).enhance(1.15)
    base = base.filter(ImageFilter.GaussianBlur(radius=.15))

    draw = ImageDraw.Draw(base, "RGBA")
    w, h = base.size
    cx, cy = w // 2, int(h * .52)

    # Approximate central knee region.
    rx, ry = int(w * .22), int(h * .29)
    draw.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), outline=(73, 183, 176, 205), width=max(3, w // 350))
    draw.line((cx-int(w*.25), cy, cx+int(w*.25), cy), fill=(24, 59, 99, 210), width=max(2, w // 500))
    draw.line((cx, cy-int(h*.22), cx, cy+int(h*.22)), fill=(73, 183, 176, 145), width=max(2, w // 500))

    # Joint-space guide lines.
    y = int(h * .52)
    for offset in (-int(h*.035), int(h*.035)):
        draw.line(
            (int(w*.30), y + offset, int(w*.70), y + offset),
            fill=(208, 168, 91, 190),
            width=max(2, w // 450),
        )

    # Corner markers.
    marker = max(8, w // 45)
    for x, yy in [(int(w*.29), int(h*.29)), (int(w*.71), int(h*.29)),
                  (int(w*.29), int(h*.75)), (int(w*.71), int(h*.75))]:
        draw.rectangle((x-marker, yy-marker, x+marker, yy+marker),
                       outline=(24, 59, 99, 220), width=max(2, w//500))

    return base


def run_prototype_analysis(image: Image.Image) -> dict:
    """
    A transparent, non-clinical prototype analysis.
    This intentionally does not claim diagnostic accuracy.
    Replace this function with trained model inference in a production build.
    """
    q = estimate_quality(image)
    gray = grayscale_array(image)
    h, w = gray.shape

    # Relative structural metrics from image statistics.
    center_band = gray[int(h*.45):int(h*.60), int(w*.22):int(w*.78)]
    outer_band = np.concatenate([
        gray[int(h*.25):int(h*.45), int(w*.22):int(w*.78)].ravel(),
        gray[int(h*.60):int(h*.80), int(w*.22):int(w*.78)].ravel(),
    ])
    joint_delta = float(np.clip(abs(center_band.mean() - outer_band.mean()) / 255, 0, 1))
    joint_space_index = int(np.clip(100 - joint_delta * 80, 0, 100))

    # Demo OA risk score, not a medical score.
    texture = float(np.clip(np.std(center_band) / 90, 0, 1))
    oa_risk = int(np.clip(30 + (100-q["score"])*.35 + texture*28, 5, 92))

    if oa_risk < 35:
        kl = "KL 0–1 (Prototype Estimate)"
    elif oa_risk < 55:
        kl = "KL 1–2 (Prototype Estimate)"
    elif oa_risk < 72:
        kl = "KL 2–3 (Prototype Estimate)"
    else:
        kl = "KL 3–4 (Prototype Estimate)"

    # Alignment proxy.
    left_mean = gray[:, :max(1, w//2)].mean()
    right_mean = gray[:, w//2:].mean()
    asym = abs(left_mean-right_mean)/255
    alignment = "Near Neutral" if asym < .035 else ("Mild Varus/Valgus Signal" if asym < .08 else "Alignment Review")

    # Fracture result is intentionally framed as a screening signal.
    fracture_signal = "No Strong Signal" if q["score"] >= 60 else "Review Recommended"
    fracture_conf = int(np.clip(55 + q["score"]*.35, 55, 90))

    findings = [
        ("Image Quality", f"{q['status']} — quality score {q['score']}/100."),
        ("Anatomical Region", "Central knee-region ROI highlighted for prototype analysis."),
        ("Joint-Space Proxy", f"Relative index {joint_space_index}/100; not a calibrated millimetre measurement."),
        ("OA / KL Proxy", f"{kl}; prototype risk score {oa_risk}/100."),
        ("Alignment", alignment + " — image-statistics proxy only."),
        ("Fracture Screen", f"{fracture_signal} ({fracture_conf}% prototype confidence)."),
    ]

    return {
        "quality": q,
        "joint_space_index": joint_space_index,
        "oa_risk": oa_risk,
        "kl": kl,
        "alignment": alignment,
        "fracture_signal": fracture_signal,
        "fracture_conf": fracture_conf,
        "findings": findings,
    }


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_text(value) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------
# Automated PDF Report
# ---------------------------------------------------------------------
def build_pdf_report(
    image: Image.Image,
    analysis: dict,
    patient_label: str,
    study_label: str,
) -> bytes:
    """Build a polished A4 PDF report fully in memory."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=14*mm,
        bottomMargin=14*mm,
        title="MedVision AI Knee Assessment Report",
        author="MedVision AI Prototype",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MVTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=26, textColor=colors.HexColor(NAVY),
        alignment=TA_LEFT, spaceAfter=5
    )
    subtitle_style = ParagraphStyle(
        "MVSub", parent=styles["Normal"], fontSize=9.5, leading=13,
        textColor=colors.HexColor(STEEL_2), spaceAfter=9
    )
    h_style = ParagraphStyle(
        "MVH", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=12.5, leading=15, textColor=colors.HexColor(NAVY),
        spaceBefore=7, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "MVBody", parent=styles["BodyText"], fontSize=9.5, leading=13,
        textColor=colors.HexColor(CHARCOAL), spaceAfter=5
    )
    small_style = ParagraphStyle(
        "MVSmall", parent=styles["BodyText"], fontSize=7.8, leading=10,
        textColor=colors.HexColor(STEEL_2)
    )
    disclaimer_style = ParagraphStyle(
        "MVDisc", parent=styles["BodyText"], fontSize=8, leading=11,
        textColor=colors.HexColor("#6B4D24")
    )

    story = []
    story.append(Paragraph("MedVision AI", title_style))
    story.append(Paragraph(
        "AI-Assisted Knee X-Ray Assessment & Explainable Clinical Decision Support",
        subtitle_style
    ))

    meta = [
        ["Study Label", safe_text(study_label), "Patient Label", safe_text(patient_label)],
        ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M"), "Prototype Version", "Hackathon MVP"],
    ]
    meta_table = Table(meta, colWidths=[29*mm, 62*mm, 29*mm, 55*mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#E4E8EB")),
        ("BOX", (0,0), (-1,-1), .6, colors.HexColor("#9AA4AD")),
        ("INNERGRID", (0,0), (-1,-1), .3, colors.HexColor("#B8C0C8")),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor(CHARCOAL)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 7))

    # Source image.
    img_copy = normalize_image(image, 1400)
    img_bytes = io.BytesIO()
    img_copy.save(img_bytes, format="JPEG", quality=88)
    img_bytes.seek(0)

    max_w, max_h = 176*mm, 86*mm
    scale = min(max_w/img_copy.width, max_h/img_copy.height)
    source_img = RLImage(img_bytes, width=img_copy.width*scale, height=img_copy.height*scale)

    overlay = create_overlay(img_copy, analysis["quality"]["score"])
    ov_bytes = io.BytesIO()
    overlay.save(ov_bytes, format="JPEG", quality=88)
    ov_bytes.seek(0)
    overlay_img = RLImage(ov_bytes, width=img_copy.width*scale, height=img_copy.height*scale)

    image_table = Table([[source_img, overlay_img]], colWidths=[88*mm, 88*mm])
    image_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#D5DADF")),
        ("BOX", (0,0), (-1,-1), .5, colors.HexColor("#A5AFB8")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(Paragraph("Source Image And Explainability Overlay", h_style))
    story.append(image_table)
    story.append(Paragraph(
        "Overlay is illustrative and demonstrates the UI workflow; it is not a validated anatomical segmentation.",
        small_style
    ))

    story.append(Paragraph("Prototype Assessment Summary", h_style))
    summary_rows = [
        ["Metric", "Result", "Interpretation"],
        ["Image Quality", f"{analysis['quality']['score']}/100", analysis["quality"]["status"]],
        ["Joint-Space Proxy", f"{analysis['joint_space_index']}/100", "Relative index only"],
        ["OA / KL Proxy", f"{analysis['oa_risk']}/100", analysis["kl"]],
        ["Alignment", analysis["alignment"], "Prototype image-statistics signal"],
        ["Fracture Screen", analysis["fracture_signal"], f"{analysis['fracture_conf']}% prototype confidence"],
    ]
    table = Table(summary_rows, colWidths=[43*mm, 50*mm, 83*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (0,-1), "Helvetica-Bold"),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#EEF1F3")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#EEF1F3"), colors.HexColor("#E2E7EA")]),
        ("TEXTCOLOR", (0,1), (-1,-1), colors.HexColor(CHARCOAL)),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#A5AFB8")),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("LEADING", (0,0), (-1,-1), 11),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)

    story.append(Paragraph("Explainable Findings", h_style))
    for name, text in analysis["findings"]:
        story.append(Paragraph(
            f"<b>{safe_text(name)}:</b> {safe_text(text)}",
            body_style
        ))

    story.append(Paragraph("Recommended Review Workflow", h_style))
    review_text = (
        "1. Confirm image quality and correct view. "
        "2. Review the highlighted anatomical region and landmarks. "
        "3. Review the relative joint-space and alignment signals. "
        "4. Review the OA/KL prototype output and fracture screening signal. "
        "5. Correlate with the original image, clinical history, and qualified professional interpretation."
    )
    story.append(Paragraph(review_text, body_style))

    story.append(Spacer(1, 7))
    disclaimer_box = Table([[
        Paragraph(
            "<b>Important:</b> This is a research/hackathon prototype and is NOT a medical diagnosis system. "
            "The metrics, segmentation, KL estimate, alignment signal, and fracture screen are illustrative "
            "prototype outputs and have not been clinically validated. Do not use this report as a substitute "
            "for qualified medical assessment.",
            disclaimer_style
        )
    ]], colWidths=[176*mm])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F0E5D3")),
        ("BOX", (0,0), (-1,-1), .6, colors.HexColor("#C6A56A")),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
    ]))
    story.append(disclaimer_box)

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#9AA4AD"))
        canvas.line(15*mm, 9*mm, 195*mm, 9*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(STEEL_2))
        canvas.drawString(15*mm, 5.5*mm, "MedVision AI | Hackathon MVP")
        canvas.drawRightString(195*mm, 5.5*mm, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# ---------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "image" not in st.session_state:
    st.session_state.image = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="mv-topbar">
        <div class="brand">Med<span>Vision</span> AI</div>
        <div class="subtitle">AI-Powered Knee X-Ray Assessment & Explainable Clinical Decision Support</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Study Setup")
    patient_label = st.text_input("Patient Label", value="Demo Patient")
    study_label = st.text_input("Study Label", value="Knee X-Ray Study")
    st.markdown("---")

    st.markdown("### Agentic Workflow")
    workflow_items = [
        "Image Input",
        "Quality Agent",
        "Anatomy Agent",
        "Landmark Agent",
        "Measurement Agent",
        "OA / Fracture Agent",
        "Explainability",
        "Report Agent",
    ]
    for i, item in enumerate(workflow_items, start=1):
        st.markdown(
            f'<div class="workflow-step {"active" if i <= 8 else ""}">{i:02d} · {item}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Safety Notice")
    st.warning(
        "Research/hackathon prototype only. Outputs are not medical diagnoses and "
        "must not be used for clinical decisions."
    )


# ---------------------------------------------------------------------
# Upload + Controls
# ---------------------------------------------------------------------
st.markdown("## Image Input")
st.markdown(
    "Upload a knee X-ray to run the complete prototype workflow: quality validation, "
    "anatomical-region highlighting, relative measurements, explainability, dashboard, and PDF report."
)

uploaded = st.file_uploader(
    "Drag And Drop A Knee X-Ray",
    type=["png", "jpg", "jpeg"],
    help="PNG, JPG, or JPEG. DICOM is intentionally not enabled in this MVP.",
)

run_col, clear_col, _ = st.columns([1.2, 1.0, 4.8])
with run_col:
    run_clicked = st.button("Run AI Analysis", use_container_width=True)
with clear_col:
    clear_clicked = st.button("Reset", use_container_width=True)

if clear_clicked:
    st.session_state.analysis = None
    st.session_state.image = None
    st.session_state.pdf_bytes = None
    st.rerun()

if uploaded is not None:
    try:
        current_image = normalize_image(Image.open(uploaded))
        st.session_state.image = current_image
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")

if run_clicked:
    if st.session_state.image is None:
        st.warning("Please upload a knee X-ray before running the analysis.")
    else:
        with st.spinner("Running MedVision AI Prototype Agents..."):
            st.session_state.analysis = run_prototype_analysis(st.session_state.image)
            st.session_state.pdf_bytes = None
        st.success("Prototype analysis completed. Review the findings below.")


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
if st.session_state.analysis and st.session_state.image:
    analysis = st.session_state.analysis

    st.markdown("## Knee Assessment Dashboard")

    q = analysis["quality"]
    cols = st.columns(5)
    with cols[0]:
        metric_card("Image Quality", f"{q['score']}/100", q["status"])
    with cols[1]:
        metric_card("Joint-Space Proxy", f"{analysis['joint_space_index']}/100", "Relative Index")
    with cols[2]:
        metric_card("OA Risk Proxy", f"{analysis['oa_risk']}/100", "Prototype Only")
    with cols[3]:
        metric_card("Alignment", analysis["alignment"], "Image-Statistics Signal")
    with cols[4]:
        metric_card("Fracture Screen", analysis["fracture_signal"], f"{analysis['fracture_conf']}% Prototype")

    st.markdown("<br>", unsafe_allow_html=True)

    img_col, report_col = st.columns([1.35, 1])
    with img_col:
        st.markdown("### Explainability View")
        overlay = create_overlay(st.session_state.image, q["score"])
        st.image(
            [st.session_state.image, overlay],
            caption=["Original X-Ray", "Illustrative Explainability Overlay"],
            use_container_width=True,
        )

    with report_col:
        st.markdown("### Automated Findings")
        for name, text in analysis["findings"]:
            st.markdown(
                f'<div class="finding"><b>{name}</b><br><span class="small">{text}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="callout"><b>Explainability:</b> The prototype combines image-quality statistics, '
            'a central-region proxy, relative joint-space features, image asymmetry, and texture signals. '
            'These are transparent demo heuristics, not validated medical AI models.</div>',
            unsafe_allow_html=True
        )

    tab1, tab2, tab3 = st.tabs(["Agent Results", "Report Generator", "Technical Notes"])

    with tab1:
        st.markdown("### Agent Results")
        agent_rows = [
            ("Quality Agent", f"{q['status']} ({q['score']}/100)", "Contrast, brightness, and sharpness proxies"),
            ("Anatomy Agent", "Knee ROI Highlighted", "Central-region prototype localization"),
            ("Landmark Agent", "Joint Center + Guide Lines", "Illustrative landmark geometry"),
            ("Measurement Agent", f"{analysis['joint_space_index']}/100", "Relative joint-space index"),
            ("OA / Fracture Agent", f"{analysis['kl']} / {analysis['fracture_signal']}", "Prototype screening signals"),
            ("Explainability Agent", "Overlay + Reasoning", "Visual and textual rationale"),
            ("Report Agent", "Ready", "A4 PDF generation"),
        ]
        st.dataframe(
            [{"Agent": a, "Result": b, "Method": c} for a, b, c in agent_rows],
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.markdown("### Automated Report Generator")
        st.write(
            "Generate a structured A4 PDF containing the source image, explainability overlay, "
            "prototype metrics, findings, review workflow, and safety disclaimer."
        )
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                st.session_state.pdf_bytes = build_pdf_report(
                    st.session_state.image,
                    analysis,
                    patient_label,
                    study_label,
                )
            st.success("PDF report generated successfully.")

        if st.session_state.pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name="medvision_ai_knee_assessment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with tab3:
        st.markdown("### Technical Notes")
        st.markdown(
            """
            - **Frontend:** Streamlit with a responsive dashboard layout.
            - **Image layer:** Pillow + NumPy for normalization and transparent prototype features.
            - **AI-agent architecture:** The UI mirrors the requested Quality, Anatomy, Landmark,
              Measurement, OA/Fracture, Explainability, Report, and Dashboard agents.
            - **PDF:** ReportLab generates the report fully in memory, so no server-side file management is required.
            - **Production upgrade path:** Replace `run_prototype_analysis()` with trained segmentation,
              landmark, OA grading, and fracture models; add DICOM handling; calibrate measurements;
              add model/version metadata; and implement clinical validation.
            """
        )

else:
    st.markdown("## Dashboard Preview")
    st.info(
        "Upload a knee X-ray and select **Run AI Analysis** to populate the dashboard and report generator."
    )

    preview_cols = st.columns(4)
    preview = [
        ("01", "Quality Agent", "Image clarity and suitability checks"),
        ("02", "Anatomy Agent", "Knee-region localization and structure review"),
        ("03", "Measurement Agent", "Relative joint-space measurement"),
        ("04", "Report Agent", "Structured PDF with explainability"),
    ]
    for col, (num, title, text) in zip(preview_cols, preview):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div style="color:{ACCENT};font-weight:800;font-size:1.1rem;">{num}</div>
                    <h3>{title}</h3>
                    <div class="small">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    f"""
    <div style="margin-top:28px;padding:14px 0;border-top:1px solid {BORDER};
                color:{MID_GREY};font-size:.82rem;text-align:center;">
        MedVision AI · Hackathon MVP · Smarter Analysis, Better Insights · Not For Clinical Diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
