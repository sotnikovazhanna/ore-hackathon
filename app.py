from __future__ import annotations

import io
import inspect
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.large_image import LoadedImage, load_uploaded_image, resize_for_display
from src.ore_pipeline import AnalysisResult, OreAnalyzer

APP_API_VERSION = "orescope-final-v1"

st.set_page_config(
    page_title="РђРЅР°Р»РёР· РїРѕР»РёСЂРѕРІР°РЅРЅС‹С… С€Р»РёС„РѕРІ",
    page_icon="рџ”¬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --card-border: rgba(148, 163, 184, 0.22);
        --card-bg: rgba(255, 255, 255, 0.035);
        --muted: #9aa4b2;
        --accent: #69a7ff;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
    }

    .app-kicker {
        color: var(--accent);
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .app-title {
        font-size: clamp(2rem, 4vw, 3.35rem);
        line-height: 1.05;
        font-weight: 800;
        margin: 0 0 0.65rem 0;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 920px;
        margin-bottom: 1.5rem;
    }

    .result-card {
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        margin: 0.35rem 0 1rem 0;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }

    .result-class {
        font-size: 1.8rem;
        line-height: 1.15;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .result-reason {
        color: #cbd5e1;
        line-height: 1.55;
    }

    .metric-card {
        min-height: 150px;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.75rem;
    }

    .metric-label {
        color: #d9e0e8;
        font-size: 0.96rem;
        font-weight: 650;
        min-height: 2.5rem;
    }

    .metric-value {
        font-size: 2.25rem;
        font-weight: 760;
        line-height: 1.1;
        margin: 0.35rem 0;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.3;
    }

    .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem 1.25rem;
        margin: 0.55rem 0 1rem 0;
        color: #cbd5e1;
        font-size: 0.92rem;
    }

    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 3px;
        display: inline-block;
    }

    .section-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: -0.15rem;
        margin-bottom: 0.9rem;
    }

    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        min-height: 2.75rem;
        font-weight: 650;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-kicker">РђРІС‚РѕРјР°С‚РёС‡РµСЃРєРёР№ Р°РЅР°Р»РёР· РёР·РѕР±СЂР°Р¶РµРЅРёР№</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">рџ”¬ РђРЅР°Р»РёР· РїРѕР»РёСЂРѕРІР°РЅРЅС‹С… С€Р»РёС„РѕРІ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Р—Р°РіСЂСѓР·РёС‚Рµ РїР°РЅРѕСЂР°РјРЅРѕРµ РёР·РѕР±СЂР°Р¶РµРЅРёРµ С€Р»РёС„Р°. '
    'РЎРёСЃС‚РµРјР° РѕС†РµРЅРёС‚ РїР»РѕС‰Р°РґСЊ Р·РѕРЅС‹ РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ, РґРѕР»СЋ СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹ Рё '
    'РїСЂРµРѕР±Р»Р°РґР°СЋС‰РёР№ С‚РёРї СЃСЂР°СЃС‚Р°РЅРёР№, Р·Р°С‚РµРј СЃС„РѕСЂРјРёСЂСѓРµС‚ Р·Р°РєР»СЋС‡РµРЅРёРµ Рё РѕС‚С‡С‘С‚.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_analyzer() -> OreAnalyzer:
    return OreAnalyzer(model_dir="models")


def image_to_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def find_report_font() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def build_pdf_report(
    filename: str,
    source_preview: np.ndarray,
    overlay_preview: np.ndarray,
    row: dict[str, Any],
) -> bytes:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    page_width, page_height = landscape(A4)
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))

    font_path = find_report_font()
    font_name = "Helvetica"
    if font_path:
        try:
            font_name = "AnalysisReportFont"
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception:
            font_name = "Helvetica"

    pdf.setFont(font_name, 18)
    pdf.drawString(30, page_height - 35, "РћС‚С‡С‘С‚ Р°РЅР°Р»РёР·Р° РїРѕР»РёСЂРѕРІР°РЅРЅРѕРіРѕ С€Р»РёС„Р°")
    pdf.setFont(font_name, 10)
    pdf.drawString(30, page_height - 55, f"Р¤Р°Р№Р»: {filename}")

    pdf.setFont(font_name, 16)
    pdf.drawString(30, page_height - 82, str(row["ore_class"]))
    pdf.setFont(font_name, 9)

    lines = [
        f"РћСЃРЅРѕРІР°РЅРёРµ Р·Р°РєР»СЋС‡РµРЅРёСЏ: {row['decision_reason']}",
        f"Р—РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ: {row['talc_share_percent']:.1f}% РїР»РѕС‰Р°РґРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        f"РЎСѓР»СЊС„РёРґРЅР°СЏ С„Р°Р·Р°: {row['sulfide_share_percent']:.1f}% РїР»РѕС‰Р°РґРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        (
            "РћР±С‹С‡РЅС‹Рµ СЃСЂР°СЃС‚Р°РЅРёСЏ: "
            f"{row['ordinary_fraction_of_sulfides']:.1f}% РѕР±РЅР°СЂСѓР¶РµРЅРЅРѕР№ СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹"
        ),
        (
            "РўРѕРЅРєРёРµ СЃСЂР°СЃС‚Р°РЅРёСЏ: "
            f"{row['fine_fraction_of_sulfides']:.1f}% РѕР±РЅР°СЂСѓР¶РµРЅРЅРѕР№ СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹"
        ),
        f"РСЃС…РѕРґРЅРѕРµ СЂР°Р·СЂРµС€РµРЅРёРµ: {row['original_width']} Г— {row['original_height']} px",
        f"Р Р°Р·СЂРµС€РµРЅРёРµ Р°РЅР°Р»РёР·Р°: {row['analysis_width']} Г— {row['analysis_height']} px",
        f"Р’СЂРµРјСЏ Р°РЅР°Р»РёР·Р°: {row['processing_seconds']:.1f} СЃ",
    ]

    y = page_height - 105
    for line in lines:
        pdf.drawString(30, y, line[:165])
        y -= 14

    pdf.setFont(font_name, 8.5)
    pdf.drawString(30, y - 2, "Р¦РІРµС‚РѕРІР°СЏ РєР°СЂС‚Р°: Р·РµР»С‘РЅС‹Р№ вЂ” РѕР±С‹С‡РЅС‹Рµ; РєСЂР°СЃРЅС‹Р№ вЂ” С‚РѕРЅРєРёРµ; СЃРёРЅРёР№ вЂ” Р·РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ.")

    image_y = 35
    image_height = 285
    image_width = (page_width - 90) / 2
    pdf.drawImage(
        ImageReader(io.BytesIO(image_to_png(source_preview))),
        30,
        image_y,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True,
        anchor="c",
    )
    pdf.drawImage(
        ImageReader(io.BytesIO(image_to_png(overlay_preview))),
        60 + image_width,
        image_y,
        width=image_width,
        height=image_height,
        preserveAspectRatio=True,
        anchor="c",
    )

    pdf.save()
    return buffer.getvalue()


def choose_profile(mode: str, is_gpu: bool) -> dict[str, int | float]:
    profiles = {
        "РђРІС‚РѕРјР°С‚РёС‡РµСЃРєРёР№": {
            "max_mp": 24.0 if is_gpu else 10.0,
            "tile_size": 384,
            "overlap": 24,
            "batch_size": 12 if is_gpu else 2,
        },
        "Р‘С‹СЃС‚СЂС‹Р№": {
            "max_mp": 16.0 if is_gpu else 6.0,
            "tile_size": 384,
            "overlap": 16,
            "batch_size": 12 if is_gpu else 2,
        },
        "Р”РµС‚Р°Р»СЊРЅС‹Р№": {
            "max_mp": 40.0 if is_gpu else 18.0,
            "tile_size": 384,
            "overlap": 48,
            "batch_size": 8 if is_gpu else 2,
        },
    }
    return profiles[mode]


def result_row(
    filename: str,
    result: AnalysisResult,
    loaded: LoadedImage,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "ore_class": result.ore_class,
        "decision_reason": result.decision_reason,
        "talc_share_percent": result.talc_share_percent,
        "sulfide_share_percent": result.sulfide_share_percent,
        "ordinary_share_percent": result.ordinary_share_percent,
        "fine_share_percent": result.fine_share_percent,
        "ordinary_fraction_of_sulfides": result.ordinary_fraction_of_sulfides,
        "fine_fraction_of_sulfides": result.fine_fraction_of_sulfides,
        "original_width": loaded.original_width,
        "original_height": loaded.original_height,
        "analysis_width": loaded.analysis_width,
        "analysis_height": loaded.analysis_height,
        "file_size_mb": loaded.file_size_bytes / 1024 / 1024,
        "processing_seconds": elapsed,
    }


def make_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    rename = {
        "filename": "Р¤Р°Р№Р»",
        "ore_class": "Р—Р°РєР»СЋС‡РµРЅРёРµ",
        "talc_share_percent": "Р—РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ, % РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        "sulfide_share_percent": "РЎСѓР»СЊС„РёРґРЅР°СЏ С„Р°Р·Р°, % РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        "ordinary_fraction_of_sulfides": "РћР±С‹С‡РЅС‹Рµ, % СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹",
        "fine_fraction_of_sulfides": "РўРѕРЅРєРёРµ, % СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹",
        "processing_seconds": "Р’СЂРµРјСЏ, СЃ",
    }
    return frame[list(rename)].rename(columns=rename)


def metric_card(label: str, value: str, note: str) -> None:
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


analyzer = load_analyzer()
pipeline_path = inspect.getfile(OreAnalyzer)
status_method = getattr(analyzer, "model_status", None)
if not callable(status_method):
    st.error("РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РїСѓСЃС‚РёС‚СЊ РјРѕРґСѓР»СЊ Р°РЅР°Р»РёР·Р°: СѓСЃС‚Р°РЅРѕРІР»РµРЅС‹ РЅРµСЃРѕРІРјРµСЃС‚РёРјС‹Рµ С„Р°Р№Р»С‹ РїСЂРѕРµРєС‚Р°.")
    st.code(f"Р—Р°РіСЂСѓР¶РµРЅРЅС‹Р№ РјРѕРґСѓР»СЊ: {pipeline_path}")
    st.stop()

status = status_method()
if status.get("api_version") != APP_API_VERSION:
    st.error("РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РїСѓСЃС‚РёС‚СЊ РјРѕРґСѓР»СЊ Р°РЅР°Р»РёР·Р°: СѓСЃС‚Р°РЅРѕРІР»РµРЅС‹ РЅРµСЃРѕРІРјРµСЃС‚РёРјС‹Рµ С„Р°Р№Р»С‹ РїСЂРѕРµРєС‚Р°.")
    st.code(f"РњРѕРґСѓР»СЊ: {pipeline_path}")
    st.stop()

if not status.get("ready"):
    st.error("РќРµ РЅР°Р№РґРµРЅС‹ РѕР±СЏР·Р°С‚РµР»СЊРЅС‹Рµ С„Р°Р№Р»С‹ РјРѕРґРµР»РµР№. РџСЂРѕРІРµСЂСЊС‚Рµ РїР°РїРєСѓ models.")
    st.stop()

with st.sidebar:
    st.header("РќР°СЃС‚СЂРѕР№РєРё")
    mode = st.radio(
        "Р РµР¶РёРј Р°РЅР°Р»РёР·Р°",
        ["РђРІС‚РѕРјР°С‚РёС‡РµСЃРєРёР№", "Р‘С‹СЃС‚СЂС‹Р№", "Р”РµС‚Р°Р»СЊРЅС‹Р№"],
        index=0,
        captions=[
            "Р РµРєРѕРјРµРЅРґСѓРµС‚СЃСЏ РґР»СЏ Р±РѕР»СЊС€РёРЅСЃС‚РІР° РёР·РѕР±СЂР°Р¶РµРЅРёР№",
            "РњРёРЅРёРјР°Р»СЊРЅРѕРµ РІСЂРµРјСЏ РѕР±СЂР°Р±РѕС‚РєРё",
            "Р‘РѕР»СЊС€Рµ РґРµС‚Р°Р»РµР№ РЅР° РєСЂСѓРїРЅС‹С… РїР°РЅРѕСЂР°РјР°С…",
        ],
    )
    st.divider()
    st.success("РЎРёСЃС‚РµРјР° РіРѕС‚РѕРІР° Рє СЂР°Р±РѕС‚Рµ")

with st.expander("РљР°Рє С„РѕСЂРјРёСЂСѓРµС‚СЃСЏ Р·Р°РєР»СЋС‡РµРЅРёРµ", expanded=False):
    st.markdown(
        """
        1. Р•СЃР»Рё Р·РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ Р·Р°РЅРёРјР°РµС‚ Р±РѕР»РµРµ **10% РїР»РѕС‰Р°РґРё РёР·РѕР±СЂР°Р¶РµРЅРёСЏ**, РѕР±СЂР°Р·РµС† РѕС‚РЅРѕСЃРёС‚СЃСЏ Рє РѕС‚Р°Р»СЊРєРѕРІР°РЅРЅРѕР№ СЂСѓРґРµ.  
        2. Р•СЃР»Рё РїРѕСЂРѕРі 10% РЅРµ РїСЂРµРІС‹С€РµРЅ, Р·Р°РєР»СЋС‡РµРЅРёРµ РѕРїСЂРµРґРµР»СЏРµС‚СЃСЏ РїРѕ РїСЂРµРѕР±Р»Р°РґР°СЋС‰РµРјСѓ С‚РёРїСѓ СЃСѓР»СЊС„РёРґРЅС‹С… СЃСЂР°СЃС‚Р°РЅРёР№.  
        3. РќР° С†РІРµС‚РѕРІРѕР№ РєР°СЂС‚Рµ РѕР±С‹С‡РЅС‹Рµ СЃСЂР°СЃС‚Р°РЅРёСЏ РѕС‚РјРµС‡РµРЅС‹ Р·РµР»С‘РЅС‹Рј, С‚РѕРЅРєРёРµ вЂ” РєСЂР°СЃРЅС‹Рј, Р·РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ вЂ” СЃРёРЅРёРј.
        """
    )

uploaded_files = st.file_uploader(
    "Р—Р°РіСЂСѓР·РёС‚Рµ TIFF, PNG РёР»Рё JPEG",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="РњРѕР¶РЅРѕ Р·Р°РіСЂСѓР·РёС‚СЊ РѕРґРЅРѕ РёР·РѕР±СЂР°Р¶РµРЅРёРµ РёР»Рё СЃРµСЂРёСЋ С„Р°Р№Р»РѕРІ. РњР°РєСЃРёРјР°Р»СЊРЅС‹Р№ СЂР°Р·РјРµСЂ РѕРґРЅРѕРіРѕ С„Р°Р№Р»Р° вЂ” 512 РњР‘.",
)

if not uploaded_files:
    st.caption("РџРѕРґРґРµСЂР¶РёРІР°СЋС‚СЃСЏ РїР°РЅРѕСЂР°РјРЅС‹Рµ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ РІС‹СЃРѕРєРѕРіРѕ СЂР°Р·СЂРµС€РµРЅРёСЏ Рё РїР°РєРµС‚РЅР°СЏ РѕР±СЂР°Р±РѕС‚РєР°.")
    st.stop()

st.write(f"Р’С‹Р±СЂР°РЅРѕ С„Р°Р№Р»РѕРІ: **{len(uploaded_files)}**")

is_gpu = str(status["device"]).startswith("cuda")
profile = choose_profile(mode, is_gpu=is_gpu)

if st.button("РќР°С‡Р°С‚СЊ Р°РЅР°Р»РёР·", type="primary", width='stretch'):
    summary_rows: list[dict[str, Any]] = []
    detailed_results: list[tuple[str, LoadedImage, AnalysisResult, dict[str, Any]]] = []
    errors: list[dict[str, str]] = []
    zip_buffer = io.BytesIO()
    overall_progress = st.progress(0.0, text="РџРѕРґРіРѕС‚РѕРІРєР° Рє Р°РЅР°Р»РёР·Сѓ")

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_index, uploaded_file in enumerate(uploaded_files):
            file_started = time.time()
            try:
                with st.status(f"РћР±СЂР°Р±РѕС‚РєР° С„Р°Р№Р»Р°: {uploaded_file.name}", expanded=True) as status_box:
                    status_box.write("РџРѕРґРіРѕС‚РѕРІРєР° РёР·РѕР±СЂР°Р¶РµРЅРёСЏ")
                    loaded = load_uploaded_image(
                        uploaded_file,
                        uploaded_file.name,
                        max_analysis_megapixels=float(profile["max_mp"]),
                    )

                    tile_progress = st.progress(0.0, text="Р Р°СЃРїРѕР·РЅР°РІР°РЅРёРµ РѕР±Р»Р°СЃС‚РµР№")

                    def on_progress(value: float, _text: str) -> None:
                        tile_progress.progress(value, text="Р Р°СЃРїРѕР·РЅР°РІР°РЅРёРµ РѕР±Р»Р°СЃС‚РµР№")

                    initial_result = analyzer.analyze(
                        loaded.rgb,
                        tile_size=int(profile["tile_size"]),
                        overlap=int(profile["overlap"]),
                        batch_size=int(profile["batch_size"]),
                        progress_callback=on_progress,
                    )

                    final_loaded = loaded
                    final_result = initial_result

                    # Р’ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРѕРј СЂРµР¶РёРјРµ РїРѕРіСЂР°РЅРёС‡РЅС‹Рµ СЂРµР·СѓР»СЊС‚Р°С‚С‹ СѓС‚РѕС‡РЅСЏСЋС‚СЃСЏ
                    # РІ Р±РѕР»СЊС€РµРј СЂР°Р·СЂРµС€РµРЅРёРё. Р­С‚Рѕ РїСЂРѕРёСЃС…РѕРґРёС‚ РЅРµР·Р°РјРµС‚РЅРѕ РґР»СЏ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ.
                    borderline = 7.0 <= initial_result.talc_share_percent <= 13.0
                    can_refine = (
                        mode == "РђРІС‚РѕРјР°С‚РёС‡РµСЃРєРёР№"
                        and borderline
                        and loaded.original_megapixels > loaded.analysis_megapixels * 1.35
                    )

                    if can_refine:
                        status_box.write("РЈС‚РѕС‡РЅРµРЅРёРµ СЂРµР·СѓР»СЊС‚Р°С‚Р°")
                        refined_mp = min(
                            loaded.original_megapixels,
                            float(profile["max_mp"]) * 1.8,
                            42.0 if is_gpu else 20.0,
                        )
                        refined_loaded = load_uploaded_image(
                            uploaded_file,
                            uploaded_file.name,
                            max_analysis_megapixels=refined_mp,
                        )
                        refined_result = analyzer.analyze(
                            refined_loaded.rgb,
                            tile_size=int(profile["tile_size"]),
                            overlap=max(int(profile["overlap"]), 32),
                            batch_size=int(profile["batch_size"]),
                            progress_callback=on_progress,
                        )
                        final_loaded = refined_loaded
                        final_result = refined_result

                    status_box.write("Р¤РѕСЂРјРёСЂРѕРІР°РЅРёРµ Р·Р°РєР»СЋС‡РµРЅРёСЏ Рё РѕС‚С‡С‘С‚Р°")
                    elapsed = time.time() - file_started
                    row = result_row(
                        filename=uploaded_file.name,
                        result=final_result,
                        loaded=final_loaded,
                        elapsed=elapsed,
                    )
                    summary_rows.append(row)
                    detailed_results.append((uploaded_file.name, final_loaded, final_result, row))

                    source_preview = resize_for_display(final_loaded.rgb)
                    overlay_preview = resize_for_display(final_result.overlay)
                    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")
                    folder = f"{file_index + 1:03d}_{safe_stem}"
                    archive.writestr(f"{folder}/color_map.png", image_to_png(final_result.overlay))
                    archive.writestr(
                        f"{folder}/result.json",
                        json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8"),
                    )
                    archive.writestr(
                        f"{folder}/report.pdf",
                        build_pdf_report(uploaded_file.name, source_preview, overlay_preview, row),
                    )

                    status_box.update(
                        label=f"{uploaded_file.name}: {final_result.ore_class} вЂ” {elapsed:.1f} СЃ",
                        state="complete",
                        expanded=False,
                    )
            except Exception as error:
                errors.append({"Р¤Р°Р№Р»": uploaded_file.name, "РћС€РёР±РєР°": str(error)})
                archive.writestr(
                    f"errors/{file_index + 1:03d}_{uploaded_file.name}.txt",
                    str(error).encode("utf-8"),
                )
                st.error(f"РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±СЂР°Р±РѕС‚Р°С‚СЊ {uploaded_file.name}: {error}")

            overall_progress.progress(
                (file_index + 1) / len(uploaded_files),
                text=f"РћР±СЂР°Р±РѕС‚Р°РЅРѕ С„Р°Р№Р»РѕРІ: {file_index + 1} РёР· {len(uploaded_files)}",
            )

        if summary_rows:
            archive.writestr(
                "summary.csv",
                make_summary_frame(summary_rows).to_csv(index=False).encode("utf-8-sig"),
            )
        if errors:
            archive.writestr(
                "errors.csv",
                pd.DataFrame(errors).to_csv(index=False).encode("utf-8-sig"),
            )

    st.session_state["analysis_results"] = detailed_results
    st.session_state["analysis_summary"] = summary_rows
    st.session_state["analysis_errors"] = errors
    st.session_state["analysis_zip"] = zip_buffer.getvalue()

if "analysis_summary" not in st.session_state:
    st.stop()

if st.session_state.get("analysis_errors"):
    with st.expander("Р¤Р°Р№Р»С‹, РєРѕС‚РѕСЂС‹Рµ РЅРµ СѓРґР°Р»РѕСЃСЊ РѕР±СЂР°Р±РѕС‚Р°С‚СЊ", expanded=False):
        st.dataframe(
            pd.DataFrame(st.session_state["analysis_errors"]),
            hide_index=True,
            width='stretch',
        )

if not st.session_state["analysis_summary"]:
    st.error("РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±СЂР°Р±РѕС‚Р°С‚СЊ РЅРё РѕРґРЅРѕРіРѕ С„Р°Р№Р»Р°.")
    st.stop()

st.subheader("РЎРІРѕРґРЅС‹Рµ СЂРµР·СѓР»СЊС‚Р°С‚С‹")
summary_frame = make_summary_frame(st.session_state["analysis_summary"])
st.dataframe(summary_frame, width='stretch', hide_index=True)

st.download_button(
    "РЎРєР°С‡Р°С‚СЊ РІСЃРµ СЂРµР·СѓР»СЊС‚Р°С‚С‹ РѕРґРЅРёРј Р°СЂС…РёРІРѕРј",
    data=st.session_state["analysis_zip"],
    file_name="analysis_results.zip",
    mime="application/zip",
    width='stretch',
)

for filename, loaded, result, row in st.session_state["analysis_results"]:
    st.divider()
    st.caption(filename)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Р—Р°РєР»СЋС‡РµРЅРёРµ</div>
            <div class="result-class">{result.ore_class}</div>
            <div class="result-reason">{result.decision_reason}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Р РµР·СѓР»СЊС‚Р°С‚С‹ РёР·РјРµСЂРµРЅРёСЏ")
    metric_columns = st.columns(4)
    with metric_columns[0]:
        metric_card(
            "Р—РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ",
            f"{result.talc_share_percent:.1f}%",
            "РґРѕР»СЏ РѕС‚ РїР»РѕС‰Р°РґРё РІСЃРµРіРѕ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        )
    with metric_columns[1]:
        metric_card(
            "РЎСѓР»СЊС„РёРґРЅР°СЏ С„Р°Р·Р°",
            f"{result.sulfide_share_percent:.1f}%",
            "РґРѕР»СЏ РѕС‚ РїР»РѕС‰Р°РґРё РІСЃРµРіРѕ РёР·РѕР±СЂР°Р¶РµРЅРёСЏ",
        )
    with metric_columns[2]:
        metric_card(
            "РћР±С‹С‡РЅС‹Рµ СЃСЂР°СЃС‚Р°РЅРёСЏ",
            f"{result.ordinary_fraction_of_sulfides:.1f}%",
            "РґРѕР»СЏ СЃСЂРµРґРё РѕР±РЅР°СЂСѓР¶РµРЅРЅРѕР№ СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹",
        )
    with metric_columns[3]:
        metric_card(
            "РўРѕРЅРєРёРµ СЃСЂР°СЃС‚Р°РЅРёСЏ",
            f"{result.fine_fraction_of_sulfides:.1f}%",
            "РґРѕР»СЏ СЃСЂРµРґРё РѕР±РЅР°СЂСѓР¶РµРЅРЅРѕР№ СЃСѓР»СЊС„РёРґРЅРѕР№ С„Р°Р·С‹",
        )

    st.markdown(
        '<div class="section-note">РћР±С‹С‡РЅС‹Рµ Рё С‚РѕРЅРєРёРµ СЃСЂР°СЃС‚Р°РЅРёСЏ РґРµР»СЏС‚ РјРµР¶РґСѓ СЃРѕР±РѕР№ '
        'РѕР±РЅР°СЂСѓР¶РµРЅРЅСѓСЋ СЃСѓР»СЊС„РёРґРЅСѓСЋ С„Р°Р·Сѓ, РїРѕСЌС‚РѕРјСѓ РёС… СЃСѓРјРјР° СЂР°РІРЅР° 100%.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Р’РёР·СѓР°Р»РёР·Р°С†РёСЏ")
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-item"><span class="legend-dot" style="background:#00d26a"></span>РѕР±С‹С‡РЅС‹Рµ СЃСЂР°СЃС‚Р°РЅРёСЏ</span>
            <span class="legend-item"><span class="legend-dot" style="background:#ff4b32"></span>С‚РѕРЅРєРёРµ СЃСЂР°СЃС‚Р°РЅРёСЏ</span>
            <span class="legend-item"><span class="legend-dot" style="background:#276ef1"></span>Р·РѕРЅР° РѕС‚Р°Р»СЊРєРѕРІР°РЅРёСЏ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_preview = resize_for_display(loaded.rgb)
    overlay_preview = resize_for_display(result.overlay)
    left, right = st.columns(2)
    with left:
        st.image(source_preview, caption="РР·РѕР±СЂР°Р¶РµРЅРёРµ, РёСЃРїРѕР»СЊР·РѕРІР°РЅРЅРѕРµ РґР»СЏ Р°РЅР°Р»РёР·Р°", width='stretch')
    with right:
        st.image(overlay_preview, caption="Р Р°СЃРїРѕР·РЅР°РЅРЅС‹Рµ РѕР±Р»Р°СЃС‚Рё", width='stretch')

    with st.expander("РЎРІРµРґРµРЅРёСЏ РѕР± РѕР±СЂР°Р±РѕС‚РєРµ", expanded=False):
        st.write(f"Р’СЂРµРјСЏ Р°РЅР°Р»РёР·Р°: {row['processing_seconds']:.1f} СЃ")
        st.write(f"РСЃС…РѕРґРЅРѕРµ СЂР°Р·СЂРµС€РµРЅРёРµ: {loaded.original_width} Г— {loaded.original_height} px")
        st.write(
            "Р Р°Р·СЂРµС€РµРЅРёРµ, РёСЃРїРѕР»СЊР·РѕРІР°РЅРЅРѕРµ РґР»СЏ Р°РЅР°Р»РёР·Р°: "
            f"{loaded.analysis_width} Г— {loaded.analysis_height} px"
        )

    json_bytes = json.dumps(row, ensure_ascii=False, indent=2).encode("utf-8")
    download_columns = st.columns(4)
    download_columns[0].download_button(
        "РўР°Р±Р»РёС†Р° CSV",
        data=make_summary_frame([row]).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{Path(filename).stem}_analysis.csv",
        mime="text/csv",
        key=f"csv_{filename}",
    )
    download_columns[1].download_button(
        "Р¦РІРµС‚РѕРІР°СЏ РєР°СЂС‚Р° PNG",
        data=image_to_png(result.overlay),
        file_name=f"{Path(filename).stem}_color_map.png",
        mime="image/png",
        key=f"png_{filename}",
    )
    download_columns[2].download_button(
        "Р”Р°РЅРЅС‹Рµ JSON",
        data=json_bytes,
        file_name=f"{Path(filename).stem}_analysis.json",
        mime="application/json",
        key=f"json_{filename}",
    )
    download_columns[3].download_button(
        "РћС‚С‡С‘С‚ PDF",
        data=build_pdf_report(filename, source_preview, overlay_preview, row),
        file_name=f"{Path(filename).stem}_report.pdf",
        mime="application/pdf",
        key=f"pdf_{filename}",
    )
