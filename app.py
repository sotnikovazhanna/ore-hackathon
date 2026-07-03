from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.ore_pipeline import OreAnalyzer


st.set_page_config(
    page_title="OreScope AI",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 OreScope AI")
st.caption(
    "Интерпретируемый анализ полированных шлифов: "
    "оталькование, обычные и тонкие срастания."
)


@st.cache_resource
def load_analyzer():
    return OreAnalyzer(model_dir="models")


analyzer = load_analyzer()

with st.sidebar:
    st.header("Параметры")

    tile_size = st.select_slider(
        "Размер тайла",
        options=[320, 384, 512, 640],
        value=512,
    )

    overlap = st.slider(
        "Перекрытие тайлов",
        min_value=32,
        max_value=160,
        value=96,
        step=16,
    )

    talc_gate_threshold = st.slider(
        "Порог оталькованной руды",
        min_value=0.50,
        max_value=0.99,
        value=0.97,
        step=0.01,
    )

    talc_mask_threshold = st.slider(
        "Порог визуальной маски талька",
        min_value=0.20,
        max_value=0.90,
        value=0.55,
        step=0.05,
    )

uploaded_file = st.file_uploader(
    "Загрузите TIFF, PNG или JPEG",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info(
        "Загрузите изображение, чтобы начать анализ."
    )
    st.stop()

image = Image.open(uploaded_file)
image = ImageOps.exif_transpose(
    image
).convert("RGB")
image_array = np.asarray(image)

st.write(
    f"Размер изображения: "
    f"{image_array.shape[1]} × {image_array.shape[0]} px"
)

if st.button(
    "Запустить анализ",
    type="primary",
):
    started = time.time()

    with st.spinner(
        "Выполняется тайловый анализ..."
    ):
        result = analyzer.analyze(
            image_array,
            tile_size=tile_size,
            overlap=overlap,
            talc_gate_threshold=talc_gate_threshold,
            talc_mask_threshold=talc_mask_threshold,
        )

    elapsed = time.time() - started

    st.success(
        f"Анализ завершён за {elapsed:.1f} с"
    )

    st.subheader(result.ore_class)

    metric_columns = st.columns(5)

    metric_columns[0].metric(
        "Вероятность >10% талька",
        f"{result.talc_probability * 100:.1f}%",
    )
    metric_columns[1].metric(
        "Оценка области талька",
        f"{result.talc_share_percent:.1f}%",
    )
    metric_columns[2].metric(
        "Сульфиды",
        f"{result.sulfide_share_percent:.1f}%",
    )
    metric_columns[3].metric(
        "Обычные",
        f"{result.ordinary_share_percent:.1f}%",
    )
    metric_columns[4].metric(
        "Тонкие",
        f"{result.fine_share_percent:.1f}%",
    )

    conclusion = (
        f"Руда классифицирована как "
        f"{result.ore_class.lower()}. "
        f"Вероятность превышения порога 10% талька — "
        f"{result.talc_probability * 100:.1f}%. "
        f"Визуальная оценка области оталькования — "
        f"{result.talc_share_percent:.1f}%. "
        f"Доля обычных срастаний — "
        f"{result.ordinary_share_percent:.1f}%, "
        f"тонких — {result.fine_share_percent:.1f}%."
    )

    st.write(conclusion)

    image_columns = st.columns(2)

    with image_columns[0]:
        st.image(
            image_array,
            caption="Исходное изображение",
            use_container_width=True,
        )

    with image_columns[1]:
        st.image(
            result.overlay,
            caption=(
                "Overlay: зелёный — обычные, "
                "красный — тонкие, синий — тальк"
            ),
            use_container_width=True,
        )

    st.subheader("Карта уверенности")
    st.image(
        result.confidence_map,
        clamp=True,
        use_container_width=True,
    )

    table = pd.DataFrame([{
        "ore_class": result.ore_class,
        "talc_probability": result.talc_probability,
        "talc_share_percent": result.talc_share_percent,
        "sulfide_share_percent": result.sulfide_share_percent,
        "ordinary_share_percent": result.ordinary_share_percent,
        "fine_share_percent": result.fine_share_percent,
        "processing_seconds": elapsed,
    }])

    csv_bytes = table.to_csv(
        index=False
    ).encode("utf-8-sig")

    overlay_buffer = io.BytesIO()
    Image.fromarray(
        result.overlay
    ).save(
        overlay_buffer,
        format="PNG",
    )

    json_bytes = json.dumps(
        {
            "conclusion": conclusion,
            **table.iloc[0].to_dict(),
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    download_columns = st.columns(3)

    download_columns[0].download_button(
        "Скачать CSV",
        data=csv_bytes,
        file_name="ore_analysis.csv",
        mime="text/csv",
    )

    download_columns[1].download_button(
        "Скачать overlay",
        data=overlay_buffer.getvalue(),
        file_name="ore_overlay.png",
        mime="image/png",
    )

    download_columns[2].download_button(
        "Скачать JSON",
        data=json_bytes,
        file_name="ore_analysis.json",
        mime="application/json",
    )
