from typing import List

import streamlit as st
from inference import LayoutSegmenter, filter_bboxes
from pdf2image.pdf2image import convert_from_bytes
from PIL import Image, ImageDraw

st.title("RESUME LAYOUT SEGMENTATION")
uploaded_pdf = st.file_uploader("", type="pdf")

def draw_rectangles(image: Image.Image, boxes: List[List[float]]) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    for box in boxes:
        draw.rectangle([(box[0], box[1]),(box[2], box[3])],outline="red", width=2) # type: ignore   
    return image

with st.spinner("The document is parsing please wait..."):
    if uploaded_pdf is not None:
        pdf_images = convert_from_bytes(uploaded_pdf.getvalue())
        for image in pdf_images:
            predicted_segments = LayoutSegmenter().segment(image)
            st.markdown("## Filtred predictions")
            filtered_segments = filter_bboxes(predicted_segments)
            st.image(draw_rectangles(image, filtered_segments.tolist()))
            st.markdown("## Raw predictions")
            st.image(draw_rectangles(image, predicted_segments.tolist()))