"""Streamlit demo: upload an ultrasound frame, get plane prediction and head circumference.

    streamlit run app/streamlit_app.py -- --cls exports/classifier.onnx --seg exports/segmenter.onnx
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import streamlit as st
from PIL import Image

from fetalscan.biometry import draw_ellipse
from fetalscan.inference import FetalScanRuntime

p = argparse.ArgumentParser()
p.add_argument("--cls", default="exports/classifier.onnx")
p.add_argument("--seg", default="exports/segmenter.onnx")
args = p.parse_args(sys.argv[1:])


@st.cache_resource
def load(cls, seg):
    return FetalScanRuntime(cls, seg)


st.title("FetalScan")
st.caption("Research demo on public datasets. Not a medical device; not for clinical use.")
rt = load(args.cls, args.seg)
up = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg"])
pix = st.number_input("Pixel size (mm)", min_value=0.01, max_value=2.0, value=0.15, step=0.01)
if up:
    gray = np.asarray(Image.open(up).convert("L"))
    c1, c2 = st.columns(2)
    c1.image(gray, caption="Input", use_container_width=True)
    res = rt.classify(gray)
    st.subheader(f"Plane: {res['plane']} ({res['confidence']:.0%})")
    st.bar_chart(res["probabilities"])
    head = rt.measure_head(gray, pix)
    if head["head_found"]:
        c2.image(draw_ellipse(head["image"], head["ellipse"])[:, :, ::-1], caption="Fitted head ellipse",
                 use_container_width=True)
        st.metric("Head circumference", f"{head['hc_mm']:.1f} mm")
    else:
        c2.info("No head region found.")
