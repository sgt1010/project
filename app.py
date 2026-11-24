import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import joblib
import numpy as np
import streamlit as st
from config import MODEL_DIR
from predict import predict_image

st.set_page_config(page_title="Road Damage Detector", page_icon="")
st.set_page_config(page_title="Road Damage Detector", page_icon="🛣️")

# Hide the top-right Deploy button and Streamlit menu
st.markdown("""
    <style>
        .stAppDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title(" Road Damage & Pothole Detector")
st.caption("Classical ML (KNN / SVM / Decision Tree) on hand-crafted image features")

# Sidebar Model Selector
model_options = {
    "Best Model (Auto-Selected)": "best_model.joblib",
    "SVM (Support Vector Machine)": "svm.joblib",
    "KNN (K-Nearest Neighbors)": "knn.joblib",
    "Decision Tree": "decision_tree.joblib",
}

selected_label = st.sidebar.selectbox("Choose Classifier Model", list(model_options.keys()))
model_path = MODEL_DIR / model_options[selected_label]

if not model_path.exists():
    st.error(f"Model file `{model_options[selected_label]}` not found. Run `python src/train_compare.py` first.")
    st.stop()

bundle = joblib.load(model_path)
model, classes, name = bundle["model"], bundle["classes"], bundle["name"]

st.sidebar.success(f"Active Model: **{name}**")
up = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "bmp"])

if up:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    label, conf, probs = predict_image(img, model, classes)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader(f"Prediction ({name}): {label}  ({conf:.1%})")
    st.bar_chart(probs)
    if label.lower() in ("normal", "negative", "plain"):
        st.success("Road surface looks fine - no action needed.")
    else:
        st.warning("Damage detected - flag for inspection / repair scheduling.")
