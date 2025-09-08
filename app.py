# app.py
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils import load_class_indices
import os

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image and get the predicted disease class and confidence score.")

MODEL_PATH = "models/plant_disease_model.h5"
CLASS_MAP_PATH = "models/class_indices.json"

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    mapping, inv = load_class_indices(CLASS_MAP_PATH)
    # Build list of class names in index order
    num_classes = len(inv)
    class_names = [inv[i] for i in range(num_classes)]
    return model, class_names

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_MAP_PATH):
    st.error("Model or class mapping not found. Please run training first (train.py).")
else:
    model, class_names = load_resources()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_resized = cv2.resize(img_rgb, (128, 128))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0) / 255.0

        preds = model.predict(x)[0]
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100.0

        st.markdown("### Prediction")
        st.write(f"**Class:** {class_names[idx]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Show probability bar for each class
        st.markdown("### Class probabilities")
        probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
        for cls, p in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            st.write(f"{cls}: {p*100:.2f}%")
