import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
import time
import io
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
MODEL_PATH = "models/best_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

st.set_page_config(page_title="NeuroScan — AI MRI Scanner", layout="wide", page_icon="🧠")

# ---------- STYLES ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 15% 10%, #101a3a 0%, #05060a 45%);
        color: #e6eef8;
        font-family: Inter, sans-serif;
    }

    /* Header Glow */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 38px;
        font-weight: 700;
        color: #DFF6FF;
        text-shadow: 0px 0px 15px rgba(0,229,255,0.35);
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 15px;
        color: #9fb5c9;
        margin-top: 0px;
        margin-bottom: 20px;
    }

    /* Holographic Card */
    .holo {
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.55);
        position: relative;
        overflow: hidden;
    }
    .holo::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 18px;
        pointer-events: none;
        background: linear-gradient(120deg, rgba(0,229,255,0.05), rgba(142,36,170,0.05));
        mix-blend-mode: overlay;
    }

    /* Diagnosis Card */
    .diagnosis {
        background: linear-gradient(90deg, rgba(0,229,255,0.14), rgba(142,36,170,0.14));
        padding: 16px;
        border-radius: 14px;
        text-align:center;
        font-weight:800;
        color: #fff;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 0 22px rgba(0,229,255,0.12);
        margin-bottom: 15px;
    }

    /* Loader Radar */
    .radar {
      width:120px;height:120px;border-radius:50%;
      background: radial-gradient(circle at 50% 45%, rgba(255,255,255,0.02), transparent 30%);
      border: 1px solid rgba(255,255,255,0.06);
      position: relative;
      margin: 10px auto;
      box-shadow: 0 10px 30px rgba(142,36,170,0.06) inset;
      overflow: hidden;
    }
    .radar::before {
      content: "";
      position:absolute; inset:0;
      background: conic-gradient(rgba(0,229,255,0.15), rgba(142,36,170,0.02));
      transform: rotate(0deg);
      animation: sweep 1.5s linear infinite;
      mix-blend-mode: screen;
    }
    @keyframes sweep { to { transform: rotate(360deg); } }

    .small-muted { color: #9fb5c9; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HELPERS ----------
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

def preprocess_pil(pil_img):
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, pil_img

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        shape = getattr(layer.output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise RuntimeError("No conv layer found")

def make_gradcam_heatmap(model, img_tensor):
    last_conv = find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)

    maxv = np.max(heatmap)
    heatmap = heatmap / (maxv if maxv != 0 else 1)

    return heatmap, int(class_idx), preds[0].numpy()

def overlay_heatmap_on_pil(heatmap, pil_img, alpha=0.55):
    heat = cv2.resize(heatmap, (pil_img.width, pil_img.height))
    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(pil_img), 1 - alpha, heat_color, alpha, 0)
    return overlay

def generate_medical_summary(pred_class):
    texts = {
        "glioma": "Gliomas are infiltrative tumors arising from glial cells. The model focuses on deep brain tissue abnormalities.",
        "meningioma": "Meningiomas often form near the outer brain surface. The model highlights boundary-adjacent regions.",
        "notumor": "No suspicious lesion detected. Activation is low and distributed, consistent with normal scans.",
        "pituitary": "Pituitary tumors occur near the skull base. The highlighted region suggests pituitary involvement."
    }
    return texts.get(pred_class, "No summary available.")

# ---------- UI ----------
st.markdown('<div class="main-title">NeuroScan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Cyber-Medical AI MRI Scanner • Tumor Classification + Grad-CAM Explainability</div>', unsafe_allow_html=True)

model = load_my_model()

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.subheader("📤 Upload MRI")
    uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    scan = st.button("⚡ Scan MRI", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.subheader("📊 Scan Results")

    if scan:
        if uploaded is None:
            st.error("Upload an MRI first.")
        else:
            # ✅ FIX: READ ONCE
            file_bytes = uploaded.getvalue()
            img = Image.open(io.BytesIO(file_bytes))

            loader = st.empty()
            with loader.container():
                st.markdown('<div class="radar"></div>', unsafe_allow_html=True)
                st.markdown('<div class="small-muted">Neural sync in progress... calibrating model</div>', unsafe_allow_html=True)

            tensor, pil_proc = preprocess_pil(img)
            heatmap, class_idx, probs = make_gradcam_heatmap(model, tensor)

            pred_name = CLASS_NAMES[class_idx]
            pred_conf = float(probs[class_idx]) * 100

            overlay = overlay_heatmap_on_pil(heatmap, pil_proc)

            time.sleep(0.4)
            loader.empty()

            st.markdown(
                f'<div class="diagnosis">Diagnosis: {pred_name.upper()} • Confidence: {pred_conf:.2f}%</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)
            with a:
                st.markdown("**Original MRI**")
                st.image(pil_proc, use_container_width=True)

            with b:
                st.markdown("**Grad-CAM Heatmap**")
                fig, ax = plt.subplots(figsize=(3.5, 3.5))
                ax.imshow(heatmap, cmap="jet")
                ax.axis("off")
                st.pyplot(fig)

            with c:
                st.markdown("**Overlay View**")
                st.image(overlay, use_container_width=True)

            st.markdown("### 📈 Probability Distribution")
            prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
            st.bar_chart(prob_dict)

            st.markdown("### 🧾 AI Medical Summary")
            st.info(generate_medical_summary(pred_name))

    else:
        st.info("Upload an MRI and press **Scan MRI** to generate the report.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>⚠️ Prototype for demonstration only. Not a medical device.</div>", unsafe_allow_html=True)
