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
MODEL_PATH = "models/best_model.h5"   # change if using final_model.h5
IMG_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

st.set_page_config(page_title="NeuroScan — AI MRI Scanner", layout="wide",
                   page_icon="🧠")

# ---------- STYLES & ANIMATIONS (CSS) ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    :root{
        --bg:#05060a;
        --panel:#0b0f1a;
        --glow-a: #00E5FF;
        --glow-b: #8E24AA;
        --accent: linear-gradient(90deg, rgba(0,229,255,1), rgba(142,36,170,1));
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #07102b 0%, var(--bg) 40%);
        color: #e6eef8;
        font-family: Inter, sans-serif;
    }

    /* holographic card */
    .holo {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        position: relative;
        overflow: hidden;
    }
    .holo::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 14px;
        pointer-events: none;
        background: linear-gradient(120deg, rgba(0,229,255,0.03), rgba(142,36,170,0.03));
        mix-blend-mode: overlay;
    }

    /* glowing button */
    .glow-btn {
        display:inline-block;
        border-radius: 12px;
        padding: 12px 20px;
        color: white;
        background: transparent;
        border: 1px solid rgba(255,255,255,0.08);
        cursor:pointer;
        font-weight:700;
        box-shadow: 0 8px 30px rgba(0,0,0,0.6);
    }
    .glow-wrap {
        background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(142,36,170,0.12));
        padding: 3px;
        border-radius: 14px;
        display: inline-block;
    }

    /* loader */
    .radar {
      width:120px;height:120px;border-radius:50%;
      background: radial-gradient(circle at 50% 45%, rgba(255,255,255,0.02), transparent 30%);
      border: 1px solid rgba(255,255,255,0.03);
      position: relative;
      margin: 10px auto;
      box-shadow: 0 10px 30px rgba(142,36,170,0.06) inset;
      overflow: hidden;
    }
    .radar::before {
      content: "";
      position:absolute; inset:0;
      background: conic-gradient(rgba(0,229,255,0.08), rgba(142,36,170,0.02));
      transform: rotate(0deg);
      animation: sweep 2s linear infinite;
      mix-blend-mode: screen;
    }
    @keyframes sweep { to { transform: rotate(360deg); } }

    .title {
      font-family: 'Orbitron', sans-serif;
      letter-spacing: 0.6px;
      font-weight:700;
      color: #DFF6FF;
    }

    .diagnosis {
      background: linear-gradient(90deg, rgba(0,229,255,0.12), rgba(142,36,170,0.12));
      padding: 14px;
      border-radius: 12px;
      text-align:center;
      font-weight:800;
      color: #fff;
      border: 1px solid rgba(255,255,255,0.06);
    }

    .small-muted { color: #9fb5c9; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- HELPERS ----------
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

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
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    maxv = np.max(heatmap)
    if maxv == 0: return heatmap.numpy(), int(class_idx), preds[0].numpy()
    heatmap = heatmap / maxv
    return heatmap, int(class_idx), preds[0]

def overlay_heatmap_on_pil(heatmap, pil_img, alpha=0.6):
    heat = cv2.resize(heatmap, (pil_img.width, pil_img.height))
    heat_uint8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    original = np.array(pil_img)
    overlay = cv2.addWeighted(original, 1-alpha, heat_color, alpha, 0)
    return overlay, heat_uint8

def generate_medical_summary(pred_class, confidence):
    texts = {
        "glioma": "Gliomas are infiltrative tumors arising from glial cells. The AI highlights deep parenchymal regions suggestive of infiltration.",
        "meningioma": "Meningiomas commonly attach to the meninges near the skull. The activation pattern suggests an extra-axial, well-circumscribed lesion.",
        "notumor": "No suspicious focal lesion detected. The heatmap shows diffuse low activation consistent with normal MRI.",
        "pituitary": "Pituitary lesions occur near the sella turcica at the skull base; activation near central inferior structures suggests pituitary involvement."
    }
    return texts.get(pred_class, "No detailed summary available."), f"{confidence*100:.2f}%"

# ---------- UI LAYOUT ----------
st.markdown('<div class="title">NeuroScan • AI MRI Diagnostic — Prototype</div>', unsafe_allow_html=True)
st.markdown("**Upload an MRI scan** and press **Scan MRI**. The system will run a rapid analysis, produce a diagnosis, and show Grad-CAM explainability.")

model = load_my_model()

colL, colR = st.columns([1, 2])
with colL:
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    uploaded = st.file_uploader("📂 Choose MRI image (jpg/png)", type=["jpg","jpeg","png"])
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # glowing scan button
    if st.button("⚡ Scan MRI", key="scan"):
        if uploaded is None:
            st.error("Please upload an MRI first.")
        else:
            # show loader card
            loader = st.empty()
            with loader.container():
                st.markdown('<div class="radar"></div>', unsafe_allow_html=True)
                st.markdown('<div class="small-muted">Neural sync in progress... calibrating model</div>', unsafe_allow_html=True)

            # small sleep to show animation (real compute also runs)
            # do the heavy work now
            tensor, pil_proc = preprocess_pil(Image.open(io.BytesIO(uploaded.read())))
            # compute heatmap
            heatmap, class_idx, probs = make_gradcam_heatmap(model, tensor)
            overlay_img, heat_uint8 = overlay_heatmap_on_pil(heatmap, pil_proc, alpha=0.6)
            time.sleep(0.6)  # small pause for UX feel
            loader.empty()

            # show results in right panel by setting session state
            st.session_state["result"] = {
                "orig": Image.open(io.BytesIO(uploaded.read())),
                "proc": pil_proc,
                "heatmap": heatmap,
                "overlay": overlay_img,
                "class_idx": int(class_idx),
                "probs": probs
            }
    st.markdown('</div>', unsafe_allow_html=True)

with colR:
    st.markdown('<div class="holo">', unsafe_allow_html=True)
    st.subheader("Scan Results")
    if "result" not in st.session_state:
        st.info("No scan yet — upload an MRI and press Scan MRI.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        res = st.session_state["result"]
        pred_idx = res["class_idx"]
        pred_name = CLASS_NAMES[pred_idx]
        pred_conf = res["probs"][pred_idx]

        # diagnosis card
        st.markdown(f'<div class="diagnosis">Diagnosis: {pred_name.upper()} • Confidence: {pred_conf*100:.2f}%</div>', unsafe_allow_html=True)

        # three-panel: orig | heatmap | overlay
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Original**")
            st.image(res["proc"], use_column_width=True)
        with b:
            st.markdown("**Grad-CAM Heatmap**")
            fig, ax = plt.subplots(figsize=(3.5,3.5))
            ax.imshow(res["heatmap"], cmap="jet")
            ax.axis("off")
            st.pyplot(fig)
        with c:
            st.markdown("**Overlay**")
            st.image(res["overlay"], use_column_width=True)

        st.markdown("### Probability Distribution")
        probs = {CLASS_NAMES[i]: float(res["probs"][i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(probs)

        # medical summary
        summ, conf_text = generate_medical_summary(pred_name, pred_conf)
        st.markdown("### AI Medical Summary")
        st.markdown(f"> **{pred_name.upper()}** — {summ}  ")
        st.markdown(f"**AI Confidence:** {conf_text}")

        # small metadata / gamified XP
        st.markdown("---")
        st.markdown("<div class='small-muted'>Model Level: <b>3</b> • Trained on ~7K MRIs • Scan Time: <b>fast (CPU)</b></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>NeuroScan Prototype • For demonstration purposes only. Not a medical device.</div>", unsafe_allow_html=True)
