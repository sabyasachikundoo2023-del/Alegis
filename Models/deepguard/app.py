"""
app.py  ──  DeepGuard v2: Professional Deepfake Detection System
Run with:   streamlit run app.py
"""

import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="DeepGuard — Deepfake Detector", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%); color: #e2e8f0; }
.main-header { text-align: center; padding: 2rem 0 1rem 0; }
.main-header h1 { font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7b2fff, #ff6b35); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px; }
.main-header p { color: #94a3b8; font-size: 1.1rem; }
.result-card { border-radius: 16px; padding: 1.5rem 2rem; margin: 1rem 0; border: 1px solid; }
.result-fake { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.4); }
.result-real { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.4); }
.stat-pill { display: inline-block; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 50px; padding: 4px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; }
section[data-testid="stSidebar"] { background: rgba(13,27,42,0.95); border-right: 1px solid rgba(255,255,255,0.06); }
.info-banner { background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.25); border-radius: 10px; padding: 0.75rem 1rem; font-size: 0.9rem; color: #94a3b8; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Auto-download model from Google Drive if not present ──────────────────────
GDRIVE_FILE_ID = "1z9rGvTz2PpuD6EYWd_ZU7LAp-DlhQlnU"
MODEL_DIR      = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL  = MODEL_DIR / "deepguard_v2_best.h5"
MODEL_PATH     = DEFAULT_MODEL


def _find_model_path():
    """Auto-detect model — checks env var first, then local model files."""
    env_path = os.environ.get("DEEPGUARD_MODEL")
    if env_path:
        env_model = Path(env_path)
        if env_model.exists():
            return env_model.resolve()

    candidates = [
        DEFAULT_MODEL,
        MODEL_DIR / "deepguard_best.h5",
        Path("models/deepguard_v2_best.h5"),
        Path("models/deepguard_best.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_MODEL


def download_model():
    target_path = DEFAULT_MODEL
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size < 10_000_000:
        target_path.unlink()

    if not target_path.exists():
        try:
            import gdown
            st.info("⬇️ Downloading model... please wait ~60 seconds")
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=str(target_path),
                quiet=False,
                fuzzy=True,
            )
        except Exception as e:
            st.error(f"Failed to download model: {e}")

MODEL_PATH = _find_model_path()
download_model()
model_exists = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000

@st.cache_resource
def load_model():
    import tensorflow as tf
    # We import these to handle the 'batch_shape' incompatibility in older TF versions
    from tensorflow.keras.layers import InputLayer
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Copy `deepguard_v2_best.h5` into `{MODEL_DIR}` or set DEEPGUARD_MODEL."
        )
    
    st.info(f"Loading model from {MODEL_PATH}")
    
    try:
        # Attempt standard load
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except ValueError as e:
        if "batch_shape" in str(e):
            # If it fails due to batch_shape, we manually override the InputLayer 
            # to ignore that specific keyword argument
            class FixedInputLayer(InputLayer):
                def __init__(self, **kwargs):
                    kwargs.pop('batch_shape', None)
                    super().__init__(**kwargs)
            
            model = tf.keras.models.load_model(
                str(MODEL_PATH), 
                compile=False, 
                custom_objects={'InputLayer': FixedInputLayer}
            )
        else:
            raise e
            
    return model


def preprocess(img_rgb):
    """Preprocess image for model input."""
    if isinstance(img_rgb, Image.Image):
        img_rgb = np.array(img_rgb)

    if not isinstance(img_rgb, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(img_rgb)}")

    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    elif img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image array, got shape {img_rgb.shape}")

    img = cv2.resize(img_rgb, (299, 299)).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ── Gauge chart ────────────────────────────────────────────────────────────────
def build_gauge(confidence, label):
    color = "#ef4444" if label == "FAKE" else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=confidence,
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        delta={"reference": 50},
        title={"text": f"<b>{label}</b> Confidence", "font": {"size": 18, "color": "#e2e8f0"}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.6},
               "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": confidence}},
    ))
    fig.update_layout(height=280, margin=dict(l=30,r=30,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig

def build_timeline(scores):
    ys = [s*100 for s in scores]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(scores))), y=ys, mode="lines+markers",
                             line=dict(color="#7b2fff", width=2),
                             marker=dict(color=["#ef4444" if s>=50 else "#22c55e" for s in ys], size=6)))
    fig.add_hline(y=50, line_dash="dash", line_color="#64748b")
    fig.update_layout(title="Per-frame Fake Probability", xaxis_title="Frame",
                      yaxis_title="Fake score (%)", yaxis_range=[0,100], height=260,
                      margin=dict(l=20,r=20,t=40,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                      font_color="#94a3b8")
    return fig

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DeepGuard v2")
    st.markdown("**Xception · 99.6% AUC**")
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown("DeepGuard uses **Xception** fine-tuned to detect face-swap deepfakes and AI-generated portraits.")
    st.divider()
    st.markdown("### ⚙️ How it works")
    st.markdown("1. Upload image or video\n2. Model analyses face region\n3. Returns REAL or FAKE verdict with confidence score")
    st.divider()
    st.markdown("<small style='color:#475569'>DeepGuard v2 · Xception · TensorFlow</small>", unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>🛡️ DeepGuard</h1><p>AI-powered deepfake detection · Xception · 99.6% AUC</p></div>', unsafe_allow_html=True)

if not model_exists:
    st.warning("⚠️ No model found! Copy `deepguard_v2_best.h5` into the `models/` folder then restart.", icon="⚠️")
else:
    st.success(f"✅ Model loaded: `{MODEL_PATH}`", icon="✅")

st.divider()

col_upload, col_settings = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader("📁 Upload image or video", type=["jpg","jpeg","png","webp","mp4","avi","mov"])
with col_settings:
    st.markdown("**Settings**")
    threshold   = st.slider("Decision threshold", 0.30, 0.80, 0.50, 0.01)
    top_k       = st.slider("Top-K frames (video)", 5, 30, 10)
    show_frames = st.checkbox("Show frame timeline", value=True)

if uploaded:
    is_video = uploaded.type.startswith("video")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### 🎞 Preview")
        if is_video:
            st.video(uploaded)
            uploaded.seek(0)
        else:
            try:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.stop()

    with right:
        st.markdown("#### 🔬 Analysis")
        if not model_exists:
            st.error("Model not loaded. Add model file to models/ folder.")
        else:
            with st.spinner("🔍 Analysing…"):
                try:
                    model = load_model()

                    if is_video:
                        pb = st.progress(0, text="Extracting frames…")
                        tmp = None
                        try:
                            suffix = Path(uploaded.name).suffix or ".mp4"
                            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                                f.write(uploaded.read())
                                tmp = f.name

                            cap = cv2.VideoCapture(tmp)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30
                            step = max(1, int(fps / 2))
                            frames = []
                            idx = 0
                            while len(frames) < 60:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                if idx % step == 0:
                                    frames.append(frame)
                                idx += 1
                            cap.release()

                            if not frames:
                                raise ValueError("No frames could be extracted from the video.")

                            scores = []
                            for i, frame in enumerate(frames):
                                x = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                score = float(model.predict(x, verbose=0)[0][0])
                                scores.append(score)
                                pb.progress((i + 1) / len(frames), text=f"Frame {i+1}/{len(frames)}")
                            pb.empty()
                            avg_score = float(np.mean(sorted(scores, reverse=True)[:top_k]))
                            frame_scores = scores
                        except Exception as e:
                            pb.empty()
                            raise
                        finally:
                            if tmp and os.path.exists(tmp):
                                try:
                                    os.remove(tmp)
                                except Exception:
                                    pass
                    else:
                        avg_score = float(model.predict(preprocess(img), verbose=0)[0][0])
                        frame_scores = None

                    # Class 0 = fake, Class 1 = real (alphabetical order)
                    # So score close to 0 = FAKE, score close to 1 = REAL
                    label = "REAL" if avg_score >= threshold else "FAKE"
                    confidence = avg_score * 100 if label == "REAL" else (1 - avg_score) * 100
                    colour = "#ef4444" if label == "FAKE" else "#22c55e"
                    icon = "🚨" if label == "FAKE" else "✅"

                    st.markdown(f'<div class="result-card result-{"fake" if label=="FAKE" else "real"}"><h2 style="color:{colour};margin:0">{icon} {label}</h2><p style="color:#94a3b8;margin:4px 0 0 0">Confidence: <span class="stat-pill">{confidence:.1f}%</span></p></div>', unsafe_allow_html=True)
                    st.plotly_chart(build_gauge(confidence, label), use_container_width=True, config={"displayModeBar": False})

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Verdict", label)
                    m2.metric("Confidence", f"{confidence:.1f}%")
                    m3.metric("Frames" if frame_scores else "Media", len(frame_scores) if frame_scores else "Image")

                    if is_video and frame_scores and show_frames:
                        st.plotly_chart(build_timeline(frame_scores), use_container_width=True, config={"displayModeBar": False})

                    st.markdown(f'<div class="info-banner">{"⚠️ Strong signs of AI manipulation detected." if label=="FAKE" else "✅ No deepfake artefacts detected. Media appears authentic."}<br><small>No detector is 100% accurate.</small></div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.exception(e)

st.divider()
st.markdown("<div style='text-align:center;color:#334155;font-size:0.8rem;'>DeepGuard v2 · 99.6% AUC · Research purposes only</div>", unsafe_allow_html=True)
