"""
src/predict.py
--------------
Unified inference engine for DeepGuard v2 (Xception model).
 
  predict_image(source)  -> (label, confidence_pct)
  predict_video(source)  -> (label, confidence_pct, per_frame_scores)
"""
 
import os
import tempfile
from pathlib import Path
 
import cv2
import numpy as np
from PIL import Image
 
# ── Constants for Xception v2 ─────────────────────────────────────────────────
_model         = None
MODEL_DIR      = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL  = MODEL_DIR / "deepguard_v2_best.h5"
IMG_SIZE       = 299      # Xception native input size
FAKE_THRESHOLD = 0.50
TOP_K_FRAMES   = 10


def _find_model_path():
    """Auto-detect model — checks env var first, then local model files."""
    env_path = os.environ.get("DEEPGUARD_MODEL")
    if env_path:
        env_model = Path(env_path)
        if env_model.exists():
            return env_model.resolve()

    for candidate in [
        DEFAULT_MODEL,
        MODEL_DIR / "deepguard_best.h5",
        Path("models/deepguard_v2_best.h5"),
        Path("models/deepguard_best.h5"),
    ]:
        if candidate.exists():
            return candidate.resolve()

    return DEFAULT_MODEL
 
 
# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global _model
    if _model is None:
        import tensorflow as tf
        model_path = _find_model_path()
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Make sure deepguard_v2_best.h5 is in the models/ folder."
            )
        _model = tf.keras.models.load_model(model_path)
        print(f"[predict] Model loaded from {model_path}")
    return _model
 
 
# ── Preprocessing ─────────────────────────────────────────────────────────────
def _preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Resize to 299x299 and normalize to [0,1] for Xception."""
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    x = img_resized.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)
 
 
def _to_rgb_array(source) -> np.ndarray:
    """Accept path / PIL Image / numpy array / Streamlit UploadedFile -> RGB ndarray."""
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.ndim == 3 and source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        if source.ndim == 3 and source.shape[2] == 3:
            return cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unsupported ndarray shape: {source.shape}")

    if isinstance(source, Image.Image):
        return np.array(source.convert("RGB"))

    if hasattr(source, "read"):
        data = source.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
 
    img = cv2.imread(str(source))
    if img is None:
        raise ValueError(f"Cannot read image from: {source}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
 
def _score_to_verdict(score: float):
    label      = "FAKE" if score >= FAKE_THRESHOLD else "REAL"
    confidence = score * 100 if label == "FAKE" else (1 - score) * 100
    return label, round(confidence, 2)
 
 
# ── Public API ────────────────────────────────────────────────────────────────
def predict_image(source):
    model   = load_model()
    img_rgb = _to_rgb_array(source)
    x       = _preprocess(img_rgb)
    score   = float(model.predict(x, verbose=0)[0][0])
    return _score_to_verdict(score)
 
 
def predict_video(source, fps=2, top_k=TOP_K_FRAMES, progress_callback=None):
    tmp_path = None
    if hasattr(source, "read"):
        suffix = Path(getattr(source, "name", "video.mp4")).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(source.read())
            tmp_path = f.name
        video_path = tmp_path
    else:
        video_path = str(source)
 
    try:
        cap    = cv2.VideoCapture(video_path)
        native = cap.get(cv2.CAP_PROP_FPS) or 30
        step   = max(1, int(round(native / fps)))
        frames = []
        idx    = 0
        while len(frames) < 60:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
 
    if not frames:
        raise ValueError("No frames could be extracted from the video.")
 
    model  = load_model()
    scores = []
 
    for i, frame_bgr in enumerate(frames):
        try:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            x       = _preprocess(img_rgb)
            score   = float(model.predict(x, verbose=0)[0][0])
            scores.append(score)
        except Exception:
            pass
 
        if progress_callback:
            progress_callback((i + 1) / len(frames))
 
    if not scores:
        raise RuntimeError("No valid frames were processed.")
 
    avg_score         = float(np.mean(sorted(scores, reverse=True)[:top_k]))
    label, confidence = _score_to_verdict(avg_score)
    return label, confidence, scores
 
 
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python src/predict.py <image_or_video_path>")
        sys.exit(1)
    ext = Path(path).suffix.lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        label, conf, scores = predict_video(path)
        print(f"Video verdict : {label}  ({conf:.1f}%)")
        print(f"Frames        : {len(scores)}  |  Mean: {np.mean(scores):.3f}")
    else:
        label, conf = predict_image(path)
        print(f"Image verdict : {label}  ({conf:.1f}%)")