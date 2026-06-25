"""
src/preprocess.py
-----------------
Handles all data preprocessing:
  - Face detection & cropping via MTCNN / MediaPipe fallback
  - Frame extraction from videos at 2 FPS
  - Augmentation pipeline using Albumentations
  - Dataset preparation and train/val split
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE       = 224          # EfficientNet-B4 native input
VIDEO_FPS      = 2            # frames to extract per second
FACE_MARGIN    = 0.3          # fractional padding around face crop
TRAIN_RATIO    = 0.85
AUGMENT_FACTOR = 2            # how many augmented copies per original image

# ── Augmentation pipelines ────────────────────────────────────────────────────
TRAIN_TRANSFORM = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.85, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
    ], p=0.4),
    A.OneOf([
        A.ImageCompression(quality_lower=50, quality_upper=90, p=1.0),
        A.Downscale(scale_min=0.6, scale_max=0.9, p=1.0),
    ], p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05, p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Face detector (MTCNN preferred, MediaPipe fallback) ───────────────────────
class FaceDetector:
    """Tries MTCNN first; falls back to MediaPipe if MTCNN unavailable."""

    def __init__(self, use_mtcnn: bool = True):
        self.detector = None
        self.backend  = None

        if use_mtcnn:
            try:
                from mtcnn import MTCNN
                self.detector = MTCNN()
                self.backend  = "mtcnn"
                print("[FaceDetector] Using MTCNN backend.")
            except ImportError:
                print("[FaceDetector] MTCNN not found — falling back to MediaPipe.")

        if self.detector is None:
            try:
                import mediapipe as mp
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                self.backend = "mediapipe"
                print("[FaceDetector] Using MediaPipe backend.")
            except ImportError:
                print("[FaceDetector] WARNING: No face detector available. Using full frame.")
                self.backend = "none"

    def detect(self, img_rgb: np.ndarray):
        """Returns (x1, y1, x2, y2) of the largest detected face, or None."""
        h, w = img_rgb.shape[:2]

        if self.backend == "mtcnn":
            results = self.detector.detect_faces(img_rgb)
            if not results:
                return None
            # take highest-confidence detection
            best = max(results, key=lambda r: r["confidence"])
            x, y, bw, bh = best["box"]
            return _add_margin(x, y, x + bw, y + bh, w, h, FACE_MARGIN)

        elif self.backend == "mediapipe":
            res = self.detector.process(img_rgb)
            if not res.detections:
                return None
            det = res.detections[0]
            bb  = det.location_data.relative_bounding_box
            x1  = int(bb.xmin * w)
            y1  = int(bb.ymin * h)
            x2  = int((bb.xmin + bb.width) * w)
            y2  = int((bb.ymin + bb.height) * h)
            return _add_margin(x1, y1, x2, y2, w, h, FACE_MARGIN)

        return None   # no detector


def _add_margin(x1, y1, x2, y2, W, H, margin):
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(W, int(x2 + margin * bw))
    y2 = min(H, int(y2 + margin * bh))
    return x1, y1, x2, y2


# ── Single-image preprocessing ────────────────────────────────────────────────
_detector = None   # module-level singleton

def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def crop_face(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Detect face, crop, resize to (size, size). Returns RGB numpy array."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    det     = get_detector()
    box     = det.detect(img_rgb)

    if box is not None:
        x1, y1, x2, y2 = box
        face = img_rgb[y1:y2, x1:x2]
    else:
        face = img_rgb   # fall back to full frame

    face = cv2.resize(face, (size, size))
    return face   # HWC, RGB, uint8


def preprocess_for_inference(img_rgb: np.ndarray) -> np.ndarray:
    """Full pipeline for a single RGB image → normalized float32 array (HWC)."""
    face = crop_face(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    transformed = VAL_TRANSFORM(image=face)["image"]
    return np.expand_dims(transformed, axis=0)   # (1, H, W, C)


# ── Video frame extraction ─────────────────────────────────────────────────────
def extract_frames(video_path: str, fps: int = VIDEO_FPS, max_frames: int = 60) -> list[np.ndarray]:
    """Extract frames at `fps` from a video. Returns list of BGR numpy arrays."""
    cap    = cv2.VideoCapture(str(video_path))
    native = cap.get(cv2.CAP_PROP_FPS) or 30
    step   = max(1, int(round(native / fps)))
    frames = []
    idx    = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# ── Dataset builder ────────────────────────────────────────────────────────────
def build_dataset(
    raw_dir: str,
    out_dir: str,
    augment: bool = True,
    train_ratio: float = TRAIN_RATIO,
):
    """
    Expects:
        raw_dir/real/  ← real face images (JPG/PNG) or videos (MP4)
        raw_dir/fake/  ← fake face images or videos

    Produces:
        out_dir/train/real, out_dir/train/fake
        out_dir/val/real,   out_dir/val/fake
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    det     = FaceDetector()

    for label in ("real", "fake"):
        src_dir     = raw_dir / label
        all_files   = sorted(
            list(src_dir.glob("*.jpg")) +
            list(src_dir.glob("*.jpeg")) +
            list(src_dir.glob("*.png")) +
            list(src_dir.glob("*.mp4")) +
            list(src_dir.glob("*.avi")) +
            list(src_dir.glob("*.mov"))
        )
        if not all_files:
            print(f"[build_dataset] No files found in {src_dir} — skipping.")
            continue

        n_train = int(len(all_files) * train_ratio)
        splits  = {"train": all_files[:n_train], "val": all_files[n_train:]}

        for split, files in splits.items():
            dest = out_dir / split / label
            dest.mkdir(parents=True, exist_ok=True)
            saved = 0

            for fpath in tqdm(files, desc=f"{label}/{split}"):
                if fpath.suffix.lower() in {".mp4", ".avi", ".mov"}:
                    frames = extract_frames(str(fpath))
                else:
                    frames = [cv2.imread(str(fpath))]

                for i, frame in enumerate(frames):
                    if frame is None:
                        continue
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    box     = det.detect(img_rgb)
                    if box:
                        x1, y1, x2, y2 = box
                        face = img_rgb[y1:y2, x1:x2]
                    else:
                        face = img_rgb

                    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    stem         = f"{fpath.stem}_f{i:04d}"
                    out_path     = dest / f"{stem}.jpg"
                    Image.fromarray(face_resized).save(out_path, quality=95)
                    saved += 1

                    # Augmentation copies (train only)
                    if augment and split == "train":
                        for aug_i in range(AUGMENT_FACTOR):
                            aug  = TRAIN_TRANSFORM(image=face_resized)["image"]
                            # de-normalize back to uint8 for saving
                            mean = np.array([0.485, 0.456, 0.406])
                            std  = np.array([0.229, 0.224, 0.225])
                            aug_uint8 = np.clip((aug * std + mean) * 255, 0, 255).astype(np.uint8)
                            aug_path  = dest / f"{stem}_aug{aug_i}.jpg"
                            Image.fromarray(aug_uint8).save(aug_path, quality=90)
                            saved += 1

            print(f"  [{label}/{split}] Saved {saved} images → {dest}")

    print("\n[build_dataset] Done. Dataset ready at:", out_dir)


if __name__ == "__main__":
    build_dataset(
        raw_dir="data/raw",
        out_dir="data/processed",
        augment=True,
    )
