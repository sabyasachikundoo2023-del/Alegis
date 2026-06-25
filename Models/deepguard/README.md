# 🛡️ DeepGuard v2

AI-powered deepfake & synthetic media detection using **Xception** deep learning with **99.6% AUC**.

> ⚠️ **Disclaimer** — For research and educational purposes only. Not intended for forensic or legal use without additional validation.

---

## 🔗 Live Demo

👉 **[Try DeepGuard Live →](https://sabya-ml-deepguard.streamlit.app/)**

---

## Overview

DeepGuard v2 detects AI-generated and face-swapped deepfake media using a fine-tuned **Xception** backbone trained on 20,000 real and fake face images. Upload any image or video and get an instant **REAL / FAKE verdict** with a confidence score and interactive gauge chart.

---

## ✨ Features

- **Xception backbone** — state-of-the-art architecture specifically designed for deepfake detection
- **Image & Video support** — JPG, PNG, WEBP images and MP4, AVI, MOV videos
- **Confidence gauge** — interactive Plotly gauge chart showing prediction confidence
- **Frame timeline** — per-frame fake probability chart for video analysis
- **Adjustable threshold** — tune the decision boundary (0.30 – 0.80)
- **Top-K frame averaging** — aggregates top K frames for robust video verdict
- **Dark UI** — clean, professional dark interface built with Streamlit
- **Auto model download** — model loads automatically from Google Drive on first run

---

## 📊 Performance

| Metric | Score |
|--------|-------|
| Val AUC | **99.67%** |
| Val Accuracy | **98.17%** |
| Val Precision | **97.82%** |
| Val Recall | **98.53%** |

Trained on **17,000 training** + **3,000 validation** face images using two-stage transfer learning.

---

## 🧠 Model Architecture

```
Input (299×299×3)
    └── Xception (ImageNet pre-trained)
            └── GlobalAveragePooling2D
                    └── BatchNormalization
                            └── Dropout(0.5)
                                    └── Dense(512, relu)
                                            └── BatchNormalization
                                                    └── Dropout(0.3)
                                                            └── Dense(256, relu)
                                                                    └── Dense(1, sigmoid)
```

**Training Strategy:**
- Stage 1: Frozen Xception backbone, train classification head only (15 epochs)
- Stage 2: Unfreeze top 30 layers, fine-tune entire network (up to 50 epochs with EarlyStopping)

**Loss:** Binary Cross-Entropy
**Optimizer:** Adam (lr=1e-3 → 1e-4 in Stage 2)
**Metrics:** AUC, Accuracy, Precision, Recall

---

## 📁 Project Structure

```
deepguard/
├── app.py                   ← Streamlit web application
├── requirements.txt
├── src/
│   ├── preprocess.py        ← Face crop, frame extraction, augmentation
│   ├── model.py             ← Model training script
│   └── predict.py           ← Unified inference engine
├── models/
│   └── deepguard_v2_best.h5 ← Trained Xception model
└── data/
    ├── raw/real/            ← Real face images
    └── raw/fake/            ← Fake/deepfake images
```

---

## 🚀 Installation

**Requirements:** Python 3.11, 8GB RAM, GPU optional

```bash
git clone https://github.com/sabyasachikundoo2023-del/Deepguard.git
cd Deepguard
pip install -r requirements.txt
```

---

## 💻 Usage

**Run the app:**
```bash
streamlit run app.py
# → http://localhost:8501
```

**CLI inference:**
```bash
# Image
python src/predict.py path/to/image.jpg

# Video
python src/predict.py path/to/video.mp4
```

**Python API:**
```python
from src.predict import predict_image, predict_video

# Image
label, confidence = predict_image("face.jpg")
print(f"{label} — {confidence:.1f}% confidence")
# FAKE — 100.0% confidence

# Video
label, confidence, frame_scores = predict_video("video.mp4")
print(f"{label} — {confidence:.1f}% confidence across {len(frame_scores)} frames")
```

---

## 🗂️ Dataset

Trained on:

| Dataset | Content | Size |
|---------|---------|------|
| [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | FFHQ real + StyleGAN2 fake | ~2 GB |
| [DFDC Preview](https://www.kaggle.com/c/deepfake-detection-challenge) | Video deepfakes | Optional |

**Download and place images into:**
```
data/raw/real/   ← real face JPGs
data/raw/fake/   ← fake face JPGs
```

**Then preprocess and train:**
```bash
python src/preprocess.py
python src/model.py --data_dir data/processed --epochs 30
```

---

## ⚙️ Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Decision threshold | 0.50 | Score ≥ threshold → FAKE |
| Top-K frames | 10 | Frames averaged for video verdict |
| Show frame timeline | On | Per-frame chart for videos |

---

## 📈 Training Results

| Stage | Epochs | Best Val AUC |
|-------|--------|--------------|
| Stage 1 (frozen backbone) | 15 | 0.918 |
| Stage 2 (fine-tuned) | 26 | **0.9967** |

---

## 🛠️ Tech Stack

**Backend:** TensorFlow 2.18 · Keras · OpenCV · Python 3.11
**Frontend:** Streamlit · Plotly
**Model:** Xception (ImageNet → fine-tuned on deepfake data)
**Training:** Kaggle P100 GPU

---

*Made by Sabyasachi Kundoo*
