"""
src/model.py
------------
EfficientNet-B4 transfer-learning model for deepfake detection.

Usage:
    python src/model.py --data_dir data/processed --epochs 30 --batch_size 32
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 1e-4
DROPOUT     = 0.5
MODEL_DIR   = Path("models")
LOG_DIR     = Path("logs")


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(
    img_size: int   = IMG_SIZE,
    dropout: float  = DROPOUT,
    fine_tune_from: int = 200,   # unfreeze layers after this index
) -> tf.keras.Model:
    """
    EfficientNet-B4 backbone → GAP → Dropout → Sigmoid output.

    Stage 1: backbone frozen, train new head only.
    Call `unfreeze_backbone(model)` for stage-2 fine-tuning.
    """
    base = EfficientNetB4(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False   # frozen for stage 1

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dropout(dropout)(x)
    x      = layers.Dense(256, activation="relu")(x)
    x      = layers.Dropout(dropout * 0.6)(x)
    output = layers.Dense(1, activation="sigmoid", name="deepfake_score")(x)

    model = models.Model(inputs, output, name="DeepGuard_EfficientNetB4")
    return model, base


def unfreeze_backbone(model: tf.keras.Model, base_model, fine_tune_from: int = 200):
    """Stage-2: unfreeze upper layers of the backbone for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    print(f"[unfreeze_backbone] Trainable layers: "
          f"{sum(1 for l in base_model.layers if l.trainable)} / {len(base_model.layers)}")


# ── Data loaders ──────────────────────────────────────────────────────────────
def build_generators(data_dir: str, batch_size: int = BATCH_SIZE):
    """Returns (train_gen, val_gen) using Keras ImageDataGenerator."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen


# ── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks(model_path: str, log_dir: str):
    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    ]


# ── Training loop ─────────────────────────────────────────────────────────────
def train(data_dir: str, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    model_path = str(MODEL_DIR / "deepguard_best.h5")

    # Generators
    train_gen, val_gen = build_generators(data_dir, batch_size)

    # Handle class imbalance
    total = train_gen.samples
    n_fake = train_gen.classes.sum()
    n_real = total - n_fake
    class_weight = {
        0: total / (2 * n_real + 1e-6),
        1: total / (2 * n_fake + 1e-6),
    }
    print(f"[train] Class weights: {class_weight}")

    # ── Stage 1: Train head only ──────────────────────────────────────────────
    print("\n[Stage 1] Training classification head with frozen backbone...")
    model, base_model = build_model()
    model.compile(
        optimizer=optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()

    history_s1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_callbacks(model_path, str(LOG_DIR / "stage1")),
    )

    # ── Stage 2: Fine-tune upper backbone ────────────────────────────────────
    print("\n[Stage 2] Fine-tuning upper backbone layers...")
    unfreeze_backbone(model, base_model, fine_tune_from=200)
    model.compile(
        optimizer=optimizers.Adam(LR / 10),   # lower LR for fine-tuning
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    history_s2 = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=get_callbacks(model_path, str(LOG_DIR / "stage2")),
    )

    # Save final weights
    model.save(str(MODEL_DIR / "deepguard_final.h5"))
    print(f"\n[train] Model saved → {MODEL_DIR}/deepguard_final.h5")

    # ── Plot training curves ──────────────────────────────────────────────────
    _plot_history(history_s1, history_s2)
    return model


def _plot_history(h1, h2):
    """Merge stage histories and plot loss + AUC."""
    def merge(key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(merge("loss"),     label="Train Loss",  color="#e74c3c")
    axes[0].plot(merge("val_loss"), label="Val Loss",    color="#3498db", linestyle="--")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(merge("auc"),      label="Train AUC",  color="#2ecc71")
    axes[1].plot(merge("val_auc"),  label="Val AUC",    color="#9b59b6", linestyle="--")
    axes[1].set_title("AUC Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("logs/training_curves.png", dpi=150)
    print("[train] Training curves saved → logs/training_curves.png")
    plt.close()


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model_path: str, data_dir: str, batch_size: int = BATCH_SIZE):
    """Load saved model and print val-set metrics + confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    model    = tf.keras.models.load_model(model_path)
    _, val_gen = build_generators(data_dir, batch_size)

    preds    = model.predict(val_gen, verbose=1).ravel()
    y_pred   = (preds >= 0.5).astype(int)
    y_true   = val_gen.classes

    print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png", dpi=150)
    print("[evaluate] Confusion matrix saved → logs/confusion_matrix.png")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepGuard deepfake detector")
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval_only",  action="store_true")
    parser.add_argument("--model_path", default="models/deepguard_best.h5")
    args = parser.parse_args()

    if args.eval_only:
        evaluate(args.model_path, args.data_dir, args.batch_size)
    else:
        train(args.data_dir, args.epochs, args.batch_size)
