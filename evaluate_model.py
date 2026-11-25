import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "landmark_model.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
VAL_DIR = ROOT / "dataset" / "validation"
RESULTS_DIR = ROOT / "results"


def main():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise SystemExit("Trained model not found. Run python train_model.py first.")

    metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, class_names=class_names,
        image_size=(image_size, image_size), batch_size=32,
        label_mode="int", shuffle=False
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    y_true, y_pred = [], []

    for images, labels in val_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(probabilities, axis=1).tolist())

    RESULTS_DIR.mkdir(exist_ok=True)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    )
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Validation accuracy: {accuracy:.4f}\n\n{report}", encoding="utf-8"
    )

    # Large class sets can make a full confusion matrix hard to read.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig_size = min(18, max(8, len(class_names) * 0.12))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(ax=ax, xticks_rotation="vertical", colorbar=False, values_format="d")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Report saved to {RESULTS_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    main()
