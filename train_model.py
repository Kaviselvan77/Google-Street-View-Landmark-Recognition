import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "train"
VAL_DIR = ROOT / "dataset" / "validation"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def build_model(num_classes: int, image_size: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        augmentation,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="landmark_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the landmark recognition CNN.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()

    if not TRAIN_DIR.exists() or not any(p.is_dir() for p in TRAIN_DIR.iterdir()):
        raise SystemExit("No training classes found. Add images to dataset/train/<class_name>/ first.")
    if not VAL_DIR.exists() or not any(p.is_dir() for p in VAL_DIR.iterdir()):
        raise SystemExit("No validation classes found. Add images to dataset/validation/<class_name>/ first.")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, label_mode="int", shuffle=True,
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, class_names=class_names,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size, label_mode="int", shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    model = build_model(len(class_names), args.image_size)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "landmark_model.keras",
            monitor="val_accuracy", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, callbacks=callbacks
    )

    # Save class labels and image size for the prediction app.
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps({"class_names": class_names, "image_size": args.image_size}, indent=2),
        encoding="utf-8"
    )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print(f"\nTraining complete. Classes: {len(class_names)}")
    print(f"Model saved to: {MODEL_DIR / 'landmark_model.keras'}")


if __name__ == "__main__":
    main()
