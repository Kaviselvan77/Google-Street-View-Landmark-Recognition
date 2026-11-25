import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "landmark_model.keras"
LABELS_PATH = ROOT / "models" / "class_names.json"
UPLOAD_DIR = ROOT / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_class_names = None
_image_size = None


def load_artifacts():
    global _model, _class_names, _image_size
    if _model is None:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise FileNotFoundError(
                "Trained model is missing. Add your dataset and run python train_model.py."
            )
        metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        _class_names = metadata["class_names"]
        _image_size = int(metadata["image_size"])
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model, _class_names, _image_size


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or uploaded.filename == "":
            error = "Please choose an image to upload."
        elif not allowed_file(uploaded.filename):
            error = "Please upload a PNG, JPG or JPEG image."
        else:
            try:
                model, class_names, image_size = load_artifacts()
                safe_name = secure_filename(uploaded.filename)
                filename = f"{uuid.uuid4().hex}_{safe_name}"
                save_path = UPLOAD_DIR / filename
                image = Image.open(uploaded.stream).convert("RGB")
                image.save(save_path)
                resized = image.resize((image_size, image_size))
                array = np.asarray(resized, dtype=np.float32)
                probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
                top_indices = np.argsort(probabilities)[::-1][:3]
                predictions = [
                    {"name": class_names[int(i)].replace("_", " "),
                     "confidence": float(probabilities[i]) * 100}
                    for i in top_indices
                ]
                return render_template(
                    "result.html", image_url=url_for("static", filename=f"uploads/{filename}"),
                    predictions=predictions
                )
            except FileNotFoundError as exc:
                error = str(exc)
            except (UnidentifiedImageError, OSError):
                error = "The uploaded file could not be read as an image."
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render_template("index.html", error=error)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template("index.html", error="Image is too large. Maximum upload size is 8 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
