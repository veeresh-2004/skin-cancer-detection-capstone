import os
import json
import base64
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from clip_utils.clip_filter import is_skin_lesion
from clip_utils.clip_stage import estimate_melanoma_stage
from flask_cors import CORS
from gradcam.gradcam_utils import compute_gradcam, overlay_gradcam
import threading
import requests
import tarfile
import shutil
import uuid

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# -------------------- APP SETUP --------------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models", "skin_cancer_cnn.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "model.weights.h5")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("🔵 Starting ML service...")

# Optional: download model archive at startup if MODEL_URL is provided and model missing
def download_and_extract_model(url):
    try:
        print(f"🔵 Downloading model from: {url}")
        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
        tmp_path = os.path.join(BASE_DIR, "models", "model_download.tmp")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)

        # Try to extract as tar.gz
        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=os.path.join(BASE_DIR, "models"))
            print("✅ Model archive extracted to models/")
            os.remove(tmp_path)
            return True
        except tarfile.ReadError:
            # Not a tarball — leave file as-is (user may have provided single files)
            print("⚠️ Downloaded file is not a tar.gz archive; please ensure model files are placed under models/skin_cancer_cnn.keras/")
            return False
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return False

# Model globals (will be populated by background loader)
model = None
model_ready = False
model_load_error = None

def load_model_background():
    global model, model_ready, model_load_error
    try:
        print("🔵 Loading model architecture in background thread...")
        with open(CONFIG_PATH, "r") as f:
            model_config = json.load(f)

        m = tf.keras.Sequential.from_config(model_config["config"])
        m.load_weights(WEIGHTS_PATH)

        # Build and warm-up the model
        m.build(input_shape=(None, 224, 224, 3))
        m.predict(np.zeros((1, 224, 224, 3)))

        model = m
        model_ready = True
        print("✅ Model loaded & built successfully (background)")
        print(f"📋 Model has {len(model.layers)} layers")
    except Exception as err:
        model_load_error = str(err)
        model_ready = False
        print("❌ Failed to load model in background:", model_load_error)

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def _find_last_conv_layer_name():
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return model.layers[i].name
    return None

def _analyze_single_image(image_path, view_label=None):
    if not is_skin_lesion(image_path):
        raise ValueError("Uploaded image is not a skin lesion")

    img_array = preprocess_image(image_path)
    score = float(model.predict(img_array)[0][0])
    probs = {"melanoma": score, "benign": 1.0 - score}

    if score > 0.5:
        label = "The given image found as Melanoma"
        display_confidence = probs["melanoma"]
    else:
        label = "The given image found as Benign"
        display_confidence = probs["benign"]

    gradcam_base64 = None
    try:
        layer_name = _find_last_conv_layer_name()
        if layer_name is not None:
            heatmap = compute_gradcam(img_array, model, layer_name)
            gradcam_overlay = overlay_gradcam(image_path, heatmap)
            _, buffer = cv2.imencode('.png', gradcam_overlay)
            gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as grad_err:
        print(f"⚠️ Grad-CAM generation failed: {str(grad_err)}")

    return {
        "view_label": view_label,
        "label": label,
        "confidence": float(display_confidence),
        "confidence_percent": round(display_confidence * 100, 2),
        "probabilities": probs,
        "gradcam_image": gradcam_base64,
        "clip_validation": "Image validated as a skin lesion",
    }

# -------------------- API --------------------
@app.route("/predict", methods=["POST"])
def predict():
    image_path = None
    try:
        if not model_ready:
            if model_load_error:
                return jsonify({"error": "Model failed to load", "details": model_load_error}), 500
            return jsonify({"error": "Model is still loading, try again shortly"}), 503
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
        image_path = os.path.join(UPLOAD_DIR, safe_name)
        file.save(image_path)

        try:
            result = _analyze_single_image(image_path)
        except ValueError as ve:
            print("🚫 CLIP filter rejected non-skin image")
            return jsonify({"error": str(ve)}), 400

        response = dict(result)
        score = result["probabilities"]["melanoma"]
        print(f"✅ PREDICTION → {result['label']} (melanoma_prob={score:.4f})")

        if "Melanoma" in result["label"]:
            try:
                stage = estimate_melanoma_stage(image_path)
                response["stage"] = stage
                print(f"🎭 CLIP-estimated melanoma stage: {stage}")
            except Exception as stage_err:
                print(f"⚠️ Stage estimation failed: {stage_err}")

        return jsonify(response)

    except Exception as e:
        print("❌ ML SERVICE ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

@app.route("/predict-multiview", methods=["POST"])
def predict_multiview():
    saved_paths = []
    try:
        if not model_ready:
            if model_load_error:
                return jsonify({"error": "Model failed to load", "details": model_load_error}), 500
            return jsonify({"error": "Model is still loading, try again shortly"}), 503

        files = request.files.getlist("images")
        if not files and "image" in request.files:
            files = [request.files["image"]]
        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        view_labels = []
        raw_labels = request.form.get("view_labels")
        if raw_labels:
            try:
                parsed = json.loads(raw_labels)
                if isinstance(parsed, list):
                    view_labels = [str(v) for v in parsed]
            except Exception:
                view_labels = []

        per_view = []
        stage_candidate = None
        for idx, file in enumerate(files):
            safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
            image_path = os.path.join(UPLOAD_DIR, safe_name)
            file.save(image_path)
            saved_paths.append(image_path)

            view_label = view_labels[idx] if idx < len(view_labels) else f"Angle {idx + 1}"
            try:
                result = _analyze_single_image(image_path, view_label=view_label)
            except ValueError as ve:
                return jsonify({"error": f"{view_label}: {str(ve)}"}), 400

            melanoma_prob = result["probabilities"]["melanoma"]
            certainty_weight = max(melanoma_prob, 1.0 - melanoma_prob)

            per_view.append({
                "index": idx + 1,
                "weight": float(certainty_weight),
                **result,
            })

            if stage_candidate is None or melanoma_prob > stage_candidate["melanoma_prob"]:
                stage_candidate = {"melanoma_prob": melanoma_prob, "image_path": image_path}

        weights = [item["weight"] for item in per_view]
        scores = [item["probabilities"]["melanoma"] for item in per_view]
        weight_sum = float(sum(weights))
        if weight_sum > 0:
            final_melanoma_score = float(sum(s * w for s, w in zip(scores, weights)) / weight_sum)
        else:
            final_melanoma_score = float(np.mean(scores))

        final_probs = {
            "melanoma": final_melanoma_score,
            "benign": 1.0 - final_melanoma_score,
        }

        if final_melanoma_score > 0.5:
            label = "The given lesion found as Melanoma (Multi-view)"
            display_confidence = final_probs["melanoma"]
        else:
            label = "The given lesion found as Benign (Multi-view)"
            display_confidence = final_probs["benign"]

        response = {
            "label": label,
            "confidence": float(display_confidence),
            "confidence_percent": round(display_confidence * 100, 2),
            "probabilities": final_probs,
            "clip_validation": "All uploaded views validated as skin lesions",
            "views": per_view,
            "view_count": len(per_view),
            "aggregation": {
                "method": "weighted_average_by_confidence",
                "weights": weights,
                "raw_melanoma_scores": scores,
            },
            "ai_explanation": "Final diagnosis is computed by aggregating multiple lesion views with confidence-based weighting.",
            "gradcam_image": per_view[0]["gradcam_image"] if per_view else None,
        }

        if per_view and len(per_view) > 1:
            response["gradcam_images"] = [item["gradcam_image"] for item in per_view]

        if "Melanoma" in label and stage_candidate is not None:
            try:
                response["stage"] = estimate_melanoma_stage(stage_candidate["image_path"])
            except Exception as stage_err:
                print(f"⚠️ Stage estimation failed: {stage_err}")

        return jsonify(response)

    except Exception as e:
        print("❌ ML SERVICE MULTI-VIEW ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        for image_path in saved_paths:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}

# Note: model warm-up is performed in the background loader.

# -------------------- RUN --------------------
if __name__ == "__main__":
    # If model weights are missing but MODEL_URL is provided, attempt to download and extract them.
    model_url = os.environ.get("MODEL_URL")
    if not os.path.exists(WEIGHTS_PATH) and model_url:
        ok = download_and_extract_model(model_url)
        if ok:
            print("🔁 Retrying model load after download...")

    # Start background thread to load the ML model while the server binds immediately.
    loader = threading.Thread(target=load_model_background, daemon=True)
    loader.start()

    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)