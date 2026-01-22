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
print("🔵 Loading model architecture...")

# -------------------- LOAD MODEL SAFELY --------------------
with open(CONFIG_PATH, "r") as f:
    model_config = json.load(f)

model = tf.keras.Sequential.from_config(model_config["config"])
model.load_weights(WEIGHTS_PATH)

# 🔑 BUILD MODEL (VERY IMPORTANT)
model.build(input_shape=(None, 224, 224, 3))
model.predict(np.zeros((1, 224, 224, 3)))

print("✅ Model loaded & built successfully")
print(f"📋 Model has {len(model.layers)} layers")
for i, layer in enumerate(model.layers):
    print(f"  Layer {i}: {layer.name} - {type(layer).__name__}")

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------- API --------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(image_path)

        # 1️⃣ CLIP FILTER (ENFORCED)
        if not is_skin_lesion(image_path):
            print("🚫 CLIP filter rejected non-skin image")
            return jsonify({
                "error": "Uploaded image is not a skin lesion"
            }), 400
        else:
            print("✅ CLIP filter passed - proceeding to CNN prediction")
            clip_validation_msg = "Image validated as a skin lesion"
       
        img_array = preprocess_image(image_path)

        # Make prediction
        score = model.predict(img_array)[0][0]
        label = "The given image found as Melanoma" if score > 0.5 else "The given image found as Benign"

        print(f"✅ PREDICTION → {label} ({score:.4f})")

        # Generate Grad-CAM heatmap
        gradcam_base64 = None
        try:
            # Find the last convolutional layer by index
            last_conv_layer = None
            for i in range(len(model.layers) - 1, -1, -1):
                layer = model.layers[i]
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = i
                    break
            
            if last_conv_layer is not None:
                layer_name = model.layers[last_conv_layer].name
                print(f"🔥 Generating Grad-CAM using layer: {layer_name} (index {last_conv_layer})")
                heatmap = compute_gradcam(img_array, model, layer_name)
                gradcam_overlay = overlay_gradcam(image_path, heatmap)
                
                # Encode Grad-CAM image to base64
                _, buffer = cv2.imencode('.png', gradcam_overlay)
                gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
                
                print("✅ Grad-CAM generated successfully")
            else:
                print("⚠️ No Conv2D layer found for Grad-CAM")
                gradcam_base64 = None
        except Exception as grad_err:
            print(f"⚠️ Grad-CAM generation failed: {str(grad_err)}")
            import traceback
            traceback.print_exc()
            gradcam_base64 = None

        # 🔹 STEP 3: MELANOMA STAGE (ONLY IF MELANOMA)
        response = {
            "label": label,
            "confidence": float(score),
            "gradcam_image": gradcam_base64,
            "clip_validation": clip_validation_msg
        }

        if "Melanoma" in label:
            try:
                stage = estimate_melanoma_stage(image_path)
                response["stage"] = stage
                print(f"🎭 CLIP-estimated melanoma stage: {stage}")
            except Exception as stage_err:
                print(f"⚠️ Stage estimation failed: {stage_err}")

        # Clean up uploaded file
        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify(response)

    except Exception as e:
        print("❌ ML SERVICE ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ML Service is running", "model_loaded": True})

# 🔑 ENSURE MODEL HAS SYMBOLIC INPUT
model.predict(np.zeros((1, 224, 224, 3)))

# -------------------- RUN --------------------
if __name__ == "__main__":
    print("🚀 ML Service running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)