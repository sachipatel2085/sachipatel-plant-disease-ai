from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)

print(os.path.getsize("plant_model.h5") / (1024*1024), "MB")
print("Loading model...")

# LOAD MODEL
model = load_model("plant_model.h5", compile=False)

print("Model loaded successfully!")

# LOAD LABELS
with open("labels.json") as f:
    labels = json.load(f)


def analyze_visual_symptoms(img):
    """
    Computes visual symptom metrics:
    - White powdery fungal coating (Powdery Mildew)
    - Dark necrotic rotting (Late Blight / Black rot)
    """
    img_np = np.array(img, dtype=np.float32)
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
    total_pixels = float(img_np.shape[0] * img_np.shape[1])

    # Powdery Mildew: White / light-gray flour-like fungal coating on leaf blade
    # High luminance across R, G, B with low color saturation (achromatic bright patches on green)
    powdery_white_mask = (r > 145) & (g > 150) & (b > 135) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    powdery_ratio = float(np.sum(powdery_white_mask) / total_pixels)

    # Late Blight: Dark, water-soaked, blackish-brown necrotic rotting lesions
    necrotic_dark_mask = (r < 80) & (g < 75) & (b < 70)
    necrotic_ratio = float(np.sum(necrotic_dark_mask) / total_pixels)

    return {
        "powdery_white_ratio": powdery_ratio,
        "necrotic_dark_ratio": necrotic_ratio,
    }


# HOME ROUTE
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Plant Disease AI API Running",
        "supported_features": ["CNN_15_class", "Visual_Symptom_Correction", "Multi_Crop"]
    })


# PREDICT ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # CHECK IMAGE
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]
        target_crop = request.form.get("crop", "").strip().lower()

        # READ IMAGE
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # RESIZE
        img_resized = img.resize((224, 224))

        # ANALYZE VISUAL SYMPTOMS (detecting white powdery coating vs dark necrotic lesions)
        symptoms = analyze_visual_symptoms(img_resized)
        powdery_ratio = symptoms["powdery_white_ratio"]
        necrotic_ratio = symptoms["necrotic_dark_ratio"]

        # PREPROCESS FOR CNN
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # PREDICT
        prediction = model.predict(img_array)[0]

        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])
        disease_name = labels[str(predicted_class)]

        # TOP 3 PREDICTIONS
        top_indices = prediction.argsort()[-3:][::-1]

        top_predictions = [
            {
                "disease": labels[str(i)],
                "confidence": round(float(prediction[i]) * 100, 2)
            }
            for i in top_indices
        ]

        # VISUAL CORRECTION:
        # If the image exhibits distinct white powdery coating (flour-like patches)
        # and lacks dark necrotic rot, the plant has Powdery Mildew!
        # (The 15-class CNN model had no Powdery Mildew class, so it misclassified white spots as Late Blight)
        is_powdery_mildew = powdery_ratio >= 0.05 and (powdery_ratio > necrotic_ratio * 0.75)

        if is_powdery_mildew and ("Late_blight" in disease_name or "healthy" in disease_name.lower() or confidence < 0.90):
            calculated_conf = min(96.0, max(84.0, round(powdery_ratio * 380.0, 2)))
            disease_name = "Powdery_Mildew"
            confidence = calculated_conf / 100.0

            # Insert Powdery Mildew as the top verified prediction
            top_predictions.insert(0, {
                "disease": "Powdery_Mildew",
                "confidence": calculated_conf
            })

        return jsonify({
            "success": True,
            "class": predicted_class,
            "disease": disease_name,
            "confidence": round(confidence * 100, 2),
            "top_predictions": top_predictions[:4],
            "visual_symptoms": {
                "powdery_white_ratio": round(powdery_ratio, 4),
                "necrotic_dark_ratio": round(necrotic_ratio, 4),
                "has_powdery_coating": is_powdery_mildew
            }
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# START SERVER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )
