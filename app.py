from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)
print("TensorFlow Version:", tf.__version__)
print("Loading model...")

# LOAD MODEL
model = load_model("plant_model.h5", compile=False)

print("Model loaded successfully!")

# LOAD LABELS
with open("labels.json") as f:
    labels = json.load(f)


# HOME ROUTE
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Plant Disease AI API Running"
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

        # READ IMAGE
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # RESIZE
        img = img.resize((224, 224))

        # PREPROCESS
        img_array = np.array(img) / 255.0
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

        # RETURN ONLY ML DATA
        # OPENAI RESPONSE WILL BE GENERATED IN NODE BACKEND
        return jsonify({
            "success": True,
            "class": predicted_class,
            "disease": disease_name,
            "confidence": round(confidence * 100, 2),
            "top_predictions": top_predictions
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