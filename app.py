from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

# Load labels
with open("labels.json") as f:
    labels = json.load(f)


def load_model():
    global model

    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(
            "plant_model.h5",
            compile=False
        )
        print("Model loaded successfully")


@app.route("/")
def home():
    return "Plant Disease AI API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load model only when needed
        load_model()

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        img = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]

        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])

        disease_name = labels[str(predicted_class)]

        top_indices = prediction.argsort()[-3:][::-1]

        top_predictions = [
            {
                "disease": labels[str(i)],
                "confidence": round(
                    float(prediction[i]) * 100,
                    2
                )
            }
            for i in top_indices
        ]

        return jsonify({
            "class": predicted_class,
            "disease": disease_name,
            "confidence": round(confidence * 100, 2),
            "top_predictions": top_predictions
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )