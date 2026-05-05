import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

# Load model + labels
model = tf.keras.models.load_model("plant_model.h5")

with open("labels.json") as f:
    labels = json.load(f)

# Load test image
img = image.load_img("test.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0]

predicted_class = int(np.argmax(prediction))
confidence = prediction[predicted_class]

print("🌿 Disease:", labels[str(predicted_class)])
print("📊 Confidence:", round(float(confidence) * 100, 2), "%")

# Top 3
top_indices = prediction.argsort()[-3:][::-1]

print("\nTop 3 Predictions:")
for i in top_indices:
    print(f"{labels[str(i)]}: {round(float(prediction[i]) * 100, 2)}%")