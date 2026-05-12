import tensorflow as tf

# Load old model
model = tf.keras.models.load_model(
    "plant_model.h5",
    compile=False
)

# Save in new format
model.save("new_plant_model.h5")

print("Model converted successfully")