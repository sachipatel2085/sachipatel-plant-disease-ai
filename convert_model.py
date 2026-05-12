import tensorflow as tf

print("Loading old model...")

model = tf.keras.models.load_model(
    "plant_model.h5",
    compile=False
)

print("Saving converted model...")

model.save("converted_model.h5")

print("DONE")