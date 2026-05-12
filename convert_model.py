import tensorflow as tf

print("Loading old model...")

model = tf.keras.models.load_model(
    "old_model.h5",
    compile=False
)

print("Saving new model...")

model.save("plant_model.h5")

print("DONE")