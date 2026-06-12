import tensorflow as tf

print("Loading old model...")

model = tf.keras.models.load_model(
    "plant_model.h5",
    compile=False
)

print("Saving SavedModel format...")

model.save("saved_model")

print("Reloading SavedModel...")

new_model = tf.keras.models.load_model("saved_model")

print("Saving final H5...")

new_model.save("fixed_plant_model.h5")

print("DONE")