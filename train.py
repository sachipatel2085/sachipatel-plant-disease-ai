import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import json

# Step 1: Load dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=32,
    subset="validation"
)

# Step 2: Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False

# Step 3: Add custom layers
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Step 4: Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train
model.fit(train_data, validation_data=val_data, epochs=5)

# Step 6: Save model
model.save("plant_model.h5")

# ✅ Save labels mapping
labels = {v: k for k, v in train_data.class_indices.items()}

with open("labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Training complete + labels saved")