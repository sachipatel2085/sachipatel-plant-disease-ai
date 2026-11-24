"""
Transfer Learning Training Script for Crop Identification.
Trains lightweight MobileNetV3Small to discriminate between:
1. Potato (Solanum tuberosum)
2. Watermelon (Citrullus lanatus)
3. Unsupported / Out-of-Scope flora
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_pipeline.augmentations import AgriculturalAugmenter
from models.common import CROPS

def build_crop_identification_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.25
) -> tf.keras.Model:
    """Build lightweight crop classifier using MobileNetV3Small."""
    augmenter = AgriculturalAugmenter(image_size=(input_shape[0], input_shape[1]))
    aug_layer = augmenter.build_keras_augmentation_model()

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        minimalistic=False
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="crop_image_input")
    x = aug_layer(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="crop_pooling")(x)
    x = layers.Dropout(dropout_rate, name="crop_dropout")(x)
    outputs = layers.Dense(len(CROPS), activation="softmax", name="crop_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CropIdentifier_MobileNetV3")
    return model

def train_crop_identifier(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    output_model_path: Path,
    epochs: int = 8,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Train and checkpoint crop classifier."""
    model = build_crop_identification_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("[Training Crop Identifier] Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))
    print(f"[Training Crop Identifier] Saved model to: {output_model_path}")
    return model
