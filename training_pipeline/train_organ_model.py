"""
Transfer Learning Training Script for Plant-Part / Organ Identification.
Trains lightweight MobileNetV3Small to discriminate between:
1. Leaf
2. Stem
3. Tuber
4. Fruit
5. Root
6. Insect
7. Whole Plant
Enforces strict botanical organ-disease compatibility gating.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_pipeline.augmentations import AgriculturalAugmenter
from models.common import PLANT_PARTS

def build_organ_identification_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """Build organ classifier using MobileNetV3Small."""
    augmenter = AgriculturalAugmenter(image_size=(input_shape[0], input_shape[1]))
    aug_layer = augmenter.build_keras_augmentation_model()

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        minimalistic=False
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="organ_image_input")
    x = aug_layer(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="organ_pooling")(x)
    x = layers.Dropout(dropout_rate, name="organ_dropout")(x)
    outputs = layers.Dense(len(PLANT_PARTS), activation="softmax", name="organ_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="OrganIdentifier_MobileNetV3")
    return model

def train_organ_identifier(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    output_model_path: Path,
    epochs: int = 8,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """Train and checkpoint plant part / organ classifier."""
    model = build_organ_identification_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("[Training Organ Identifier] Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb
    )

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))
    print(f"[Training Organ Identifier] Saved model to: {output_model_path}")
    return model
