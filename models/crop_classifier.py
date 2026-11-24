"""
Module A: Crop Identification Module.
Routes incoming plant imagery to Potato, Watermelon, or Unsupported / Non-Crop categories.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .common import (
    CLASSIFIER_INPUT_SHAPE,
    CROPS,
    CropPrediction,
    preprocess_for_classifier
)

class CropIdentifier:
    """Classifies agricultural photographs into supported crops (Potato, Watermelon) or flags as unsupported."""

    def __init__(self, weights_path: Optional[str] = None, backbone: str = "MobileNetV3Small"):
        self.backbone_name = backbone
        self.classes = CROPS
        self.input_shape = CLASSIFIER_INPUT_SHAPE
        self.has_trained_weights = False
        self.model = self._build_model(backbone)
        if weights_path:
            self.model.load_weights(weights_path)
            self.has_trained_weights = True

    def _build_model(self, backbone: str) -> keras.Model:
        """Construct the lightweight crop classification network."""
        inputs = keras.Input(shape=self.input_shape, name="image_input")

        if backbone == "MobileNetV3Small":
            base_model = keras.applications.MobileNetV3Small(
                input_shape=self.input_shape,
                include_top=False,
                weights=None  # Can be initialized from ImageNet when downloading
            )
        elif backbone == "EfficientNetV2B0":
            base_model = keras.applications.EfficientNetV2B0(
                input_shape=self.input_shape,
                include_top=False,
                weights=None
            )
        else:
            base_model = keras.applications.MobileNetV3Small(
                input_shape=self.input_shape,
                include_top=False,
                weights=None
            )

        x = base_model(inputs)
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="bn")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        x = layers.Dense(128, activation="relu", name="dense_embed")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        outputs = layers.Dense(len(self.classes), activation="softmax", name="crop_prob")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="CropIdentificationModel")
        return model

    def predict(self, img: Image.Image, confidence_threshold: float = 0.50) -> CropPrediction:
        """
        Classify crop identity from image.
        Returns CropPrediction with confidence, probabilities, and is_supported flag.
        """
        tensor = preprocess_for_classifier(img, (self.input_shape[0], self.input_shape[1]))
        probs = self.model(tensor, training=False).numpy()[0]

        prob_dict = {crop: float(probs[i]) for i, crop in enumerate(self.classes)}
        top_idx = int(np.argmax(probs))
        top_crop = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        # If confidence is below threshold or classified as unsupported
        is_supported = (top_crop != "unsupported") and (top_conf >= confidence_threshold)

        return CropPrediction(
            crop=top_crop if is_supported else "unsupported",
            confidence=top_conf,
            is_supported=is_supported,
            probabilities=prob_dict
        )
