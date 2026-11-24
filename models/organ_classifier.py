"""
Module B: Plant-Part Identification Module.
Identifies whether the image contains a leaf, stem, tuber, fruit, root, insect, or whole plant.
Enforces biological organ compatibility rules before diagnosing conditions.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .common import (
    CLASSIFIER_INPUT_SHAPE,
    PLANT_PARTS,
    OrganPrediction,
    preprocess_for_classifier
)

class OrganIdentifier:
    """Classifies plant organ anatomy and validates diagnostic organ compatibility."""

    def __init__(self, weights_path: Optional[str] = None, backbone: str = "MobileNetV3Small"):
        self.backbone_name = backbone
        self.classes = PLANT_PARTS
        self.input_shape = CLASSIFIER_INPUT_SHAPE
        self.has_trained_weights = False
        self.model = self._build_model(backbone)
        if weights_path:
            self.model.load_weights(weights_path)
            self.has_trained_weights = True

    def _build_model(self, backbone: str) -> keras.Model:
        """Construct plant-part identification classifier."""
        inputs = keras.Input(shape=self.input_shape, name="image_input")

        if backbone == "EfficientNetV2B0":
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
        x = layers.Dense(128, activation="relu", name="dense_features")(x)
        outputs = layers.Dense(len(self.classes), activation="softmax", name="organ_prob")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="OrganIdentificationModel")
        return model

    def predict(self, img: Image.Image) -> OrganPrediction:
        """Classify plant organ anatomy from image."""
        tensor = preprocess_for_classifier(img, (self.input_shape[0], self.input_shape[1]))
        probs = self.model(tensor, training=False).numpy()[0]

        prob_dict = {organ: float(probs[i]) for i, organ in enumerate(self.classes)}
        top_idx = int(np.argmax(probs))
        top_organ = self.classes[top_idx]
        top_conf = float(probs[top_idx])

        return OrganPrediction(
            organ=top_organ,
            confidence=top_conf,
            probabilities=prob_dict
        )

    @staticmethod
    def validate_organ_compatibility(
        detected_organ: str,
        affected_parts_str: str,
        condition_name: str
    ) -> Tuple[bool, str]:
        """
        Guardrail: Verifies that the detected organ is biologically compatible with the candidate condition.
        Prevents diagnosing tuber scab from foliar images or subterranean nematodes without root imagery.
        """
        allowed_parts = [p.strip().lower() for p in affected_parts_str.split(",")]
        norm_detected = detected_organ.strip().lower()

        # Direct match or whole plant view compatibility
        if norm_detected in allowed_parts or norm_detected == "whole_plant" or "whole plant" in allowed_parts:
            return True, f"Organ '{detected_organ}' is biologically compatible with {condition_name}."

        # Stem base vs stem mapping
        if norm_detected == "stem" and any("stem" in p for p in allowed_parts):
            return True, f"Organ '{detected_organ}' is biologically compatible with {condition_name}."

        # Incompatible mismatch
        return False, (
            f"Anatomical Mismatch: '{condition_name}' manifests primarily on [{', '.join(allowed_parts)}], "
            f"but the uploaded photograph was identified as a '{detected_organ}'. "
            f"Diagnostic rules prohibit confirming this condition without direct evidence on affected plant parts."
        )
