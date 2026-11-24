"""
Module C: Disease & Pathogen Classification Module.
High-capacity deep neural network covering all 69 conditions from the Phase 1 taxonomy,
with hierarchical crop and organ compatibility conditioning.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .common import (
    CLASSIFIER_INPUT_SHAPE,
    DiseaseCandidate,
    preprocess_for_classifier
)
from .organ_classifier import OrganIdentifier
from dataset_pipeline.taxonomy_mapper import TaxonomyMapper

class DiseaseClassifier:
    """Classifies plant pathology, pest damage, nutritional deficiencies, and physiological disorders."""

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        weights_path: Optional[str] = None,
        backbone: str = "EfficientNetV2B0",
        temperature: float = 1.2
    ):
        self.backbone_name = backbone
        self.temperature = temperature
        self.mapper = TaxonomyMapper(taxonomy_path)
        self.condition_ids = sorted(list(self.mapper.conditions_by_id.keys()))
        self.num_classes = len(self.condition_ids)
        self.input_shape = CLASSIFIER_INPUT_SHAPE
        self.has_trained_weights = False
        self.model = self._build_model(backbone)
        if weights_path:
            self.model.load_weights(weights_path)
            self.has_trained_weights = True

    def _build_model(self, backbone: str) -> keras.Model:
        """Construct the disease classification network."""
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
        x = layers.Dropout(0.4, name="dropout_1")(x)
        x = layers.Dense(256, activation="relu", name="dense_features")(x)
        x = layers.Dropout(0.2, name="dropout_2")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="disease_softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="DiseasePathologyClassifier")
        return model

    def predict(
        self,
        img: Image.Image,
        crop_filter: Optional[str] = None,
        organ_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[DiseaseCandidate]:
        """
        Generate differential disease diagnosis candidates.
        Applies temperature calibration, crop filtering, and organ compatibility checks.
        """
        tensor = preprocess_for_classifier(img, (self.input_shape[0], self.input_shape[1]))
        raw_probs = self.model(tensor, training=False).numpy()[0]

        # Apply temperature calibration: p_calibrated = softmax(log(p) / T)
        # Avoid log(0) with epsilon
        eps = 1e-7
        log_probs = np.log(np.clip(raw_probs, eps, 1.0))
        calibrated_logits = log_probs / max(0.1, self.temperature)
        exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        candidates: List[DiseaseCandidate] = []

        for idx, cond_id in enumerate(self.condition_ids):
            cond_record = self.mapper.get_condition_by_id(cond_id)
            if not cond_record:
                continue

            cond_crop = cond_record["Crop"].lower()
            # If crop filter is provided and does not match, penalize/mask
            if crop_filter and crop_filter.lower() in ("potato", "watermelon"):
                if cond_crop != crop_filter.lower():
                    continue

            # Check organ compatibility
            affected_parts = cond_record["Affected_Plant_Part"]
            cond_name = cond_record["Disease_or_Pest"]

            is_compatible = True
            if organ_filter:
                is_compatible, _ = OrganIdentifier.validate_organ_compatibility(
                    detected_organ=organ_filter,
                    affected_parts_str=affected_parts,
                    condition_name=cond_name
                )

            candidates.append(DiseaseCandidate(
                condition_id=cond_id,
                condition_name=cond_name,
                scientific_name=cond_record["Scientific_Name"],
                category=cond_record.get("Category", "Disease"),
                visual_detectability=cond_record.get("Visual_Detectability", "A"),
                raw_probability=float(raw_probs[idx]),
                calibrated_probability=float(calibrated_probs[idx]),
                organ_compatible=is_compatible
            ))

        # Sort by calibrated probability descending
        candidates.sort(key=lambda c: c.calibrated_probability, reverse=True)
        return candidates[:top_k]
