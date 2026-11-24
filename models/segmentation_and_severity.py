"""
Modules E & F: Disease Lesion Segmentation & Agronomic Severity Estimation Module.
Implements U-Net encoder-decoder architecture with skip connections to segment pathological lesions
and calculate affected tissue surface area percentage.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .common import (
    SEGMENTATION_INPUT_SHAPE,
    SeverityAssessment,
    get_severity_tier,
    preprocess_for_segmentation
)

class LesionSegmentationUNet:
    """U-Net architecture with skip connections for pixel-level plant disease lesion segmentation."""

    def __init__(self, weights_path: Optional[str] = None):
        self.input_shape = SEGMENTATION_INPUT_SHAPE
        self.has_trained_weights = False
        self.model = self._build_unet()
        if weights_path:
            self.model.load_weights(weights_path)
            self.has_trained_weights = True

    def _build_unet(self) -> keras.Model:
        """Construct standard U-Net with contracting and expanding paths."""
        inputs = keras.Input(shape=self.input_shape, name="segmentation_input")

        # --- ENCODER (Contracting Path) ---
        # Block 1
        c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_c1_1")(inputs)
        c1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="enc_c1_2")(c1)
        p1 = layers.MaxPooling2D((2, 2), name="enc_p1")(c1)

        # Block 2
        c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_c2_1")(p1)
        c2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="enc_c2_2")(c2)
        p2 = layers.MaxPooling2D((2, 2), name="enc_p2")(c2)

        # Block 3
        c3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_c3_1")(p2)
        c3 = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="enc_c3_2")(c3)
        p3 = layers.MaxPooling2D((2, 2), name="enc_p3")(c3)

        # --- BOTTLENECK ---
        b = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="bottleneck_1")(p3)
        b = layers.Dropout(0.3, name="bottleneck_drop")(b)
        b = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="bottleneck_2")(b)

        # --- DECODER (Expanding Path with Skip Connections) ---
        # UpBlock 3
        u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="dec_u3")(b)
        u3 = layers.concatenate([u3, c3], name="dec_skip3")
        c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_c4_1")(u3)
        c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="dec_c4_2")(c4)

        # UpBlock 2
        u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="dec_u2")(c4)
        u2 = layers.concatenate([u2, c2], name="dec_skip2")
        c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_c5_1")(u2)
        c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="dec_c5_2")(c5)

        # UpBlock 1
        u1 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same", name="dec_u1")(c5)
        u1 = layers.concatenate([u1, c1], name="dec_skip1")
        c6 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_c6_1")(u1)
        c6 = layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="dec_c6_2")(c6)

        # Binary Lesion Mask Output (1 channel sigmoid)
        outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", name="lesion_mask_out")(c6)

        model = keras.Model(inputs=inputs, outputs=outputs, name="UNetLesionSegmentation")
        return model

    def estimate_severity(
        self,
        img: Image.Image,
        threshold: float = 0.50
    ) -> Tuple[SeverityAssessment, np.ndarray]:
        """
        Segment pathological lesions and calculate affected surface area percentage.
        Returns: (SeverityAssessment, binary_mask_256x256)
        """
        tensor = preprocess_for_segmentation(img, (self.input_shape[0], self.input_shape[1]))
        pred_mask = self.model(tensor, training=False).numpy()[0, :, :, 0]

        # Extract plant tissue mask (exclude non-plant background)
        orig_resized = np.array(img.convert("RGB").resize((self.input_shape[0], self.input_shape[1])))
        hsv = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2HSV)

        # Mask non-background pixels (plant tissue: green, yellow, brown, reddish)
        # Exclude pure black (V < 25) or pure white background (S < 15, V > 240)
        is_not_black = hsv[:, :, 2] >= 25
        is_not_white = ~((hsv[:, :, 1] < 15) & (hsv[:, :, 2] > 240))
        tissue_mask = is_not_black & is_not_white
        total_tissue_pixels = int(np.sum(tissue_mask))

        # Binary lesion mask constrained to plant tissue
        binary_lesions = (pred_mask >= threshold) & tissue_mask
        lesion_pixels = int(np.sum(binary_lesions))

        if total_tissue_pixels > 0:
            severity_pct = (lesion_pixels / float(total_tissue_pixels)) * 100.0
        else:
            severity_pct = 0.0

        severity_tier = get_severity_tier(severity_pct)

        assessment = SeverityAssessment(
            lesion_pixels=lesion_pixels,
            tissue_pixels=total_tissue_pixels,
            severity_percentage=round(severity_pct, 2),
            severity_tier=severity_tier,
            mask_available=True
        )

        return assessment, binary_lesions.astype(np.uint8) * 255
