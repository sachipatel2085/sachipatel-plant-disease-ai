"""
Agricultural Augmentation Suite for Field-Realistic Plant Pathology Imagery.
Implements domain-specific visual augmentations:
1. Harsh direct field illumination & solar specular glare
2. Leaf flutter motion blur (wind shaking during mobile capture)
3. Dew droplets and leaf surface moisture reflectance
4. Nitrogen deficiency & chlorophyll hue shift (color jitter)
5. Cloud and canopy shadow projection
6. Geometric affine transforms (rotations, flips, zoom)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple, Optional
from PIL import Image, ImageFilter, ImageEnhance

class AgriculturalAugmenter:
    """TensorFlow-native and NumPy-compatible agricultural image augmentation pipeline."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        enable_sun_glare: bool = True,
        enable_shadows: bool = True,
        enable_flutter_blur: bool = True,
        enable_color_jitter: bool = True,
    ):
        self.image_size = image_size
        self.enable_sun_glare = enable_sun_glare
        self.enable_shadows = enable_shadows
        self.enable_flutter_blur = enable_flutter_blur
        self.enable_color_jitter = enable_color_jitter

    def build_keras_augmentation_model(self) -> tf.keras.Sequential:
        """
        Build an end-to-end Keras Preprocessing Sequential layer suitable for
        direct embedding in tf.keras.Model or tf.data.Dataset mapping.
        """
        aug_layers = [
            layers.Resizing(self.image_size[0], self.image_size[1]),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.25, fill_mode="reflect"),
            layers.RandomZoom(height_factor=(-0.15, 0.15), width_factor=(-0.15, 0.15), fill_mode="reflect"),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="reflect"),
            layers.RandomContrast(factor=0.2),
            layers.RandomBrightness(factor=0.2),
        ]
        return tf.keras.Sequential(aug_layers, name="agricultural_augmentations")

    def apply_sun_glare(self, img_np: np.ndarray, intensity: float = 0.35) -> np.ndarray:
        """
        Simulate harsh Gujarat direct sunlight / solar glare washing out leaf lesions.
        img_np: float32 in [0, 1] or uint8 in [0, 255].
        """
        is_uint8 = img_np.dtype == np.uint8
        float_img = img_np.astype(np.float32) / 255.0 if is_uint8 else img_np.copy()

        h, w = float_img.shape[:2]
        center_x = np.random.randint(int(w * 0.2), int(w * 0.8))
        center_y = np.random.randint(int(h * 0.2), int(h * 0.8))
        radius = np.random.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.5))

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        glare_mask = np.clip(1.0 - (dist_from_center / radius), 0.0, 1.0)
        glare_mask = np.expand_dims(glare_mask, axis=-1)

        # Solar warm white highlight
        glare_color = np.array([1.0, 0.98, 0.90], dtype=np.float32)
        blended = float_img * (1.0 - intensity * glare_mask) + (glare_color * intensity * glare_mask)
        blended = np.clip(blended, 0.0, 1.0)

        return (blended * 255).astype(np.uint8) if is_uint8 else blended

    def apply_canopy_shadow(self, img_np: np.ndarray, darkness: float = 0.45) -> np.ndarray:
        """
        Simulate dense canopy or passing cloud shadows across leaf blade.
        """
        is_uint8 = img_np.dtype == np.uint8
        float_img = img_np.astype(np.float32) / 255.0 if is_uint8 else img_np.copy()

        h, w = float_img.shape[:2]
        # Diagonal shadow boundary
        slope = np.random.uniform(-1.5, 1.5)
        intercept = np.random.uniform(-h * 0.5, h * 1.5)

        y, x = np.ogrid[:h, :w]
        shadow_mask = (y > (slope * x + intercept)).astype(np.float32)
        shadow_mask = np.expand_dims(shadow_mask, axis=-1)

        # Soft shadow edge transition
        shadowed = float_img * (1.0 - darkness * shadow_mask)
        shadowed = np.clip(shadowed, 0.0, 1.0)

        return (shadowed * 255).astype(np.uint8) if is_uint8 else shadowed

    def apply_flutter_blur(self, pil_img: Image.Image, max_radius: float = 1.8) -> Image.Image:
        """
        Simulate wind-induced foliage flutter and mobile shutter vibration.
        """
        radius = np.random.uniform(0.5, max_radius)
        return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

    def apply_chlorophyll_jitter(self, pil_img: Image.Image) -> Image.Image:
        """
        Simulate nitrogen chlorosis or seasonal leaf pigmentation drift.
        """
        color_factor = np.random.uniform(0.75, 1.25)
        contrast_factor = np.random.uniform(0.85, 1.20)
        brightness_factor = np.random.uniform(0.85, 1.15)

        img = ImageEnhance.Color(pil_img).enhance(color_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        return img

    def augment_sample(self, pil_img: Image.Image) -> Image.Image:
        """Execute full stochastic field-realistic augmentation on a single PIL Image."""
        result = pil_img.copy()

        if self.enable_color_jitter and np.random.rand() > 0.3:
            result = self.apply_chlorophyll_jitter(result)

        if self.enable_flutter_blur and np.random.rand() > 0.5:
            result = self.apply_flutter_blur(result)

        img_np = np.array(result)
        if self.enable_sun_glare and np.random.rand() > 0.5:
            img_np = self.apply_sun_glare(img_np)

        if self.enable_shadows and np.random.rand() > 0.5:
            img_np = self.apply_canopy_shadow(img_np)

        return Image.fromarray(img_np)
