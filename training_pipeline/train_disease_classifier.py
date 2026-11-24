"""
Multi-Phase Transfer Learning Pipeline for Crop Disease & Pest Classification.
Features:
- Two-phase training: Phase 1 (Frozen Backbone Feature Extraction) + Phase 2 (Selective Fine-Tuning)
- Backbones: EfficientNetV2B0, MobileNetV3Small, ConvNeXt
- Automatic balanced class weighting for rare agricultural conditions
- Cosine Decay with Warmup learning rate schedule
- EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau
- Evaluation artifact logging (JSON history & loss curves)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_pipeline.augmentations import AgriculturalAugmenter
from dataset_pipeline.balancer_and_splitter import DatasetBalancerAndSplitter

def build_backbone(architecture: str, input_shape: Tuple[int, int, int] = (224, 224, 3)):
    """Instantiate pre-trained feature extraction backbone with ImageNet weights."""
    arch = architecture.lower()
    if "efficientnetv2" in arch:
        base = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    elif "mobilenetv3" in arch:
        base = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    elif "convnext" in arch:
        try:
            base = tf.keras.applications.ConvNeXtTiny(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
        except AttributeError:
            print("[Warning] ConvNeXt not available in this Keras version, falling back to EfficientNetV2B0.")
            base = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
    else:
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape
        )
    return base

def build_disease_classifier_model(
    num_classes: int,
    architecture: str = "EfficientNetV2B0",
    dropout_rate: float = 0.35,
    dense_units: int = 256,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build complete diagnostic classification model with data augmentation head.
    Returns: (full_model, base_backbone)
    """
    augmenter = AgriculturalAugmenter(image_size=(input_shape[0], input_shape[1]))
    aug_layer = augmenter.build_keras_augmentation_model()

    base_backbone = build_backbone(architecture, input_shape=input_shape)
    base_backbone.trainable = False  # Frozen for Phase 1

    inputs = layers.Input(shape=input_shape, name="plant_image_input")
    x = aug_layer(inputs)
    x = base_backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pooling")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    x = layers.Dense(dense_units, activation="relu", name="projection_dense")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="projection_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_probabilities")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"CropHealth_{architecture}")
    return model, base_backbone

def compute_class_weights_from_manifest(train_manifest_path: Path) -> Dict[int, float]:
    """Compute balanced class weights inversely proportional to frequency."""
    if not train_manifest_path.exists():
        return {}

    with open(train_manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    class_counts: Dict[int, int] = {}
    for r in records:
        lbl = r.get("class_idx", r.get("label"))
        if lbl is not None:
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

    total_samples = len(records)
    n_classes = len(class_counts)
    if n_classes == 0:
        return {}

    class_weights = {}
    for c_idx, count in class_counts.items():
        class_weights[c_idx] = float(total_samples / (n_classes * count))

    return class_weights

class DiseaseModelTrainer:
    """Orchestrates 2-phase transfer learning and checkpointing."""

    def __init__(
        self,
        num_classes: int,
        checkpoint_dir: Path,
        architecture: str = "EfficientNetV2B0",
        learning_rate_p1: float = 1e-3,
        learning_rate_p2: float = 5e-5
    ):
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.architecture = architecture
        self.lr_p1 = learning_rate_p1
        self.lr_p2 = learning_rate_p2

        self.model, self.base_backbone = build_disease_classifier_model(
            num_classes=num_classes,
            architecture=architecture
        )

    def train_phase_1(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 10,
        class_weights: Optional[Dict[int, float]] = None
    ) -> tf.keras.callbacks.History:
        """Phase 1: Feature Extraction with frozen backbone."""
        print(f"\n[Training] === Phase 1: Feature Extraction ({self.architecture}) ===")
        print(f"[Training] Base layers frozen: {len(self.base_backbone.layers)}")

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr_p1),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, self.num_classes), name="top_3_acc")
            ]
        )

        p1_checkpoint = self.checkpoint_dir / "phase1_best.weights.h5"
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            callbacks.ModelCheckpoint(filepath=str(p1_checkpoint), save_best_only=True, save_weights_only=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
        ]

        history_p1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=cb
        )
        return history_p1

    def train_phase_2(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        unfreeze_top_layers: int = 35,
        epochs: int = 15,
        class_weights: Optional[Dict[int, float]] = None
    ) -> tf.keras.callbacks.History:
        """Phase 2: Fine-Tuning top backbone layers with low learning rate."""
        print(f"\n[Training] === Phase 2: Fine-Tuning (Top {unfreeze_top_layers} layers) ===")

        self.base_backbone.trainable = True
        for layer in self.base_backbone.layers[:-unfreeze_top_layers]:
            layer.trainable = False

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr_p2),
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, self.num_classes), name="top_3_acc")
            ]
        )

        p2_checkpoint = self.checkpoint_dir / "phase2_best.weights.h5"
        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            callbacks.ModelCheckpoint(filepath=str(p2_checkpoint), save_best_only=True, save_weights_only=True, monitor="val_loss"),
            callbacks.CosineDecayRestarts(initial_learning_rate=self.lr_p2, first_decay_steps=5) if hasattr(callbacks, "CosineDecayRestarts") else callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
        ]

        history_p2 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=cb
        )
        return history_p2

    def save_final_model(self, output_path: Path):
        """Save fully trained diagnostic model."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_path))
        print(f"[Training] Final model saved successfully: {output_path}")
