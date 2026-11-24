"""
Common Data Structures, Constants, and Preprocessing Utilities for Multi-Model Architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

# Standard network input resolutions
CLASSIFIER_INPUT_SHAPE = (224, 224, 3)
SEGMENTATION_INPUT_SHAPE = (256, 256, 3)
DETECTION_INPUT_SHAPE = (640, 640, 3)

# Canonical Crop Labels
CROPS = ["potato", "watermelon", "unsupported"]

# Canonical Plant Organs
PLANT_PARTS = [
    "leaf",
    "stem",
    "tuber",
    "fruit",
    "root",
    "insect",
    "whole_plant"
]

# Agronomic Severity Tiers
SEVERITY_TIERS = {
    "HEALTHY": (0.0, 0.0),
    "MILD": (0.01, 10.0),
    "MODERATE": (10.01, 25.0),
    "SEVERE": (25.01, 50.0),
    "CRITICAL": (50.01, 100.0)
}

def get_severity_tier(percentage: float) -> str:
    """Classify affected leaf/fruit/tuber area percentage into agricultural severity tier."""
    if percentage <= 0.05:
        return "HEALTHY"
    elif percentage <= 10.0:
        return "MILD"
    elif percentage <= 25.0:
        return "MODERATE"
    elif percentage <= 50.0:
        return "SEVERE"
    else:
        return "CRITICAL"

def preprocess_for_classifier(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize image and convert to float32 batch tensor normalized to [0, 1]."""
    resized = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_for_segmentation(img: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Preprocess image for U-Net segmentation model."""
    resized = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@dataclass
class CropPrediction:
    crop: str
    confidence: float
    is_supported: bool
    probabilities: Dict[str, float] = field(default_factory=dict)

@dataclass
class OrganPrediction:
    organ: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)

@dataclass
class DiseaseCandidate:
    condition_id: str
    condition_name: str
    scientific_name: str
    category: str
    visual_detectability: str
    raw_probability: float
    calibrated_probability: float
    organ_compatible: bool

@dataclass
class PestDetection:
    pest_name: str
    scientific_name: str
    confidence: float
    bbox_normalized: List[float]  # [ymin, xmin, ymax, xmax]
    damage_type: str

@dataclass
class SeverityAssessment:
    lesion_pixels: int
    tissue_pixels: int
    severity_percentage: float
    severity_tier: str
    mask_available: bool

@dataclass
class UncertaintyEvaluation:
    max_confidence: float
    shannon_entropy: float
    margin_top1_top2: float
    is_rejected: bool
    is_differential: bool
    rejection_reason: Optional[str] = None
    differential_warning: Optional[str] = None
