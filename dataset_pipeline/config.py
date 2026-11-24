"""
Central Configuration for Crop Health Dataset Engineering Pipeline.
Defines storage hierarchies, image validation constraints, perceptual hash deduplication parameters,
split ratios, and taxonomy mappings.
"""

import os
from pathlib import Path

# Base directories
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PIPELINE_DIR.parent
TAXONOMY_JSON_PATH = PROJECT_DIR / "taxonomy" / "crop_health_taxonomy.json"
TAXONOMY_CSV_PATH = PROJECT_DIR / "taxonomy" / "crop_health_taxonomy.csv"

# Dataset root directories
DATA_ROOT = PROJECT_DIR / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
SPLITS_DIR = DATA_ROOT / "splits"
MANIFESTS_DIR = DATA_ROOT / "manifests"
REJECTS_DIR = DATA_ROOT / "rejects"

# Image constraints
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_COLORSPACE = "RGB"
MIN_IMAGE_WIDTH = 128
MIN_IMAGE_HEIGHT = 128
MAX_IMAGE_WIDTH = 4096
MAX_IMAGE_HEIGHT = 4096
MIN_FILE_SIZE_BYTES = 1024  # Reject files smaller than 1 KB (empty or corrupted)
MIN_LAPLACIAN_VAR = 15.0     # Sharpness threshold (values < 15 indicate severe out-of-focus blur)

# Deduplication parameters
PHASH_IMAGE_SIZE = 32        # Size for 32x32 DCT computation
PHASH_HASH_SIZE = 8          # 8x8 = 64-bit perceptual hash
PHASH_HAMMING_THRESHOLD = 4  # Hamming distance <= 4 considered near-duplicate

# Dataset Partitioning
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_RANDOM_SEED = 42

# Canonical Plant Organs
SUPPORTED_ORGANS = [
    "leaf",
    "stem",
    "tuber",
    "fruit",
    "root",
    "insect",
    "whole_plant"
]

# Supported Crops
SUPPORTED_CROPS = [
    "potato",
    "watermelon"
]

def ensure_directories():
    """Ensure all required pipeline directories exist on disk."""
    for path in [
        DATA_ROOT,
        RAW_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        SPLITS_DIR / "train",
        SPLITS_DIR / "val",
        SPLITS_DIR / "test",
        MANIFESTS_DIR,
        REJECTS_DIR,
        REJECTS_DIR / "corrupted",
        REJECTS_DIR / "undersized",
        REJECTS_DIR / "blurry",
        REJECTS_DIR / "duplicates",
        REJECTS_DIR / "unknown_label"
    ]:
        path.mkdir(parents=True, exist_ok=True)
