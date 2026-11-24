"""
Deduplication and Leakage Prevention Engine.
Implements SHA-256 exact matching and 64-bit Perceptual Hashing (pHash & dHash)
to eliminate near-duplicates, redundant web captures, and train/test dataset contamination.
"""

import hashlib
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

from .config import (
    PHASH_IMAGE_SIZE,
    PHASH_HASH_SIZE,
    PHASH_HAMMING_THRESHOLD
)

class ImageDeduplicator:
    """Computes exact and perceptual hashes to detect identical and near-duplicate agricultural images."""

    @staticmethod
    def compute_sha256(img: Image.Image) -> str:
        """Compute cryptographic SHA-256 digest of raw uncompressed RGB pixel bytes."""
        pixel_bytes = img.tobytes()
        return hashlib.sha256(pixel_bytes).hexdigest()

    @staticmethod
    def compute_phash(img: Image.Image) -> str:
        """
        Compute 64-bit Perceptual Hash (pHash) using 2D Discrete Cosine Transform (DCT).
        Robust against JPEG recompression, minor lighting shifts, and small scaling artifacts.
        """
        # 1. Convert to grayscale and resize to 32x32
        np_img = np.array(img.convert("L"))
        resized = cv2.resize(np_img, (PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        # 2. Compute 2D DCT
        dct = cv2.dct(np.float32(resized))

        # 3. Extract top-left 8x8 low-frequency matrix (64 coefficients)
        dct_low = dct[:PHASH_HASH_SIZE, :PHASH_HASH_SIZE]

        # 4. Compute median excluding DC term [0, 0]
        med = float(np.median(dct_low.flatten()[1:]))

        # 5. Threshold coefficients into 64 boolean bits
        bits = dct_low > med
        flat_bits = bits.flatten()

        # 6. Convert 64 bits to 16-character hexadecimal string
        hash_int = 0
        for bit in flat_bits:
            hash_int = (hash_int << 1) | int(bit)
        return f"{hash_int:016x}"

    @staticmethod
    def compute_dhash(img: Image.Image) -> str:
        """
        Compute 64-bit Difference Hash (dHash) based on horizontal pixel gradients.
        Fast and effective complementary hash.
        """
        np_img = np.array(img.convert("L"))
        resized = cv2.resize(np_img, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        flat_bits = diff.flatten()

        hash_int = 0
        for bit in flat_bits:
            hash_int = (hash_int << 1) | int(bit)
        return f"{hash_int:016x}"

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Compute bitwise Hamming distance between two 16-character hex hashes."""
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count('1')


class DeduplicationIndex:
    """In-memory index to detect duplicate and near-duplicate images in real time."""

    def __init__(self, hamming_threshold: int = PHASH_HAMMING_THRESHOLD):
        self.hamming_threshold = hamming_threshold
        self.exact_hashes: Dict[str, str] = {}  # sha256 -> identifier
        self.phash_entries: List[Dict[str, Any]] = []

    def check_duplicate(self, img: Image.Image) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
        """
        Inspect whether an image is an exact or near-duplicate of an already registered image.
        Returns:
            - is_duplicate (bool)
            - match_type ("exact" | "near_duplicate" | None)
            - hamming_distance (int or None)
            - matched_id (str or None)
        """
        # 1. Exact SHA-256 check
        sha256_hash = ImageDeduplicator.compute_sha256(img)
        if sha256_hash in self.exact_hashes:
            return True, "exact", 0, self.exact_hashes[sha256_hash]

        # 2. Perceptual Hash check
        phash = ImageDeduplicator.compute_phash(img)
        for entry in self.phash_entries:
            dist = ImageDeduplicator.hamming_distance(phash, entry["phash"])
            if dist <= self.hamming_threshold:
                return True, "near_duplicate", dist, entry["identifier"]

        return False, None, None, None

    def register(self, identifier: str, img: Image.Image, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """
        Register a verified unique image into the deduplication index.
        Returns (sha256_hash, phash_hash).
        """
        sha256_hash = ImageDeduplicator.compute_sha256(img)
        phash_hash = ImageDeduplicator.compute_phash(img)

        self.exact_hashes[sha256_hash] = identifier
        entry = {
            "identifier": identifier,
            "sha256": sha256_hash,
            "phash": phash_hash,
            "metadata": metadata or {}
        }
        self.phash_entries.append(entry)
        return sha256_hash, phash_hash

    def count(self) -> int:
        return len(self.exact_hashes)
