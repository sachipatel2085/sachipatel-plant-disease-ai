"""
Image Validation and Integrity Engine.
Verifies file structure, dimensions, color channels, sharpness, and corrupted byte streams.
Purges unreadable or severely degraded images to maintain high-quality training sets.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFile

# Ensure truncated images trigger errors rather than silent corruption
ImageFile.LOAD_TRUNCATED_IMAGES = False

from .config import (
    SUPPORTED_EXTENSIONS,
    MIN_IMAGE_WIDTH,
    MIN_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_IMAGE_HEIGHT,
    MIN_FILE_SIZE_BYTES,
    MIN_LAPLACIAN_VAR,
    TARGET_COLORSPACE
)

class ImageValidator:
    """Validates raw image files against rigorous agricultural computer vision quality constraints."""

    @staticmethod
    def validate_file(image_path: Path) -> Dict[str, Any]:
        """
        Validate an image file on disk.
        Returns a dictionary containing:
            - is_valid: bool
            - reason: Optional rejection cause
            - width: int
            - height: int
            - channels: int
            - sharpness: float (Laplacian variance)
            - file_size_bytes: int
            - sanitized_image: Optional PIL.Image in RGB format
        """
        if not image_path.exists():
            return {
                "is_valid": False,
                "reason": "File does not exist",
                "path": str(image_path)
            }

        # 1. Extension check
        ext = image_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "is_valid": False,
                "reason": f"Unsupported extension '{ext}'. Must be one of {SUPPORTED_EXTENSIONS}",
                "path": str(image_path)
            }

        # 2. File size check
        file_size = image_path.stat().st_size
        if file_size < MIN_FILE_SIZE_BYTES:
            return {
                "is_valid": False,
                "reason": f"File size too small ({file_size} bytes < {MIN_FILE_SIZE_BYTES} bytes)",
                "path": str(image_path),
                "file_size_bytes": file_size
            }

        # 3. PIL load & EXIF auto-rotation & format integrity
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify underlying file stream integrity
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Corrupted file stream: {str(e)}",
                "path": str(image_path)
            }

        # Re-open after verify() (verify invalidates the file pointer in PIL)
        try:
            with Image.open(image_path) as img:
                # Transpose according to EXIF orientation tag if present
                img = ImageOps.exif_transpose(img)
                # Convert to standard RGB
                rgb_img = img.convert("RGB")
                width, height = rgb_img.size
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"Failed to decode RGB pixels: {str(e)}",
                "path": str(image_path)
            }

        # 4. Dimension constraints
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            return {
                "is_valid": False,
                "reason": f"Undersized resolution ({width}x{height} < minimum {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT})",
                "path": str(image_path),
                "width": width,
                "height": height
            }

        if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
            return {
                "is_valid": False,
                "reason": f"Oversized resolution ({width}x{height} > maximum {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT})",
                "path": str(image_path),
                "width": width,
                "height": height
            }

        # 5. OpenCV Sharpness & Blurriness Check (Laplacian Variance)
        try:
            np_img = np.array(rgb_img)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as e:
            return {
                "is_valid": False,
                "reason": f"OpenCV matrix conversion failed: {str(e)}",
                "path": str(image_path)
            }

        if laplacian_var < MIN_LAPLACIAN_VAR:
            return {
                "is_valid": False,
                "reason": f"Severe out-of-focus blur detected (Laplacian variance {laplacian_var:.2f} < threshold {MIN_LAPLACIAN_VAR})",
                "path": str(image_path),
                "width": width,
                "height": height,
                "sharpness": laplacian_var
            }

        # 6. Uniform color / blank image check
        std_dev = float(np.std(gray))
        if std_dev < 5.0:
            return {
                "is_valid": False,
                "reason": f"Blank or monochrome image detected (Standard deviation {std_dev:.2f} < 5.0)",
                "path": str(image_path),
                "width": width,
                "height": height,
                "sharpness": laplacian_var
            }

        return {
            "is_valid": True,
            "reason": None,
            "path": str(image_path),
            "width": width,
            "height": height,
            "channels": 3,
            "sharpness": round(laplacian_var, 2),
            "file_size_bytes": file_size,
            "aspect_ratio": round(width / float(height), 3),
            "sanitized_image": rgb_img
        }

    @staticmethod
    def save_sanitized(rgb_img: Image.Image, output_path: Path, quality: int = 95):
        """Save sanitized RGB image as JPEG with consistent baseline parameters."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_img.save(output_path, "JPEG", quality=quality, optimize=True)
