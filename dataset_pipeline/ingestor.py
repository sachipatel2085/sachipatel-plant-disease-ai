"""
Dataset Ingestion and Processing Pipeline.
Coordinates validation, taxonomy mapping, cryptographic deduplication,
stratified partitioning, and physical directory organization.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from .config import (
    ensure_directories,
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    REJECTS_DIR,
    MANIFESTS_DIR
)
from .validator import ImageValidator
from .deduplicator import DeduplicationIndex
from .taxonomy_mapper import TaxonomyMapper
from .sources import get_source_metadata
from .balancer_and_splitter import DatasetBalancerAndSplitter
from .manifest_generator import ManifestGenerator

class DatasetIngestor:
    """Executes the complete dataset processing pipeline."""

    def __init__(self):
        ensure_directories()
        self.validator = ImageValidator()
        self.dedup_index = DeduplicationIndex()
        self.mapper = TaxonomyMapper()
        self.balancer = DatasetBalancerAndSplitter()
        self.manifest_gen = ManifestGenerator()

    def process_directory(
        self,
        input_dir: Path,
        default_source_id: str = "plantvillage",
        copy_to_splits: bool = True
    ) -> Dict[str, Any]:
        """
        Process an entire directory tree of agricultural images.
        Supports structures:
            input_dir/<condition_label>/image.jpg
            input_dir/<crop>/<condition_label>/image.jpg
            input_dir/<source_id>/<crop>/<condition_label>/image.jpg
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            image_files.extend(list(input_dir.glob(f"**/*{ext}")))
            image_files.extend(list(input_dir.glob(f"**/*{ext.upper()}")))

        image_files = sorted(list(set(image_files)))
        print(f"[Pipeline] Found {len(image_files)} raw candidate image files in {input_dir}")

        valid_records: List[Dict[str, Any]] = []
        rejected_records: List[Dict[str, Any]] = []

        for idx, file_path in enumerate(image_files, start=1):
            # 1. Determine metadata from directory hierarchy
            rel_parts = file_path.relative_to(input_dir).parts
            source_id = default_source_id
            crop_hint = None
            raw_label = None

            if len(rel_parts) >= 3:
                source_id = rel_parts[0]
                crop_hint = rel_parts[1]
                raw_label = rel_parts[2]
            elif len(rel_parts) == 2:
                raw_label = rel_parts[0]
            else:
                raw_label = file_path.stem

            source_meta = get_source_metadata(source_id)

            # 2. Taxonomy mapping
            tax_record = self.mapper.normalize_label(raw_label, crop_hint=crop_hint)
            if not tax_record:
                # Attempt using parent folder
                tax_record = self.mapper.normalize_label(file_path.parent.name, crop_hint=crop_hint)

            if not tax_record:
                reject_info = {
                    "path": str(file_path),
                    "reason": f"Unknown or unmapped label '{raw_label}'",
                    "file_name": file_path.name
                }
                rejected_records.append(reject_info)
                # Copy to rejects
                dst = REJECTS_DIR / "unknown_label" / file_path.name
                shutil.copy2(file_path, dst)
                continue

            # 3. Image validation (format, corruption, resolution, blur)
            val_result = self.validator.validate_file(file_path)
            if not val_result["is_valid"]:
                reject_info = {
                    "path": str(file_path),
                    "reason": val_result["reason"],
                    "details": {k: v for k, v in val_result.items() if k not in ("is_valid", "sanitized_image")}
                }
                rejected_records.append(reject_info)
                # Group reject by failure category
                category = "corrupted"
                if "blur" in val_result["reason"].lower():
                    category = "blurry"
                elif "undersized" in val_result["reason"].lower():
                    category = "undersized"
                dst = REJECTS_DIR / category / file_path.name
                shutil.copy2(file_path, dst)
                continue

            sanitized_img = val_result["sanitized_image"]

            # 4. Deduplication & Near-duplicate prevention (pHash & SHA-256)
            is_dup, match_type, dist, matched_id = self.dedup_index.check_duplicate(sanitized_img)
            if is_dup:
                reject_info = {
                    "path": str(file_path),
                    "reason": f"Duplicate detected ({match_type}, distance={dist}) matching '{matched_id}'",
                    "matched_with": matched_id,
                    "match_type": match_type,
                    "hamming_distance": dist
                }
                rejected_records.append(reject_info)
                dst = REJECTS_DIR / "duplicates" / f"dup_{idx}_{file_path.name}"
                shutil.copy2(file_path, dst)
                continue

            # 5. Image passed all gates -> Register into Index
            image_id = f"{tax_record['condition_id']}_{idx:06d}"
            sha256_hash, phash_hash = self.dedup_index.register(image_id, sanitized_img)

            # Save sanitized image into processed directory
            processed_rel_path = Path(tax_record["Crop"].lower()) / tax_record["condition_id"] / f"{image_id}.jpg"
            processed_abs_path = PROCESSED_DIR / processed_rel_path
            self.validator.save_sanitized(sanitized_img, processed_abs_path)

            record = {
                "image_id": image_id,
                "crop": tax_record["Crop"].lower(),
                "condition_id": tax_record["condition_id"],
                "condition_name": tax_record["Disease_or_Pest"],
                "scientific_name": tax_record["Scientific_Name"],
                "category": tax_record.get("Category", "Disease"),
                "primary_plant_part": tax_record.get("primary_plant_part", "leaf"),
                "visual_detectability": tax_record.get("Visual_Detectability", "A"),
                "source_id": source_meta.get("source_id", source_id),
                "source_license": source_meta.get("license", "Open"),
                "relative_path": str(processed_rel_path).replace("\\", "/"),
                "absolute_path": str(processed_abs_path),
                "width": val_result["width"],
                "height": val_result["height"],
                "sharpness": val_result["sharpness"],
                "file_size_bytes": processed_abs_path.stat().st_size,
                "sha256": sha256_hash,
                "phash": phash_hash
            }
            valid_records.append(record)

        print(f"[Pipeline] Validation Complete: {len(valid_records)} valid, {len(rejected_records)} rejected.")

        # 6. Stratified Split (70/15/15)
        splits = self.balancer.stratified_split(valid_records)
        is_clean, violations = self.balancer.verify_split_leakage(splits)
        if not is_clean:
            print(f"[WARNING] Leakage detected across splits: {violations}")

        # Physical copy to split directories
        if copy_to_splits:
            for split_name, split_records in splits.items():
                for r in split_records:
                    src_file = Path(r["absolute_path"])
                    dst_file = SPLITS_DIR / split_name / r["condition_id"] / f"{r['image_id']}.jpg"
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    r["split_relative_path"] = str(dst_file.relative_to(SPLITS_DIR.parent)).replace("\\", "/")

        # 7. Class weighting & augmentation policy
        class_weights = self.balancer.calculate_class_weights(valid_records)

        # 8. Manifest Generation
        json_path, csv_path = self.manifest_gen.generate_manifest(
            valid_records=valid_records,
            rejected_records=rejected_records,
            class_weights=class_weights
        )

        return {
            "total_processed": len(image_files),
            "valid_count": len(valid_records),
            "rejected_count": len(rejected_records),
            "splits": {k: len(v) for k, v in splits.items()},
            "leakage_free": is_clean,
            "manifest_json": str(json_path),
            "manifest_csv": str(csv_path),
            "class_weights": class_weights
        }
