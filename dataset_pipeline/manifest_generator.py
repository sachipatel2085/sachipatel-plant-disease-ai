"""
Dataset Manifest Generator.
Generates comprehensive JSON manifests and CSV summaries recording metadata,
provenance tracking, cryptographic digests, perceptual hashes, and split distribution.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .config import MANIFESTS_DIR

class ManifestGenerator:
    """Produces immutable dataset manifests with full cryptographic and provenance tracking."""

    @staticmethod
    def generate_manifest(
        valid_records: List[Dict[str, Any]],
        rejected_records: List[Dict[str, Any]],
        class_weights: Dict[str, Dict[str, Any]],
        output_dir: Path = MANIFESTS_DIR
    ) -> Tuple[Path, Path]:
        """
        Assemble and persist dataset_manifest.json and manifest_summary.csv.
        Returns paths to the generated files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute summary metrics
        split_counts = {"train": 0, "val": 0, "test": 0}
        crop_counts = {}
        part_counts = {}
        source_counts = {}
        class_counts = {}

        for r in valid_records:
            split = r.get("split", "unassigned")
            split_counts[split] = split_counts.get(split, 0) + 1

            crop = r.get("crop", "unknown")
            crop_counts[crop] = crop_counts.get(crop, 0) + 1

            part = r.get("primary_plant_part", "unknown")
            part_counts[part] = part_counts.get(part, 0) + 1

            source = r.get("source_id", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

            cond_id = r.get("condition_id", "unknown")
            class_counts[cond_id] = class_counts.get(cond_id, 0) + 1

        manifest = {
            "metadata": {
                "generated_at": timestamp,
                "pipeline_version": "2.0.0-phase2",
                "total_valid_images": len(valid_records),
                "total_rejected_images": len(rejected_records),
                "split_counts": split_counts,
                "crop_counts": crop_counts,
                "plant_part_counts": part_counts,
                "source_attribution_counts": source_counts,
                "total_conditions_represented": len(class_counts),
                "class_weights_and_augmentation_policies": class_weights
            },
            "rejected_samples_summary": [
                {
                    "path": r.get("path"),
                    "reason": r.get("reason"),
                    "details": {k: v for k, v in r.items() if k not in ("path", "reason", "sanitized_image")}
                }
                for r in rejected_records
            ],
            "images": [
                {
                    "image_id": r.get("image_id"),
                    "split": r.get("split"),
                    "crop": r.get("crop"),
                    "condition_id": r.get("condition_id"),
                    "condition_name": r.get("condition_name"),
                    "scientific_name": r.get("scientific_name"),
                    "category": r.get("category"),
                    "primary_plant_part": r.get("primary_plant_part"),
                    "visual_detectability": r.get("visual_detectability"),
                    "source_id": r.get("source_id"),
                    "source_license": r.get("source_license"),
                    "relative_path": r.get("relative_path"),
                    "width": r.get("width"),
                    "height": r.get("height"),
                    "sharpness_laplacian_var": r.get("sharpness"),
                    "file_size_bytes": r.get("file_size_bytes"),
                    "sha256": r.get("sha256"),
                    "phash": r.get("phash")
                }
                for r in valid_records
            ]
        }

        # Write JSON Manifest
        json_path = output_dir / "dataset_manifest.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Write CSV Summary
        csv_path = output_dir / "manifest_summary.csv"
        csv_fields = [
            "image_id", "split", "crop", "condition_id", "condition_name",
            "scientific_name", "category", "primary_plant_part",
            "visual_detectability", "source_id", "source_license",
            "width", "height", "sharpness_laplacian_var", "sha256", "phash"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in manifest["images"]:
                writer.writerow(r)

        return json_path, csv_path
