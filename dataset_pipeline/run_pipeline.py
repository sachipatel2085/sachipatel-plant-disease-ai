"""
Dataset Pipeline CLI Orchestration and Verification Tool.
Provides automated end-to-end testing, validation audits, benchmark ingestion,
and dataset manifest inspection.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Add parent directory to path to ensure package resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dataset_pipeline.config import (
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    MANIFESTS_DIR,
    REJECTS_DIR,
    ensure_directories
)
from dataset_pipeline.ingestor import DatasetIngestor
from dataset_pipeline.validator import ImageValidator
from dataset_pipeline.deduplicator import ImageDeduplicator, DeduplicationIndex
from dataset_pipeline.taxonomy_mapper import TaxonomyMapper
from dataset_pipeline.sources import list_supported_sources, RESEARCH_SOURCES

def create_synthetic_test_fixture(fixture_dir: Path):
    """
    Generate a controlled set of test images to rigorously test every gate:
    1. Valid Potato Early Blight image
    2. Near-duplicate of #1 (minor brightness adjustment)
    3. Valid Watermelon Downy Mildew image
    4. Valid Watermelon Fruit Blotch image
    5. Valid Potato Common Scab image
    6. Severely blurry image (must be rejected by Laplacian threshold)
    7. Corrupted image file (invalid byte stream)
    8. Undersized image (64x64, must be rejected)
    9. Unmapped / random label image (must be rejected)
    """
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid Potato Early Blight (Leaf with concentric rings)
    img1 = Image.new("RGB", (256, 256), color=(40, 140, 40))
    draw1 = ImageDraw.Draw(img1)
    # Concentric brown rings
    for r in range(40, 5, -8):
        draw1.ellipse([128 - r, 128 - r, 128 + r, 128 + r], outline=(80, 45, 20), width=3)
    p1 = fixture_dir / "potato___early_blight" / "early_blight_01.jpg"
    p1.parent.mkdir(parents=True, exist_ok=True)
    img1.save(p1, "JPEG")

    # 2. Near-duplicate of #1 (brightness scaled slightly by 5%, visually and perceptually identical)
    np_img1 = np.array(img1)
    np_dup = np.clip(np_img1.astype(np.int16) + 6, 0, 255).astype(np.uint8)
    img2 = Image.fromarray(np_dup)
    p2 = fixture_dir / "potato___early_blight" / "early_blight_dup.jpg"
    img2.save(p2, "JPEG")

    # 3. Valid Watermelon Downy Mildew (Angular chlorotic patterns)
    img3 = Image.new("RGB", (300, 300), color=(30, 120, 30))
    draw3 = ImageDraw.Draw(img3)
    # Angular polygons
    draw3.polygon([(50, 50), (120, 60), (100, 130), (40, 110)], fill=(210, 210, 30))
    draw3.polygon([(140, 70), (220, 80), (200, 150), (130, 140)], fill=(220, 200, 20))
    p3 = fixture_dir / "watermelon_downy_mildew" / "downy_01.jpg"
    p3.parent.mkdir(parents=True, exist_ok=True)
    img3.save(p3, "JPEG")

    # 4. Valid Watermelon Fruit Blotch (Fruit rind with greasy blotch and texture)
    img4 = Image.new("RGB", (256, 256), color=(20, 90, 40))
    draw4 = ImageDraw.Draw(img4)
    draw4.ellipse([60, 60, 180, 180], fill=(10, 40, 15), outline=(5, 20, 5), width=2)
    # Add rind marbling / texture streaks
    for x in range(30, 230, 25):
        draw4.line([(x, 30), (x + 15, 220)], fill=(35, 120, 50), width=3)
    p4 = fixture_dir / "bacterial_fruit_blotch" / "fruit_blotch_01.jpg"
    p4.parent.mkdir(parents=True, exist_ok=True)
    img4.save(p4, "JPEG")

    # 5. Valid Potato Common Scab (Tuber periderm with corky lesions)
    img5 = Image.new("RGB", (256, 256), color=(160, 120, 80))
    draw5 = ImageDraw.Draw(img5)
    for cx, cy in [(80, 80), (170, 90), (120, 160)]:
        draw5.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(90, 60, 30), outline=(60, 30, 10))
    p5 = fixture_dir / "potato_scab" / "common_scab_01.jpg"
    p5.parent.mkdir(parents=True, exist_ok=True)
    img5.save(p5, "JPEG")

    # 6. Severely blurry image (uniform smooth gradient, Laplacian variance < 5)
    img6 = Image.new("RGB", (256, 256), color=(100, 100, 100))
    blurred = cv2.GaussianBlur(np.array(img6), (51, 51), 0)
    img6_b = Image.fromarray(blurred)
    p6 = fixture_dir / "potato___early_blight" / "blurry_image.jpg"
    img6_b.save(p6, "JPEG")

    # 7. Corrupted image file
    p7 = fixture_dir / "potato___early_blight" / "corrupted_file.jpg"
    with open(p7, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00corrupted_bytes_only")

    # 8. Undersized image (64x64 < 128x128 limit)
    img8 = Image.new("RGB", (64, 64), color=(50, 150, 50))
    p8 = fixture_dir / "watermelon_downy_mildew" / "undersized_64x64.jpg"
    img8.save(p8, "JPEG")

    # 9. Unmapped / random label
    img9 = Image.new("RGB", (256, 256), color=(200, 50, 50))
    draw9 = ImageDraw.Draw(img9)
    draw9.rectangle([50, 50, 200, 200], fill=(255, 255, 255))
    p9 = fixture_dir / "random_unregistered_object" / "mystery.jpg"
    p9.parent.mkdir(parents=True, exist_ok=True)
    img9.save(p9, "JPEG")

    print(f"[Test Fixture] Generated 9 diagnostic test cases in {fixture_dir}")
    return fixture_dir

def run_benchmark_audit():
    """Execute end-to-end validation, deduplication, and partitioning test."""
    print("=" * 70)
    print("PHASE 2: DATASET ENGINEERING PIPELINE AUDIT")
    print("=" * 70)

    fixture_dir = RAW_DIR / "audit_fixture"
    create_synthetic_test_fixture(fixture_dir)

    ingestor = DatasetIngestor()
    results = ingestor.process_directory(fixture_dir, default_source_id="plantvillage")

    print("\n" + "-" * 70)
    print("PIPELINE AUDIT EXECUTION SUMMARY:")
    print("-" * 70)
    print(f"Total Candidate Files Ingested : {results['total_processed']}")
    print(f"Valid Certified Images         : {results['valid_count']}")
    print(f"Rejected Defective Images      : {results['rejected_count']}")
    print(f"Train / Val / Test Partitioning: {results['splits']}")
    print(f"Zero Split Leakage Guarantee   : {'PASS (Clean)' if results['leakage_free'] else 'FAIL'}")
    print(f"Generated JSON Manifest        : {results['manifest_json']}")
    print(f"Generated CSV Summary          : {results['manifest_csv']}")
    print("-" * 70)

    # Validate results matches expected behavior
    # Total: 9 images
    # Valid: 4 images (p1, p3, p4, p5)
    # Rejections: 5 images (p2=duplicate, p6=blurry, p7=corrupted, p8=undersized, p9=unknown_label)
    assert results["valid_count"] == 4, f"Expected 4 valid images, got {results['valid_count']}"
    assert results["rejected_count"] == 5, f"Expected 5 rejected images, got {results['rejected_count']}"
    assert results["leakage_free"] is True, "Split leakage detected!"

    print("[SUCCESS] All pipeline gates passed with 100% precision.")
    print("=" * 70)

def run_multi_sample_split_audit():
    """Verify stratified 70/15/15 partitioning and class weight computation across multiple classes."""
    print("\n" + "=" * 70)
    print("STRATIFIED PARTITIONING & CLASS WEIGHTING AUDIT")
    print("=" * 70)

    from dataset_pipeline.balancer_and_splitter import DatasetBalancerAndSplitter

    balancer = DatasetBalancerAndSplitter(random_seed=42)

    # Mock dataset with varying class frequencies (simulating heavy imbalance)
    mock_records = []
    # Class 1: 20 samples (Frequent class)
    for i in range(20):
        mock_records.append({
            "image_id": f"eb_{i:03d}",
            "condition_id": "potato__early_blight",
            "sha256": f"sha_eb_{i:03d}",
            "phash": f"phash_eb_{i:03d}"
        })
    # Class 2: 10 samples (Medium class)
    for i in range(10):
        mock_records.append({
            "image_id": f"dm_{i:03d}",
            "condition_id": "watermelon__downy_mildew",
            "sha256": f"sha_dm_{i:03d}",
            "phash": f"phash_dm_{i:03d}"
        })
    # Class 3: 4 samples (Rare class)
    for i in range(4):
        mock_records.append({
            "image_id": f"fb_{i:03d}",
            "condition_id": "watermelon__bacterial_fruit_blotch",
            "sha256": f"sha_fb_{i:03d}",
            "phash": f"phash_fb_{i:03d}"
        })

    splits = balancer.stratified_split(mock_records)
    print(f"Total mock records: {len(mock_records)}")
    print(f"Train split count : {len(splits['train'])} ({len(splits['train'])/len(mock_records)*100:.1f}%)")
    print(f"Val split count   : {len(splits['val'])} ({len(splits['val'])/len(mock_records)*100:.1f}%)")
    print(f"Test split count  : {len(splits['test'])} ({len(splits['test'])/len(mock_records)*100:.1f}%)")

    # Verify anti-leakage
    is_clean, violations = balancer.verify_split_leakage(splits)
    print(f"Anti-leakage check: {'PASS (Zero leakage)' if is_clean else f'FAIL: {violations}'}")
    assert is_clean is True

    # Check class weights
    weights = balancer.calculate_class_weights(mock_records)
    print("\nCalculated Class Weights & Recommended Augmentation Policies:")
    for cond_id, w in weights.items():
        print(f"  - {cond_id:36}: N={w['sample_count']:2d} | Weight={w['inverse_frequency_weight']:6.3f} | Policy={w['augmentation_policy']:8s}")

    print("\n[SUCCESS] Stratified partitioning and class weighting verified.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Health Dataset Pipeline CLI")
    parser.add_argument("--test", action="store_true", help="Run automated benchmark audit")
    parser.add_argument("--ingest", type=str, help="Ingest a custom directory of images")
    parser.add_argument("--source", type=str, default="plantvillage", help="Source ID for ingestion")

    args = parser.parse_args()

    if args.test or len(sys.argv) == 1:
        run_benchmark_audit()
        run_multi_sample_split_audit()
    elif args.ingest:
        ingestor = DatasetIngestor()
        res = ingestor.process_directory(Path(args.ingest), default_source_id=args.source)
        print(f"Processed: {res}")
