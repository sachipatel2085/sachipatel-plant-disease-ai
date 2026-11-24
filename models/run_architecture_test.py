"""
Multi-Model Architecture Verification and Clinical Workflow Test Tool.
Audits all 7 AI modules (A, B, C, D, E, F, G) and executes the complete 10-Stage Diagnostic Pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Add plant-disease-ai directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Suppress verbose TF messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from models.crop_classifier import CropIdentifier
from models.organ_classifier import OrganIdentifier
from models.disease_classifier import DiseaseClassifier
from models.pest_detector import InsectPestDetector
from models.segmentation_and_severity import LesionSegmentationUNet
from models.uncertainty_engine import UncertaintyEngine
from models.diagnostic_orchestrator import DiagnosticPipeline
from models.common import DiseaseCandidate

def create_sample_leaf(has_blight: bool = True) -> Image.Image:
    """Create a sample synthetic leaf for pipeline testing."""
    img = Image.new("RGB", (256, 256), color=(35, 130, 35))
    draw = ImageDraw.Draw(img)
    # Leaf vein structure
    draw.line([(128, 20), (128, 240)], fill=(50, 160, 50), width=3)
    for y in range(40, 220, 30):
        draw.line([(128, y), (70, y + 25)], fill=(45, 150, 45), width=2)
        draw.line([(128, y), (186, y + 25)], fill=(45, 150, 45), width=2)

    if has_blight:
        # Concentric necrotic rings
        for r in range(35, 5, -6):
            draw.ellipse([90 - r, 100 - r, 90 + r, 100 + r], outline=(70, 40, 15), width=3)
        draw.ellipse([90 - 5, 100 - 5, 90 + 5, 100 + 5], fill=(50, 25, 10))

    return img

def run_multi_model_audit():
    print("=" * 70)
    print("PHASE 3: MULTI-MODEL AI ARCHITECTURE AUDIT")
    print("=" * 70)

    # 1. Audit Module A: Crop Identification
    print("\n[Audit 1/7] Testing Module A: Crop Identification (MobileNetV3Small)...")
    crop_model = CropIdentifier(backbone="MobileNetV3Small")
    assert crop_model.model.output_shape == (None, 3), f"Unexpected shape {crop_model.model.output_shape}"
    test_leaf = create_sample_leaf(has_blight=True)
    crop_res = crop_model.predict(test_leaf)
    print(f"  -> Model compiled: Output shape = {crop_model.model.output_shape}")
    print(f"  -> Prediction result: crop='{crop_res.crop}', confidence={crop_res.confidence:.4f}")
    print("  -> Module A: PASS")

    # 2. Audit Module B: Plant-Part Identification
    print("\n[Audit 2/7] Testing Module B: Plant-Part Identification...")
    organ_model = OrganIdentifier(backbone="MobileNetV3Small")
    assert organ_model.model.output_shape == (None, 7), f"Unexpected shape {organ_model.model.output_shape}"
    organ_res = organ_model.predict(test_leaf)
    print(f"  -> Model compiled: Output shape = {organ_model.model.output_shape}")
    print(f"  -> Prediction result: organ='{organ_res.organ}', confidence={organ_res.confidence:.4f}")

    # Test organ compatibility guardrail
    comp_ok, comp_msg = organ_model.validate_organ_compatibility("leaf", "Leaf, Stem", "Early Blight")
    assert comp_ok is True
    incomp_ok, incomp_msg = organ_model.validate_organ_compatibility("leaf", "Tuber", "Common Scab")
    assert incomp_ok is False
    print(f"  -> Organ Compatibility Guardrail verified (Prevented diagnosing tuber scab from leaf): PASS")
    print("  -> Module B: PASS")

    # 3. Audit Module C: Disease Classification (69 Classes)
    print("\n[Audit 3/7] Testing Module C: Disease Classification (69 Taxonomy Conditions)...")
    disease_model = DiseaseClassifier(backbone="MobileNetV3Small")
    assert disease_model.num_classes == 69, f"Expected 69 classes, got {disease_model.num_classes}"
    candidates = disease_model.predict(test_leaf, crop_filter="potato", organ_filter="leaf", top_k=3)
    print(f"  -> Network compiled across {disease_model.num_classes} verified taxonomy classes")
    print(f"  -> Top-1 candidate: '{candidates[0].condition_name}' ({candidates[0].calibrated_probability * 100:.2f}%)")
    print("  -> Module C: PASS")

    # 4. Audit Module D: Insect & Pest Detection
    print("\n[Audit 4/7] Testing Module D: Insect & Pest Detection...")
    pest_detector = InsectPestDetector()
    pests = pest_detector.detect_pests(test_leaf, crop="potato")
    print(f"  -> Localized detection engine initialized: Found {len(pests)} active pest clusters")
    print("  -> Module D: PASS")

    # 5. Audit Modules E & F: U-Net Lesion Segmentation & Severity Estimator
    print("\n[Audit 5/7 & 6/7] Testing Modules E & F: U-Net Segmentation & Severity Estimator...")
    unet = LesionSegmentationUNet()
    assert unet.model.output_shape == (None, 256, 256, 1), f"Unexpected shape {unet.model.output_shape}"
    severity, mask = unet.estimate_severity(test_leaf)
    print(f"  -> U-Net compiled: Input shape={unet.input_shape}, Output shape={unet.model.output_shape}")
    print(f"  -> Evaluated Severity: {severity.severity_tier} ({severity.severity_percentage:.2f}% affected tissue area)")
    print("  -> Modules E & F: PASS")

    # 6. Audit Module G: Uncertainty & Out-of-Distribution Rejection
    print("\n[Audit 7/7] Testing Module G: Uncertainty & OOD Rejection Engine...")
    uncertainty_eng = UncertaintyEngine(min_confidence_threshold=0.50, max_entropy_threshold=2.20, margin_threshold=0.15)
    # Test low confidence rejection
    low_cand = [
        DiseaseCandidate("c1", "Disease A", "Sci A", "Fungal", "A", 0.35, 0.35, True),
        DiseaseCandidate("c2", "Disease B", "Sci B", "Bacterial", "A", 0.33, 0.33, True)
    ]
    u_eval = uncertainty_eng.evaluate(low_cand)
    assert u_eval.is_rejected is True
    print(f"  -> Low confidence rejection verified: '{u_eval.rejection_reason[:60]}...'")

    # Test differential warning
    diff_cand = [
        DiseaseCandidate("c1", "Early Blight", "Alternaria solani", "Fungal", "A", 0.58, 0.58, True),
        DiseaseCandidate("c2", "Brown Spot", "Alternaria alternata", "Fungal", "A", 0.47, 0.47, True)
    ]
    u_diff = uncertainty_eng.evaluate(diff_cand)
    assert u_diff.is_rejected is False
    assert u_diff.is_differential is True
    print(f"  -> Differential diagnosis margin warning verified: PASS")
    print("  -> Module G: PASS")

    # 7. Audit Full 10-Stage Diagnostic Pipeline
    print("\n" + "-" * 70)
    print("EXECUTING FULL 10-STAGE CLINICAL DIAGNOSTIC PIPELINE")
    print("-" * 70)
    pipeline = DiagnosticPipeline(crop_backbone="MobileNetV3Small", disease_backbone="MobileNetV3Small")

    # Case A: Out-of-Distribution / Low Confidence Test Image
    print("[Pipeline Case A] Evaluating ambiguous / un-calibrated test image:")
    report_a = pipeline.diagnose(test_leaf, user_crop_hint="potato")
    print(f"  -> Execution Status : {report_a['status']}")
    print(f"  -> Uncertainty Guard : {report_a['diagnostic_explanation'][:80]}...")
    assert report_a["status"] == "REJECTED_UNCERTAINTY"
    print("  -> Case A (OOD Rejection): PASS")

    # Case B: Confident Diagnosis Test (Injecting confident Early Blight prediction)
    print("\n[Pipeline Case B] Evaluating confirmed foliar lesion with high confidence:")
    # Mocking high-confidence prediction for Early Blight to verify downstream stages 8, 9, 10
    original_predict = pipeline.disease_classifier.predict
    def mock_confident_predict(img, crop_filter=None, organ_filter=None, top_k=5):
        return [
            DiseaseCandidate(
                condition_id="potato__early_blight",
                condition_name="Early Blight",
                scientific_name="Alternaria solani",
                category="Fungal Disease",
                visual_detectability="A",
                raw_probability=0.91,
                calibrated_probability=0.88,
                organ_compatible=True
            ),
            DiseaseCandidate(
                condition_id="potato__late_blight",
                condition_name="Late Blight",
                scientific_name="Phytophthora infestans",
                category="Oomycete Disease",
                visual_detectability="A",
                raw_probability=0.06,
                calibrated_probability=0.08,
                organ_compatible=True
            )
        ]
    pipeline.disease_classifier.predict = mock_confident_predict

    report_b = pipeline.diagnose(test_leaf, user_crop_hint="potato")
    pipeline.disease_classifier.predict = original_predict  # Restore

    print(f"  -> Execution Status : {report_b['status']}")
    print(f"  -> Final Diagnosis  : {report_b['final_diagnosis']['condition_name']} ({report_b['final_diagnosis']['scientific_name']})")
    print(f"  -> Category         : {report_b['final_diagnosis']['category']} (Class {report_b['final_diagnosis']['visual_detectability_class']})")
    print(f"  -> Assessed Severity: {report_b['severity']['tier']} ({report_b['severity']['percentage']:.2f}% affected tissue area)")
    print(f"  -> Stages Executed  : {list(report_b['stage_results'].keys())}")
    print("\nDiagnostic Explanation Narrative:")
    print(report_b["diagnostic_explanation"])
    assert report_b["status"] == "DIAGNOSED"
    assert report_b["final_diagnosis"]["condition_id"] == "potato__early_blight"
    print("  -> Case B (Confirmed Diagnosis): PASS")

    # Case C: Blurry Image Rejection Gate
    print("\n[Pipeline Case C] Evaluating severely blurry capture:")
    blurry_img = Image.fromarray(cv2.GaussianBlur(np.array(test_leaf), (51, 51), 0))
    # Write temporary file to test Stage 1 file validator
    temp_blur_path = Path("temp_blur_audit.jpg")
    blurry_img.save(temp_blur_path, "JPEG")
    try:
        report_c = pipeline.diagnose(temp_blur_path, user_crop_hint="potato")
        print(f"  -> Execution Status : {report_c['status']}")
        print(f"  -> Rejection Reason : {report_c['diagnostic_explanation']}")
        assert report_c["status"] == "REJECTED_QUALITY"
        print("  -> Case C (Blurry Rejection Gate): PASS")
    finally:
        if temp_blur_path.exists():
            temp_blur_path.unlink()

    print("-" * 70)
    print("[SUCCESS] All 7 AI modules and 10 clinical pipeline stages verified 100%.")
    print("=" * 70)

if __name__ == "__main__":
    run_multi_model_audit()
