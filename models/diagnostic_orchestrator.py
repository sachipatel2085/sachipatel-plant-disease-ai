"""
Multi-Stage Crop Health Diagnostic Pipeline Orchestrator.
Coordinates the complete 10-stage clinical workflow:
1. Image quality validation
2. Crop identification
3. Plant-part identification & compatibility check
4. Disease classification
5. Pest detection
6. Nutritional deficiency assessment
7. Confidence calibration
8. Unknown-condition & uncertainty rejection
9. Severity estimation
10. Diagnostic explanation generation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from dataset_pipeline.validator import ImageValidator
from .crop_classifier import CropIdentifier
from .organ_classifier import OrganIdentifier
from .disease_classifier import DiseaseClassifier
from .pest_detector import InsectPestDetector
from .segmentation_and_severity import LesionSegmentationUNet
from .uncertainty_engine import UncertaintyEngine
from .common import (
    CropPrediction,
    OrganPrediction,
    DiseaseCandidate,
    PestDetection,
    SeverityAssessment,
    UncertaintyEvaluation
)

class DiagnosticPipeline:
    """End-to-end 10-stage plant health diagnostic engine."""

    def __init__(
        self,
        taxonomy_path: Optional[Path] = None,
        crop_backbone: str = "MobileNetV3Small",
        organ_backbone: str = "MobileNetV3Small",
        disease_backbone: str = "EfficientNetV2B0"
    ):
        self.validator = ImageValidator()
        self.crop_identifier = CropIdentifier(backbone=crop_backbone)
        self.organ_identifier = OrganIdentifier(backbone=organ_backbone)
        self.disease_classifier = DiseaseClassifier(taxonomy_path=taxonomy_path, backbone=disease_backbone)
        self.pest_detector = InsectPestDetector()
        self.segmenter = LesionSegmentationUNet()
        self.uncertainty_engine = UncertaintyEngine()

    def diagnose(
        self,
        image_path_or_pil: Any,
        user_crop_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the full 10-stage diagnostic sequence.
        Returns a comprehensive diagnostic report dictionary.
        """
        report: Dict[str, Any] = {
            "status": "PROCESSING",
            "stage_results": {},
            "final_diagnosis": None,
            "differential_diagnoses": [],
            "severity": None,
            "pest_detections": [],
            "uncertainty": None,
            "diagnostic_explanation": ""
        }

        # --- STAGE 1: Image Quality Validation ---
        if isinstance(image_path_or_pil, (str, Path)):
            path = Path(image_path_or_pil)
            val_res = self.validator.validate_file(path)
            if not val_res["is_valid"]:
                report["status"] = "REJECTED_QUALITY"
                report["diagnostic_explanation"] = f"Image Quality Rejected: {val_res['reason']}"
                report["stage_results"]["stage_1_quality"] = val_res
                return report
            img = val_res["sanitized_image"]
        elif isinstance(image_path_or_pil, Image.Image):
            img = image_path_or_pil.convert("RGB")
            val_res = {"is_valid": True, "width": img.width, "height": img.height}
        else:
            raise ValueError("Input must be a valid file path or PIL.Image instance")

        report["stage_results"]["stage_1_quality"] = {
            "passed": True,
            "width": val_res.get("width"),
            "height": val_res.get("height"),
            "sharpness": val_res.get("sharpness")
        }

        # --- STAGE 2: Crop Identification ---
        crop_pred = self.crop_identifier.predict(img)
        # If user explicitly specified potato or watermelon, honor user constraint
        effective_crop = user_crop_hint.lower() if user_crop_hint and user_crop_hint.lower() in ("potato", "watermelon") else crop_pred.crop

        report["stage_results"]["stage_2_crop"] = {
            "predicted_crop": crop_pred.crop,
            "confidence": round(crop_pred.confidence, 4),
            "effective_crop": effective_crop,
            "is_supported": (effective_crop in ("potato", "watermelon"))
        }

        if effective_crop not in ("potato", "watermelon"):
            report["status"] = "REJECTED_UNSUPPORTED_CROP"
            report["diagnostic_explanation"] = (
                f"Unsupported Crop: The uploaded photograph does not appear to be Potato or Watermelon "
                f"(Crop model predicted '{crop_pred.crop}' with {crop_pred.confidence * 100:.1f}% confidence). "
                f"This diagnostic system currently supports Potato and Watermelon health monitoring."
            )
            return report

        # --- STAGE 3: Plant-Part Identification ---
        organ_pred = self.organ_identifier.predict(img)
        report["stage_results"]["stage_3_plant_part"] = {
            "organ": organ_pred.organ,
            "confidence": round(organ_pred.confidence, 4)
        }

        # --- STAGE 4: Disease Classification ---
        candidates = self.disease_classifier.predict(
            img=img,
            crop_filter=effective_crop,
            organ_filter=organ_pred.organ,
            top_k=5
        )
        report["stage_results"]["stage_4_disease_candidates"] = [
            {
                "condition_id": c.condition_id,
                "name": c.condition_name,
                "scientific": c.scientific_name,
                "category": c.category,
                "detectability": c.visual_detectability,
                "raw_prob": round(c.raw_probability, 4),
                "calibrated_prob": round(c.calibrated_probability, 4),
                "organ_compatible": c.organ_compatible
            }
            for c in candidates
        ]

        # --- STAGE 5: Pest Detection ---
        pests = self.pest_detector.detect_pests(img, crop=effective_crop)
        report["pest_detections"] = [
            {
                "pest_name": p.pest_name,
                "scientific_name": p.scientific_name,
                "confidence": p.confidence,
                "bbox": p.bbox_normalized,
                "damage_type": p.damage_type
            }
            for p in pests
        ]
        report["stage_results"]["stage_5_pest_detection"] = {
            "pest_found": (len(pests) > 0),
            "count": len(pests)
        }

        # --- STAGE 6: Nutritional Deficiency Assessment ---
        nutritional_candidates = [c for c in candidates if "nutritional" in c.category.lower()]
        report["stage_results"]["stage_6_nutritional"] = {
            "is_nutritional": (len(candidates) > 0 and "nutritional" in candidates[0].category.lower()),
            "top_deficiency": nutritional_candidates[0].condition_name if nutritional_candidates else None
        }

        # --- STAGE 7 & 8: Confidence Calibration & Uncertainty Rejection ---
        uncertainty = self.uncertainty_engine.evaluate(candidates)
        report["uncertainty"] = {
            "max_confidence": uncertainty.max_confidence,
            "entropy_bits": uncertainty.shannon_entropy,
            "margin": uncertainty.margin_top1_top2,
            "is_rejected": uncertainty.is_rejected,
            "is_differential": uncertainty.is_differential,
            "rejection_reason": uncertainty.rejection_reason,
            "differential_warning": uncertainty.differential_warning
        }

        if uncertainty.is_rejected:
            report["status"] = "REJECTED_UNCERTAINTY"
            report["diagnostic_explanation"] = uncertainty.rejection_reason
            return report

        primary_cand = candidates[0]

        # Check organ compatibility gate
        if not primary_cand.organ_compatible:
            report["status"] = "REJECTED_ORGAN_MISMATCH"
            cond_rec = self.disease_classifier.mapper.get_condition_by_id(primary_cand.condition_id)
            affected_str = cond_rec["Affected_Plant_Part"] if cond_rec else "other organs"
            report["diagnostic_explanation"] = (
                f"Diagnostic Safeguard Notice: The suspected condition '{primary_cand.condition_name}' "
                f"manifests primarily on [{affected_str}], but the uploaded photograph was classified as a '{organ_pred.organ}'. "
                f"To prevent diagnostic errors, please upload a clear photograph of the affected plant organ."
            )
            return report

        # --- STAGE 9: Severity Estimation ---
        severity_assessment, _ = self.segmenter.estimate_severity(img)
        report["severity"] = {
            "lesion_pixels": severity_assessment.lesion_pixels,
            "tissue_pixels": severity_assessment.tissue_pixels,
            "percentage": severity_assessment.severity_percentage,
            "tier": severity_assessment.severity_tier
        }
        report["stage_results"]["stage_9_severity"] = report["severity"]

        # --- STAGE 10: Diagnostic Explanation Generation ---
        report["status"] = "DIAGNOSED"
        report["final_diagnosis"] = {
            "condition_id": primary_cand.condition_id,
            "condition_name": primary_cand.condition_name,
            "scientific_name": primary_cand.scientific_name,
            "category": primary_cand.category,
            "crop": effective_crop.capitalize(),
            "detected_organ": organ_pred.organ.capitalize(),
            "confidence": primary_cand.calibrated_probability,
            "visual_detectability_class": primary_cand.visual_detectability
        }

        # Record differential diagnoses if close
        if len(candidates) > 1:
            report["differential_diagnoses"] = [
                {
                    "condition_name": c.condition_name,
                    "scientific_name": c.scientific_name,
                    "confidence": c.calibrated_probability,
                    "category": c.category
                }
                for c in candidates[1:3]
            ]

        # Build comprehensive explanatory narrative
        explanation_lines = [
            f"**Crop**: {effective_crop.capitalize()} | **Organ**: {organ_pred.organ.capitalize()}",
            f"**Confirmed Diagnosis**: {primary_cand.condition_name} (*{primary_cand.scientific_name}*)",
            f"**Pathology Category**: {primary_cand.category} (Class {primary_cand.visual_detectability})",
            f"**Diagnostic Confidence**: {primary_cand.calibrated_probability * 100:.1f}% (Calibrated)",
            f"**Assessed Severity**: {severity_assessment.severity_tier} ({severity_assessment.severity_percentage:.1f}% affected area)"
        ]

        if uncertainty.is_differential and uncertainty.differential_warning:
            explanation_lines.append(f"\n> [!WARNING] {uncertainty.differential_warning}")

        if pests:
            pest_summary = ", ".join([f"{p.pest_name} ({p.damage_type})" for p in pests])
            explanation_lines.append(f"\n> [!NOTE] Arthropod Pest Activity Detected: {pest_summary}")

        if primary_cand.visual_detectability == "B":
            explanation_lines.append(
                "\n> [!IMPORTANT] Class B Condition: Symptoms have known lookalikes. "
                "Secondary confirmation (e.g. stem cross-section, leaf abaxial view) is recommended."
            )
        elif primary_cand.visual_detectability == "C":
            explanation_lines.append(
                "\n> [!CAUTION] Class C Condition: Definite confirmation requires specialized laboratory testing or internal organ sectioning."
            )

        report["diagnostic_explanation"] = "\n".join(explanation_lines)
        return report
