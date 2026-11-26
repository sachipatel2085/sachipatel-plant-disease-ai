"""
Production FastAPI Crop Health Diagnostic Service — v2.
Replaces the legacy Flask prototype with the full multi-model 10-stage pipeline,
knowledge base, and recommendation engine.

Endpoints:
  POST /api/v2/diagnose       — Full image → diagnosis → advisory pipeline
  GET  /api/v2/conditions      — List all 69 supported conditions
  GET  /api/v2/conditions/{id} — Full clinical profile for one condition
  GET  /api/v2/advisory/{id}   — Generate advisory with optional context params
  GET  /api/v2/health          — Service health check and model status

Legacy compatibility:
  POST /predict                — Legacy endpoint (mirrors old Flask response shape)
"""

import os
import sys
import io
import json
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any

# Suppress TF noise before anything else
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure our package root (plant-disease-ai/) is on sys.path so that
# `from models.…`, `from knowledge_base.…` etc. resolve correctly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import our Phase 3 & 4 modules
# ---------------------------------------------------------------------------
from models.diagnostic_orchestrator import DiagnosticPipeline
from knowledge_base.recommendation_engine import RecommendationEngine
from knowledge_base.conditions_db import AgriculturalKnowledgeBase

# ---------------------------------------------------------------------------
# Legacy model support (for backward-compatible /predict endpoint)
# ---------------------------------------------------------------------------
LEGACY_MODEL = None
LEGACY_LABELS = None

def _load_legacy_model():
    """Load the old 10 MB plant_model.h5 + labels.json if they exist."""
    global LEGACY_MODEL, LEGACY_LABELS
    h5_path = PROJECT_ROOT / "plant_model.h5"
    labels_path = PROJECT_ROOT / "labels.json"
    if h5_path.exists() and labels_path.exists():
        try:
            from tensorflow.keras.models import load_model
            LEGACY_MODEL = load_model(str(h5_path), compile=False)
            with open(labels_path) as f:
                LEGACY_LABELS = json.load(f)
            print(f"[v2] Legacy model loaded: {h5_path.stat().st_size / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"[v2] Legacy model load skipped: {e}")


# Mapping from legacy PlantVillage CNN labels to Phase 4 Knowledge Base condition_ids
LEGACY_TO_KB_MAP: Dict[str, str] = {
    "Potato___Early_blight": "potato__early_blight",
    "Potato___Late_blight": "potato__late_blight",
    "Potato___healthy": "potato__healthy_potato_plant",
}


def _warmup_models():
    """Run a synthetic forward pass at startup so TF graph tracing finishes before requests arrive."""
    try:
        if LEGACY_MODEL is not None:
            dummy_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = LEGACY_MODEL(dummy_tensor, training=False)
        pipeline = _get_pipeline()
        if getattr(pipeline.disease_classifier, "has_trained_weights", False):
            dummy_img = Image.new("RGB", (256, 256), color=(34, 139, 34))
            _ = pipeline.diagnose(image_path_or_pil=dummy_img, user_crop_hint="potato")
        print("[v2] Model graphs warmed up successfully.")
    except Exception as e:
        print(f"[v2] Warmup warning (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    _load_legacy_model()
    try:
        _get_knowledge_base()
        _get_recommendation_engine()
        _get_pipeline()
        _warmup_models()
    except Exception as e:
        print(f"[v2] Pre-warm warning (non-fatal): {e}")
    print("[v2] Crop Health Diagnostic Service v2 ready.")
    yield


# ---------------------------------------------------------------------------
# Initialize application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Health Diagnostic Service",
    version="2.0.0",
    description="Multi-model AI diagnostic pipeline for Potato & Watermelon crops.",
    lifespan=lifespan,
)

# CORS — allow the React dev server and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5000",
        "https://sachipatel-farm-advisory.vercel.app",
        "https://sachipatel-farm-advisory-back.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared singletons (lazy-initialized on first request)
# ---------------------------------------------------------------------------
_pipeline: Optional[DiagnosticPipeline] = None
_recommendation_engine: Optional[RecommendationEngine] = None
_knowledge_base: Optional[AgriculturalKnowledgeBase] = None


def _get_pipeline() -> DiagnosticPipeline:
    global _pipeline
    if _pipeline is None:
        taxonomy_path = PROJECT_ROOT / "taxonomy" / "crop_health_taxonomy.json"
        _pipeline = DiagnosticPipeline(
            taxonomy_path=taxonomy_path if taxonomy_path.exists() else None
        )
        print("[v2] DiagnosticPipeline initialized.")
    return _pipeline


def _get_recommendation_engine() -> RecommendationEngine:
    global _recommendation_engine
    if _recommendation_engine is None:
        kb_path = PROJECT_ROOT / "knowledge_base" / "conditions_db.json"
        _recommendation_engine = RecommendationEngine(
            kb_path=kb_path if kb_path.exists() else None
        )
        print("[v2] RecommendationEngine initialized.")
    return _recommendation_engine


def _get_knowledge_base() -> AgriculturalKnowledgeBase:
    global _knowledge_base
    if _knowledge_base is None:
        kb_path = PROJECT_ROOT / "knowledge_base" / "conditions_db.json"
        _knowledge_base = AgriculturalKnowledgeBase(
            json_path=kb_path if kb_path.exists() else None
        )
        print("[v2] AgriculturalKnowledgeBase initialized.")
    return _knowledge_base


# ---------------------------------------------------------------------------
# Startup handled via lifespan context manager above
# ---------------------------------------------------------------------------


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v2/health
# ═══════════════════════════════════════════════════════════════════════════
@app.api_route("/api/v2/health", methods=["GET", "HEAD"])
async def health_check():
    pipeline = _get_pipeline()
    engine = _get_recommendation_engine()
    kb = _get_knowledge_base()
    conditions_count = len(kb.database) if kb and hasattr(kb, "database") else 0
    return {
        "status": "ok",
        "version": "2.0.0",
        "models": {
            "diagnostic_pipeline": pipeline is not None,
            "recommendation_engine": engine is not None,
            "knowledge_base_conditions": conditions_count,
            "legacy_cnn": LEGACY_MODEL is not None,
        },
    }


def _filter_top_candidates(
    candidates: List[Dict[str, Any]],
    confidence_key: str = "confidence",
    min_confidence_pct: float = 50.0,
    fallback_top_n: int = 2,
) -> List[Dict[str, Any]]:
    """Keep only results above 50% confidence; if none exceed 50%, keep only top 2."""
    if not candidates:
        return []
    sorted_cands = sorted(
        candidates,
        key=lambda c: float(c.get(confidence_key, 0.0) or 0.0),
        reverse=True,
    )
    high_conf = [c for c in sorted_cands if float(c.get(confidence_key, 0.0) or 0.0) > min_confidence_pct]
    return high_conf[:fallback_top_n] if high_conf else sorted_cands[:fallback_top_n]


def _extract_products_from_advisory(adv: Optional[Dict[str, Any]]) -> tuple:
    """Extract formatted product cards and text lines from a Phase 4 advisory dict."""
    products: List[Dict[str, Any]] = []
    chem_lines: List[str] = []
    bio_lines: List[str] = []
    if not adv or not isinstance(adv, dict):
        return products, "", ""

    chem_items = adv.get("chemical_treatments") or adv.get("chemical_management") or []
    if isinstance(chem_items, list):
        for item in chem_items:
            if isinstance(item, dict):
                trade_info = f" ({item['trade_names']})" if item.get("trade_names") else ""
                ai_name = item.get("active_ingredient") or item.get("product") or item.get("name", "")
                dose = item.get("dose_per_liter") or item.get("dose") or item.get("dosage", "")
                phi = f" | PHI: {item['phi_days']} days" if item.get("phi_days") is not None else ""
                products.append({
                    "name": f"{ai_name}{trade_info}",
                    "dosage": dose,
                    "applicationNotes": f"{item.get('notes', '')}{phi}".strip(),
                    "productType": "chemical",
                })
                chem_lines.append(f"• {ai_name}{trade_info} @ {dose}{phi}")
            elif isinstance(item, str):
                chem_lines.append(item)

    bio_items = adv.get("organic_or_biological_options") or adv.get("biological_management") or []
    if isinstance(bio_items, list):
        for item in bio_items:
            if isinstance(item, dict):
                agent_name = item.get("agent") or item.get("name") or item.get("product", "")
                app_methods = item.get("application", {})
                app_str = ""
                if isinstance(app_methods, dict):
                    app_str = "; ".join(f"{k}: {v}" for k, v in app_methods.items())
                elif isinstance(app_methods, str):
                    app_str = app_methods
                products.append({
                    "name": agent_name,
                    "dosage": app_str[:80] if app_str else "As recommended",
                    "applicationNotes": item.get("notes", ""),
                    "productType": "organic",
                })
                bio_lines.append(f"• {agent_name}: {app_str}")
            elif isinstance(item, str):
                bio_lines.append(item)

    return products, "\n".join(chem_lines), "\n".join(bio_lines)


def _estimate_hsv_lesion_severity(img: Image.Image) -> Dict[str, Any]:
    """Estimate foliage lesion percentage via HSV color analysis when U-Net weights are untrained."""
    from models.common import get_severity_tier
    resized = np.array(img.convert("RGB").resize((256, 256)))
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

    is_not_black = hsv[:, :, 2] >= 25
    is_not_white = ~((hsv[:, :, 1] < 15) & (hsv[:, :, 2] > 240))
    tissue_mask = is_not_black & is_not_white
    tissue_pixels = int(np.sum(tissue_mask))

    # Necrotic/chlorotic tones: brown/yellow/dark spots (H < 30 or H > 150, or very dark/desaturated spots)
    lesion_hue = (hsv[:, :, 0] < 32) | (hsv[:, :, 0] > 140)
    lesion_mask = tissue_mask & lesion_hue & (hsv[:, :, 1] > 25)
    lesion_pixels = int(np.sum(lesion_mask))

    pct = round((lesion_pixels / float(tissue_pixels)) * 100.0, 2) if tissue_pixels > 0 else 0.0
    return {
        "lesion_pixels": lesion_pixels,
        "tissue_pixels": tissue_pixels,
        "percentage": pct,
        "tier": get_severity_tier(pct),
    }


def _run_legacy_cnn_predictions(img: Image.Image) -> List[Dict[str, Any]]:
    """Run the loaded legacy PlantVillage CNN and return sorted class predictions."""
    resized = img.convert("RGB").resize((224, 224))
    arr = (np.array(resized, dtype=np.float32) / 255.0)[None, ...]
    probs = LEGACY_MODEL(arr, training=False).numpy()[0]
    top_indices = np.argsort(probs)[::-1]
    results = []
    for idx in top_indices:
        label = (LEGACY_LABELS or {}).get(str(int(idx)), f"Class_{int(idx)}")
        results.append({
            "index": int(idx),
            "label": label,
            "prob": float(probs[int(idx)]),
        })
    return results


def _run_diagnosis_sync(tmp_path: str, crop: Optional[str]) -> Dict[str, Any]:
    """
    Execute diagnosis synchronously (intended for threadpool execution).
    Uses full trained v2 models when weights exist; when v2 DiseaseClassifier is untrained
    and legacy plant_model.h5 is loaded, routes Potato/supported legacy classes through
    the Phase 4 Knowledge Base & Recommendation Engine while enforcing Stage 1 & Stage 5 checks.
    """
    pipeline = _get_pipeline()
    kb = _get_knowledge_base()
    has_v2_weights = getattr(pipeline.disease_classifier, "has_trained_weights", False)

    if has_v2_weights or LEGACY_MODEL is None:
        return pipeline.diagnose(image_path_or_pil=tmp_path, user_crop_hint=crop)

    # --- Hybrid Mode: Stage 1 Validation + Legacy CNN + Phase 4 KB/Engine ---
    val_res = pipeline.validator.validate_file(Path(tmp_path))
    if not val_res["is_valid"]:
        return {
            "status": "REJECTED_QUALITY",
            "stage_results": {"stage_1_quality": val_res},
            "final_diagnosis": None,
            "differential_diagnoses": [],
            "severity": None,
            "pest_detections": [],
            "uncertainty": {"is_rejected": True, "rejection_reason": val_res["reason"]},
            "diagnostic_explanation": f"Image Quality Rejected: {val_res['reason']}",
        }

    img = val_res["sanitized_image"]
    legacy_preds = _run_legacy_cnn_predictions(img)
    top_legacy = legacy_preds[0] if legacy_preds else {"label": "Unknown", "prob": 0.0}
    crop_lower = crop.lower().strip() if crop else ""

    # Determine if Potato hybrid path applies
    is_potato_request = crop_lower == "potato" or (not crop_lower and top_legacy["label"].startswith("Potato___"))

    if is_potato_request:
        potato_preds = [p for p in legacy_preds if p["label"] in LEGACY_TO_KB_MAP]
        # Normalize probabilities across Potato classes if user explicitly selected Potato
        total_pot_prob = sum(p["prob"] for p in potato_preds) or 1.0
        candidates = []
        for p in potato_preds:
            cid = LEGACY_TO_KB_MAP[p["label"]]
            rec = kb.get_condition(cid) or {}
            norm_prob = p["prob"] / total_pot_prob if crop_lower == "potato" else p["prob"]
            candidates.append({
                "condition_id": cid,
                "name": rec.get("condition_name", p["label"]),
                "scientific": rec.get("scientific_name", ""),
                "category": rec.get("category", "Disease"),
                "detectability": rec.get("visual_detectability_class", "A"),
                "raw_prob": round(p["prob"], 4),
                "calibrated_prob": round(norm_prob, 4),
                "organ_compatible": True,
            })

        candidates.sort(key=lambda c: c["calibrated_prob"], reverse=True)
        primary = candidates[0]

        pests = pipeline.pest_detector.detect_pests(img, crop="potato")
        pest_dicts = [
            {
                "pest_name": pt.pest_name,
                "scientific_name": pt.scientific_name,
                "confidence": pt.confidence,
                "bbox": pt.bbox_normalized,
                "damage_type": pt.damage_type,
            }
            for pt in pests
        ]
        sev = _estimate_hsv_lesion_severity(img)

        return {
            "status": "DIAGNOSED",
            "stage_results": {
                "stage_1_quality": {
                    "passed": True,
                    "width": val_res.get("width"),
                    "height": val_res.get("height"),
                    "sharpness": val_res.get("sharpness"),
                },
                "stage_2_crop": {
                    "predicted_crop": "potato",
                    "confidence": round(primary["calibrated_prob"], 4),
                    "effective_crop": "potato",
                    "is_supported": True,
                },
                "stage_3_plant_part": {
                    "organ": "leaf",
                    "confidence": 0.90,
                },
                "stage_4_disease_candidates": candidates,
            },
            "final_diagnosis": {
                "condition_id": primary["condition_id"],
                "condition_name": primary["name"],
                "scientific_name": primary["scientific"],
                "category": primary["category"],
                "crop": "Potato",
                "detected_organ": "Leaf",
                "confidence": primary["calibrated_prob"],
                "visual_detectability_class": primary["detectability"],
            },
            "differential_diagnoses": [
                {
                    "condition_id": c["condition_id"],
                    "condition_name": c["name"],
                    "scientific_name": c["scientific"],
                    "confidence": c["calibrated_prob"],
                    "category": c["category"],
                }
                for c in candidates[1:3]
            ],
            "severity": sev,
            "pest_detections": pest_dicts,
            "uncertainty": {
                "max_confidence": primary["calibrated_prob"],
                "is_rejected": False,
                "is_differential": len(candidates) > 1 and (candidates[0]["calibrated_prob"] - candidates[1]["calibrated_prob"]) < 0.15,
            },
            "diagnostic_explanation": f"Diagnosed {primary['name']} via hybrid CNN + Phase 4 Knowledge Base.",
        }

    # If auto crop & legacy CNN confidently detected Tomato or Pepper
    if not crop_lower and top_legacy["prob"] >= 0.45:
        sev = _estimate_hsv_lesion_severity(img)
        clean_candidates = [
            {
                "condition_id": p["label"],
                "name": p["label"].replace("___", " ").replace("__", " ").replace("_", " ").strip(),
                "scientific": "",
                "category": "Fungal / Foliar Condition",
                "calibrated_prob": round(p["prob"], 4),
                "raw_prob": round(p["prob"], 4),
            }
            for p in legacy_preds[:4]
        ]
        detected_crop = "tomato" if "tomato" in top_legacy["label"].lower() else "pepper" if "pepper" in top_legacy["label"].lower() else "potato"
        return {
            "status": "DIAGNOSED_LEGACY",
            "stage_results": {
                "stage_1_quality": {"passed": True, "width": val_res.get("width"), "height": val_res.get("height")},
                "stage_2_crop": {"predicted_crop": detected_crop, "confidence": round(top_legacy["prob"], 4), "effective_crop": detected_crop, "is_supported": True},
                "stage_3_plant_part": {"organ": "leaf", "confidence": 0.90},
                "stage_4_disease_candidates": clean_candidates,
            },
            "final_diagnosis": {
                "condition_id": top_legacy["label"],
                "condition_name": clean_candidates[0]["name"],
                "scientific_name": "",
                "category": "Plant Condition",
                "crop": detected_crop.capitalize(),
                "detected_organ": "Leaf",
                "confidence": round(top_legacy["prob"], 4),
            },
            "differential_diagnoses": clean_candidates[1:3],
            "severity": sev,
            "pest_detections": [],
            "uncertainty": {"max_confidence": round(top_legacy["prob"], 4), "is_rejected": False, "is_differential": False},
            "diagnostic_explanation": f"Detected {clean_candidates[0]['name']} ({top_legacy['prob']*100:.1f}%).",
        }

    # Watermelon or unsupported crop when v2 weights aren't trained yet:
    # Fast, honest uncertainty rejection without running slow untrained neural networks
    pests = pipeline.pest_detector.detect_pests(img, crop=crop_lower or "watermelon") if crop_lower == "watermelon" else []
    pest_dicts = [
        {
            "pest_name": pt.pest_name,
            "scientific_name": pt.scientific_name,
            "confidence": pt.confidence,
            "bbox": pt.bbox_normalized,
            "damage_type": pt.damage_type,
        }
        for pt in pests
    ]
    sev = _estimate_hsv_lesion_severity(img)
    explanation = (
        "Watermelon multi-class model weights are not yet trained (Phase 6 training scripts are ready in training_pipeline/). "
        "The system refuses to force an unverified diagnosis. Please verify visual symptoms against the Watermelon Knowledge Base."
        if crop_lower == "watermelon"
        else "Unable to classify this crop image with sufficient confidence. Please upload a clear Potato leaf photograph or select the crop explicitly."
    )
    return {
        "status": "REJECTED_UNCERTAINTY",
        "stage_results": {
            "stage_1_quality": {"passed": True, "width": val_res.get("width"), "height": val_res.get("height")},
            "stage_2_crop": {"predicted_crop": crop_lower or "unknown", "confidence": 0.0, "effective_crop": crop_lower or "unknown", "is_supported": crop_lower in ("potato", "watermelon")},
            "stage_3_plant_part": {"organ": "leaf", "confidence": 0.0},
            "stage_4_disease_candidates": [],
        },
        "final_diagnosis": None,
        "differential_diagnoses": [],
        "severity": sev,
        "pest_detections": pest_dicts,
        "uncertainty": {
            "max_confidence": 0.0,
            "is_rejected": True,
            "is_differential": False,
            "rejection_reason": explanation,
        },
        "diagnostic_explanation": explanation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v2/diagnose  — The main endpoint
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/v2/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    district: Optional[str] = Form(None),
    growth_stage: Optional[str] = Form(None),
    weather_temp: Optional[float] = Form(None),
    weather_humidity: Optional[float] = Form(None),
    weather_foggy: Optional[bool] = Form(None),
):
    """
    Full 10-stage diagnosis pipeline:
    Image → validation → crop ID → organ ID → disease classification →
    pest detection → severity → uncertainty → recommendation → advisory.
    """
    tmp_path: Optional[str] = None
    try:
        # 1. Read & validate image bytes
        contents = await image.read()
        if len(contents) < 100:
            raise HTTPException(status_code=400, detail="Uploaded file is too small or empty.")
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image exceeds 10 MB size limit.")

        # Save to temp file for the pipeline (which expects a file path)
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tempfile.gettempdir()) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # 2. Run the diagnostic pipeline in a worker thread so the asyncio loop never blocks
        diagnosis_report = await run_in_threadpool(_run_diagnosis_sync, tmp_path, crop)

        # 3. Extract primary results with robust dataclass/dict conversion
        pipeline_status = diagnosis_report.get("status", "unknown")

        def _to_dict(obj):
            if obj is None:
                return {}
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            if isinstance(obj, dict):
                return obj
            return {}

        # Stage 2 Crop
        stage_crop = diagnosis_report.get("stage_results", {}).get("stage_2_crop", {})
        predicted_crop = stage_crop.get("effective_crop") or stage_crop.get("predicted_crop") or (crop or "potato")

        # Stage 3 Organ
        stage_organ = diagnosis_report.get("stage_results", {}).get("stage_3_plant_part", {})
        predicted_organ = stage_organ.get("organ", "leaf")

        # Final Diagnosis
        raw_final = diagnosis_report.get("final_diagnosis")
        final_dx = _to_dict(raw_final) if raw_final else None

        # Differential Candidates (prefer full stage_4 list which includes condition_id & top-1)
        raw_candidates = (
            diagnosis_report.get("stage_results", {}).get("stage_4_disease_candidates")
            or diagnosis_report.get("differential_diagnoses")
            or []
        )
        candidates = [_to_dict(c) for c in raw_candidates]

        # Severity
        raw_sev = diagnosis_report.get("severity")
        severity_dict = _to_dict(raw_sev) if raw_sev else {}
        severity_tier = severity_dict.get("tier") or severity_dict.get("severity_tier") or "MODERATE"
        raw_pct = severity_dict.get("percentage") if severity_dict.get("percentage") is not None else severity_dict.get("severity_percentage", 0.0)
        affected_pct = round(float(raw_pct or 0.0), 1)

        # Pests
        raw_pests = diagnosis_report.get("pest_detections", [])
        pest_detections = [_to_dict(p) for p in raw_pests]

        # 4. Build a list of conditions to generate advisories for
        condition_ids_for_advisory = []
        if final_dx and final_dx.get("condition_id"):
            condition_ids_for_advisory.append(final_dx["condition_id"])
        for cand in candidates:
            cid = cand.get("condition_id")
            if cid and cid not in condition_ids_for_advisory:
                condition_ids_for_advisory.append(cid)

        # 5. Generate advisory for conditions
        engine = _get_recommendation_engine()
        kb = _get_knowledge_base()
        primary_advisory = None
        all_advisories = []

        for cid in condition_ids_for_advisory:
            if not kb.get_condition(cid):
                continue
            try:
                adv = engine.generate_advisory(
                    crop=predicted_crop,
                    condition_id_or_name=cid,
                    severity=severity_tier,
                    growth_stage=growth_stage or "vegetative",
                    plant_part=predicted_organ,
                    location=district or location or "Banaskantha",
                    weather={
                        "temperature": weather_temp,
                        "humidity": weather_humidity,
                        "foggy": weather_foggy,
                    } if any([weather_temp, weather_humidity, weather_foggy]) else None,
                )
                if primary_advisory is None:
                    primary_advisory = adv
                all_advisories.append({"condition_id": cid, "advisory": adv})
            except Exception as adv_err:
                all_advisories.append({"condition_id": cid, "error": str(adv_err)})

        # 6. Build the response in a shape the Node.js forwarding layer can use
        #    while also being much richer than legacy
        detected_name = "Unknown Issue"
        detected_confidence = 0.0
        scientific_name = ""
        category = ""
        condition_severity = "medium"
        symptoms_list = []
        cause_text = ""
        chemical_text = ""
        organic_text = ""
        prevention_text = ""
        recommended_products = []

        if final_dx:
            detected_name = final_dx.get("condition_name", "Unknown Issue")
            raw_conf = final_dx.get("confidence") if final_dx.get("confidence") is not None else final_dx.get("calibrated_confidence", 0.0)
            detected_confidence = round(float(raw_conf or 0.0) * 100, 2)
        elif diagnosis_report.get("diagnostic_explanation"):
            cause_text = diagnosis_report["diagnostic_explanation"]

        # Enrich from KB
        if final_dx and final_dx.get("condition_id"):
            profile = kb.get_condition(final_dx["condition_id"])
            if profile:
                scientific_name = profile.get("scientific_name", "")
                category = profile.get("category", "")
                condition_severity = profile.get("severity_default", "medium")
                raw_syms = profile.get("symptoms", [])
                symptoms_list = raw_syms if isinstance(raw_syms, list) else [raw_syms] if raw_syms else []
                cause_text = profile.get("favourable_environmental_conditions") or profile.get("favourable_conditions", "")
                raw_chem = profile.get("chemical_management", "")
                chemical_text = "\n".join(raw_chem) if isinstance(raw_chem, list) else str(raw_chem)
                raw_bio = profile.get("biological_management", "")
                organic_text = "\n".join(raw_bio) if isinstance(raw_bio, list) else str(raw_bio)
                raw_prev = profile.get("preventive_measures", "")
                prevention_text = "\n".join(raw_prev) if isinstance(raw_prev, list) else str(raw_prev)

        # Augment from primary advisory (which may add weather-specific, severity-calibrated text)
        if primary_advisory:
            if primary_advisory.get("immediate_actions"):
                prevention_text = "\n".join(primary_advisory["immediate_actions"]) + "\n\n" + prevention_text
            elif primary_advisory.get("treatment_strategy_note"):
                prevention_text = primary_advisory["treatment_strategy_note"] + "\n\n" + prevention_text

            extracted_prods, extra_chem, extra_bio = _extract_products_from_advisory(primary_advisory)
            recommended_products.extend(extracted_prods)
            if extra_chem:
                chemical_text = (chemical_text + "\n" + extra_chem).strip()
            if extra_bio:
                organic_text = (organic_text + "\n" + extra_bio).strip()

            if primary_advisory.get("cultural_management"):
                cult = primary_advisory["cultural_management"]
                if isinstance(cult, list):
                    prevention_text += "\n" + "\n".join(
                        [c if isinstance(c, str) else str(c) for c in cult]
                    )
                elif isinstance(cult, str):
                    prevention_text += "\n" + cult

            if primary_advisory.get("safety_precautions"):
                safe = primary_advisory["safety_precautions"]
                if isinstance(safe, list):
                    cause_text += "\n\n🛡️ Safety:\n" + "\n".join(safe)
            if primary_advisory.get("weather_alert"):
                cause_text += f"\n\n⚠️ Weather Trigger: {primary_advisory['weather_alert']}"

        # Build enriched probable_diseases list (what the frontend expects)
        probable_diseases = []
        for idx, cand in enumerate(candidates):
            cand_id = cand.get("condition_id", "")
            cand_name = cand.get("name") or cand.get("condition_name") or "Unknown"
            raw_c_prob = (
                cand.get("calibrated_prob")
                if cand.get("calibrated_prob") is not None
                else cand.get("calibrated_confidence")
                if cand.get("calibrated_confidence") is not None
                else cand.get("confidence")
                if cand.get("confidence") is not None
                else cand.get("raw_prob", 0.0)
            )
            cand_profile = kb.get_condition(cand_id) if cand_id else None
            cand_syms = cand_profile.get("symptoms", []) if cand_profile else []
            # Find the advisory for this candidate if we generated one
            cand_adv_entry = next((a for a in all_advisories if a.get("condition_id") == cand_id), None)
            cand_advisory = cand_adv_entry.get("advisory") if cand_adv_entry else None

            cand_chem = ""
            cand_organic = ""
            cand_prevention = ""
            cand_cause = ""
            cand_products = []

            if cand_profile:
                raw_c_chem = cand_profile.get("chemical_management", "")
                cand_chem = "\n".join(raw_c_chem) if isinstance(raw_c_chem, list) else str(raw_c_chem)
                raw_c_bio = cand_profile.get("biological_management", "")
                cand_organic = "\n".join(raw_c_bio) if isinstance(raw_c_bio, list) else str(raw_c_bio)
                raw_c_prev = cand_profile.get("preventive_measures", "")
                cand_prevention = "\n".join(raw_c_prev) if isinstance(raw_c_prev, list) else str(raw_c_prev)
                cand_cause = cand_profile.get("favourable_environmental_conditions") or cand_profile.get("favourable_conditions", "")

            if cand_advisory and isinstance(cand_advisory, dict):
                cand_products, extra_c_chem, extra_c_bio = _extract_products_from_advisory(cand_advisory)
                if extra_c_chem:
                    cand_chem = (cand_chem + "\n" + extra_c_chem).strip()
                if extra_c_bio:
                    cand_organic = (cand_organic + "\n" + extra_c_bio).strip()

            probable_diseases.append({
                "disease": cand_id or cand_name,
                "displayName": cand_name,
                "confidence": round(float(raw_c_prob or 0.0) * 100, 2),
                "category": cand_profile.get("category", "") if cand_profile else cand.get("category", ""),
                "severity": cand_profile.get("severity_default", "medium") if cand_profile else "medium",
                "scientificName": cand_profile.get("scientific_name", "") if cand_profile else cand.get("scientific", ""),
                "symptoms": cand_syms[:5] if cand_syms else [
                    "Irregular spots or discoloration on leaf surface",
                    "Signs of plant stress or foliage decay",
                ],
                "advisory": {
                    "name": cand_name,
                    "scientificName": cand_profile.get("scientific_name", "") if cand_profile else cand.get("scientific", ""),
                    "category": cand_profile.get("category", "") if cand_profile else cand.get("category", ""),
                    "severity": cand_profile.get("severity_default", "medium") if cand_profile else "medium",
                    "symptoms": cand_syms,
                    "cause": cand_cause,
                    "chemicalTreatment": cand_chem,
                    "organicSolution": cand_organic,
                    "prevention": cand_prevention,
                    "recommendedProducts": cand_products,
                },
                "solution": f"Cause:\n{cand_cause}\n\nTreatment:\n{cand_chem}\n\nOrganic Solution:\n{cand_organic}\n\nPrevention:\n{cand_prevention}",
            })

        probable_diseases = _filter_top_candidates(probable_diseases, confidence_key="confidence", min_confidence_pct=50.0, fallback_top_n=2)

        # Build the solution text (formatted just like diseaseController.js does)
        solution_text = ""
        if cause_text.strip():
            solution_text += f"Cause:\n{cause_text.strip()}\n\n"
        if chemical_text.strip():
            solution_text += f"Treatment:\n{chemical_text.strip()}\n\n"
        if organic_text.strip():
            solution_text += f"Organic Solution:\n{organic_text.strip()}\n\n"
        if prevention_text.strip():
            solution_text += f"Prevention:\n{prevention_text.strip()}"
        solution_text = solution_text.strip()

        # Pest section
        pests_found = []
        for pd_item in pest_detections:
            pests_found.append({
                "pest_name": pd_item.get("pest_name", ""),
                "confidence": round(pd_item.get("confidence", 0.0) * 100, 2),
                "bbox": pd_item.get("bbox"),
            })

        # Severity info
        severity_info = {
            "tier": severity_tier.lower(),
            "affected_percent": affected_pct,
            "description": f"{severity_tier.capitalize()} ({affected_pct}% foliage affected)",
        }

        # === The unified response ===
        response = {
            # Fields the frontend currently reads (backward compatible)
            "disease": detected_name,
            "rawDisease": final_dx.get("condition_id", "") if final_dx else "",
            "confidence": detected_confidence,
            "targetCrop": crop or predicted_crop,
            "probable_diseases": probable_diseases,
            "top_predictions": probable_diseases,
            "solution": solution_text,
            "advisory": {
                "name": detected_name,
                "scientificName": scientific_name,
                "category": category,
                "severity": condition_severity,
                "symptoms": symptoms_list,
                "cause": cause_text.strip(),
                "chemicalTreatment": chemical_text.strip(),
                "organicSolution": organic_text.strip(),
                "prevention": prevention_text.strip(),
                "recommendedProducts": recommended_products,
            },
            "source": "ai_pipeline_v2",

            # New v2 enrichments
            "v2": {
                "pipeline_status": pipeline_status,
                "crop_identification": {
                    "predicted_crop": predicted_crop,
                    "confidence": stage_crop.get("confidence", 0.0),
                },
                "organ_identification": {
                    "predicted_organ": predicted_organ,
                    "confidence": stage_organ.get("confidence", 0.0),
                },
                "severity": severity_info,
                "pests": pests_found,
                "uncertainty": _to_dict(diagnosis_report.get("uncertainty")),
                "advisories": all_advisories,
                "primary_advisory": primary_advisory,
            },
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "detail": "Internal diagnostic pipeline error. Please try again with a different image.",
            },
        )
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v2/conditions  — List all supported conditions
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/conditions")
async def list_conditions(
    crop: Optional[str] = Query(None, description="Filter by crop: potato, watermelon"),
    category: Optional[str] = Query(None, description="Filter by category: fungal, bacterial, etc."),
):
    kb = _get_knowledge_base()
    database = kb.database if hasattr(kb, "database") else {}

    results = []
    for cond_id, cond in database.items():
        cond_crop = cond.get("crop", "")
        if crop and crop.lower() != cond_crop.lower() and crop.lower() not in [c.lower() for c in cond.get("crops", [])]:
            continue
        if category and category.lower() != cond.get("category", "").lower():
            continue
        results.append({
            "condition_id": cond.get("condition_id", cond_id),
            "name": cond.get("condition_name", ""),
            "scientific_name": cond.get("scientific_name", ""),
            "crop": cond_crop,
            "category": cond.get("category", ""),
            "affected_plant_parts": cond.get("affected_plant_parts", []),
            "visual_detectability": cond.get("visual_detectability_class", ""),
        })

    return {"total": len(results), "conditions": results}


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v2/conditions/{condition_id}  — Full clinical profile
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/conditions/{condition_id}")
async def get_condition(condition_id: str):
    kb = _get_knowledge_base()
    profile = kb.get_condition(condition_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Condition '{condition_id}' not found.")
    return {"condition": profile}


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v2/advisory/{condition_id}  — Generate advisory with context
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/advisory/{condition_id}")
async def get_advisory(
    condition_id: str,
    crop: Optional[str] = Query("potato"),
    severity: Optional[str] = Query("moderate"),
    growth_stage: Optional[str] = Query("vegetative"),
    location: Optional[str] = Query("Banaskantha"),
):
    engine = _get_recommendation_engine()
    try:
        advisory = engine.generate_advisory(
            crop=crop,
            condition_id_or_name=condition_id,
            severity=severity,
            growth_stage=growth_stage,
            plant_part="leaf",
            location=location or "Banaskantha",
        )
        return {"condition_id": condition_id, "advisory": advisory}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# POST /predict  — Legacy compatibility endpoint (matches old Flask shape)
# ═══════════════════════════════════════════════════════════════════════════
def _legacy_predict_sync(contents: bytes, tmp_path: str, crop: Optional[str]) -> Dict[str, Any]:
    pipeline = _get_pipeline()
    has_v2_weights = getattr(pipeline.disease_classifier, "has_trained_weights", False)

    if has_v2_weights:
        report = pipeline.diagnose(image_path_or_pil=tmp_path, user_crop_hint=crop)
        final_dx = report.get("final_diagnosis")
        candidates = (
            report.get("stage_results", {}).get("stage_4_disease_candidates")
            or report.get("differential_diagnoses")
            or []
        )
        if final_dx and final_dx.get("condition_name"):
            disease_name = final_dx["condition_name"]
            raw_conf = final_dx.get("confidence") if final_dx.get("confidence") is not None else final_dx.get("calibrated_confidence", 0.0)
            confidence = round(float(raw_conf or 0.0) * 100, 2)
            top_preds = [
                {
                    "disease": c.get("name") or c.get("condition_name", "Unknown"),
                    "confidence": round(float(c.get("calibrated_prob") or c.get("confidence") or 0.0) * 100, 2),
                }
                for c in candidates
            ]
            top_preds = _filter_top_candidates(top_preds, confidence_key="confidence", min_confidence_pct=50.0, fallback_top_n=2)
            return {
                "success": True,
                "class": 0,
                "disease": disease_name,
                "confidence": confidence,
                "top_predictions": top_preds or [{"disease": disease_name, "confidence": confidence}],
                "visual_symptoms": {
                    "powdery_white_ratio": 0.0,
                    "necrotic_dark_ratio": 0.0,
                    "has_powdery_coating": False,
                },
            }

    if LEGACY_MODEL is not None:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        legacy_preds = _run_legacy_cnn_predictions(img)
        if crop and crop.lower() == "potato":
            pot_preds = [p for p in legacy_preds if p["label"].startswith("Potato___")]
            if pot_preds:
                legacy_preds = pot_preds
        top = legacy_preds[0]
        disease_name = top["label"]
        confidence = round(top["prob"] * 100, 2)
        top_preds = [
            {"disease": p["label"], "confidence": round(p["prob"] * 100, 2)}
            for p in legacy_preds
        ]
        top_preds = _filter_top_candidates(top_preds, confidence_key="confidence", min_confidence_pct=50.0, fallback_top_n=2)
        return {
            "success": True,
            "class": top["index"],
            "disease": disease_name,
            "confidence": confidence,
            "top_predictions": top_preds,
            "visual_symptoms": {
                "powdery_white_ratio": 0.0,
                "necrotic_dark_ratio": 0.0,
                "has_powdery_coating": False,
            },
        }

    return {
        "success": True,
        "class": 0,
        "disease": "Unknown Issue",
        "confidence": 0.0,
        "top_predictions": [{"disease": "Unknown Issue", "confidence": 0.0}],
        "visual_symptoms": {
            "powdery_white_ratio": 0.0,
            "necrotic_dark_ratio": 0.0,
            "has_powdery_coating": False,
        },
    }


@app.post("/predict")
async def legacy_predict(
    image: UploadFile = File(...),
    crop: Optional[str] = Form(None),
):
    """
    Backward-compatible endpoint that returns the same JSON shape as the
    old Flask app.py.  Used by the Node.js /api/ai/detect route.
    """
    tmp_path: Optional[str] = None
    try:
        contents = await image.read()
        if len(contents) < 100:
            return JSONResponse(status_code=400, content={"error": "No image uploaded"})

        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = await run_in_threadpool(_legacy_predict_sync, contents, tmp_path, crop)
        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# GET / — Root info
# ═══════════════════════════════════════════════════════════════════════════
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "message": "Crop Health Diagnostic Service v2",
        "version": "2.0.0",
        "docs": "/docs",
        "supported_features": [
            "10_stage_diagnostic_pipeline",
            "69_condition_knowledge_base",
            "rule_based_recommendation_engine",
            "severity_estimation",
            "pest_detection",
            "uncertainty_rejection",
            "legacy_cnn_fallback",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"[v2] Starting on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
