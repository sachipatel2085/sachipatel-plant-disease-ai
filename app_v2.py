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
from pathlib import Path
from typing import Optional, List, Dict, Any

# Suppress TF noise before anything else
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
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


# ---------------------------------------------------------------------------
# Initialize application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Health Diagnostic Service",
    version="2.0.0",
    description="Multi-model AI diagnostic pipeline for Potato & Watermelon crops.",
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
            kb_path=kb_path if kb_path.exists() else None
        )
        print("[v2] AgriculturalKnowledgeBase initialized.")
    return _knowledge_base


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    _load_legacy_model()
    # Pre-warm the lazy singletons so the first real request is fast
    try:
        _get_knowledge_base()
        _get_recommendation_engine()
        _get_pipeline()
    except Exception as e:
        print(f"[v2] Pre-warm warning (non-fatal): {e}")
    print("[v2] Crop Health Diagnostic Service v2 ready.")


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v2/health
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/v2/health")
async def health_check():
    kb = _get_knowledge_base()
    conditions_count = len(kb.conditions) if kb and hasattr(kb, "conditions") else 0
    return {
        "status": "ok",
        "version": "2.0.0",
        "models": {
            "diagnostic_pipeline": _pipeline is not None,
            "recommendation_engine": _recommendation_engine is not None,
            "knowledge_base_conditions": conditions_count,
            "legacy_cnn": LEGACY_MODEL is not None,
        },
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

        # 2. Run the 10-stage diagnostic pipeline
        pipeline = _get_pipeline()
        diagnosis_report = pipeline.diagnose(
            image_path_or_pil=tmp_path,
            user_crop_hint=crop
        )

        # 3. Extract primary results
        pipeline_status = diagnosis_report.get("pipeline_status", "unknown")
        final_dx = diagnosis_report.get("final_diagnosis")
        candidates = diagnosis_report.get("differential_candidates", [])
        severity = diagnosis_report.get("severity_assessment", {})
        pest_detections = diagnosis_report.get("pest_detections", [])
        crop_id = diagnosis_report.get("crop_identification", {})
        organ_id = diagnosis_report.get("organ_identification", {})

        # 4. Build a list of conditions to generate advisories for
        condition_ids_for_advisory = []
        if final_dx and final_dx.get("condition_id"):
            condition_ids_for_advisory.append(final_dx["condition_id"])
        for cand in candidates:
            cid = cand.get("condition_id")
            if cid and cid not in condition_ids_for_advisory:
                condition_ids_for_advisory.append(cid)

        # 5. Generate advisory for the top condition
        engine = _get_recommendation_engine()
        kb = _get_knowledge_base()
        primary_advisory = None
        all_advisories = []

        for cid in condition_ids_for_advisory:
            try:
                adv = engine.generate_advisory(
                    crop=crop or crop_id.get("predicted_crop", "potato"),
                    condition_id=cid,
                    severity_tier=severity.get("severity_tier", "moderate"),
                    growth_stage=growth_stage or "vegetative",
                    plant_part=organ_id.get("predicted_organ", "leaf"),
                    location=district or location or None,
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
            detected_confidence = round(final_dx.get("calibrated_confidence", 0.0) * 100, 2)

        # Enrich from KB
        if final_dx and final_dx.get("condition_id"):
            profile = kb.get_condition(final_dx["condition_id"])
            if profile:
                scientific_name = profile.get("scientific_name", "")
                category = profile.get("category", "")
                condition_severity = profile.get("severity_default", "medium")
                symptoms_list = profile.get("symptoms", [])
                cause_text = profile.get("favourable_conditions", "")
                chemical_text = "\n".join(profile.get("chemical_management", []))
                organic_text = "\n".join(profile.get("biological_management", []))
                prevention_text = "\n".join(profile.get("preventive_measures", []))

        # Augment from primary advisory (which may add weather-specific, severity-calibrated text)
        if primary_advisory:
            if primary_advisory.get("immediate_actions"):
                prevention_text = "\n".join(primary_advisory["immediate_actions"]) + "\n\n" + prevention_text
            if primary_advisory.get("chemical_management"):
                chem_items = primary_advisory["chemical_management"]
                if isinstance(chem_items, list):
                    for item in chem_items:
                        if isinstance(item, dict):
                            recommended_products.append({
                                "name": item.get("product", item.get("name", "")),
                                "dosage": item.get("dose", item.get("dosage", "")),
                                "applicationNotes": item.get("notes", item.get("applicationNotes", "")),
                                "productType": "chemical",
                            })
                        elif isinstance(item, str):
                            chemical_text += "\n" + item
            if primary_advisory.get("biological_management"):
                bio_items = primary_advisory["biological_management"]
                if isinstance(bio_items, list):
                    for item in bio_items:
                        if isinstance(item, dict):
                            recommended_products.append({
                                "name": item.get("product", item.get("name", "")),
                                "dosage": item.get("dose", item.get("dosage", "")),
                                "applicationNotes": item.get("notes", item.get("applicationNotes", "")),
                                "productType": "organic",
                            })
                        elif isinstance(item, str):
                            organic_text += "\n" + item
            if primary_advisory.get("cultural_management"):
                cult = primary_advisory["cultural_management"]
                if isinstance(cult, list):
                    prevention_text += "\n" + "\n".join(
                        [c if isinstance(c, str) else str(c) for c in cult]
                    )
            if primary_advisory.get("expert_warnings"):
                cause_text += "\n\n⚠️ " + "\n⚠️ ".join(primary_advisory["expert_warnings"])

        # Build enriched probable_diseases list (what the frontend expects)
        probable_diseases = []
        for idx, cand in enumerate(candidates):
            cand_id = cand.get("condition_id", "")
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
                cand_chem = "\n".join(cand_profile.get("chemical_management", []))
                cand_organic = "\n".join(cand_profile.get("biological_management", []))
                cand_prevention = "\n".join(cand_profile.get("preventive_measures", []))
                cand_cause = cand_profile.get("favourable_conditions", "")

            if cand_advisory and isinstance(cand_advisory, dict):
                if cand_advisory.get("chemical_management"):
                    for item in cand_advisory["chemical_management"]:
                        if isinstance(item, dict):
                            cand_products.append({
                                "name": item.get("product", item.get("name", "")),
                                "dosage": item.get("dose", item.get("dosage", "")),
                                "applicationNotes": item.get("notes", ""),
                                "productType": "chemical",
                            })
                if cand_advisory.get("biological_management"):
                    for item in cand_advisory["biological_management"]:
                        if isinstance(item, dict):
                            cand_products.append({
                                "name": item.get("product", item.get("name", "")),
                                "dosage": item.get("dose", item.get("dosage", "")),
                                "applicationNotes": item.get("notes", ""),
                                "productType": "organic",
                            })

            probable_diseases.append({
                "disease": cand_id or cand.get("condition_name", "Unknown"),
                "displayName": cand.get("condition_name", "Unknown"),
                "confidence": round(cand.get("calibrated_confidence", cand.get("raw_score", 0.0)) * 100, 2),
                "category": cand_profile.get("category", "") if cand_profile else "",
                "severity": cand_profile.get("severity_default", "medium") if cand_profile else "medium",
                "scientificName": cand_profile.get("scientific_name", "") if cand_profile else "",
                "symptoms": cand_syms[:5] if cand_syms else [
                    "Irregular spots or discoloration on leaf surface",
                    "Signs of plant stress or foliage decay",
                ],
                "advisory": {
                    "name": cand.get("condition_name", ""),
                    "scientificName": cand_profile.get("scientific_name", "") if cand_profile else "",
                    "category": cand_profile.get("category", "") if cand_profile else "",
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
            "tier": severity.get("severity_tier", "unknown"),
            "affected_percent": round(severity.get("affected_percent", 0.0), 1),
            "description": severity.get("severity_description", ""),
        }

        # === The unified response ===
        response = {
            # Fields the frontend currently reads (backward compatible)
            "disease": detected_name,
            "rawDisease": final_dx.get("condition_id", "") if final_dx else "",
            "confidence": detected_confidence,
            "targetCrop": crop or crop_id.get("predicted_crop", ""),
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
                "crop_identification": crop_id,
                "organ_identification": organ_id,
                "severity": severity_info,
                "pests": pests_found,
                "uncertainty": diagnosis_report.get("uncertainty", {}),
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
    conditions = kb.conditions if hasattr(kb, "conditions") else []

    results = []
    for cond in conditions:
        if crop and crop.lower() not in [c.lower() for c in cond.get("crops", [])]:
            continue
        if category and category.lower() != cond.get("category", "").lower():
            continue
        results.append({
            "condition_id": cond.get("condition_id", ""),
            "name": cond.get("condition_name", ""),
            "scientific_name": cond.get("scientific_name", ""),
            "crops": cond.get("crops", []),
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
    location: Optional[str] = Query(None),
):
    engine = _get_recommendation_engine()
    try:
        advisory = engine.generate_advisory(
            crop=crop,
            condition_id=condition_id,
            severity_tier=severity,
            growth_stage=growth_stage,
            plant_part="leaf",
            location=location,
        )
        return {"condition_id": condition_id, "advisory": advisory}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# POST /predict  — Legacy compatibility endpoint (matches old Flask shape)
# ═══════════════════════════════════════════════════════════════════════════
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

        # First attempt: use the new pipeline
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        pipeline = _get_pipeline()
        report = pipeline.diagnose(image_path_or_pil=tmp_path, user_crop_hint=crop)

        final_dx = report.get("final_diagnosis")
        candidates = report.get("differential_candidates", [])

        if final_dx and final_dx.get("condition_name"):
            disease_name = final_dx["condition_name"]
            confidence = round(final_dx.get("calibrated_confidence", 0.0) * 100, 2)
        elif LEGACY_MODEL is not None:
            # Fallback to legacy CNN
            img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = LEGACY_MODEL.predict(img_array, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = round(float(prediction[predicted_class]) * 100, 2)
            disease_name = LEGACY_LABELS.get(str(predicted_class), "Unknown")
        else:
            disease_name = "Unknown Issue"
            confidence = 0.0

        # Build top_predictions in legacy shape
        top_preds = []
        for cand in candidates[:4]:
            top_preds.append({
                "disease": cand.get("condition_name", "Unknown"),
                "confidence": round(cand.get("calibrated_confidence", cand.get("raw_score", 0.0)) * 100, 2),
            })
        if not top_preds:
            top_preds.append({"disease": disease_name, "confidence": confidence})

        return JSONResponse(content={
            "success": True,
            "class": 0,
            "disease": disease_name,
            "confidence": confidence,
            "top_predictions": top_preds,
            "visual_symptoms": {
                "powdery_white_ratio": 0.0,
                "necrotic_dark_ratio": 0.0,
                "has_powdery_coating": False,
            },
        })

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
@app.get("/")
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
