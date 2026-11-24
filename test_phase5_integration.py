"""
Phase 5 End-to-End Integration Test Suite.
Validates the complete FastAPI diagnostic service:
1. Health & model status endpoints
2. 69-condition knowledge base queries
3. Clinical profile retrieval
4. Context-aware advisory generation (with Gujarat regional profiles)
5. Multi-stage diagnostic pipeline execution via POST /api/v2/diagnose
6. Legacy compatibility via POST /predict
"""

import sys
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Ensure plant-disease-ai root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def create_synthetic_leaf(path: Path):
    """Generate a synthetic leaf image with texture to pass validation."""
    img = Image.new("RGB", (256, 256), color=(34, 139, 34)) # Forest green
    draw = ImageDraw.Draw(img)
    # Add leaf vein structures & spots
    draw.line([(128, 20), (128, 236)], fill=(20, 100, 20), width=3)
    for y in range(40, 220, 25):
        draw.line([(128, y), (50, y - 20)], fill=(25, 110, 25), width=2)
        draw.line([(128, y), (206, y - 20)], fill=(25, 110, 25), width=2)
    # Add concentric early blight-like spots
    draw.ellipse([(80, 80), (140, 140)], fill=(80, 50, 20))
    draw.ellipse([(90, 90), (130, 130)], fill=(50, 30, 10))
    draw.ellipse([(100, 100), (120, 120)], fill=(30, 15, 5))
    img.save(path, "JPEG", quality=90)

def run_all_tests():
    print("=" * 70)
    print("PHASE 5: PRODUCTION BACKEND & INTEGRATION AUDIT")
    print("=" * 70)

    with TestClient(app) as client:
        # -----------------------------------------------------------------------
        # TEST 1: Service Root & Health
        # -----------------------------------------------------------------------
        print("\n[TEST 1] Auditing Root and Health Check Endpoints...")
        res_root = client.get("/")
        assert res_root.status_code == 200, f"Root returned {res_root.status_code}"
        root_data = res_root.json()
        assert "version" in root_data and root_data["version"] == "2.0.0"
        print("  -> GET / : OK (v2.0.0)")

        res_health = client.get("/api/v2/health")
        assert res_health.status_code == 200, f"Health returned {res_health.status_code}"
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["models"]["diagnostic_pipeline"] is True
        assert health_data["models"]["recommendation_engine"] is True
        assert health_data["models"]["knowledge_base_conditions"] == 69
        print(f"  -> GET /api/v2/health : OK ({health_data['models']['knowledge_base_conditions']} conditions verified)")

        # -----------------------------------------------------------------------
        # TEST 2: Knowledge Base Listing (69 conditions)
        # -----------------------------------------------------------------------
        print("\n[TEST 2] Auditing Knowledge Base Listing (/api/v2/conditions)...")
        res_all = client.get("/api/v2/conditions")
        assert res_all.status_code == 200
        all_data = res_all.json()
        assert all_data["total"] == 69, f"Expected 69 conditions, got {all_data['total']}"
        print(f"  -> Total conditions registered: {all_data['total']}")

        res_pot = client.get("/api/v2/conditions?crop=potato")
        assert res_pot.status_code == 200
        pot_data = res_pot.json()
        print(f"  -> Potato conditions: {pot_data['total']}")
        assert pot_data["total"] >= 27

        res_wm = client.get("/api/v2/conditions?crop=watermelon")
        assert res_wm.status_code == 200
        wm_data = res_wm.json()
        print(f"  -> Watermelon conditions: {wm_data['total']}")
        assert wm_data["total"] >= 26

        # -----------------------------------------------------------------------
        # TEST 3: Clinical Profile Retrieval
        # -----------------------------------------------------------------------
        print("\n[TEST 3] Auditing Single Condition Profile (/api/v2/conditions/{id})...")
        res_profile = client.get("/api/v2/conditions/potato__early_blight")
        assert res_profile.status_code == 200, f"Expected 200, got {res_profile.status_code}: {res_profile.text}"
        profile_data = res_profile.json()["condition"]
        assert profile_data["condition_name"] == "Early Blight"
        assert "Alternaria solani" in profile_data["scientific_name"]
        assert len(profile_data["chemical_management"]) > 0
        assert len(profile_data["preventive_measures"]) > 0
        print(f"  -> potato__early_blight: {profile_data['condition_name']} ({profile_data['scientific_name']})")
        print(f"     Chemicals: {len(profile_data['chemical_management'])} chars, Prevention rules: {len(profile_data['preventive_measures'])} chars")

        # -----------------------------------------------------------------------
        # TEST 4: Context-Aware Advisory Generation with Gujarat Profile
        # -----------------------------------------------------------------------
        print("\n[TEST 4] Auditing Context-Aware Advisory (/api/v2/advisory/{id})...")
        res_adv = client.get("/api/v2/advisory/potato__late_blight?crop=potato&severity=severe&location=banaskantha")
        assert res_adv.status_code == 200, f"Expected 200, got {res_adv.status_code}: {res_adv.text}"
        adv_data = res_adv.json()["advisory"]
        assert adv_data["crop"].lower() == "potato"
        assert adv_data["severity_tier"].lower() == "severe"
        assert "regional_context" in adv_data and adv_data["regional_context"] is not None
        assert "Deesa-Banaskantha" in adv_data["regional_context"]["hub_name"]
        print(f"  -> Late Blight Severe in Banaskantha:")
        print(f"     Regional Hub: {adv_data['regional_context']['hub_name']}")
        print(f"     Strategy: {adv_data.get('treatment_strategy_note', '')[:70]}...")
        print(f"     Chemical options: {len(adv_data.get('chemical_treatments', []))}")

        # -----------------------------------------------------------------------
        # TEST 5: Full 10-Stage Diagnostic Pipeline (POST /api/v2/diagnose)
        # -----------------------------------------------------------------------
        print("\n[TEST 5] Auditing Full Diagnostic Pipeline (POST /api/v2/diagnose)...")
        test_leaf_path = PROJECT_ROOT / "test_leaf_sample.jpg"
        create_synthetic_leaf(test_leaf_path)

        with open(test_leaf_path, "rb") as f:
            res_diag = client.post(
                "/api/v2/diagnose",
                files={"image": ("test_leaf.jpg", f, "image/jpeg")},
                data={"crop": "potato", "district": "banaskantha", "growth_stage": "vegetative"}
            )

        assert res_diag.status_code == 200, f"Diagnose returned {res_diag.status_code}: {res_diag.text}"
        diag_data = res_diag.json()

        print(f"  -> Primary Disease: {diag_data['disease']}")
        print(f"  -> Calibrated Confidence: {diag_data['confidence']}%")
        print(f"  -> Target Crop: {diag_data['targetCrop']}")
        print(f"  -> Probable candidates returned: {len(diag_data['probable_diseases'])}")
        print(f"  -> Source: {diag_data['source']}")
        print(f"  -> Pipeline status: {diag_data['v2']['pipeline_status']}")
        print(f"  -> Organ: {diag_data['v2']['organ_identification']['predicted_organ']}")
        print(f"  -> Severity Tier: {diag_data['v2']['severity']['tier']} ({diag_data['v2']['severity']['affected_percent']}%)")

        # Verify response contains the frontend's expected advisory fields
        assert "advisory" in diag_data
        assert "symptoms" in diag_data["advisory"]
        assert "cause" in diag_data["advisory"]
        assert "chemicalTreatment" in diag_data["advisory"]
        assert "organicSolution" in diag_data["advisory"]
        assert "prevention" in diag_data["advisory"]

        # -----------------------------------------------------------------------
        # TEST 6: Legacy Compatibility Endpoint (POST /predict)
        # -----------------------------------------------------------------------
        print("\n[TEST 6] Auditing Legacy Compatibility (POST /predict)...")
        with open(test_leaf_path, "rb") as f:
            res_legacy = client.post(
                "/predict",
                files={"image": ("test_leaf.jpg", f, "image/jpeg")},
                data={"crop": "potato"}
            )

        assert res_legacy.status_code == 200, f"Legacy predict returned {res_legacy.status_code}"
        legacy_data = res_legacy.json()
        assert legacy_data["success"] is True
        assert "disease" in legacy_data
        assert "confidence" in legacy_data
        assert "top_predictions" in legacy_data
        print(f"  -> Legacy response disease: {legacy_data['disease']} ({legacy_data['confidence']}%)")
        print(f"  -> Top predictions count: {len(legacy_data['top_predictions'])}")

        # Clean up test artifact
        if test_leaf_path.exists():
            test_leaf_path.unlink()

        print("\n" + "=" * 70)
        print("ALL PHASE 5 INTEGRATION AUDITS PASSED WITH ZERO ERRORS!")
        print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
