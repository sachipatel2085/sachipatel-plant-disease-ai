"""
Phase 4 Agricultural Knowledge Base & Recommendation Engine Verification Tool.
Audits the 69-condition database, validates CIB&RC chemical safety, and tests
severity-calibrated agronomic advisory scenarios across Gujarat farming belts.
"""

import sys
from pathlib import Path

# Add plant-disease-ai directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from knowledge_base.conditions_db import AgriculturalKnowledgeBase
from knowledge_base.chemical_registry import (
    validate_chemical_safety,
    BANNED_OR_RESTRICTED_CHEMICALS,
    APPROVED_REGISTRY
)
from knowledge_base.recommendation_engine import RecommendationEngine

def run_phase_4_audit():
    print("=" * 70)
    print("PHASE 4: KNOWLEDGE BASE & RECOMMENDATION ENGINE AUDIT")
    print("=" * 70)

    # 1. Knowledge Base Audit
    print("\n[Audit 1/4] Auditing Agricultural Knowledge Base Structure & Integrity...")
    kb = AgriculturalKnowledgeBase()
    assert kb.count() == 69, f"Expected 69 conditions, got {kb.count()}"
    print(f"  -> Total verified conditions loaded: {kb.count()}")

    required_fields = [
        "condition_id", "crop", "condition_name", "scientific_name", "category",
        "symptoms", "affected_plant_parts", "disease_causes_or_pest_biology",
        "favourable_environmental_conditions", "transmission_mechanism",
        "diagnostic_characteristics", "similar_looking_conditions",
        "cultural_management", "biological_management", "chemical_management",
        "preventive_measures", "authoritative_references", "verification_status"
    ]

    for cond_id, rec in kb.database.items():
        for field in required_fields:
            assert field in rec, f"Missing field '{field}' in condition {cond_id}"
            assert str(rec[field]).strip(), f"Empty field '{field}' in condition {cond_id}"

    print(f"  -> All 18 mandatory agricultural fields verified across 69 conditions: PASS")

    # 2. Chemical Registry & Safety Blacklist Audit
    print("\n[Audit 2/4] Auditing CIB&RC Agrochemical Registry & Banned Substance Filters...")
    # Test banned substances
    for banned in ["endosulfan", "monocrotophos", "methyl parathion", "carbofuran"]:
        is_ok, msg = validate_chemical_safety(banned, "potato")
        assert is_ok is False, f"Banned chemical '{banned}' was incorrectly approved!"
        assert "SAFETY VIOLATION" in msg
    print(f"  -> Banned/restricted pesticide filter verified (100% blocked): PASS")

    # Test approved chemicals
    is_ok, msg = validate_chemical_safety("mancozeb_75_wp", "potato")
    assert is_ok is True
    is_ok, msg = validate_chemical_safety("dimethomorph_50_wp", "watermelon")
    assert is_ok is True
    print(f"  -> CIB&RC approved formulation registry verified: PASS")

    # 3. Clinical Scenario Audits
    print("\n[Audit 3/4] Testing Severity-Calibrated Agronomic Advisory Scenarios...")
    engine = RecommendationEngine()

    # Scenario A: Healthy Plant -> ZERO chemicals constraint
    print("  -> Scenario A: Healthy Potato Canopy...")
    adv_a = engine.generate_advisory(crop="Potato", condition_id_or_name="potato__healthy_potato_plant", severity="HEALTHY")
    assert adv_a["status"] == "HEALTHY"
    assert len(adv_a["chemical_treatments"]) == 0, "Chemical was prescribed for a healthy plant!"
    assert "PESTICIDE RESTRICTION NOTICE" in adv_a["chemical_restriction_note"]
    print("     [PASS] Healthy plant: Zero chemicals prescribed, restriction note present.")

    # Scenario B: Mild Early Blight in Banaskantha -> Biological prioritized
    print("  -> Scenario B: Mild Early Blight in Banaskantha...")
    adv_b = engine.generate_advisory(
        crop="Potato",
        condition_id_or_name="potato__early_blight",
        severity="MILD",
        location="Banaskantha"
    )
    assert adv_b["severity_tier"] == "MILD"
    assert "Prioritize non-chemical cultural sanitation" in adv_b["treatment_strategy_note"]
    assert len(adv_b["organic_or_biological_options"]) > 0
    print("     [PASS] Mild disease: Biological and cultural methods prioritized.")

    # Scenario C: Severe Late Blight under Foggy Weather -> Curative + Alert
    print("  -> Scenario C: Severe Late Blight in Banaskantha under Cold Fog...")
    adv_c = engine.generate_advisory(
        crop="Potato",
        condition_id_or_name="potato__late_blight",
        severity="SEVERE",
        location="Banaskantha",
        weather={"temp_c": 14, "humidity_pct": 95, "foggy": True}
    )
    assert adv_c["severity_tier"] == "SEVERE"
    assert adv_c["weather_alert"] is not None
    assert "CRITICAL WEATHER ALERT" in adv_c["weather_alert"]
    assert len(adv_c["chemical_treatments"]) >= 1
    # Check exact dose per liter and acre
    top_chem = adv_c["chemical_treatments"][0]
    assert "dose_per_liter" in top_chem and "phi_days" in top_chem
    print(f"     [PASS] Severe blight: Triggered weather alert and prescribed {top_chem['active_ingredient']} (Dose: {top_chem['dose_per_liter']}, PHI: {top_chem['phi_days']}d).")

    # Scenario D: Blossom End Rot on Watermelon -> ZERO pesticides, Calcium prescribed
    print("  -> Scenario D: Watermelon Blossom End Rot (Abiotic Disorder)...")
    adv_d = engine.generate_advisory(
        crop="Watermelon",
        condition_id_or_name="watermelon__blossom_end_rot",
        severity="MODERATE",
        growth_stage="Fruit Development",
        plant_part="fruit",
        location="Anand"
    )
    assert adv_d["status"] == "ABIOTIC_DISORDER"
    assert len(adv_d["chemical_treatments"]) == 0, "Pesticides were prescribed for an abiotic disorder!"
    assert len(adv_d["remedial_fertigation_options"]) >= 1
    fert_opt = adv_d["remedial_fertigation_options"][0]
    assert "Calcium" in fert_opt["nutrient_treatment"]
    print("     [PASS] Blossom End Rot: Zero pesticides, prescribed remedial Calcium fertigation.")

    # Scenario E: Watermelon Sucking Pest & Pollinator Protection
    print("  -> Scenario E: Melon Aphids during Flowering Stage in Kheda...")
    adv_e = engine.generate_advisory(
        crop="Watermelon",
        condition_id_or_name="watermelon__melon_aphid",
        severity="MODERATE",
        growth_stage="Flowering",
        location="Kheda"
    )
    assert any("POLLINATOR PROTECTION NOTICE" in p for p in adv_e["safety_precautions"])
    print("     [PASS] Pollinator protection guardrail active during flowering.")

    # 4. Regional Gujarat Agronomy Verification
    print("\n[Audit 4/4] Verifying Gujarat Regional Soil & Climate Contexts...")
    for dist in ["Banaskantha", "Sabarkantha", "Anand", "Kheda", "Saurashtra"]:
        adv_loc = engine.generate_advisory(crop="Potato", condition_id_or_name="potato__early_blight", location=dist)
        assert any(dist.lower() in p.lower() or "regional" in p.lower() for p in adv_loc["preventive_measures"])
    print("  -> Regional contextualization verified across all Gujarat key districts: PASS")

    print("\n" + "-" * 70)
    print("[SUCCESS] All Phase 4 Knowledge Base and Recommendation Engine tests passed 100%.")
    print("=" * 70)

if __name__ == "__main__":
    run_phase_4_audit()
