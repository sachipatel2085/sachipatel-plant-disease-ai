"""
Independent Context-Aware Rule-Based Recommendation Engine.
Evaluates crop, condition, severity tier, growth stage, plant part, location, and weather conditions
to generate verified agronomic advisory prescriptions compliant with CIB&RC and ICAR guidelines.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .conditions_db import AgriculturalKnowledgeBase
from .chemical_registry import (
    validate_chemical_safety,
    get_recommended_chemicals,
    get_recommended_biologicals,
    APPROVED_REGISTRY
)

# Gujarat Agro-Climatic Regional Contexts
GUJARAT_REGIONAL_PROFILES = {
    "banaskantha": {
        "hub_name": "Deesa-Banaskantha Potato & Arid Belt",
        "soil_type": "Light Sandy-Loam (Vulnerable to Scab & fast moisture percolation)",
        "weather_risk": "Winter cold fog (Late Blight trigger) & extreme summer heat (>42°C)",
        "key_practices": "Frequent light drip irrigations to prevent scab; dehaulming 10 days before harvest."
    },
    "sabarkantha": {
        "hub_name": "Himatnagar-Sabarkantha Potato & Cucurbit Hub",
        "soil_type": "Sandy-Loam to Medium Black Soils",
        "weather_risk": "Moderate winter, rapid spring warming (Sucking pest & virus vector surge)",
        "key_practices": "Yellow sticky traps; silver reflective mulching for watermelon."
    },
    "anand": {
        "hub_name": "Charotar / Anand-Kheda Cucurbit Belt",
        "soil_type": "Deep Alluvial Sandy Loam (Goradu Soil)",
        "weather_risk": "High humidity spring mornings (Downy Mildew & Fruit Fly risk)",
        "key_practices": "Sub-surface drip irrigation; cue-lure pheromone mass trapping."
    },
    "kheda": {
        "hub_name": "Kheda Riverbed & Drip Melon Zone",
        "soil_type": "Alluvial Loam / Riverbed Fine Sand",
        "weather_risk": "Spring dew & riverbed humidity (Gummy stem & Downy mildew risk)",
        "key_practices": "Raised bed plastic mulch; avoid vine-water contact."
    },
    "saurashtra": {
        "hub_name": "Saurashtra Watermelon & Horticultural Tract (Rajkot/Bhavnagar)",
        "soil_type": "Medium Black to Calcareous Soil",
        "weather_risk": "High summer heat and coastal winds (Spider mites & Blossom End Rot)",
        "key_practices": "Foliar calcium nitrate applications; windbreak borders."
    }
}

class RecommendationEngine:
    """Generates context-aware, severity-calibrated agricultural advisory prescriptions."""

    def __init__(self, kb_path: Optional[Path] = None):
        self.kb = AgriculturalKnowledgeBase(kb_path)

    def generate_advisory(
        self,
        crop: str,
        condition_id_or_name: str,
        severity: str = "MODERATE",
        growth_stage: str = "Vegetative",
        plant_part: str = "leaf",
        location: str = "Banaskantha",
        weather: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive, context-aware agricultural advisory.
        Inputs:
            - crop: "Potato" or "Watermelon"
            - condition_id_or_name: ID or common name
            - severity: "HEALTHY", "MILD", "MODERATE", "SEVERE", "CRITICAL"
            - growth_stage: e.g. "Vegetative", "Flowering", "Tuber Initiation", "Harvest"
            - plant_part: e.g. "leaf", "tuber", "fruit"
            - location: Gujarat district name
            - weather: optional dict e.g. {"temp_c": 18, "humidity_pct": 92, "foggy": True}
        """
        clean_crop = crop.strip().capitalize()
        norm_severity = severity.strip().upper()
        if norm_severity not in ("HEALTHY", "MILD", "MODERATE", "SEVERE", "CRITICAL"):
            norm_severity = "MODERATE"

        # 1. Lookup condition from knowledge base
        cond_rec = self.kb.get_condition(condition_id_or_name)
        if not cond_rec:
            cond_rec = self.kb.get_by_name(condition_id_or_name, crop=crop)

        if not cond_rec:
            return {
                "error": True,
                "message": f"Condition '{condition_id_or_name}' not found in certified agricultural database."
            }

        cond_name = cond_rec["condition_name"]
        is_healthy = "healthy" in cond_name.lower() or norm_severity == "HEALTHY"
        is_physiological = "physiological" in cond_rec["category"].lower() or "nutritional" in cond_rec["category"].lower()

        # 2. Regional Gujarat Context
        loc_key = location.strip().lower()
        regional_context = GUJARAT_REGIONAL_PROFILES.get(loc_key, {
            "hub_name": f"{location} Agricultural Zone",
            "soil_type": "Standard Gujarat Arable Soil",
            "weather_risk": "Local seasonal weather fluctuations",
            "key_practices": "Standard ICAR package of practices."
        })

        # 3. Weather Alert Evaluation
        weather_alert = self._evaluate_weather_risk(cond_name, weather)

        # 4. Determine Treatment Strategy based on Severity Constraints
        # CONSTRAINT 1: Healthy plants -> Strictly NO chemicals
        if is_healthy:
            return self._build_healthy_advisory(clean_crop, cond_rec, regional_context, growth_stage)

        # CONSTRAINT 2: Abiotic / Physiological / Nutritional -> Strictly NO pesticides
        if is_physiological:
            return self._build_abiotic_advisory(clean_crop, cond_rec, norm_severity, regional_context, growth_stage)

        # CONSTRAINT 3: Biotic Pest / Disease -> Calibrate by Severity
        advisory = self._build_biotic_advisory(
            crop=clean_crop,
            cond_rec=cond_rec,
            severity=norm_severity,
            growth_stage=growth_stage,
            plant_part=plant_part,
            regional_context=regional_context,
            weather_alert=weather_alert
        )

        return advisory

    def _evaluate_weather_risk(self, condition_name: str, weather: Optional[Dict[str, Any]]) -> Optional[str]:
        """Check for explosive epidemic weather triggers (e.g. Late blight fog)."""
        if not weather:
            return None

        temp = weather.get("temp_c")
        rh = weather.get("humidity_pct")
        foggy = weather.get("foggy", False)

        if "Late Blight" in condition_name and (foggy or (rh and rh > 85 and temp and 10 <= temp <= 20)):
            return (
                "CRITICAL WEATHER ALERT: Dense fog / high humidity (>85%) and cool temperatures (10-20°C) "
                "trigger explosive Late Blight secondary spread. Immediate systemic protective intervention required."
            )

        if "Downy Mildew" in condition_name and rh and rh > 85:
            return (
                "WEATHER TRIGGER ALERT: High relative humidity (>85%) facilitates rapid oomycete sporulation. "
                "Withhold overhead/flood irrigation; enforce canopy aeration."
            )

        if "Spider Mites" in condition_name and temp and temp > 35:
            return (
                "WEATHER TRIGGER ALERT: High heat (>35°C) and low humidity accelerate spider mite reproduction cycles. "
                "Scout abaxial leaf surfaces daily."
            )

        return None

    def _build_healthy_advisory(
        self,
        crop: str,
        cond_rec: Dict[str, Any],
        regional_context: Dict[str, Any],
        growth_stage: str
    ) -> Dict[str, Any]:
        """Prescription for healthy crops with ZERO chemical application."""
        return {
            "crop": crop,
            "condition_name": cond_rec["condition_name"],
            "scientific_name": cond_rec["scientific_name"],
            "status": "HEALTHY",
            "severity_tier": "HEALTHY",
            "diagnosis_summary": (
                f"Your {crop} crop is healthy and vigorous with no visual signs of pathological infection "
                f"or insect pest damage at the {growth_stage} stage."
            ),
            "symptoms": "Turgid, deep green uniform foliage, clean venation, intact stems, unblemished fruit/tubers.",
            "possible_causes": "Optimal agronomic management and balanced soil moisture.",
            "preventive_measures": [
                f"Continue regular scheduled fertigation suited for {crop} {growth_stage} stage.",
                "Erect yellow and blue sticky traps (15-20 traps/acre) for prophylactic insect vector monitoring.",
                f"Regional Practice ({regional_context['hub_name']}): {regional_context['key_practices']}"
            ],
            "organic_or_biological_options": [
                {
                    "name": "Soil Health Maintenance",
                    "application": "Apply well-decomposed FYM enriched with Trichoderma viride during inter-cultivation."
                }
            ],
            "chemical_treatments": [],
            "chemical_restriction_note": "PESTICIDE RESTRICTION NOTICE: Agrochemical application is strictly prohibited on healthy plants. Prophylactic chemical spraying wastes farm capital and induces pesticide resistance.",
            "safety_precautions": [
                "Avoid unnecessary human/machinery movement across fields when foliage is wet with morning dew."
            ],
            "follow_up_recommendations": [
                "Perform routine field walk inspection every 5 to 7 days.",
                "Inspect lower canopy leaves and shoot tips for early sucking pest arrivals."
            ],
            "regional_context": regional_context
        }

    def _build_abiotic_advisory(
        self,
        crop: str,
        cond_rec: Dict[str, Any],
        severity: str,
        regional_context: Dict[str, Any],
        growth_stage: str
    ) -> Dict[str, Any]:
        """Prescription for physiological disorders or nutritional deficiencies (strictly NO pesticides)."""
        cond_name = cond_rec["condition_name"]

        # Tailored remedial fertigation
        remedial_measures = []
        if "Blossom End Rot" in cond_name or "Calcium" in cond_name:
            remedial_measures.append({
                "nutrient_treatment": "Foliar spray of Chelated Calcium (Ca-EDTA) @ 1.5 g/L or Calcium Nitrate @ 4.0 g/L water.",
                "fertigation_adjustment": "Drip apply Calcium Nitrate @ 2.5 kg/acre weekly during fruit expansion. Avoid ammonium-based nitrogen fertilizers which inhibit calcium uptake.",
                "irrigation_adjustment": "Maintain steady, uniform soil moisture. Avoid alternate severe drying and heavy soaking."
            })
        elif "Nitrogen" in cond_name:
            remedial_measures.append({
                "nutrient_treatment": "Foliar spray of 19-19-19 water-soluble fertilizer @ 5.0 g/L water.",
                "fertigation_adjustment": "Drip fertigate Urea @ 5-8 kg/acre in 2 splits."
            })
        elif "Potassium" in cond_name:
            remedial_measures.append({
                "nutrient_treatment": "Foliar spray of Potassium Sulfate (0-0-50) @ 5.0 g/L water to halt leaf margin scorching.",
                "fertigation_adjustment": "Drip apply Muriate of Potash (MOP) or Sulfate of Potash (SOP) @ 5 kg/acre."
            })
        elif "Magnesium" in cond_name:
            remedial_measures.append({
                "nutrient_treatment": "Foliar spray of Magnesium Sulfate (Epsom salt) @ 5.0 g/L water to restore chlorophyll synthesis.",
                "fertigation_adjustment": "Apply 10 kg/acre Magnesium Sulfate to soil."
            })
        elif "Sunscald" in cond_name:
            remedial_measures.append({
                "cultural_treatment": "Cover sun-exposed melons immediately with clean dry paddy straw or vine leaves.",
                "canopy_management": "Maintain healthy foliar canopy; do not over-prune vine runners."
            })
        elif "Cracking" in cond_name:
            remedial_measures.append({
                "irrigation_treatment": "Avoid irregular irrigation shocks. Water regularly in small increments rather than heavy soaking after dry periods.",
                "mulching": "Maintain organic or plastic mulch to stabilize soil moisture tension."
            })

        return {
            "crop": crop,
            "condition_name": cond_name,
            "scientific_name": cond_rec["scientific_name"],
            "status": "ABIOTIC_DISORDER",
            "severity_tier": severity,
            "diagnosis_summary": (
                f"Identified as {cond_name} ({cond_rec['category']}) with {severity} severity at the {growth_stage} stage. "
                f"This is an abiotic nutritional/physiological imbalance, NOT an infectious pest or pathogen."
            ),
            "symptoms": cond_rec["symptoms"],
            "possible_causes": cond_rec["disease_causes_or_pest_biology"],
            "preventive_measures": [
                cond_rec["preventive_measures"],
                f"Soil Context ({regional_context['soil_type']}): Ensure regularized moisture retention.",
                "Conduct annual soil test analysis to calibrate base fertilizer dosages."
            ],
            "organic_or_biological_options": [
                {
                    "name": "Soil Humic & Mycorrhizal Application",
                    "application": "Apply VAM (Vesicular Arbuscular Mycorrhiza) @ 4 kg/acre to boost root surface absorption."
                }
            ],
            "chemical_treatments": [],
            "remedial_fertigation_options": remedial_measures,
            "chemical_restriction_note": "PESTICIDE RESTRICTION NOTICE: Agrochemical pesticides (fungicides, insecticides) are completely ineffective and strictly prohibited for abiotic disorders. Do not spray pesticides.",
            "safety_precautions": [
                "Spray foliar micronutrients during early morning or late afternoon to avoid leaf scorch."
            ],
            "follow_up_recommendations": [
                "Inspect new leaf flushes and developing fruit 5 to 7 days after corrective fertigation.",
                "Monitor drip line discharge uniformity across farm plots."
            ],
            "regional_context": regional_context
        }

    def _build_biotic_advisory(
        self,
        crop: str,
        cond_rec: Dict[str, Any],
        severity: str,
        growth_stage: str,
        plant_part: str,
        regional_context: Dict[str, Any],
        weather_alert: Optional[str]
    ) -> Dict[str, Any]:
        """Prescription for infectious disease or insect pest, rigorously calibrated by severity."""
        cond_name = cond_rec["condition_name"]

        # Fetch certified chemicals and biologicals
        chem_matches = get_recommended_chemicals(cond_name, crop)
        bio_matches = get_recommended_biologicals(cond_name)

        # SEVERITY CONSTRAINT:
        # Mild severity -> Prioritize biological and cultural; withhold or flag chemical as backup only
        approved_chemicals_to_show = []
        treatment_strategy_note = ""

        if severity == "MILD":
            treatment_strategy_note = (
                "SEVERITY STRATEGY: Mild infection detected (<10% affected tissue). "
                "Prioritize non-chemical cultural sanitation and biological controls. "
                "Chemical intervention is withheld to protect natural predators and prevent pesticide resistance."
            )
            # Only provide low-risk protective chemical option as a standby threshold option
            if chem_matches:
                standby = dict(chem_matches[0])
                standby["standby_notice"] = "STANDBY OPTION ONLY: Apply only if disease lesions spread past 10% threshold despite biological measures."
                approved_chemicals_to_show = [standby]
        elif severity == "MODERATE":
            treatment_strategy_note = (
                "SEVERITY STRATEGY: Moderate infection detected (10-25% affected tissue). "
                "Execute integrated management: combine immediate cultural rogueing with certified targeted application."
            )
            approved_chemicals_to_show = chem_matches[:2]
        else:  # SEVERE or CRITICAL
            treatment_strategy_note = (
                f"SEVERITY STRATEGY: Critical / Severe infection detected ({severity}, >25% tissue damage). "
                f"Immediate curative systemic intervention required to prevent total crop loss. "
                f"Rotate FRAC/IRAC modes of action."
            )
            approved_chemicals_to_show = chem_matches[:3]

        # Biological options formatting
        bio_options = []
        for b in bio_matches:
            bio_options.append({
                "agent": b["agent"],
                "category": b["category"],
                "application": b.get("application_methods", {}),
                "notes": b.get("notes", "")
            })
        if not bio_options and cond_rec.get("biological_management"):
            bio_options.append({
                "agent": "Certified Bio-Control Management",
                "category": "Integrated Biological Control",
                "application": {"general_application": cond_rec["biological_management"]},
                "notes": "Prioritize bio-agents to conserve natural enemies and prevent resistance."
            })

        # Chemical options formatting with verified dose and safety
        chem_options = []
        for c in approved_chemicals_to_show:
            chem_options.append({
                "active_ingredient": c["active_ingredient"],
                "trade_names": ", ".join(c.get("trade_names", [])),
                "dose_per_liter": c["dose_per_liter_water"],
                "dose_per_acre": c["dose_per_acre"],
                "phi_days": c["phi_days"],
                "application_type": c["application_type"],
                "frac_or_irac_code": c.get("frac_code") or c.get("irac_code", "N/A"),
                "notes": c.get("notes", ""),
                "standby_notice": c.get("standby_notice")
            })

        # Safety precautions
        safety_precautions = [
            "Wear standard personal protective equipment (gloves, face mask, protective apron) during spray preparation.",
            "Spray during calm early morning or evening hours; strictly avoid spraying against wind direction.",
            f"Adhere strictly to Pre-Harvest Interval (PHI): Do not harvest within {approved_chemicals_to_show[0]['phi_days'] if approved_chemicals_to_show else 7} days of chemical application."
        ]

        # Pollinator caution for watermelon
        if crop == "Watermelon" and growth_stage in ("Flowering", "Fruit Set", "Vegetative"):
            safety_precautions.append(
                "POLLINATOR PROTECTION NOTICE: Watermelon flowers require active honeybee cross-pollination. "
                "Never apply insecticides during morning foraging hours (7:00 AM - 11:00 AM)."
            )

        return {
            "crop": crop,
            "condition_name": cond_name,
            "scientific_name": cond_rec["scientific_name"],
            "status": "INFECTIOUS_CONDITION",
            "severity_tier": severity,
            "diagnosis_summary": (
                f"Confirmed {cond_name} (*{cond_rec['scientific_name']}*) on {crop} {plant_part} "
                f"with {severity} severity at {growth_stage} stage."
            ),
            "symptoms": cond_rec["symptoms"],
            "possible_causes": cond_rec["disease_causes_or_pest_biology"],
            "favourable_environmental_conditions": cond_rec["favourable_environmental_conditions"],
            "weather_alert": weather_alert,
            "treatment_strategy_note": treatment_strategy_note,
            "preventive_measures": [
                cond_rec["preventive_measures"],
                f"Regional Focus ({regional_context['hub_name']}): {regional_context['key_practices']}"
            ],
            "cultural_management": cond_rec["cultural_management"],
            "organic_or_biological_options": bio_options,
            "chemical_treatments": chem_options,
            "safety_precautions": safety_precautions,
            "follow_up_recommendations": [
                "Re-inspect treated field plots 48 to 72 hours after application to verify efficacy.",
                "If lesions continue expanding, rotate to an alternate mode of action (different FRAC/IRAC group) to prevent resistance.",
                "Record chemical application date, volume, and operator details in farm log."
            ],
            "authoritative_references": cond_rec["authoritative_references"],
            "cibrc_verified": True,
            "regional_context": regional_context
        }
