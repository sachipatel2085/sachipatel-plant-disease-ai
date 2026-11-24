"""
CIB&RC (Central Insecticides Board & Registration Committee, Government of India) Chemical Registry.
Enforces verified active ingredients, formulations, exact dosages, pre-harvest intervals (PHI),
and strict filtering of banned and restricted pesticides.
"""

from typing import Dict, Any, List, Optional, Tuple

# Strictly Banned and Restricted Agrochemicals in India (Gazette of India / CIB&RC Notifications)
BANNED_OR_RESTRICTED_CHEMICALS = {
    "endosulfan": "Globally banned under Stockholm Convention and Supreme Court of India order.",
    "monocrotophos": "Strictly banned for use on vegetables and fruit crops due to high acute mammalian toxicity.",
    "methyl parathion": "Banned in India due to extreme neurotoxicity.",
    "carbofuran": "Banned / restricted on short-duration vegetable crops.",
    "phorate": "Banned / restricted; hazardous organophosphate.",
    "dichlorvos": "Banned for domestic/horticultural use.",
    "triazophos": "Restricted; prohibited on tender vegetables.",
    "methomyl": "Banned in India.",
    "paraquat": "Restricted non-selective herbicide; prohibited inside standing crop canopy."
}

# CIB&RC Approved Agrochemicals for Solanaceous (Potato) and Cucurbit (Watermelon) Crops
APPROVED_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- FUNGICIDES & OOMYCETICIDES ---
    "mancozeb_75_wp": {
        "chemical_id": "mancozeb_75_wp",
        "trade_names": ["Dithane M-45", "Indofil M-45"],
        "active_ingredient": "Mancozeb 75% WP",
        "category": "Broad-spectrum Contact Dithiocarbamate",
        "frac_code": "M03",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Early Blight", "Late Blight", "Downy Mildew", "Anthracnose", "Alternaria Leaf Blight"],
        "dose_per_liter_water": "2.0 - 2.5 g",
        "dose_per_acre": "600 - 800 g in 200 L water",
        "phi_days": 7,
        "application_type": "Foliar Spray (Protective)",
        "notes": "Excellent multi-site protective fungicide. Spray before disease onset or at first appearance of lesions."
    },
    "cymoxanil_8_mancozeb_64_wp": {
        "chemical_id": "cymoxanil_8_mancozeb_64_wp",
        "trade_names": ["Curzate M8", "Sectin"],
        "active_ingredient": "Cymoxanil 8% + Mancozeb 64% WP",
        "category": "Systemic + Contact Combination",
        "frac_code": "27 + M03",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Late Blight", "Downy Mildew"],
        "dose_per_liter_water": "2.5 - 3.0 g",
        "dose_per_acre": "600 g in 200 L water",
        "phi_days": 10,
        "application_type": "Foliar Spray (Curative + Protective)",
        "notes": "Fast kick-back action; stops fungal sporulation within 48 hours of infection."
    },
    "dimethomorph_50_wp": {
        "chemical_id": "dimethomorph_50_wp",
        "trade_names": ["Acrobat"],
        "active_ingredient": "Dimethomorph 50% WP",
        "category": "Cinnamic acid amide (CAA) Systemic Oomyceticide",
        "frac_code": "40",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Late Blight", "Downy Mildew", "Phytophthora Blight"],
        "dose_per_liter_water": "1.0 - 1.25 g",
        "dose_per_acre": "200 - 250 g in 200 L water",
        "phi_days": 7,
        "application_type": "Foliar Spray (Locally Systemic)",
        "notes": "Inhibits cell wall synthesis of oomycetes. Tank mix with Mancozeb to prevent resistance."
    },
    "azoxystrobin_difenoconazole": {
        "chemical_id": "azoxystrobin_difenoconazole",
        "trade_names": ["Amistar Top"],
        "active_ingredient": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC",
        "category": "Strobilurin + Triazole Broad-Spectrum Systemic",
        "frac_code": "11 + 3",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Early Blight", "Anthracnose", "Powdery Mildew", "Downy Mildew", "Gummy Stem Blight", "Cercospora Leaf Spot"],
        "dose_per_liter_water": "1.0 ml",
        "dose_per_acre": "200 ml in 200 L water",
        "phi_days": 7,
        "application_type": "Foliar Spray",
        "notes": "Dual mode of action providing both preventative and curative control."
    },
    "copper_oxychloride_50_wp": {
        "chemical_id": "copper_oxychloride_50_wp",
        "trade_names": ["Blitox 50", "Cupramar"],
        "active_ingredient": "Copper Oxychloride 50% WP",
        "category": "Inorganic Copper Bactericide & Fungicide",
        "frac_code": "M01",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Early Blight", "Late Blight", "Bacterial Fruit Blotch", "Angular Leaf Spot", "Gummy Stem Blight", "Anthracnose"],
        "dose_per_liter_water": "2.5 - 3.0 g",
        "dose_per_acre": "600 g in 200 L water",
        "phi_days": 5,
        "application_type": "Foliar Spray",
        "notes": "Essential for bacterial leaf spots and fruit blotch. Avoid spraying in direct hot midday sun."
    },
    "hexaconazole_5_sc": {
        "chemical_id": "hexaconazole_5_sc",
        "trade_names": ["Contaf"],
        "active_ingredient": "Hexaconazole 5% SC",
        "category": "Triazole Systemic Fungicide",
        "frac_code": "3",
        "crops": ["watermelon"],
        "target_conditions": ["Powdery Mildew", "Anthracnose"],
        "dose_per_liter_water": "1.0 - 1.5 ml",
        "dose_per_acre": "250 ml in 200 L water",
        "phi_days": 14,
        "application_type": "Foliar Spray",
        "notes": "Highly effective ergosterol biosynthesis inhibitor for cucurbit powdery mildew."
    },

    # --- INSECTICIDES & ACARICIDES ---
    "imidacloprid_17_8_sl": {
        "chemical_id": "imidacloprid_17_8_sl",
        "trade_names": ["Confidor"],
        "active_ingredient": "Imidacloprid 17.8% SL",
        "category": "Neonicotinoid Systemic Insecticide",
        "irac_code": "4A",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Aphids", "Whiteflies", "Leafhoppers / Jassids", "Melon Aphid"],
        "dose_per_liter_water": "0.3 - 0.5 ml",
        "dose_per_acre": "60 - 80 ml in 200 L water",
        "phi_days": 15,
        "application_type": "Foliar Spray / Drench",
        "notes": "Effective against sucking vectors of PLRV and potyviruses. Avoid spraying during active honeybee foraging in flowering watermelon."
    },
    "flonicamid_50_wg": {
        "chemical_id": "flonicamid_50_wg",
        "trade_names": ["Ulala"],
        "active_ingredient": "Flonicamid 50% WG",
        "category": "Pyridinecarboxamide Sucking Pest Blocker",
        "irac_code": "29",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Aphids", "Whiteflies", "Thrips", "Melon Aphid"],
        "dose_per_liter_water": "0.3 g",
        "dose_per_acre": "60 g in 200 L water",
        "phi_days": 7,
        "application_type": "Foliar Spray",
        "notes": "Rapidly halts insect stylet penetration within 30 minutes, preventing virus transmission. Safe for beneficial pollinators."
    },
    "cyantraniliprole_10_26_od": {
        "chemical_id": "cyantraniliprole_10_26_od",
        "trade_names": ["Benevia"],
        "active_ingredient": "Cyantraniliprole 10.26% OD",
        "category": "Diamide Cross-Spectrum Insecticide",
        "irac_code": "28",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Leaf Miner / Serpentine Leafminer", "Melon Caterpillar", "Fruit Fly", "Whiteflies", "Potato Tuber Moth"],
        "dose_per_liter_water": "1.8 - 2.0 ml",
        "dose_per_acre": "360 - 400 ml in 200 L water",
        "phi_days": 5,
        "application_type": "Foliar Spray",
        "notes": "Exceptional translaminar control of serpentine leafminer larvae feeding within leaf laminae."
    },
    "spinosad_45_sc": {
        "chemical_id": "spinosad_45_sc",
        "trade_names": ["Tracer"],
        "active_ingredient": "Spinosad 45% SC",
        "category": "Spinosyn Biological Insecticide (Naturally derived)",
        "irac_code": "5",
        "crops": ["potato", "watermelon"],
        "target_conditions": ["Thrips", "Melon Caterpillar", "Potato Tuber Moth", "Flea Beetle", "Red Pumpkin Beetle"],
        "dose_per_liter_water": "0.3 - 0.4 ml",
        "dose_per_acre": "60 - 75 ml in 200 L water",
        "phi_days": 3,
        "application_type": "Foliar Spray",
        "notes": "Fast knock-down with very low pre-harvest interval (3 days). Safe for integrated pest management."
    },
    "abamectin_1_9_ec": {
        "chemical_id": "abamectin_1_9_ec",
        "trade_names": ["Vertimec"],
        "active_ingredient": "Abamectin 1.9% EC",
        "category": "Avermectin Insecticide & Acaricide",
        "irac_code": "6",
        "crops": ["watermelon"],
        "target_conditions": ["Spider Mites / Red Spider Mite", "Thrips"],
        "dose_per_liter_water": "0.75 - 1.0 ml",
        "dose_per_acre": "150 - 200 ml in 200 L water",
        "phi_days": 7,
        "application_type": "Foliar Spray (Abaxial Coverage)",
        "notes": "Target lower leaf undersides where red spider mite colonies and webbing reside."
    }
}

# Approved Bio-Pesticides, Botanicals, and Agronomic Biologicals
BIOLOGICAL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "trichoderma_viride": {
        "bio_id": "trichoderma_viride",
        "agent": "Trichoderma viride 1% WP (2x10^8 CFU/g)",
        "category": "Antagonistic Bio-fungicide",
        "target_conditions": ["Early Blight", "Black Scurf", "Fusarium Dry Rot", "Fusarium Wilt", "Root Rot / Pythium Damping Off", "Charcoal Rot", "Downy Mildew", "Anthracnose"],
        "application_methods": {
            "tuber_treatment": "5 g/kg seed tuber slurry before sowing",
            "soil_application": "2.5 kg mixed with 250 kg well-decomposed FYM applied per acre",
            "foliar_spray": "5 g per liter water applied as preventive bio-fungicide",
            "drenching": "10 g per liter water around plant collar"
        },
        "notes": "Colonizes plant surface and rhizosphere, suppresses pathogen germination, and secretes chitinase."
    },
    "pseudomonas_fluorescens": {
        "bio_id": "pseudomonas_fluorescens",
        "agent": "Pseudomonas fluorescens 1% WP",
        "category": "Plant Growth Promoting Bio-bactericide",
        "target_conditions": ["Bacterial Wilt / Brown Rot", "Bacterial Soft Rot", "Bacterial Fruit Blotch", "Blackleg"],
        "application_methods": {
            "seed_treatment": "10 g/kg seed",
            "soil_drench": "10 g/L water into root zone"
        },
        "notes": "Produces siderophores and phenazines that suppress bacterial vascular colonizers."
    },
    "neem_oil_azadirachtin": {
        "bio_id": "neem_oil_azadirachtin",
        "agent": "Azadirachtin 1500 ppm / Cold-Pressed Neem Oil",
        "category": "Botanical Insecticide & Antifeedant",
        "target_conditions": ["Melon Aphid", "Whiteflies", "Thrips", "Leaf Miner", "Red Pumpkin Beetle", "Aphids"],
        "application_methods": {
            "foliar_spray": "4.0 - 5.0 ml per liter water with 0.5 ml liquid soap emulsifier"
        },
        "notes": "Acts as oviposition deterrent, antifeedant, and insect growth regulator (IGR). Safe for natural enemies."
    },
    "cue_lure_pheromone_traps": {
        "bio_id": "cue_lure_pheromone_traps",
        "agent": "Cue-lure Male Annihilation Traps (MAT)",
        "category": "Pheromone Mass Trapping",
        "target_conditions": ["Melon Fruit Fly"],
        "application_methods": {
            "field_installation": "Install 6 - 8 traps per acre at vine canopy level before flowering"
        },
        "notes": "Captures male Bactrocera cucurbitae flies, collapsing reproductive mating cycles."
    },
    "yellow_sticky_traps": {
        "bio_id": "yellow_sticky_traps",
        "agent": "Yellow & Blue Chromotropic Sticky Traps",
        "category": "Physical / Behavioral Pest Trapping",
        "target_conditions": ["Whiteflies", "Aphids", "Leaf Miner", "Thrips"],
        "application_methods": {
            "field_installation": "Erect 15 - 20 yellow sticky traps (and 5 blue traps for thrips) per acre 15 cm above canopy"
        },
        "notes": "Continuous mass trapping and early warning economic threshold monitoring."
    }
}

def validate_chemical_safety(chemical_name: str, crop: str) -> Tuple[bool, str]:
    """
    Guardrail: Checks whether a chemical is banned, restricted, or unregistered for the target crop.
    Returns (is_approved: bool, message: str).
    """
    clean_name = chemical_name.strip().lower()

    # 1. Check banned registry
    for banned_key, ban_reason in BANNED_OR_RESTRICTED_CHEMICALS.items():
        if banned_key in clean_name:
            return False, f"SAFETY VIOLATION: '{chemical_name}' is strictly BANNED in India. Reason: {ban_reason}"

    # 2. Check approved registry
    found_approved = False
    for chem_id, chem_data in APPROVED_REGISTRY.items():
        if chem_id in clean_name or clean_name in chem_data["active_ingredient"].lower():
            if crop.lower() in chem_data["crops"]:
                return True, f"Approved for {crop.capitalize()} under CIB&RC standards."
            else:
                return False, f"Crop Mismatch: '{chemical_name}' is registered in India, but NOT approved for {crop.capitalize()}."

    return False, f"Unverified Agrochemical: '{chemical_name}' is not in the certified CIB&RC database."

def get_recommended_chemicals(condition_name: str, crop: str) -> List[Dict[str, Any]]:
    """Retrieve certified chemicals specifically approved for the target disease or pest and crop."""
    matches = []
    clean_crop = crop.lower()
    for chem_id, data in APPROVED_REGISTRY.items():
        if clean_crop in data["crops"] and condition_name in data["target_conditions"]:
            matches.append(data)
    return matches

def get_recommended_biologicals(condition_name: str) -> List[Dict[str, Any]]:
    """Retrieve certified biological controls, pheromones, and botanicals for condition."""
    matches = []
    for bio_id, data in BIOLOGICAL_REGISTRY.items():
        if condition_name in data["target_conditions"]:
            matches.append(data)
    return matches
