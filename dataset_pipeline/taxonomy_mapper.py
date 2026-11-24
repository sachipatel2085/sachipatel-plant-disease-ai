"""
Taxonomy Mapper and Label Normalization Engine.
Maps diverse public dataset folder names and annotations (PlantVillage, PlantDoc, IP102)
to canonical Phase 1 taxonomy conditions, scientific names, and plant organs.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .config import TAXONOMY_JSON_PATH, SUPPORTED_CROPS, SUPPORTED_ORGANS

class TaxonomyMapper:
    """Validates raw image labels against the verified agricultural taxonomy."""

    def __init__(self, taxonomy_path: Optional[Path] = None):
        self.taxonomy_path = taxonomy_path or TAXONOMY_JSON_PATH
        self.conditions_by_id: Dict[str, Dict[str, Any]] = {}
        self.alias_map: Dict[str, str] = {}
        self._load_taxonomy()
        self._build_aliases()

    def _generate_condition_id(self, crop: str, condition_name: str) -> str:
        """Create a standardized snake_case identifier e.g. 'potato__early_blight'."""
        clean_crop = crop.strip().lower()
        clean_cond = re.sub(r'[^a-zA-Z0-9]+', '_', condition_name.strip().lower()).strip('_')
        return f"{clean_crop}__{clean_cond}"

    def _load_taxonomy(self):
        """Load and index the 69 taxonomy conditions."""
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found at: {self.taxonomy_path}")

        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_list = json.load(f)

        for item in taxonomy_list:
            cond_id = self._generate_condition_id(item["Crop"], item["Disease_or_Pest"])
            item_record = dict(item)
            item_record["condition_id"] = cond_id
            # Determine primary organ
            parts = [p.strip().lower() for p in item["Affected_Plant_Part"].split(",")]
            primary_part = parts[0] if parts else "leaf"
            item_record["primary_plant_part"] = primary_part
            self.conditions_by_id[cond_id] = item_record

    def _build_aliases(self):
        """Build mapping from common dataset naming conventions to condition_id."""
        # 1. Exact condition_id
        for cond_id in self.conditions_by_id:
            self.alias_map[cond_id] = cond_id

        # 2. PlantVillage exact formats
        pv_mappings = {
            "potato___early_blight": "potato__early_blight",
            "potato___late_blight": "potato__late_blight",
            "potato___healthy": "potato__healthy_potato_plant",
            "potato_early_blight": "potato__early_blight",
            "potato_late_blight": "potato__late_blight",
            "potato_healthy": "potato__healthy_potato_plant",
            "early_blight": "potato__early_blight",
            "late_blight": "potato__late_blight"
        }
        self.alias_map.update(pv_mappings)

        # 3. IP102 and general insect pest mappings
        pest_mappings = {
            "potato_tuber_moth": "potato__potato_tuber_moth",
            "aphids": "potato__aphids",
            "potato_aphids": "potato__aphids",
            "whiteflies": "potato__whiteflies",
            "potato_whiteflies": "potato__whiteflies",
            "thrips": "potato__thrips",
            "potato_thrips": "potato__thrips",
            "cutworms": "potato__cutworms",
            "leafhoppers": "potato__leafhoppers_jassids",
            "jassids": "potato__leafhoppers_jassids",
            "melon_aphid": "watermelon__melon_aphid",
            "watermelon_aphids": "watermelon__melon_aphid",
            "red_pumpkin_beetle": "watermelon__red_pumpkin_beetle",
            "melon_fruit_fly": "watermelon__melon_fruit_fly",
            "fruit_fly": "watermelon__melon_fruit_fly",
            "spider_mites": "watermelon__spider_mites_red_spider_mite",
            "red_spider_mite": "watermelon__spider_mites_red_spider_mite",
            "leafminer": "watermelon__leaf_miner_serpentine_leafminer",
            "serpentine_leafminer": "watermelon__leaf_miner_serpentine_leafminer",
            "melon_caterpillar": "watermelon__melon_caterpillar_pickleworm",
            "pickleworm": "watermelon__melon_caterpillar_pickleworm"
        }
        self.alias_map.update(pest_mappings)

        # 4. Watermelon diseases and disorders
        wm_mappings = {
            "downy_mildew": "watermelon__downy_mildew",
            "watermelon_downy_mildew": "watermelon__downy_mildew",
            "powdery_mildew": "watermelon__powdery_mildew",
            "watermelon_powdery_mildew": "watermelon__powdery_mildew",
            "anthracnose": "watermelon__anthracnose",
            "watermelon_anthracnose": "watermelon__anthracnose",
            "gummy_stem_blight": "watermelon__gummy_stem_blight",
            "bacterial_fruit_blotch": "watermelon__bacterial_fruit_blotch",
            "blossom_end_rot": "watermelon__blossom_end_rot",
            "sunscald": "watermelon__sunscald",
            "fruit_cracking": "watermelon__fruit_cracking_splitting",
            "watermelon_healthy": "watermelon__healthy_watermelon_plant"
        }
        self.alias_map.update(wm_mappings)

        # 5. Potato tuber conditions
        tuber_mappings = {
            "common_scab": "potato__common_scab",
            "potato_scab": "potato__common_scab",
            "black_scurf": "potato__black_scurf",
            "rhizoctonia": "potato__black_scurf",
            "fusarium_dry_rot": "potato__fusarium_dry_rot",
            "bacterial_soft_rot": "potato__bacterial_soft_rot",
            "tuber_greening": "potato__tuber_greening_solanine_accumulation"
        }
        self.alias_map.update(tuber_mappings)

    def normalize_label(self, raw_label: str, crop_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Normalize a raw label string to a canonical taxonomy record.
        Returns the full taxonomy dictionary or None if unmapped.
        """
        if not raw_label:
            return None

        clean = re.sub(r'[^a-zA-Z0-9]+', '_', raw_label.strip().lower()).strip('_')

        # Direct alias check
        if clean in self.alias_map:
            cond_id = self.alias_map[clean]
            return self.conditions_by_id.get(cond_id)

        # Combined crop + label check
        if crop_hint:
            combined = f"{crop_hint.lower()}__{clean}"
            if combined in self.alias_map:
                cond_id = self.alias_map[combined]
                return self.conditions_by_id.get(cond_id)
            if combined in self.conditions_by_id:
                return self.conditions_by_id[combined]

        # Substring search through known condition IDs
        for cond_id, record in self.conditions_by_id.items():
            cond_name_clean = re.sub(r'[^a-zA-Z0-9]+', '_', record["Disease_or_Pest"].lower()).strip('_')
            if cond_name_clean == clean or (clean in cond_id and (not crop_hint or crop_hint.lower() in cond_id)):
                return record

        return None

    def get_condition_by_id(self, cond_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve condition by canonical ID."""
        return self.conditions_by_id.get(cond_id)

    def total_conditions(self) -> int:
        return len(self.conditions_by_id)
