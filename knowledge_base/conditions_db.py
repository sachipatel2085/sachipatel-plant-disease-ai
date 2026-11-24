"""
Agricultural Knowledge Base Loader and Query Interface.
Loads the verified 69-condition database and provides indexed queries for the recommendation engine.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

KB_JSON_PATH = Path(__file__).resolve().parent / "conditions_db.json"

class AgriculturalKnowledgeBase:
    """Provides indexed access to certified plant pathology, pest biology, and management profiles."""

    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or KB_JSON_PATH
        self.database: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.database = json.load(f)

    def get_condition(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve condition by canonical condition_id or slug key."""
        if condition_id in self.database:
            return self.database[condition_id]
        clean_id = condition_id.strip().upper()
        for key, record in self.database.items():
            if record.get("condition_id", "").upper() == clean_id:
                return record
            if key.upper() == clean_id:
                return record
        return None

    def get_by_name(self, condition_name: str, crop: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find condition by common name, optionally filtered by crop."""
        clean_name = condition_name.strip().lower()
        for cond_id, record in self.database.items():
            if crop and record["crop"].lower() != crop.lower():
                continue
            if record["condition_name"].lower() == clean_name:
                return record
            if clean_name in record["condition_name"].lower():
                return record
        return None

    def list_conditions(self, crop: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all condition records, optionally filtered by crop."""
        if not crop:
            return list(self.database.values())
        return [r for r in self.database.values() if r["crop"].lower() == crop.lower()]

    def count(self) -> int:
        return len(self.database)
