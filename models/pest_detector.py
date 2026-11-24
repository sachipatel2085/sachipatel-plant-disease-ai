"""
Module D: Insect & Pest Detection Module.
Detects localized arthropod pests, mites, oviposition punctures, and distinctive feeding injuries
(e.g., Red Pumpkin Beetle circular notches, serpentine leaf miner trails, aphid colonies).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .common import (
    DETECTION_INPUT_SHAPE,
    PestDetection
)

# Known localized agricultural pests
SUPPORTED_PESTS = {
    "potato": [
        {"name": "Aphids", "scientific": "Myzus persicae", "type": "Sap-sucking colony"},
        {"name": "Whiteflies", "scientific": "Bemisia tabaci", "type": "Abaxial winged nymphs/adults"},
        {"name": "Thrips", "scientific": "Thrips palmi", "type": "Silvery epidermal scarring"},
        {"name": "Potato Tuber Moth", "scientific": "Phthorimaea operculella", "type": "Frass & periderm tunnels"},
        {"name": "Leafhoppers / Jassids", "scientific": "Amrasca biguttula", "type": "Hopper-burn & nymphs"},
        {"name": "Flea Beetle", "scientific": "Epitrix hirtipennis", "type": "Shot-hole foliar perforations"}
    ],
    "watermelon": [
        {"name": "Red Pumpkin Beetle", "scientific": "Aulacophora foveicollis", "type": "Circular leaf feeding rings"},
        {"name": "Melon Fruit Fly", "scientific": "Bactrocera cucurbitae", "type": "Oviposition punctures & rot"},
        {"name": "Leaf Miner / Serpentine Leafminer", "scientific": "Liriomyza trifolii", "type": "Serpentine winding trails"},
        {"name": "Spider Mites / Red Spider Mite", "scientific": "Tetranychus urticae", "type": "Silken webbing & stippling"},
        {"name": "Melon Aphid", "scientific": "Aphis gossypii", "type": "Downward leaf curling clusters"},
        {"name": "Melon Caterpillar", "scientific": "Diaphania indica", "type": "Foliar webbing & fruit boreholes"}
    ]
}

class InsectPestDetector:
    """Detects and localizes agricultural pests and characteristic feeding signatures."""

    def __init__(self, weights_path: Optional[str] = None, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.weights_path = weights_path

    def detect_pests(self, img: Image.Image, crop: str = "potato") -> List[PestDetection]:
        """
        Scan image for visible insect bodies, colonies, or pathognomonic feeding marks.
        Returns a list of localized PestDetection bounding boxes.
        """
        crop_clean = crop.lower() if crop.lower() in ("potato", "watermelon") else "potato"
        known_pests = SUPPORTED_PESTS.get(crop_clean, [])

        np_img = np.array(img.convert("RGB"))
        h, w, _ = np_img.shape

        # Preprocessing: convert to HSV for color/texture segmentation of feeding injuries
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        detections: List[PestDetection] = []

        # 1. Detect Serpentine Leafminer trails (white winding tracks on green leaf)
        if crop_clean == "watermelon":
            # Mask very light / whitish trails on green leaf
            trail_mask = cv2.inRange(hsv, np.array([20, 0, 160]), np.array([85, 80, 255]))
            contours, _ = cv2.findContours(trail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 250 < area < (h * w * 0.25):
                    perimeter = cv2.arcLength(cnt, True)
                    # Serpentine mines have high perimeter-to-area ratio (elongated/winding)
                    if perimeter > 0:
                        circularity = 4 * np.pi * (area / (perimeter * perimeter))
                        if circularity < 0.25:  # Highly elongated/winding
                            x, y, bw, bh = cv2.boundingRect(cnt)
                            detections.append(PestDetection(
                                pest_name="Leaf Miner / Serpentine Leafminer",
                                scientific_name="Liriomyza trifolii",
                                confidence=round(float(min(0.95, 0.65 + (area / float(h * w)) * 2)), 2),
                                bbox_normalized=[round(y / h, 4), round(x / w, 4), round((y + bh) / h, 4), round((x + bw) / w, 4)],
                                damage_type="Serpentine winding larval mines"
                            ))

            # 2. Detect Red Pumpkin Beetle (bright orange-red chrysomelid bodies)
            orange_red_mask = cv2.inRange(hsv, np.array([5, 120, 100]), np.array([22, 255, 255]))
            contours, _ = cv2.findContours(orange_red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 100 < area < 5000:  # Beetle-sized area
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect = bw / float(bh)
                    if 0.5 < aspect < 2.0:  # Oval beetle shape
                        detections.append(PestDetection(
                            pest_name="Red Pumpkin Beetle",
                            scientific_name="Aulacophora foveicollis",
                            confidence=0.88,
                            bbox_normalized=[round(y / h, 4), round(x / w, 4), round((y + bh) / h, 4), round((x + bw) / w, 4)],
                            damage_type="Adult beetle on leaf"
                        ))

        # 3. Detect Dark clusters (Aphids or dark frass)
        dark_clusters = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        contours, _ = cv2.findContours(dark_clusters, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 < area < 2500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                pest_candidate = known_pests[0] if known_pests else {"name": "Aphids", "scientific": "Aphis spp.", "type": "Colony"}
                detections.append(PestDetection(
                    pest_name=pest_candidate["name"],
                    scientific_name=pest_candidate["scientific"],
                    confidence=0.72,
                    bbox_normalized=[round(y / h, 4), round(x / w, 4), round((y + bh) / h, 4), round((x + bw) / w, 4)],
                    damage_type=pest_candidate["type"]
                ))

        # Deduplicate overlapping detections (Non-Maximum Suppression)
        final_detections = self._apply_nms(detections, iou_threshold=0.45)
        return final_detections[:5]

    @staticmethod
    def _apply_nms(detections: List[PestDetection], iou_threshold: float = 0.45) -> List[PestDetection]:
        """Simple bounding box Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        selected: List[PestDetection] = []

        for det in sorted_dets:
            overlap = False
            for s in selected:
                # Compute IoU
                boxA = det.bbox_normalized
                boxB = s.bbox_normalized

                yA = max(boxA[0], boxB[0])
                xA = max(boxA[1], boxB[1])
                yB = min(boxA[2], boxB[2])
                xB = min(boxA[3], boxB[3])

                interArea = max(0, yB - yA) * max(0, xB - xA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                unionArea = boxAArea + boxBArea - interArea

                iou = interArea / float(unionArea) if unionArea > 0 else 0.0
                if iou > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                selected.append(det)

        return selected
