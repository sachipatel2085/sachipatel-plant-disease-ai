"""
Stratified Dataset Partitioner and Class Imbalance Engine.
Performs 70/15/15 stratified partitioning, computes inverse frequency and effective sample weights,
and provides tailored data augmentation recommendations to counteract class imbalance.
"""

import math
import random
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from .config import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SPLIT_RANDOM_SEED
)

class DatasetBalancerAndSplitter:
    """Manages stratified dataset partitioning, class frequency weighting, and balance analysis."""

    def __init__(self, random_seed: int = SPLIT_RANDOM_SEED):
        self.random_seed = random_seed

    def stratified_split(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Partition records into train (70%), validation (15%), and test (15%) splits
        stratified by condition_id.
        """
        # Group by condition_id
        grouped = defaultdict(list)
        for r in records:
            cond_id = r.get("condition_id", "unknown")
            grouped[cond_id].append(r)

        splits: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "val": [],
            "test": []
        }

        # Deterministic shuffle per class
        rng = random.Random(self.random_seed)

        for cond_id, class_records in grouped.items():
            shuffled = list(class_records)
            rng.shuffle(shuffled)
            n = len(shuffled)

            if n == 1:
                # If only 1 sample, place in train
                shuffled[0]["split"] = "train"
                splits["train"].append(shuffled[0])
            elif n == 2:
                shuffled[0]["split"] = "train"
                shuffled[1]["split"] = "val"
                splits["train"].append(shuffled[0])
                splits["val"].append(shuffled[1])
            elif n < 5:
                # Small sample: 1 test, 1 val, remainder train
                shuffled[0]["split"] = "train"
                splits["train"].append(shuffled[0])
                if n >= 2:
                    shuffled[1]["split"] = "val"
                    splits["val"].append(shuffled[1])
                if n >= 3:
                    shuffled[2]["split"] = "test"
                    splits["test"].append(shuffled[2])
                for item in shuffled[3:]:
                    item["split"] = "train"
                    splits["train"].append(item)
            else:
                n_train = int(round(n * TRAIN_RATIO))
                n_val = int(round(n * VAL_RATIO))
                # Ensure at least 1 in val and test
                n_val = max(1, n_val)
                n_test = max(1, n - n_train - n_val)
                n_train = n - n_val - n_test

                train_items = shuffled[:n_train]
                val_items = shuffled[n_train:n_train + n_val]
                test_items = shuffled[n_train + n_val:]

                for item in train_items:
                    item["split"] = "train"
                    splits["train"].append(item)
                for item in val_items:
                    item["split"] = "val"
                    splits["val"].append(item)
                for item in test_items:
                    item["split"] = "test"
                    splits["test"].append(item)

        return splits

    @staticmethod
    def calculate_class_weights(records: List[Dict[str, Any]], beta: float = 0.9999) -> Dict[str, Dict[str, Any]]:
        """
        Compute class frequency weights to counteract pathological dataset imbalance:
        1. Inverse Class Frequency (ICF)
        2. Effective Number of Samples (Cui et al., CVPR 2019)
        3. Tailored augmentation intensity recommendations
        """
        counts = defaultdict(int)
        for r in records:
            counts[r.get("condition_id", "unknown")] += 1

        total_samples = len(records)
        num_classes = len(counts)
        if total_samples == 0 or num_classes == 0:
            return {}

        median_count = sorted(counts.values())[len(counts) // 2]
        weights = {}

        for cond_id, count in counts.items():
            # Standard inverse frequency
            icf = total_samples / float(num_classes * count)

            # Effective Number of Samples: E_n = (1 - beta^N) / (1 - beta)
            effective_num = (1.0 - math.pow(beta, count)) / (1.0 - beta) if count > 0 else 1.0
            effective_weight = 1.0 / effective_num

            # Determine augmentation policy recommendation based on representation ratio
            ratio = count / float(median_count) if median_count > 0 else 1.0
            if ratio < 0.2:
                aug_policy = "HEAVY"
                aug_recipe = "ElasticTransform, RandomPerspective, ColorJitter(hue=0.1, sat=0.3), CLAHE, Mixup"
            elif ratio < 0.6:
                aug_policy = "MODERATE"
                aug_recipe = "RandomHorizontalFlip, RandomRotation(30), ColorJitter(brightness=0.2, contrast=0.2)"
            else:
                aug_policy = "STANDARD"
                aug_recipe = "RandomHorizontalFlip, RandomResizedCrop(scale=(0.8, 1.0))"

            weights[cond_id] = {
                "sample_count": count,
                "percentage": round(count / float(total_samples) * 100.0, 2),
                "inverse_frequency_weight": round(icf, 4),
                "effective_sample_weight": round(effective_weight * 1000.0, 4),
                "augmentation_policy": aug_policy,
                "recommended_augmentations": aug_recipe
            }

        return weights

    @staticmethod
    def verify_split_leakage(splits: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, List[str]]:
        """
        Verify that no exact SHA-256 or near-duplicate pHash leaks across train, val, and test splits.
        Returns (is_clean: bool, violations: List[str]).
        """
        train_shas = {r["sha256"]: r["image_id"] for r in splits["train"] if "sha256" in r}
        train_phashes = {r["phash"]: r["image_id"] for r in splits["train"] if "phash" in r}

        violations = []

        # Check val
        for r in splits["val"]:
            sha = r.get("sha256")
            if sha and sha in train_shas:
                violations.append(f"Exact leakage: Image {r['image_id']} in Val matches Train {train_shas[sha]}")
            ph = r.get("phash")
            if ph and ph in train_phashes:
                violations.append(f"Perceptual leakage: Image {r['image_id']} in Val matches Train {train_phashes[ph]}")

        # Check test
        for r in splits["test"]:
            sha = r.get("sha256")
            if sha and sha in train_shas:
                violations.append(f"Exact leakage: Image {r['image_id']} in Test matches Train {train_shas[sha]}")
            ph = r.get("phash")
            if ph and ph in train_phashes:
                violations.append(f"Perceptual leakage: Image {r['image_id']} in Test matches Train {train_phashes[ph]}")

        return (len(violations) == 0), violations
