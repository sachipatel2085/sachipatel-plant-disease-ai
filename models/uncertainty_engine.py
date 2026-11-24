"""
Module G: Uncertainty Estimation & Out-of-Distribution (OOD) Rejection Engine.
Computes Shannon entropy, margin between top-1 and top-2 candidates,
temperature scaling, and rejects ambiguous or out-of-scope imagery.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .common import (
    DiseaseCandidate,
    UncertaintyEvaluation
)

class UncertaintyEngine:
    """Evaluates prediction certainty, computes entropy, and flags OOD or differential diagnoses."""

    def __init__(
        self,
        min_confidence_threshold: float = 0.50,
        max_entropy_threshold: float = 2.20,
        margin_threshold: float = 0.15
    ):
        self.min_confidence = min_confidence_threshold
        self.max_entropy = max_entropy_threshold
        self.margin_threshold = margin_threshold

    def evaluate(self, candidates: List[DiseaseCandidate]) -> UncertaintyEvaluation:
        """
        Assess diagnostic uncertainty across the candidate predictions.
        Enforces strict rejection when evidence is insufficient or conflicting.
        """
        if not candidates:
            return UncertaintyEvaluation(
                max_confidence=0.0,
                shannon_entropy=0.0,
                margin_top1_top2=0.0,
                is_rejected=True,
                is_differential=False,
                rejection_reason="No compatible disease or pest candidates found for the uploaded image."
            )

        top1 = candidates[0]
        top1_prob = top1.calibrated_probability
        top2_prob = candidates[1].calibrated_probability if len(candidates) > 1 else 0.0

        # Margin between top-1 and top-2
        margin = top1_prob - top2_prob

        # Shannon Entropy across candidates: H(P) = -sum(p * log2(p))
        probs = np.array([c.calibrated_probability for c in candidates])
        probs = probs / np.sum(probs)  # Re-normalize over top candidates
        eps = 1e-9
        entropy = -float(np.sum(probs * np.log2(probs + eps)))

        # 1. Check for Out-of-Distribution / Low Confidence Rejection
        if top1_prob < self.min_confidence:
            return UncertaintyEvaluation(
                max_confidence=round(top1_prob, 4),
                shannon_entropy=round(entropy, 3),
                margin_top1_top2=round(margin, 4),
                is_rejected=True,
                is_differential=False,
                rejection_reason=(
                    f"Low Confidence Diagnostic Rejection: The primary candidate '{top1.condition_name}' "
                    f"achieved only {top1_prob * 100:.1f}% confidence (below safety threshold {self.min_confidence * 100:.0f}%). "
                    f"The system refuses to force an uncertain diagnosis. Please provide a sharper, well-lit photograph."
                )
            )

        # 2. Check for High Ambiguity / Entropy Rejection
        if entropy > self.max_entropy:
            return UncertaintyEvaluation(
                max_confidence=round(top1_prob, 4),
                shannon_entropy=round(entropy, 3),
                margin_top1_top2=round(margin, 4),
                is_rejected=True,
                is_differential=False,
                rejection_reason=(
                    f"Diagnostic Ambiguity Rejection: High Shannon entropy ({entropy:.2f} bits > {self.max_entropy:.2f}). "
                    f"The neural network detected conflicting visual patterns resembling multiple distinct pathologies. "
                    f"Diagnostic safety rules require additional field context or clear laboratory inspection."
                )
            )

        # 3. Check for Differential Diagnosis Warning (Close Margin)
        is_differential = (margin < self.margin_threshold) and (len(candidates) > 1)
        differential_warning = None
        if is_differential:
            top2 = candidates[1]
            differential_warning = (
                f"Differential Diagnosis Warning: Close visual resemblance between "
                f"'{top1.condition_name}' ({top1_prob * 100:.1f}%) and '{top2.condition_name}' ({top2_prob * 100:.1f}%). "
                f"Margin of {margin * 100:.1f}% is below {self.margin_threshold * 100:.0f}%. "
                f"Both possibilities are reported with characteristic symptom differentiators."
            )

        return UncertaintyEvaluation(
            max_confidence=round(top1_prob, 4),
            shannon_entropy=round(entropy, 3),
            margin_top1_top2=round(margin, 4),
            is_rejected=False,
            is_differential=is_differential,
            differential_warning=differential_warning
        )
