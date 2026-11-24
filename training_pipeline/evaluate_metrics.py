"""
Clinical Evaluation and Confidence Calibration Evaluation Suite.
Computes:
1. Top-1 and Top-3 Accuracy
2. Per-class Precision, Recall, and F1-Score (with Macro & Weighted averages)
3. Full Confusion Matrix (CSV and JSON export)
4. Expected Calibration Error (ECE) across 10 confidence bins
5. Maximum Calibration Error (MCE) and Reliability Diagram data
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class AgriculturalModelEvaluator:
    """Computes comprehensive clinical metrics and confidence calibration statistics."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate full test set predictions.
        y_true: integer array of true class labels (N,)
        y_pred_probs: predicted probability distribution (N, num_classes)
        """
        n_samples = len(y_true)
        if n_samples == 0:
            raise ValueError("Evaluation array y_true cannot be empty.")

        y_pred_top1 = np.argmax(y_pred_probs, axis=1)
        top1_confidences = np.max(y_pred_probs, axis=1)

        # 1. Top-1 Accuracy
        correct_top1 = (y_pred_top1 == y_true)
        top1_acc = float(np.mean(correct_top1))

        # 2. Top-3 Accuracy
        k = min(3, self.num_classes)
        top_k_indices = np.argsort(y_pred_probs, axis=1)[:, -k:]
        correct_top3 = [y_true[i] in top_k_indices[i] for i in range(n_samples)]
        top3_acc = float(np.mean(correct_top3))

        # 3. Per-Class Precision, Recall, F1-Score & Confusion Matrix
        conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(y_true, y_pred_top1):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                conf_matrix[t, p] += 1

        per_class_metrics = {}
        precisions = []
        recalls = []
        f1_scores = []
        supports = []

        for c_idx in range(self.num_classes):
            tp = conf_matrix[c_idx, c_idx]
            fp = np.sum(conf_matrix[:, c_idx]) - tp
            fn = np.sum(conf_matrix[c_idx, :]) - tp
            support = int(np.sum(conf_matrix[c_idx, :]))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * (prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

            c_name = self.class_names[c_idx] if c_idx < len(self.class_names) else f"Class_{c_idx}"
            per_class_metrics[c_name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": support
            }
            if support > 0:
                precisions.append(prec)
                recalls.append(rec)
                f1_scores.append(f1)
                supports.append(support)

        macro_prec = float(np.mean(precisions)) if precisions else 0.0
        macro_rec = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

        total_support = sum(supports)
        weighted_prec = float(np.sum([p * s for p, s in zip(precisions, supports)]) / total_support) if total_support > 0 else 0.0
        weighted_rec = float(np.sum([r * s for r, s in zip(recalls, supports)]) / total_support) if total_support > 0 else 0.0
        weighted_f1 = float(np.sum([f * s for f, s in zip(f1_scores, supports)]) / total_support) if total_support > 0 else 0.0

        # 4. Expected Calibration Error (ECE) & Reliability Diagram
        ece, mce, reliability_bins = self.compute_ece(
            confidences=top1_confidences,
            correctness=correct_top1.astype(int),
            num_bins=num_bins
        )

        return {
            "total_samples": n_samples,
            "overall_accuracy": {
                "top_1_accuracy": round(top1_acc * 100, 2),
                "top_3_accuracy": round(top3_acc * 100, 2)
            },
            "macro_averages": {
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "f1_score": round(macro_f1, 4)
            },
            "weighted_averages": {
                "precision": round(weighted_prec, 4),
                "recall": round(weighted_rec, 4),
                "f1_score": round(weighted_f1, 4)
            },
            "calibration": {
                "expected_calibration_error_ece": round(ece * 100, 2),
                "max_calibration_error_mce": round(mce * 100, 2),
                "num_bins": num_bins,
                "reliability_diagram": reliability_bins
            },
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": conf_matrix.tolist()
        }

    def compute_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        num_bins: int = 10
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Compute Expected Calibration Error (ECE) across M equally-spaced confidence bins.
        Returns: (ECE, MCE, reliability_bins)
        """
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        mce = 0.0
        n_total = len(confidences)
        reliability_bins = []

        for m in range(num_bins):
            low = bin_edges[m]
            high = bin_edges[m + 1]

            # Samples falling in bin (low, high]
            if m == 0:
                in_bin = (confidences >= low) & (confidences <= high)
            else:
                in_bin = (confidences > low) & (confidences <= high)

            bin_size = int(np.sum(in_bin))
            if bin_size > 0:
                bin_acc = float(np.mean(correctness[in_bin]))
                bin_conf = float(np.mean(confidences[in_bin]))
                gap = abs(bin_acc - bin_conf)
                ece += (bin_size / n_total) * gap
                mce = max(mce, gap)

                reliability_bins.append({
                    "bin_index": m + 1,
                    "bin_range": f"{low:.2f}-{high:.2f}",
                    "sample_count": bin_size,
                    "accuracy": round(bin_acc, 4),
                    "confidence": round(bin_conf, 4),
                    "calibration_gap": round(gap, 4)
                })
            else:
                reliability_bins.append({
                    "bin_index": m + 1,
                    "bin_range": f"{low:.2f}-{high:.2f}",
                    "sample_count": 0,
                    "accuracy": 0.0,
                    "confidence": round((low + high) / 2.0, 4),
                    "calibration_gap": 0.0
                })

        return float(ece), float(mce), reliability_bins

    def export_confusion_matrix_csv(self, conf_matrix: List[List[int]], output_path: Path):
        """Export confusion matrix in standard CSV format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            header = "True_Class," + ",".join([f"Pred_{c}" for c in self.class_names]) + "\n"
            f.write(header)
            for i, row in enumerate(conf_matrix):
                c_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
                row_str = f"{c_name}," + ",".join([str(val) for val in row]) + "\n"
                f.write(row_str)
        print(f"[Evaluator] Confusion matrix exported to: {output_path}")
