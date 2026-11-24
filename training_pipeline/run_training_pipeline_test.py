"""
Phase 6 Automated Training Pipeline & Edge Export Verification Suite.
Validates:
1. Research repository harvester & BibTeX citation generator
2. Agricultural field-realistic augmentation suite
3. 2-phase transfer learning model construction & dry-run training
4. Clinical metrics calculation (Top-1, Top-3, F1, Confusion Matrix, ECE)
5. Model export (TFLite FP32 & INT8) and CPU latency benchmarking
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from training_pipeline.sources_harvester import AgriculturalDataHarvester, RESEARCH_DATASETS_REGISTRY
from training_pipeline.augmentations import AgriculturalAugmenter
from training_pipeline.train_disease_classifier import build_disease_classifier_model, DiseaseModelTrainer
from training_pipeline.train_crop_model import build_crop_identification_model
from training_pipeline.evaluate_metrics import AgriculturalModelEvaluator
from training_pipeline.export_and_benchmark import ModelExporterAndBenchmarker

def create_sample_leaf_image(path: Path) -> Image.Image:
    """Create a textured synthetic leaf image for augmentation testing."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    draw.line([(112, 10), (112, 214)], fill=(20, 100, 20), width=3)
    draw.ellipse([(60, 60), (120, 120)], fill=(90, 50, 20))
    img.save(path, "JPEG")
    return img

def run_phase6_audit():
    print("=" * 70)
    print("PHASE 6: MODEL TRAINING PIPELINE & DATASET SCRIPTS AUDIT")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # -------------------------------------------------------------------
        # AUDIT 1: Sources Harvester & Citation Tracker
        # -------------------------------------------------------------------
        print("\n[AUDIT 1] Testing Agricultural Dataset Harvester & Attribution...")
        harvester = AgriculturalDataHarvester(raw_data_dir=tmp_path / "raw")
        registry = harvester.get_source_registry()
        assert len(registry) >= 5, f"Expected at least 5 research datasets, got {len(registry)}"
        print(f"  -> Certified research repositories registered: {len(registry)}")
        for k, v in registry.items():
            print(f"     • {k}: {v['title']} (License: {v['license']})")

        bibtex_path = tmp_path / "citations.bib"
        bibtex_str = harvester.export_citations_bibtex(bibtex_path)
        assert bibtex_path.exists() and len(bibtex_str) > 100
        print(f"  -> Exported BibTeX citations ({len(bibtex_str)} chars) to citations.bib")

        # -------------------------------------------------------------------
        # AUDIT 2: Agricultural Augmentation Suite
        # -------------------------------------------------------------------
        print("\n[AUDIT 2] Testing Field-Realistic Agricultural Augmentations...")
        sample_img_path = tmp_path / "sample_leaf.jpg"
        sample_pil = create_sample_leaf_image(sample_img_path)

        augmenter = AgriculturalAugmenter(image_size=(224, 224))

        # Test individual field transforms
        glared_np = augmenter.apply_sun_glare(np.array(sample_pil))
        assert glared_np.shape == (224, 224, 3)
        print("  -> Sun Glare Simulation: PASS")

        shadowed_np = augmenter.apply_canopy_shadow(np.array(sample_pil))
        assert shadowed_np.shape == (224, 224, 3)
        print("  -> Canopy Shadow Projection: PASS")

        flutter_pil = augmenter.apply_flutter_blur(sample_pil)
        assert flutter_pil.size == (224, 224)
        print("  -> Leaf Flutter Blur: PASS")

        jitter_pil = augmenter.apply_chlorophyll_jitter(sample_pil)
        assert jitter_pil.size == (224, 224)
        print("  -> Chlorophyll Jitter: PASS")

        keras_aug = augmenter.build_keras_augmentation_model()
        dummy_tensor = tf.random.uniform((2, 224, 224, 3), 0.0, 1.0)
        aug_output = keras_aug(dummy_tensor)
        assert aug_output.shape == (2, 224, 224, 3)
        print("  -> Keras Preprocessing Augmentation Layer: PASS")

        # -------------------------------------------------------------------
        # AUDIT 3: Transfer Learning Training Loop (Dry-Run)
        # -------------------------------------------------------------------
        print("\n[AUDIT 3] Testing Transfer Learning Model Construction & Training...")
        num_test_classes = 4
        model, backbone = build_disease_classifier_model(
            num_classes=num_test_classes,
            architecture="MobileNetV3Small"
        )
        assert len(model.layers) > 5
        print(f"  -> Model constructed ({model.name}): Total params: {model.count_params():,}")

        # Synthetic dry-run dataset (4 batches of 4 images)
        x_dummy = np.random.uniform(0.0, 1.0, size=(16, 224, 224, 3)).astype(np.float32)
        y_dummy = np.random.randint(0, num_test_classes, size=(16,)).astype(np.int32)
        train_ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy)).batch(4)
        val_ds = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy)).batch(4)

        trainer = DiseaseModelTrainer(
            num_classes=num_test_classes,
            checkpoint_dir=tmp_path / "checkpoints",
            architecture="MobileNetV3Small"
        )
        h1 = trainer.train_phase_1(train_ds, val_ds, epochs=1)
        assert "loss" in h1.history and "val_loss" in h1.history
        print(f"  -> Phase 1 Feature Extraction 1-epoch dry-run: PASS (Loss: {h1.history['loss'][0]:.4f})")

        h2 = trainer.train_phase_2(train_ds, val_ds, unfreeze_top_layers=10, epochs=1)
        assert "loss" in h2.history
        print(f"  -> Phase 2 Fine-Tuning 1-epoch dry-run: PASS (Loss: {h2.history['loss'][0]:.4f})")

        # -------------------------------------------------------------------
        # AUDIT 4: Clinical Metrics & ECE Calibration Suite
        # -------------------------------------------------------------------
        print("\n[AUDIT 4] Testing Clinical Evaluation & ECE Calibration...")
        classes = ["Early_Blight", "Late_Blight", "Anthracnose", "Healthy"]
        evaluator = AgriculturalModelEvaluator(class_names=classes)

        # Generate synthetic evaluation predictions (100 samples)
        np.random.seed(42)
        n_eval = 100
        y_true_eval = np.random.randint(0, 4, size=n_eval)
        # Add high probability on true class with realistic noise
        raw_logits = np.random.uniform(0.1, 0.5, size=(n_eval, 4))
        for i in range(n_eval):
            raw_logits[i, y_true_eval[i]] += np.random.uniform(0.8, 2.5)
        y_pred_probs = np.exp(raw_logits) / np.sum(np.exp(raw_logits), axis=1, keepdims=True)

        eval_report = evaluator.evaluate_predictions(y_true_eval, y_pred_probs, num_bins=10)

        assert "overall_accuracy" in eval_report
        assert "macro_averages" in eval_report
        assert "calibration" in eval_report
        assert "expected_calibration_error_ece" in eval_report["calibration"]
        assert len(eval_report["confusion_matrix"]) == 4

        print(f"  -> Top-1 Accuracy: {eval_report['overall_accuracy']['top_1_accuracy']}%")
        print(f"  -> Top-3 Accuracy: {eval_report['overall_accuracy']['top_3_accuracy']}%")
        print(f"  -> Macro F1-Score: {eval_report['macro_averages']['f1_score']}")
        print(f"  -> Expected Calibration Error (ECE): {eval_report['calibration']['expected_calibration_error_ece']}%")

        cm_csv_path = tmp_path / "confusion_matrix.csv"
        evaluator.export_confusion_matrix_csv(eval_report["confusion_matrix"], cm_csv_path)
        assert cm_csv_path.exists()
        print("  -> Confusion matrix CSV exported: PASS")

        # -------------------------------------------------------------------
        # AUDIT 5: Model Export & Latency Benchmarking
        # -------------------------------------------------------------------
        print("\n[AUDIT 5] Testing TFLite Export & Latency Benchmarking...")
        exporter = ModelExporterAndBenchmarker(export_dir=tmp_path / "exports")

        # Export TFLite Float32
        tflite_fp32 = exporter.export_tflite_float32(trainer.model, "test_disease_model")
        assert tflite_fp32.exists() and tflite_fp32.stat().st_size > 1000
        print(f"  -> TFLite FP32 Size: {tflite_fp32.stat().st_size / (1024*1024):.2f} MB")

        # Export TFLite Dynamic Range INT8
        tflite_int8 = exporter.export_tflite_int8_dynamic(trainer.model, "test_disease_model")
        assert tflite_int8.exists() and tflite_int8.stat().st_size > 1000
        print(f"  -> TFLite INT8 Size: {tflite_int8.stat().st_size / (1024*1024):.2f} MB")

        # Benchmark Keras Model on CPU
        keras_perf = exporter.benchmark_keras_model(trainer.model, warmup_runs=3, benchmark_runs=10)
        print(f"  -> Keras CPU Latency: {keras_perf['latency_ms']['mean']} ms ({keras_perf['throughput_fps']} FPS)")

        # Benchmark TFLite Interpreter
        tflite_perf = exporter.benchmark_tflite_model(tflite_int8, warmup_runs=3, benchmark_runs=10)
        print(f"  -> TFLite INT8 Latency: {tflite_perf['latency_ms']['mean']} ms ({tflite_perf['throughput_fps']} FPS)")

    print("\n" + "=" * 70)
    print("ALL 5 PHASE 6 TRAINING & EXPORT AUDITS PASSED WITH ZERO ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    run_phase6_audit()
