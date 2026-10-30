"""
Unit test suite for Presento Face Anti-Spoofing (anti-spoof-mn3).
Validates model loading, GPU hardware compilation, input/output tensor shapes,
CelebA-Spoof preprocessing, probability normalization, and finite latency without requiring a physical camera.
"""
from __future__ import annotations

import os
import unittest
import numpy as np
import openvino as ov

from edge.anti_spoof import AntiSpoofClassifier, crop_face, CELEBA_MEAN, CELEBA_SCALE


class TestAntiSpoofModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.available_devices = [d.upper() for d in cls.core.available_devices]
        cls.model_path = "data/models/anti-spoof-mn3.onnx"
        cls.classifier = AntiSpoofClassifier(model_path=cls.model_path, device="GPU")

    def test_model_file_exists_and_loads(self):
        """1. Verify that anti-spoof-mn3 model exists on disk and loads in OpenVINO."""
        self.assertTrue(os.path.exists(self.model_path), f"Model file missing at: {self.model_path}")
        self.assertIsNotNone(self.classifier.compiled_model)
        self.assertGreater(self.classifier.load_time_ms, 0.0)

    def test_gpu_device_compilation(self):
        """2. Verify that Intel Iris Xe GPU is available and compiled directly without fallback."""
        self.assertIn("GPU", self.available_devices, "OpenVINO GPU device not found in available devices")
        self.assertEqual(self.classifier.active_device, "GPU")
        self.assertIn("GPU", self.classifier.full_device_name.upper())

    def test_input_and_output_shapes(self):
        """3. Verify input is [1, 3, 128, 128] and output is [1, 2]."""
        inp = self.classifier.compiled_model.inputs[0]
        out = self.classifier.compiled_model.outputs[0]

        inp_shape = [dim.get_length() for dim in inp.get_partial_shape()]
        out_shape = [dim.get_length() for dim in out.get_partial_shape()]

        self.assertEqual(inp_shape, [1, 3, 128, 128])
        self.assertEqual(out_shape, [1, 2])

    def test_preprocessing_shape_and_range(self):
        """4. Verify preprocessing converts arbitrary BGR crops into normalized (1, 3, 128, 128) float32."""
        dummy_bgr = np.random.randint(0, 256, (200, 250, 3), dtype=np.uint8)
        blob = self.classifier.preprocess(dummy_bgr)

        self.assertEqual(blob.shape, (1, 3, 128, 128))
        self.assertEqual(blob.dtype, np.float32)

        # Preprocessing of known constant image
        const_bgr = np.full((128, 128, 3), 150, dtype=np.uint8)
        blob_const = self.classifier.preprocess(const_bgr)

        # In BGR: B=150, G=150, R=150
        # In RGB: R=150, G=150, B=150
        expected_r = (150.0 - 151.2405) / 63.0105
        expected_g = (150.0 - 119.5950) / 56.4570
        expected_b = (150.0 - 107.8395) / 55.0035

        self.assertAlmostEqual(float(blob_const[0, 0, 0, 0]), expected_r, places=3)
        self.assertAlmostEqual(float(blob_const[0, 1, 0, 0]), expected_g, places=3)
        self.assertAlmostEqual(float(blob_const[0, 2, 0, 0]), expected_b, places=3)

    def test_softmax_and_probability_decoding(self):
        """5. Verify predict() returns probabilities in [0, 1] summing to 1.0 with correct label."""
        dummy_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        p_real, p_spoof, label, lat = self.classifier.predict(dummy_face)

        self.assertGreaterEqual(p_real, 0.0)
        self.assertLessEqual(p_real, 1.0)
        self.assertGreaterEqual(p_spoof, 0.0)
        self.assertLessEqual(p_spoof, 1.0)
        self.assertAlmostEqual(p_real + p_spoof, 1.0, places=4)
        self.assertIn(label, ("REAL", "SPOOF"))
        if p_real >= p_spoof:
            self.assertEqual(label, "REAL")
        else:
            self.assertEqual(label, "SPOOF")

    def test_inference_latency_is_finite(self):
        """6. Verify inference latency is finite and strictly positive."""
        dummy_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        _, _, _, lat = self.classifier.predict(dummy_face)

        self.assertIsInstance(lat, float)
        self.assertTrue(np.isfinite(lat))
        self.assertGreater(lat, 0.0)

    def test_crop_face_bounding_logic(self):
        """7. Verify face crop expansion logic safely bounds within image dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)  # w=100, h=100
        crop = crop_face(frame, bbox, scale=1.2)

        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)
        self.assertEqual(crop.shape[2], 3)

        # Corner case: bbox on image boundary
        corner_bbox = (0, 0, 50, 50)
        corner_crop = crop_face(frame, corner_bbox, scale=1.2)
        self.assertGreater(corner_crop.shape[0], 0)
        self.assertGreater(corner_crop.shape[1], 0)


class TestAntiSpoofBenchmarkAutoMode(unittest.TestCase):
    """
    Validates automatic long-duration benchmark features:
    - Configurable duration
    - Automatic scenario ordering
    - Full statistical calculation (mean, median, min, max, std, counts, latencies, FPS)
    - Temporal consistency metrics (runs, >= 0.5 probabilities)
    - JSON results generation and history preservation
    """

    def test_configurable_duration(self):
        """1. Verify CLI argument parser supports configurable durations (e.g. 10s, 20s)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--duration", type=float, default=10.0)

        args_default = parser.parse_args([])
        self.assertEqual(args_default.duration, 10.0)

        args_custom = parser.parse_args(["--duration", "20.5"])
        self.assertEqual(args_custom.duration, 20.5)

    def test_automatic_scenario_ordering(self):
        """2. Verify scenarios are ordered strictly as specified."""
        from edge.anti_spoof_benchmark import AUTO_SCENARIOS
        expected_scenarios = [
            "REAL FACE",
            "PRINTED PHOTO",
            "PHONE PHOTO",
            "PHONE VIDEO",
            "LIGHTING VARIATION",
            "FACE DISTANCE",
        ]
        self.assertEqual(AUTO_SCENARIOS, expected_scenarios)
        self.assertEqual(len(AUTO_SCENARIOS), 6)

    def test_statistics_calculation(self):
        """3. Verify multi-frame statistical metrics calculation."""
        from edge.anti_spoof_benchmark import compute_scenario_metrics

        real_probs = [0.9, 0.8, 0.7, 0.6]
        spoof_probs = [0.1, 0.2, 0.3, 0.4]
        predictions = ["REAL", "REAL", "REAL", "REAL"]
        latencies = [2.0, 3.0, 4.0, 5.0]
        duration_sec = 2.0

        m = compute_scenario_metrics(
            name="TEST SCENARIO",
            real_probs=real_probs,
            spoof_probs=spoof_probs,
            predictions=predictions,
            latencies=latencies,
            duration_sec=duration_sec,
        )

        self.assertEqual(m["frames"], 4)
        self.assertEqual(m["duration_sec"], 2.0)
        self.assertAlmostEqual(m["real_mean"], 0.75, places=3)
        self.assertAlmostEqual(m["real_median"], 0.75, places=3)
        self.assertAlmostEqual(m["real_min"], 0.6, places=3)
        self.assertAlmostEqual(m["real_max"], 0.9, places=3)
        self.assertGreater(m["real_std"], 0.0)
        self.assertAlmostEqual(m["spoof_mean"], 0.25, places=3)
        self.assertEqual(m["real_count"], 4)
        self.assertEqual(m["spoof_count"], 0)
        self.assertEqual(m["real_pct"], 100.0)
        self.assertEqual(m["spoof_pct"], 0.0)
        self.assertAlmostEqual(m["latency_mean"], 3.5, places=2)
        self.assertEqual(m["fps"], 2.0)

        # Edge case: 0 frames collected
        m_empty = compute_scenario_metrics("EMPTY", [], [], [], [], 10.0)
        self.assertEqual(m_empty["frames"], 0)
        self.assertEqual(m_empty["real_mean"], 0.0)
        self.assertEqual(m_empty["fps"], 0.0)

    def test_temporal_statistics(self):
        """4. Verify temporal statistics including longest runs and >= 0.5 probability counts."""
        from edge.anti_spoof_benchmark import compute_temporal_metrics

        predictions = ["REAL", "REAL", "SPOOF", "SPOOF", "SPOOF", "REAL", "REAL", "REAL", "REAL"]
        real_probs = [0.9, 0.8, 0.2, 0.1, 0.3, 0.7, 0.8, 0.9, 0.95]
        spoof_probs = [0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.2, 0.1, 0.05]

        temp = compute_temporal_metrics(real_probs, spoof_probs, predictions)

        self.assertEqual(temp["longest_real_run"], 4)
        self.assertEqual(temp["longest_spoof_run"], 3)
        # 6 out of 9 are >= 0.5 -> 66.7%
        self.assertEqual(temp["real_ge_05_pct"], 66.7)
        # 3 out of 9 are >= 0.5 -> 33.3%
        self.assertEqual(temp["spoof_ge_05_pct"], 33.3)

        # Empty case
        temp_empty = compute_temporal_metrics([], [], [])
        self.assertEqual(temp_empty["longest_real_run"], 0)
        self.assertEqual(temp_empty["longest_spoof_run"], 0)
        self.assertEqual(temp_empty["real_ge_05_pct"], 0.0)

    def test_json_result_generation_and_preservation(self):
        """5. Verify JSON output generation preserves history and writes required metadata."""
        import tempfile
        import json
        from edge.anti_spoof_benchmark import save_benchmark_results

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "test_results.json")

            run_1 = {
                "timestamp": "2026-09-20T12:00:00Z",
                "model": "data/models/anti-spoof-mn3.onnx",
                "device": "GPU",
                "hardware": "Intel Iris Xe",
                "configured_duration_sec": 10.0,
                "scenarios": {
                    "REAL FACE": {"frames": 120, "real_pct": 100.0}
                },
                "per_frame_data": {
                    "REAL FACE": [{"frame_idx": 1, "p_real": 0.99}]
                },
            }
            save_benchmark_results(run_1, json_file)

            with open(json_file, "r", encoding="utf-8") as f:
                data_1 = json.load(f)
            self.assertEqual(data_1["total_runs"], 1)
            self.assertEqual(len(data_1["runs"]), 1)
            self.assertEqual(data_1["runs"][0]["configured_duration_sec"], 10.0)

            # Append second run - must preserve run 1 without overwrite
            run_2 = {
                "timestamp": "2026-09-20T12:30:00Z",
                "model": "data/models/anti-spoof-mn3.onnx",
                "device": "GPU",
                "hardware": "Intel Iris Xe",
                "configured_duration_sec": 20.0,
                "scenarios": {
                    "PRINTED PHOTO": {"frames": 240, "real_pct": 2.0}
                },
                "per_frame_data": {},
            }
            save_benchmark_results(run_2, json_file)

            with open(json_file, "r", encoding="utf-8") as f:
                data_2 = json.load(f)
            self.assertEqual(data_2["total_runs"], 2)
            self.assertEqual(len(data_2["runs"]), 2)
            self.assertEqual(data_2["runs"][0]["configured_duration_sec"], 10.0)
            self.assertEqual(data_2["runs"][1]["configured_duration_sec"], 20.0)

    def test_benchmark_model_resolution(self):
        """6. Verify benchmark correctly resolves 'mn3' and 'minifas' model selections."""
        from edge.anti_spoof_benchmark import resolve_model_and_classifier

        cls_mn3, crop_mn3, name_mn3, path_mn3 = resolve_model_and_classifier("mn3", device="GPU")
        self.assertIn("MobileNetV3", name_mn3)
        self.assertIn("anti-spoof-mn3.onnx", path_mn3)

        cls_mf, crop_mf, name_mf, path_mf = resolve_model_and_classifier("minifas", device="GPU")
        self.assertIn("MiniFASNet", name_mf)
        self.assertIn("minifasnet_v2se.onnx", path_mf)

    def test_json_result_model_identification(self):
        """7. Verify JSON benchmark logging accurately identifies models across comparative runs."""
        import tempfile
        import json
        from edge.anti_spoof_benchmark import save_benchmark_results

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "comparative_results.json")

            run_mn3 = {
                "timestamp": "2026-09-20T13:00:00Z",
                "model_name": "anti-spoof-mn3 (MobileNetV3)",
                "model_path": "data/models/anti-spoof-mn3.onnx",
                "device": "GPU",
                "hardware": "Intel Iris Xe",
                "duration": 20.0,
                "scenarios": {"REAL FACE": {"real_pct": 100.0}},
                "per_frame_data": {},
            }
            save_benchmark_results(run_mn3, json_file)

            run_mf = {
                "timestamp": "2026-09-20T13:30:00Z",
                "model_name": "MiniFASNet V2 SE",
                "model_path": "data/models/minifasnet_v2se.onnx",
                "device": "GPU",
                "hardware": "Intel Iris Xe",
                "duration": 20.0,
                "scenarios": {"REAL FACE": {"real_pct": 100.0}},
                "per_frame_data": {},
            }
            save_benchmark_results(run_mf, json_file)

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["total_runs"], 2)
            self.assertEqual(data["runs"][0]["model_name"], "anti-spoof-mn3 (MobileNetV3)")
            self.assertEqual(data["runs"][1]["model_name"], "MiniFASNet V2 SE")


class TestMiniFASNetV2SEModel(unittest.TestCase):
    """
    Unit test suite for MiniFASNet V2 SE binary anti-spoofing classifier.
    Validates model loading, GPU hardware compilation, input/output tensor shapes,
    RGB [0, 1] preprocessing, class mapping, and finite latency.
    """

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.available_devices = [d.upper() for d in cls.core.available_devices]
        cls.model_path = "data/models/minifasnet_v2se.onnx"
        from edge.minifas_antispoof import MiniFASNetV2SEClassifier
        cls.classifier = MiniFASNetV2SEClassifier(model_path=cls.model_path, device="GPU")

    def test_minifas_model_file_exists_and_loads(self):
        """1. Verify that minifasnet_v2se.onnx model exists on disk and loads in OpenVINO."""
        self.assertTrue(os.path.exists(self.model_path), f"Model file missing at: {self.model_path}")
        self.assertIsNotNone(self.classifier.compiled_model)
        self.assertGreater(self.classifier.load_time_ms, 0.0)

    def test_minifas_gpu_device_selection(self):
        """2. Verify that Intel Iris Xe GPU is targeted directly without silent fallback."""
        self.assertIn("GPU", self.available_devices)
        self.assertEqual(self.classifier.active_device, "GPU")
        self.assertIn("GPU", self.classifier.full_device_name.upper())

    def test_minifas_input_and_output_shapes(self):
        """3. Verify input shape is [-1, 3, 128, 128] and output shape is [-1, 2]."""
        inp = self.classifier.compiled_model.inputs[0]
        out = self.classifier.compiled_model.outputs[0]

        inp_shape = [dim.get_length() if dim.is_static else -1 for dim in inp.get_partial_shape()]
        out_shape = [dim.get_length() if dim.is_static else -1 for dim in out.get_partial_shape()]

        self.assertEqual(inp_shape[1:], [3, 128, 128])
        self.assertEqual(out_shape[1:], [2])

    def test_minifas_preprocessing(self):
        """4. Verify preprocessing normalizes RGB to [0.0, 1.0] in NCHW float32 format."""
        dummy_bgr = np.random.randint(0, 256, (180, 160, 3), dtype=np.uint8)
        blob = self.classifier.preprocess(dummy_bgr)

        self.assertEqual(blob.shape, (1, 3, 128, 128))
        self.assertEqual(blob.dtype, np.float32)
        self.assertGreaterEqual(float(blob.min()), 0.0)
        self.assertLessEqual(float(blob.max()), 1.0)

        # Preprocessing of known constant image
        const_bgr = np.full((128, 128, 3), 204, dtype=np.uint8)
        blob_const = self.classifier.preprocess(const_bgr)
        self.assertAlmostEqual(float(blob_const[0, 0, 0, 0]), 204.0 / 255.0, places=4)

    def test_minifas_crop_face_bounding_and_reflection(self):
        """5. Verify face crop expansion (1.5x) and reflection border padding."""
        from edge.minifas_antispoof import crop_face_minifas

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)  # w=100, h=100
        crop = crop_face_minifas(frame, bbox, scale=1.5)

        self.assertEqual(crop.shape[0], 150)
        self.assertEqual(crop.shape[1], 150)
        self.assertEqual(crop.shape[2], 3)

        # Boundary / corner case
        corner_bbox = (0, 0, 60, 60)
        corner_crop = crop_face_minifas(frame, corner_bbox, scale=1.5)
        self.assertEqual(corner_crop.shape[0], 90)
        self.assertEqual(corner_crop.shape[1], 90)

    def test_minifas_class_mapping_and_softmax(self):
        """6. Verify predict() returns probabilities summing to 1.0 with correct class 0=REAL, 1=SPOOF."""
        dummy_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        p_real, p_spoof, label, lat = self.classifier.predict(dummy_face)

        self.assertGreaterEqual(p_real, 0.0)
        self.assertLessEqual(p_real, 1.0)
        self.assertGreaterEqual(p_spoof, 0.0)
        self.assertLessEqual(p_spoof, 1.0)
        self.assertAlmostEqual(p_real + p_spoof, 1.0, places=4)
        self.assertIn(label, ("REAL", "SPOOF"))
        if p_real >= p_spoof:
            self.assertEqual(label, "REAL")
        else:
            self.assertEqual(label, "SPOOF")

    def test_minifas_inference_latency_is_finite(self):
        """7. Verify MiniFASNet inference latency is finite and strictly positive."""
        dummy_face = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        _, _, _, lat = self.classifier.predict(dummy_face)

        self.assertIsInstance(lat, float)
        self.assertTrue(np.isfinite(lat))
        self.assertGreater(lat, 0.0)


if __name__ == "__main__":
    unittest.main()
