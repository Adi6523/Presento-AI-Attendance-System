from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure project root is in sys.path when running tests individually
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from edge.recognition_benchmark import (
    FaceObservation,
    calculate_distribution_stats,
    calculate_overall_distributions,
    calculate_scenario_statistics,
    compute_histogram,
    compute_margin,
    evaluate_observation_correctness,
    generate_summary_table,
    save_benchmark_results,
)


class TestRecognitionBenchmark(unittest.TestCase):
    """
    Unit test suite for Phase 2B recognition benchmark validation tool:
    - statistics
    - margin calculation
    - correct/incorrect identity calculation
    - unknown classification statistics
    - JSON serialization
    - distribution aggregation and summary formatting
    """

    # -------------------------------------------------------------------------
    # 1. Statistics Calculation Tests
    # -------------------------------------------------------------------------
    def test_statistics_calculation_standard_values(self):
        """Verify min, max, mean, median, p5, p95 on known values."""
        values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        stats = calculate_distribution_stats(values)

        self.assertEqual(stats["count"], 10)
        self.assertAlmostEqual(stats["min"], 0.10, places=4)
        self.assertAlmostEqual(stats["max"], 1.00, places=4)
        self.assertAlmostEqual(stats["mean"], 0.55, places=4)
        self.assertAlmostEqual(stats["median"], 0.55, places=4)
        self.assertAlmostEqual(stats["p5"], 0.145, places=3)
        self.assertAlmostEqual(stats["p95"], 0.955, places=3)
        self.assertGreater(stats["std"], 0.0)

    def test_statistics_calculation_empty_list(self):
        """Verify empty value list returns safe zeroed dictionary."""
        stats = calculate_distribution_stats([])
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 0.0)
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["median"], 0.0)
        self.assertEqual(stats["p5"], 0.0)
        self.assertEqual(stats["p95"], 0.0)
        self.assertEqual(stats["std"], 0.0)

    def test_statistics_calculation_single_value(self):
        """Verify single observation returns consistent min, max, mean, median, p5, p95."""
        stats = calculate_distribution_stats([0.75])
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["min"], 0.75)
        self.assertAlmostEqual(stats["max"], 0.75)
        self.assertAlmostEqual(stats["mean"], 0.75)
        self.assertAlmostEqual(stats["median"], 0.75)
        self.assertAlmostEqual(stats["p5"], 0.75)
        self.assertAlmostEqual(stats["p95"], 0.75)
        self.assertAlmostEqual(stats["std"], 0.0)

    # -------------------------------------------------------------------------
    # 2. Margin Calculation Tests
    # -------------------------------------------------------------------------
    def test_margin_calculation_with_second_best(self):
        """Verify margin is best_similarity - second_best_similarity."""
        margin = compute_margin(best_similarity=0.82, second_similarity=0.45)
        self.assertAlmostEqual(margin, 0.37, places=5)

    def test_margin_calculation_without_second_best(self):
        """Verify margin falls back to best_similarity when no second candidate exists."""
        margin = compute_margin(best_similarity=0.76, second_similarity=None)
        self.assertAlmostEqual(margin, 0.76, places=5)

    # -------------------------------------------------------------------------
    # 3. Correct / Incorrect Identity Calculation Tests
    # -------------------------------------------------------------------------
    def test_evaluate_observation_correctness(self):
        """Verify top candidate matches expected student ID correctly (case-insensitive)."""
        # Exact match
        self.assertTrue(evaluate_observation_correctness("23UAM090", "23UAM090"))
        # Case-insensitive match with whitespace
        self.assertTrue(evaluate_observation_correctness("  23uam090  ", "23UAM090"))
        # Incorrect match
        self.assertFalse(evaluate_observation_correctness("23UAM092", "23UAM090"))
        # Unknown scenario where expected is None
        self.assertIsNone(evaluate_observation_correctness("23UAM090", None))

    def test_scenario_statistics_top1_accuracy_calculation(self):
        """Verify top-1 accuracy percentage is calculated correctly for known person."""
        obs_list = []
        # 8 correct matches for 23UAM090, 2 mismatches (assigned to 23UAM092)
        for i in range(8):
            obs_list.append(
                FaceObservation(
                    timestamp=float(i),
                    frame_id=i,
                    face_id=0,
                    best_student_id="23UAM090",
                    best_student_name="Aditya Patil",
                    best_similarity=0.78,
                    second_student_id="23UAM092",
                    second_student_name="Piyush",
                    second_similarity=0.35,
                    margin=0.43,
                    is_correct=True,
                )
            )
        for i in range(8, 10):
            obs_list.append(
                FaceObservation(
                    timestamp=float(i),
                    frame_id=i,
                    face_id=0,
                    best_student_id="23UAM092",
                    best_student_name="Piyush",
                    best_similarity=0.68,
                    second_student_id="23UAM090",
                    second_student_name="Aditya Patil",
                    second_similarity=0.65,
                    margin=0.03,
                    is_correct=False,
                )
            )

        sc_stats = calculate_scenario_statistics(
            scenario_name="KNOWN - ADITYA",
            expected_id="23UAM090",
            expected_name="Aditya Patil",
            is_known=True,
            observations=obs_list,
            duration_seconds=20.0,
        )

        self.assertEqual(sc_stats["total_observations"], 10)
        self.assertEqual(sc_stats["top1_correct_count"], 8)
        self.assertAlmostEqual(sc_stats["top1_correct_percent"], 80.0, places=2)
        self.assertAlmostEqual(sc_stats["similarity_stats"]["mean"], 0.76, places=2)
        self.assertAlmostEqual(sc_stats["similarity_stats"]["min"], 0.68, places=2)
        self.assertAlmostEqual(sc_stats["similarity_stats"]["max"], 0.78, places=2)

    # -------------------------------------------------------------------------
    # 4. Unknown Classification Statistics Tests
    # -------------------------------------------------------------------------
    def test_unknown_classification_distribution(self):
        """Verify UNKNOWN scenario properly breaks down top candidate assignments across enrolled IDs."""
        enrolled = [
            {"student_id": "23UAM090", "name": "Aditya Patil"},
            {"student_id": "23UAM092", "name": "Piyush"},
            {"student_id": "23UAM084", "name": "Paryusha"},
        ]

        obs_list = []
        # 6 frames assigned to 23UAM090, 3 to 23UAM092, 1 to 23UAM084
        for i in range(6):
            obs_list.append(
                FaceObservation(
                    timestamp=float(i),
                    frame_id=i,
                    face_id=0,
                    best_student_id="23UAM090",
                    best_student_name="Aditya Patil",
                    best_similarity=0.32,
                    second_student_id="23UAM092",
                    second_student_name="Piyush",
                    second_similarity=0.25,
                    margin=0.07,
                    is_correct=None,
                )
            )
        for i in range(6, 9):
            obs_list.append(
                FaceObservation(
                    timestamp=float(i),
                    frame_id=i,
                    face_id=0,
                    best_student_id="23UAM092",
                    best_student_name="Piyush",
                    best_similarity=0.29,
                    second_student_id="23UAM090",
                    second_student_name="Aditya Patil",
                    second_similarity=0.20,
                    margin=0.09,
                    is_correct=None,
                )
            )
        obs_list.append(
            FaceObservation(
                timestamp=9.0,
                frame_id=9,
                face_id=0,
                best_student_id="23UAM084",
                best_student_name="Paryusha",
                best_similarity=0.25,
                second_student_id="23UAM090",
                second_student_name="Aditya Patil",
                second_similarity=0.18,
                margin=0.07,
                is_correct=None,
            )
        )

        sc_stats = calculate_scenario_statistics(
            scenario_name="UNKNOWN",
            expected_id=None,
            expected_name="Unenrolled Person",
            is_known=False,
            observations=obs_list,
            duration_seconds=20.0,
            enrolled_students=enrolled,
        )

        self.assertIsNone(sc_stats["top1_correct_count"])
        self.assertIsNone(sc_stats["top1_correct_percent"])

        cand_dist = sc_stats["candidate_distribution"]
        self.assertEqual(cand_dist["23UAM090"]["count"], 6)
        self.assertAlmostEqual(cand_dist["23UAM090"]["percent"], 60.0, places=2)

        self.assertEqual(cand_dist["23UAM092"]["count"], 3)
        self.assertAlmostEqual(cand_dist["23UAM092"]["percent"], 30.0, places=2)

        self.assertEqual(cand_dist["23UAM084"]["count"], 1)
        self.assertAlmostEqual(cand_dist["23UAM084"]["percent"], 10.0, places=2)

    # -------------------------------------------------------------------------
    # 5. Overall Distributions & Summary Table Tests
    # -------------------------------------------------------------------------
    def test_overall_distributions_and_separation(self):
        """Verify aggregation of KNOWN and UNKNOWN similarity distributions and separation metrics."""
        known_sc = {
            "is_known": True,
            "observations": [
                {"best_similarity": 0.75, "margin": 0.35},
                {"best_similarity": 0.80, "margin": 0.40},
                {"best_similarity": 0.70, "margin": 0.30},
            ],
        }
        unknown_sc = {
            "is_known": False,
            "observations": [
                {"best_similarity": 0.25, "margin": 0.05},
                {"best_similarity": 0.30, "margin": 0.08},
                {"best_similarity": 0.20, "margin": 0.04},
            ],
        }

        dists = calculate_overall_distributions([known_sc, unknown_sc])

        self.assertEqual(dists["known"]["stats"]["count"], 3)
        self.assertAlmostEqual(dists["known"]["stats"]["mean"], 0.75, places=4)
        self.assertAlmostEqual(dists["known"]["stats"]["min"], 0.70, places=4)

        self.assertEqual(dists["unknown"]["stats"]["count"], 3)
        self.assertAlmostEqual(dists["unknown"]["stats"]["mean"], 0.25, places=4)
        self.assertAlmostEqual(dists["unknown"]["stats"]["max"], 0.30, places=4)

        # Separation check
        sep = dists["separation"]
        self.assertFalse(sep["has_overlap"])
        self.assertGreater(sep["separation_gap_min_to_max"], 0.35)

    def test_generate_summary_table_formatting(self):
        """Verify summary table contains required headers and columns."""
        sc1 = {
            "scenario_name": "KNOWN - ADITYA",
            "expected_id": "23UAM090",
            "top1_correct_percent": 100.0,
            "similarity_stats": {"mean": 0.7654, "min": 0.6900, "max": 0.8200},
            "margin_stats": {"mean": 0.3800},
        }
        sc2 = {
            "scenario_name": "UNKNOWN",
            "expected_id": None,
            "top1_correct_percent": None,
            "similarity_stats": {"mean": 0.2750, "min": 0.1800, "max": 0.3600},
            "margin_stats": {"mean": 0.0750},
        }

        table = generate_summary_table([sc1, sc2])
        self.assertIn("Scenario", table)
        self.assertIn("Expected", table)
        self.assertIn("Top-1 Correct %", table)
        self.assertIn("Best Sim Mean", table)
        self.assertIn("Best Sim Min", table)
        self.assertIn("Best Sim Max", table)
        self.assertIn("Margin Mean", table)
        self.assertIn("KNOWN - ADITYA", table)
        self.assertIn("100.0%", table)
        self.assertIn("UNKNOWN", table)
        self.assertIn("N/A", table)

    # -------------------------------------------------------------------------
    # 6. JSON Serialization Tests
    # -------------------------------------------------------------------------
    def test_json_serialization_with_numpy_types(self):
        """Verify save_benchmark_results seamlessly handles numpy types and saves valid JSON."""
        data_to_save = {
            "timestamp": "2026-09-21T12:00:00Z",
            "float_val": np.float32(0.754),
            "float_val64": np.float64(0.8123),
            "int_val": np.int64(42),
            "int_val32": np.int32(10),
            "array_val": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "nested": {
                "metric": np.float32(0.99),
                "items": [np.int64(1), np.int64(2)],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "test_results.json")
            saved = save_benchmark_results(data_to_save, out_file)

            self.assertTrue(os.path.exists(saved))
            with open(saved, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertAlmostEqual(loaded["float_val"], 0.754, places=3)
            self.assertEqual(loaded["int_val"], 42)
            self.assertEqual(loaded["array_val"], [0.10000000149011612, 0.20000000298023224, 0.30000001192092896][:3])
            self.assertAlmostEqual(loaded["nested"]["metric"], 0.99, places=2)
            self.assertEqual(loaded["nested"]["items"], [1, 2])

    # -------------------------------------------------------------------------
    # 7. End-to-End Benchmark Execution Test
    # -------------------------------------------------------------------------
    @patch("edge.recognition_benchmark.open_camera")
    @patch("edge.recognition_benchmark.SCRFDDetector")
    @patch("edge.recognition_benchmark.ArcFaceEmbedder")
    def test_end_to_end_benchmark_run(self, mock_embedder_cls, mock_detector_cls, mock_open_cam):
        """Verify full run_recognition_benchmark execution across all 6 scenarios."""
        from unittest.mock import MagicMock
        from edge.scrfd_detector import SCRFDFace
        from edge.recognition_benchmark import run_recognition_benchmark

        mock_cam = MagicMock()
        dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        mock_cam.read.return_value = (True, dummy_frame)
        mock_open_cam.return_value = mock_cam

        mock_detector = MagicMock()
        mock_face = SCRFDFace(
            bbox=(100, 100, 300, 300),
            confidence=0.98,
            landmarks=np.array([[150, 150], [250, 150], [200, 200], [160, 260], [240, 260]], dtype=np.float32),
            face_id=0,
        )
        mock_detector.detect.return_value = [mock_face]
        mock_detector_cls.return_value = mock_detector

        mock_embedder = MagicMock()
        dummy_emb = np.zeros(512, dtype=np.float32)
        dummy_emb[0] = 1.0
        mock_embedder.get_embedding.return_value = dummy_emb
        mock_embedder_cls.return_value = mock_embedder

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "bench_results.json")
            results = run_recognition_benchmark(
                source="0",
                duration=0.03,
                output_path=out_file,
                headless=True,
                auto=True,
            )

            self.assertEqual(len(results["scenarios"]), 6)
            self.assertTrue(os.path.exists(out_file))
            self.assertIn("summary_table", results)
            self.assertIn("distributions", results)
            self.assertIn("separation", results["distributions"])


if __name__ == "__main__":
    unittest.main()
