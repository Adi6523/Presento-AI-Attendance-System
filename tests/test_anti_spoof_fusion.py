"""
Unit test suite for Presento Hybrid Anti-Spoof Fusion Engine (edge/anti_spoof_fusion.py).

Validates:
- AND logic
- OR logic
- AVERAGE probability and thresholding
- TEMPORAL REAL condition (>= 80% consensus)
- TEMPORAL SPOOF condition (>= 80% consensus)
- TEMPORAL UNCERTAIN condition (< 80% consensus)
- Configurable window sizing and rolling eviction
- Empty history handling
- Disagreement handling
- Invalid probability validation (bounds, NaN, Inf, non-numeric)
"""
from __future__ import annotations

import unittest
import numpy as np

from edge.anti_spoof_fusion import (
    AntiSpoofFusionEngine,
    FrameEvaluation,
    fuse_and,
    fuse_or,
    fuse_average,
    fuse_temporal,
    validate_probability,
)


class TestAntiSpoofFusion(unittest.TestCase):

    def test_and_logic(self):
        """1. Verify AND strategy requires strict consensus (both REAL)."""
        self.assertEqual(fuse_and(True, True), "REAL")
        self.assertEqual(fuse_and(True, False), "SPOOF")
        self.assertEqual(fuse_and(False, True), "SPOOF")
        self.assertEqual(fuse_and(False, False), "SPOOF")

    def test_or_logic(self):
        """2. Verify OR strategy accepts lenient consensus (either REAL)."""
        self.assertEqual(fuse_or(True, True), "REAL")
        self.assertEqual(fuse_or(True, False), "REAL")
        self.assertEqual(fuse_or(False, True), "REAL")
        self.assertEqual(fuse_or(False, False), "SPOOF")

    def test_average_probability(self):
        """3. Verify AVERAGE strategy combines real probabilities and respects 0.50 boundary."""
        avg_r, avg_s, label = fuse_average(0.9, 0.3)
        self.assertAlmostEqual(avg_r, 0.60, places=4)
        self.assertAlmostEqual(avg_s, 0.40, places=4)
        self.assertEqual(label, "REAL")

        avg_r, avg_s, label = fuse_average(0.2, 0.4)
        self.assertAlmostEqual(avg_r, 0.30, places=4)
        self.assertAlmostEqual(avg_s, 0.70, places=4)
        self.assertEqual(label, "SPOOF")

        # Boundary at exactly 0.50
        avg_r, avg_s, label = fuse_average(0.5, 0.5)
        self.assertAlmostEqual(avg_r, 0.50, places=4)
        self.assertEqual(label, "REAL")

        # Custom threshold
        _, _, label_high = fuse_average(0.6, 0.6, threshold=0.70)
        self.assertEqual(label_high, "SPOOF")

    def test_temporal_real_condition(self):
        """4. Verify TEMPORAL strategy outputs REAL when >= 80% of window frames are both_real."""
        engine = AntiSpoofFusionEngine(window_size=15, real_ratio_threshold=0.80)

        # Feed 12 both_real frames and 3 disagreement frames (12/15 = 80%)
        for _ in range(12):
            res = engine.update(mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.90, mf_p_spoof=0.10)
        for _ in range(3):
            res = engine.update(mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.10, mf_p_spoof=0.90)

        self.assertEqual(res.temporal_label, "REAL")
        self.assertAlmostEqual(res.both_real_ratio, 12 / 15, places=3)
        self.assertEqual(res.window_frames, 15)

    def test_temporal_spoof_condition(self):
        """5. Verify TEMPORAL strategy outputs SPOOF when >= 80% of window frames are both_spoof."""
        engine = AntiSpoofFusionEngine(window_size=15, spoof_ratio_threshold=0.80)

        # Feed 13 both_spoof frames and 2 both_real frames (13/15 = 86.7%)
        for _ in range(13):
            res = engine.update(mn3_p_real=0.05, mn3_p_spoof=0.95, mf_p_real=0.10, mf_p_spoof=0.90)
        for _ in range(2):
            res = engine.update(mn3_p_real=0.90, mn3_p_spoof=0.10, mf_p_real=0.90, mf_p_spoof=0.10)

        self.assertEqual(res.temporal_label, "SPOOF")
        self.assertAlmostEqual(res.both_spoof_ratio, 13 / 15, places=3)

    def test_temporal_uncertain_condition(self):
        """6. Verify TEMPORAL strategy outputs UNCERTAIN when neither consensus reaches 80%."""
        engine = AntiSpoofFusionEngine(window_size=15, real_ratio_threshold=0.80, spoof_ratio_threshold=0.80)

        # Feed 7 both_real frames and 8 both_spoof frames (neither >= 80%)
        for _ in range(7):
            res = engine.update(mn3_p_real=0.95, mn3_p_spoof=0.05, mf_p_real=0.90, mf_p_spoof=0.10)
        for _ in range(8):
            res = engine.update(mn3_p_real=0.05, mn3_p_spoof=0.95, mf_p_real=0.05, mf_p_spoof=0.95)

        self.assertEqual(res.temporal_label, "UNCERTAIN")

    def test_window_size_and_eviction(self):
        """7. Verify configurable window size limits history and rolls out older frames."""
        engine = AntiSpoofFusionEngine(window_size=5)

        # Feed 5 SPOOF frames
        for _ in range(5):
            res = engine.update(0.1, 0.9, 0.1, 0.9)
        self.assertEqual(res.temporal_label, "SPOOF")
        self.assertEqual(len(engine.history), 5)

        # Feed 5 REAL frames -> old SPOOF frames must be completely evicted
        for _ in range(5):
            res = engine.update(0.9, 0.1, 0.9, 0.1)
        self.assertEqual(res.temporal_label, "REAL")
        self.assertEqual(len(engine.history), 5)
        self.assertEqual(res.both_real_ratio, 1.0)
        self.assertEqual(res.both_spoof_ratio, 0.0)

    def test_empty_history_handling(self):
        """8. Verify empty history returns UNCERTAIN with 0.0 ratios and engine reset works."""
        decision, r_ratio, s_ratio = fuse_temporal([])
        self.assertEqual(decision, "UNCERTAIN")
        self.assertEqual(r_ratio, 0.0)
        self.assertEqual(s_ratio, 0.0)

        engine = AntiSpoofFusionEngine(window_size=10)
        engine.update(0.9, 0.1, 0.9, 0.1)
        self.assertEqual(len(engine.history), 1)
        engine.reset()
        self.assertEqual(len(engine.history), 0)

    def test_disagreement_handling(self):
        """9. Verify persistent disagreement between models produces UNCERTAIN temporal output."""
        engine = AntiSpoofFusionEngine(window_size=15)

        for _ in range(15):
            res = engine.update(
                mn3_p_real=0.95, mn3_p_spoof=0.05,  # MN3 says REAL
                mf_p_real=0.05, mf_p_spoof=0.95,   # MiniFAS says SPOOF
            )

        self.assertTrue(res.disagreement)
        self.assertFalse(res.both_real)
        self.assertFalse(res.both_spoof)
        self.assertEqual(res.and_label, "SPOOF")
        self.assertEqual(res.or_label, "REAL")
        self.assertAlmostEqual(res.avg_p_real, 0.50, places=3)
        self.assertEqual(res.temporal_label, "UNCERTAIN")

    def test_invalid_probability_handling(self):
        """10. Verify validation rejects negative values, > 1.0, NaN, Inf, and non-numeric types."""
        with self.assertRaises(ValueError):
            validate_probability(-0.01)
        with self.assertRaises(ValueError):
            validate_probability(1.01)
        with self.assertRaises(ValueError):
            validate_probability(float("nan"))
        with self.assertRaises(ValueError):
            validate_probability(float("inf"))
        with self.assertRaises(ValueError):
            validate_probability("0.5")  # type: ignore

        # Valid boundaries
        self.assertEqual(validate_probability(0.0), 0.0)
        self.assertEqual(validate_probability(1.0), 1.0)
        self.assertEqual(validate_probability(0.5), 0.5)

        # Engine rejection
        engine = AntiSpoofFusionEngine()
        with self.assertRaises(ValueError):
            engine.update(-0.1, 1.1, 0.5, 0.5)


if __name__ == "__main__":
    unittest.main()
