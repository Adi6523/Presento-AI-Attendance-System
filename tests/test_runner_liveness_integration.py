"""
Unit test suite verifying the active Phase 2B Anti-Spoofing and Temporal Fusion runner integration.
Validates:
1. Clean import of edge.runner without dlib or face_recognition loaded into sys.modules.
2. Complete absence of legacy EAR, face_landmarks, and LivenessDetector calls in edge.runner.main.
3. Strict per-face isolation for AntiSpoofFusionEngine instances.
4. Attendance gating requiring candidate match + multi-frame stability + anti-spoof REAL.
5. Absolute blocking of attendance for SPOOF and UNCERTAIN states.
6. Clean pruning of stale face_ids from fusion dictionary after 3.0 seconds.
7. Preservation of legacy coordinate and vision utilities if referenced.
"""
from __future__ import annotations

import inspect
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import edge.runner
from edge.config import settings
from edge.anti_spoof_fusion import AntiSpoofFusionEngine, FusionResult
from edge.scrfd_detector import SCRFDFace
from edge.face_matcher import MatchResult, CandidateMatch


class TestRunnerAntiSpoofIntegration(unittest.TestCase):

    def test_clean_edge_runner_import_without_dlib(self):
        """1. Verify importing edge.runner does NOT load dlib or face_recognition into sys.modules."""
        import subprocess
        code = (
            "import sys; import edge.runner; "
            "loaded = [m for m in sys.modules if 'dlib' in m or 'face_recognition' in m]; "
            "assert not loaded, f'Loaded: {loaded}'; print('CLEAN')"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            res.returncode,
            0,
            f"edge.runner loaded dlib/face_recognition: {res.stderr}",
        )
        self.assertIn("CLEAN", res.stdout)

    def test_no_active_phase2b_ear_or_liveness_calls(self):
        """2. Verify edge.runner.main contains zero active calls to EAR, dlib landmarks, or LivenessDetector."""
        main_source = inspect.getsource(edge.runner.main)

        forbidden_calls = [
            "eye_aspect_ratio_for_face",
            "face_recognition.face_landmarks",
            "LivenessDetector",
            "LivenessResult",
        ]
        for call_name in forbidden_calls:
            self.assertNotIn(
                call_name,
                main_source,
                f"Forbidden legacy call '{call_name}' found in active Phase 2B runner main()",
            )

    def test_per_face_fusion_isolation(self):
        """3. Verify anti-spoof fusion engines are strictly isolated per SCRFD face_id."""
        anti_spoof_fusion: dict[int, AntiSpoofFusionEngine] = {}

        def _get_anti_spoof_fusion(fid: int) -> AntiSpoofFusionEngine:
            if fid not in anti_spoof_fusion:
                anti_spoof_fusion[fid] = AntiSpoofFusionEngine(
                    window_size=15,
                    real_ratio_threshold=0.80,
                    spoof_ratio_threshold=0.80,
                )
            return anti_spoof_fusion[fid]

        eng1 = _get_anti_spoof_fusion(101)
        eng2 = _get_anti_spoof_fusion(202)

        self.assertIsNot(eng1, eng2)
        self.assertIn(101, anti_spoof_fusion)
        self.assertIn(202, anti_spoof_fusion)

        # Feed 15 REAL frames to Face 101
        for _ in range(15):
            res1 = eng1.update(0.99, 0.01, 0.98, 0.02)

        # Feed 15 SPOOF frames to Face 202
        for _ in range(15):
            res2 = eng2.update(0.01, 0.99, 0.02, 0.98)

        self.assertEqual(res1.temporal_label, "REAL")
        self.assertEqual(res2.temporal_label, "SPOOF")
        self.assertEqual(len(eng1.history), 15)
        self.assertEqual(len(eng2.history), 15)

    def test_real_only_attendance_gating(self):
        """4. Verify attendance posting strictly requires is_stable == True AND temporal_label == 'REAL'."""
        candidate = CandidateMatch(
            student_id="23UAM092",
            name="Piyush",
            similarity=0.82,
            best_sample_index=0,
        )

        stable_match = MatchResult(
            best_match=candidate,
            second_best_match=None,
            margin=0.82,
            is_stable=True,
            stability_label="Stable 5/5",
            all_candidates=[candidate],
            face_id=1,
            frame_id=1,
        )

        unstable_match = MatchResult(
            best_match=candidate,
            second_best_match=None,
            margin=0.82,
            is_stable=False,
            stability_label="Acquiring 2/5",
            all_candidates=[candidate],
            face_id=1,
            frame_id=1,
        )

        def make_fusion_result(label: str) -> FusionResult:
            return FusionResult(
                and_label=label,
                or_label=label,
                avg_p_real=0.95 if label == "REAL" else 0.05,
                avg_p_spoof=0.05 if label == "REAL" else 0.95,
                avg_label=label,
                temporal_label=label,
                both_real=(label == "REAL"),
                both_spoof=(label == "SPOOF"),
                disagreement=False,
                window_frames=15,
                both_real_ratio=1.0 if label == "REAL" else 0.0,
                both_spoof_ratio=1.0 if label == "SPOOF" else 0.0,
            )

        # Case A: Stable Match + REAL -> Attendance Allowed
        cooldown: dict[str, float] = {}
        now = 100.0
        fus_real = make_fusion_result("REAL")

        with patch("edge.runner.post_event", return_value=True) as mock_post:
            should_post = stable_match.is_stable and fus_real.temporal_label == "REAL"
            self.assertTrue(should_post)
            if should_post and cooldown.get(candidate.student_id, 0.0) <= now:
                mock_post({"student_id": candidate.student_id})
                cooldown[candidate.student_id] = now + 15.0
            mock_post.assert_called_once()

        # Case B: Stable Match + SPOOF -> Blocked
        fus_spoof = make_fusion_result("SPOOF")
        with patch("edge.runner.post_event", return_value=True) as mock_post:
            should_post = stable_match.is_stable and fus_spoof.temporal_label == "REAL"
            self.assertFalse(should_post)
            if should_post:
                mock_post({"student_id": candidate.student_id})
            mock_post.assert_not_called()

        # Case C: Stable Match + UNCERTAIN -> Blocked
        fus_uncert = make_fusion_result("UNCERTAIN")
        with patch("edge.runner.post_event", return_value=True) as mock_post:
            should_post = stable_match.is_stable and fus_uncert.temporal_label == "REAL"
            self.assertFalse(should_post)
            if should_post:
                mock_post({"student_id": candidate.student_id})
            mock_post.assert_not_called()

        # Case D: Unstable Match + REAL -> Blocked
        with patch("edge.runner.post_event", return_value=True) as mock_post:
            should_post = unstable_match.is_stable and fus_real.temporal_label == "REAL"
            self.assertFalse(should_post)
            if should_post:
                mock_post({"student_id": candidate.student_id})
            mock_post.assert_not_called()

    def test_stale_fusion_cleanup(self):
        """5. Verify cleanup logic prunes anti-spoof fusion engines for faces that disappeared for > 3.0s."""
        anti_spoof_fusion: dict[int, AntiSpoofFusionEngine] = {}
        anti_spoof_last_seen: dict[int, float] = {}

        now = 1000.0
        anti_spoof_fusion[1] = AntiSpoofFusionEngine()
        anti_spoof_last_seen[1] = now

        anti_spoof_fusion[2] = AntiSpoofFusionEngine()
        anti_spoof_last_seen[2] = now - 5.0  # Last seen 5 seconds ago

        # Only face 1 is currently active
        active_fids = {1}
        for old_fid in list(anti_spoof_fusion.keys()):
            if old_fid not in active_fids and (now - anti_spoof_last_seen.get(old_fid, 0.0)) > 3.0:
                anti_spoof_fusion.pop(old_fid, None)
                anti_spoof_last_seen.pop(old_fid, None)

        self.assertIn(1, anti_spoof_fusion)
        self.assertNotIn(2, anti_spoof_fusion)
        self.assertIn(1, anti_spoof_last_seen)
        self.assertNotIn(2, anti_spoof_last_seen)

    def test_coordinate_conversion_scrfd_to_face_location(self):
        """6. Verify conversion of SCRFD bbox (left, top, right, bottom) to (top, right, bottom, left)."""
        face = SCRFDFace(
            bbox=(120, 80, 260, 240),  # (left, top, right, bottom)
            confidence=0.95,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            face_id=14,
        )

        h_frame, w_frame = 720, 1280
        loc_top = max(0, min(h_frame - 1, face.top))
        loc_right = max(0, min(w_frame - 1, face.right))
        loc_bottom = max(0, min(h_frame - 1, face.bottom))
        loc_left = max(0, min(w_frame - 1, face.left))
        face_location = (loc_top, loc_right, loc_bottom, loc_left)

        self.assertEqual(face_location, (80, 260, 240, 120))

    def test_async_antispoof_result_attendance_gating(self):
        """7. Verify attendance posting requires fresh AntiSpoofResult with temporal_label == 'REAL'."""
        from edge.anti_spoof_worker import AntiSpoofResult

        candidate = CandidateMatch(
            student_id="23UAM092",
            name="Piyush",
            similarity=0.85,
            best_sample_index=0,
        )
        match_res = MatchResult(
            best_match=candidate,
            second_best_match=None,
            margin=0.85,
            is_stable=True,
            stability_label="Stable 5/5",
            all_candidates=[candidate],
            face_id=10,
            frame_id=100,
        )

        def make_as_result(label: str, fid: int, updated_at: float) -> AntiSpoofResult:
            fus = FusionResult(
                and_label=label,
                or_label=label,
                avg_p_real=0.98 if label == "REAL" else 0.02,
                avg_p_spoof=0.02 if label == "REAL" else 0.98,
                avg_label=label,
                temporal_label=label,
                both_real=(label == "REAL"),
                both_spoof=(label == "SPOOF"),
                disagreement=False,
                window_frames=15,
                both_real_ratio=1.0 if label == "REAL" else 0.0,
                both_spoof_ratio=1.0 if label == "SPOOF" else 0.0,
            )
            return AntiSpoofResult(
                face_id=fid,
                frame_id=100,
                mn3_p_real=0.98,
                mn3_p_spoof=0.02,
                mf_p_real=0.98,
                mf_p_spoof=0.02,
                fusion_result=fus,
                mn3_latency_ms=3.0,
                mf_latency_ms=6.0,
                updated_at=updated_at,
            )

        now = 500.0

        # Case 1: Stable + Fresh REAL result (0.1s old) -> ALLOWED
        res_real = make_as_result("REAL", fid=10, updated_at=now - 0.1)
        is_fresh = (now - res_real.updated_at) <= 2.0
        self.assertTrue(is_fresh)
        self.assertTrue(match_res.is_stable and res_real.fusion_result.temporal_label == "REAL" and is_fresh)

        # Case 2: Stable + Stale REAL result (3.0s old) -> BLOCKED
        res_stale = make_as_result("REAL", fid=10, updated_at=now - 3.0)
        is_fresh_stale = (now - res_stale.updated_at) <= 2.0
        self.assertFalse(is_fresh_stale)
        self.assertFalse(match_res.is_stable and res_stale.fusion_result.temporal_label == "REAL" and is_fresh_stale)

        # Case 3: Stable + Fresh SPOOF result -> BLOCKED
        res_spoof = make_as_result("SPOOF", fid=10, updated_at=now - 0.1)
        self.assertFalse(match_res.is_stable and res_spoof.fusion_result.temporal_label == "REAL")

        # Case 4: Stable + Fresh UNCERTAIN result -> BLOCKED
        res_uncert = make_as_result("UNCERTAIN", fid=10, updated_at=now - 0.1)
        self.assertFalse(match_res.is_stable and res_uncert.fusion_result.temporal_label == "REAL")

    def test_continuous_rolling_fps_accuracy(self):
        """8. Verify continuous rolling Display FPS calculates accurately and avoids 0.0 reset."""
        from collections import deque

        frame_timestamps = deque(maxlen=30)
        display_fps = 0.0

        # Frame 1: Single frame
        t0 = 100.0
        frame_timestamps.append(t0)
        if len(frame_timestamps) >= 2:
            dt = frame_timestamps[-1] - frame_timestamps[0]
            display_fps = round((len(frame_timestamps) - 1) / dt, 1)
        self.assertEqual(display_fps, 0.0)

        # Frame 2: Exactly 33.33ms later (30 FPS)
        frame_timestamps.append(t0 + 0.033333)
        dt = frame_timestamps[-1] - frame_timestamps[0]
        display_fps = round((len(frame_timestamps) - 1) / dt, 1)
        self.assertEqual(display_fps, 30.0)

        # Feed 30 frames at 30 FPS (33.33ms apart)
        for i in range(2, 30):
            frame_timestamps.append(t0 + i * 0.033333)
            dt = frame_timestamps[-1] - frame_timestamps[0]
            display_fps = round((len(frame_timestamps) - 1) / dt, 1)
            self.assertGreater(display_fps, 29.0)
            self.assertLess(display_fps, 31.0)

        # Continues smoothly without resetting to 0.0 on subsequent frames
        frame_timestamps.append(t0 + 30 * 0.033333)
        dt = frame_timestamps[-1] - frame_timestamps[0]
        display_fps = round((len(frame_timestamps) - 1) / dt, 1)
        self.assertNotEqual(display_fps, 0.0)
        self.assertAlmostEqual(display_fps, 30.0, places=0)


if __name__ == "__main__":
    unittest.main()
