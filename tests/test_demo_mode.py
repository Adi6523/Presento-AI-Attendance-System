"""
Unit and integration test suite for Presento AI Vision Demo mode.
Validates:
1. State classification engine (UNKNOWN, RECOGNIZED, CHECKING, SPOOF, UNCERTAIN, PRESENT).
2. DemoMetricsTracker accumulation and event logging.
3. All HUD and canvas rendering functions across Modes 1 through 5.
4. Guaranteed startup and shutdown lifecycle (release of workers and CameraSession).
5. Clean import without dlib in sys.modules.
"""
from __future__ import annotations

import collections
import inspect
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import edge.demo as demo
from edge.scrfd_detector import SCRFDFace
from edge.face_matcher import CandidateMatch, MatchResult
from edge.anti_spoof_worker import AntiSpoofResult
from edge.anti_spoof_fusion import FusionResult


class TestDemoMode(unittest.TestCase):

    def test_clean_import_without_dlib(self):
        """Verify importing edge.demo does NOT load dlib or face_recognition into sys.modules."""
        import subprocess
        code = (
            "import sys; import edge.demo; "
            "loaded = [m for m in sys.modules if 'dlib' in m or 'face_recognition' in m]; "
            "assert not loaded, f'Loaded: {loaded}'; print('CLEAN_DEMO')"
        )
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"edge.demo loaded dlib: {res.stderr}")
        self.assertIn("CLEAN_DEMO", res.stdout)

    def test_state_classification_checking_when_embedding_none(self):
        """State is CHECKING when embedding is still being extracted."""
        face = SCRFDFace(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            embedding=None,
            face_id=1,
        )
        state, col, reason = demo.classify_face_state(
            face=face,
            match_res=None,
            as_res=None,
            cooldown_rem=0.0,
            tolerance=0.48,
        )
        self.assertEqual(state, demo.STATE_CHECKING)
        self.assertIn("Extracting", reason)

    def test_state_classification_unknown_when_no_match_or_low_similarity(self):
        """State is UNKNOWN when similarity is below tolerance or no candidate matches."""
        face = SCRFDFace(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            embedding=np.zeros(512, dtype=np.float32),
            face_id=2,
        )
        # Low similarity match (0.30 < 0.48)
        cand = CandidateMatch(student_id="DEMO001", name="Demo Student", similarity=0.30, best_sample_index=0)
        match_res = MatchResult(
            best_match=cand,
            second_best_match=None,
            margin=0.30,
            is_stable=False,
            stability_label="Unstable",
            all_candidates=[cand],
            face_id=2,
            frame_id=1,
        )
        state, col, reason = demo.classify_face_state(
            face=face,
            match_res=match_res,
            as_res=None,
            cooldown_rem=0.0,
            tolerance=0.48,
        )
        self.assertEqual(state, demo.STATE_UNKNOWN)

    def test_state_classification_spoof_and_uncertain(self):
        """State correctly reflects SPOOF and UNCERTAIN fusion labels."""
        face = SCRFDFace(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            embedding=np.zeros(512, dtype=np.float32),
            face_id=3,
        )
        cand = CandidateMatch(student_id="DEMO001", name="Demo Student", similarity=0.75, best_sample_index=0)
        match_res = MatchResult(
            best_match=cand,
            second_best_match=None,
            margin=0.75,
            is_stable=True,
            stability_label="Candidate stable",
            all_candidates=[cand],
            face_id=3,
            frame_id=1,
        )

        # 1. Spoof Result
        spoof_fus = FusionResult(
            and_label="SPOOF",
            or_label="SPOOF",
            avg_p_real=0.05,
            avg_p_spoof=0.95,
            avg_label="SPOOF",
            temporal_label="SPOOF",
            both_real=False,
            both_spoof=True,
            disagreement=False,
            window_frames=15,
            both_real_ratio=0.0,
            both_spoof_ratio=0.95,
        )
        as_spoof = AntiSpoofResult(
            face_id=3,
            frame_id=1,
            mn3_p_real=0.05,
            mn3_p_spoof=0.95,
            mf_p_real=0.05,
            mf_p_spoof=0.95,
            fusion_result=spoof_fus,
            mn3_latency_ms=10.0,
            mf_latency_ms=10.0,
            updated_at=100.0,
        )
        state_spoof, col_s, _ = demo.classify_face_state(face, match_res, as_spoof, 0.0, 0.48)
        self.assertEqual(state_spoof, demo.STATE_SPOOF)

        # 2. Uncertain Result
        unc_fus = FusionResult(
            and_label="SPOOF",
            or_label="REAL",
            avg_p_real=0.50,
            avg_p_spoof=0.50,
            avg_label="REAL",
            temporal_label="UNCERTAIN",
            both_real=False,
            both_spoof=False,
            disagreement=True,
            window_frames=15,
            both_real_ratio=0.5,
            both_spoof_ratio=0.5,
        )
        as_unc = AntiSpoofResult(
            face_id=3,
            frame_id=1,
            mn3_p_real=0.90,
            mn3_p_spoof=0.10,
            mf_p_real=0.10,
            mf_p_spoof=0.90,
            fusion_result=unc_fus,
            mn3_latency_ms=10.0,
            mf_latency_ms=10.0,
            updated_at=100.0,
        )
        state_unc, col_u, _ = demo.classify_face_state(face, match_res, as_unc, 0.0, 0.48)
        self.assertEqual(state_unc, demo.STATE_UNCERTAIN)

    def test_state_classification_present_and_recognized(self):
        """State is PRESENT when confirmed stable and real, or actively in cooldown."""
        face = SCRFDFace(
            bbox=(100, 100, 200, 200),
            confidence=0.95,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            embedding=np.zeros(512, dtype=np.float32),
            face_id=4,
        )
        cand = CandidateMatch(student_id="DEMO001", name="Demo Student", similarity=0.75, best_sample_index=0)
        match_stable = MatchResult(
            best_match=cand,
            second_best_match=None,
            margin=0.75,
            is_stable=True,
            stability_label="Candidate stable",
            all_candidates=[cand],
            face_id=4,
            frame_id=1,
        )
        real_fus = FusionResult(
            and_label="REAL",
            or_label="REAL",
            avg_p_real=0.95,
            avg_p_spoof=0.05,
            avg_label="REAL",
            temporal_label="REAL",
            both_real=True,
            both_spoof=False,
            disagreement=False,
            window_frames=15,
            both_real_ratio=0.95,
            both_spoof_ratio=0.0,
        )
        as_real = AntiSpoofResult(
            face_id=4,
            frame_id=1,
            mn3_p_real=0.95,
            mn3_p_spoof=0.05,
            mf_p_real=0.95,
            mf_p_spoof=0.05,
            fusion_result=real_fus,
            mn3_latency_ms=10.0,
            mf_latency_ms=10.0,
            updated_at=100.0,
        )

        # Stable + REAL -> PRESENT
        state, col, _ = demo.classify_face_state(face, match_stable, as_real, 0.0, 0.48)
        self.assertEqual(state, demo.STATE_PRESENT)

        # In cooldown -> PRESENT
        state_cd, _, _ = demo.classify_face_state(face, match_stable, as_real, 45.0, 0.48)
        self.assertEqual(state_cd, demo.STATE_PRESENT)

        # Unstable + REAL -> RECOGNIZED
        match_unstable = MatchResult(
            best_match=cand,
            second_best_match=None,
            margin=0.75,
            is_stable=False,
            stability_label="Candidate (2/5)",
            all_candidates=[cand],
            face_id=4,
            frame_id=1,
        )
        state_rec, _, _ = demo.classify_face_state(face, match_unstable, as_real, 0.0, 0.48)
        self.assertEqual(state_rec, demo.STATE_RECOGNIZED)

    def test_metrics_tracker(self):
        """Verify DemoMetricsTracker increments counts and records attendance log."""
        tracker = demo.DemoMetricsTracker()
        face = SCRFDFace(
            bbox=(10, 10, 50, 50),
            confidence=0.9,
            landmarks=np.zeros((5, 2)),
            face_id=10,
        )
        tracker.register_faces(
            faces=[face],
            states={10: demo.STATE_PRESENT},
            matches={},
            anti_spoof={},
        )
        self.assertEqual(len(tracker.seen_face_ids), 1)
        self.assertEqual(tracker.recognized_encounters, 1)

        tracker.log_attendance("Demo Student 1", "DEMO001", 0.78, "REAL (95%)")
        self.assertEqual(tracker.attendance_events, 1)
        self.assertEqual(len(tracker.recent_logs), 1)
        self.assertEqual(tracker.recent_logs[0].name, "Demo Student 1")

    def test_hud_rendering_functions(self):
        """Verify that all HUD renderers execute cleanly across all 5 modes without error."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Glass rect, corners, badge
        demo.draw_glass_rect(frame, 10, 10, 100, 100, bg_color=(20, 20, 20), alpha=0.5)
        demo.draw_tech_corners(frame, 50, 50, 150, 150, color=(0, 255, 0))
        demo.draw_pill_badge(frame, "TEST", 20, 20, bg_color=(0, 255, 0))

        # 2. Top Header HUD & Bottom Mode Banner for Modes 1-5
        for mode in range(1, 6):
            demo.render_top_header_hud(
                img=frame,
                current_mode=mode,
                display_fps=29.8,
                det_fps=30.0,
                arc_fps=14.5,
                arc_lat=12.2,
                enrolled_count=5,
                camera_active=True,
                is_paused=False,
            )
            demo.render_bottom_mode_banner(img=frame, current_mode=mode)

        # 3. Pipeline Debug Panel
        demo.render_pipeline_debug_panel(
            img=frame,
            det_fps=30.0,
            arc_fps=14.5,
            arc_lat=12.2,
            as_lat=18.5,
            enrolled_count=5,
            active_faces=1,
            camera_source="10.0.0.50:5001",
        )

        # 4. Mode 5 Summary Dashboard
        tracker = demo.DemoMetricsTracker()
        tracker.log_attendance("Demo Student 1", "DEMO001", 0.784, "REAL (93%)")
        demo.render_mode5_summary_dashboard(
            img=frame,
            tracker=tracker,
            display_fps=29.8,
            det_fps=30.0,
            arc_fps=14.5,
            enrolled_count=5,
        )

        # 5. Face Callout Card across Modes 1-5
        face = SCRFDFace(
            bbox=(200, 150, 450, 400),
            confidence=0.98,
            landmarks=np.array([[250, 220], [350, 220], [300, 270], [270, 320], [330, 320]], dtype=np.float32),
            embedding=np.zeros(512, dtype=np.float32),
            face_id=1,
        )
        cand = CandidateMatch(student_id="DEMO001", name="Demo Student 1", similarity=0.78, best_sample_index=0)
        cand2 = CandidateMatch(student_id="DEMO002", name="Demo Student 2", similarity=0.25, best_sample_index=1)
        match_res = MatchResult(
            best_match=cand,
            second_best_match=cand2,
            margin=0.53,
            is_stable=True,
            stability_label="STABLE",
            all_candidates=[cand, cand2],
            face_id=1,
            frame_id=1,
        )
        fus = FusionResult(
            and_label="REAL",
            or_label="REAL",
            avg_p_real=0.94,
            avg_p_spoof=0.06,
            avg_label="REAL",
            temporal_label="REAL",
            both_real=True,
            both_spoof=False,
            disagreement=False,
            window_frames=15,
            both_real_ratio=0.94,
            both_spoof_ratio=0.0,
        )
        as_res = AntiSpoofResult(
            face_id=1,
            frame_id=1,
            mn3_p_real=0.93,
            mn3_p_spoof=0.07,
            mf_p_real=0.95,
            mf_p_spoof=0.05,
            fusion_result=fus,
            mn3_latency_ms=8.0,
            mf_latency_ms=9.0,
            updated_at=10.0,
        )

        for mode in range(1, 6):
            demo.render_face_callout_card(
                img=frame,
                face=face,
                m_bbox=(200, 150, 450, 400),
                m_landmarks=face.landmarks,
                match_res=match_res,
                as_res=as_res,
                cooldown_rem=0.0,
                current_mode=mode,
                tolerance=0.48,
            )

        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_run_demo_lifecycle_and_cleanup(self):
        """Verify run_demo handles mock camera, runs N frames headless, and releases resources."""
        mock_camera = MagicMock()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_camera.read.return_value = (True, dummy_frame)
        mock_camera.isOpened.return_value = True

        mock_session = MagicMock()

        with patch("edge.demo.CameraSession", return_value=mock_session), \
             patch("cv2.VideoCapture", return_value=mock_camera), \
             patch("edge.demo.SCRFDWorker") as mock_detector_cls, \
             patch("edge.demo.AntiSpoofWorker") as mock_as_cls, \
             patch("edge.demo.AttendanceIOWorker") as mock_att_cls:

            mock_det_instance = mock_detector_cls.return_value
            mock_det_instance.get_latest_detections.return_value = []
            mock_det_instance.get_stats.return_value = {"detection_fps": 30.0, "arcface_fps": 15.0}

            demo.run_demo(
                source="mock",
                initial_mode=1,
                headless=True,
                max_frames=3,
            )

            # Assert workers were started
            mock_det_instance.start.assert_called_once()
            mock_as_cls.return_value.start.assert_called_once()
            mock_att_cls.return_value.start.assert_called_once()

            # Assert cleanup was called
            mock_det_instance.stop.assert_called_once()
            mock_as_cls.return_value.stop.assert_called_once()
            mock_att_cls.return_value.stop.assert_called_once()
            mock_session.stop.assert_called_once()

    def test_in_gui_enrollment_dialog_renderers(self):
        """Verify all non-blocking in-window enrollment dialogs render without error on canvas."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Step 1: ID input modal
        demo.render_enrollment_input_dialog(
            img=canvas,
            step_title="DEMO ENROLLMENT - ENTER STUDENT ID (Step 1 of 3)",
            field_label="STUDENT ID (e.g. 23UAM123):",
            value="23UAM888",
            hint_text="Type ID, then press [ENTER] | [ESC] Cancel",
            extra_info="Enroll new student into Presento local face database",
            cursor_visible=True,
        )
        self.assertEqual(canvas.shape, (720, 1280, 3))

        # 2. Step 2: Name input modal
        demo.render_enrollment_input_dialog(
            img=canvas,
            step_title="DEMO ENROLLMENT - ENTER STUDENT NAME (Step 2 of 3)",
            field_label="FULL NAME (e.g. John Doe):",
            value="Alex Morgan",
            hint_text="Type name, then press [ENTER] | [ESC] Cancel",
            extra_info="Target Student ID: 23UAM888",
            cursor_visible=False,
        )

        # 3. Step 3: Confirmation modal
        demo.render_enrollment_confirm_dialog(
            img=canvas,
            student_id="23UAM888",
            student_name="Alex Morgan",
            target_samples=16,
        )

        # 4. Step 4: Capture overlay with guidance and progress bar
        demo.render_enrollment_capture_overlay(
            img=canvas,
            student_name="Alex Morgan",
            student_id="23UAM888",
            current_count=5,
            target_samples=16,
            min_samples=10,
            current_pose_instruction="Pose 5/16: Tilt head slightly DOWN (~10 deg)",
            feedback_msg="Sample 5/16 Captured Successfully!",
            feedback_color=(30, 220, 120),
        )

        # 5. Step 5: Enrollment complete card
        demo.render_enrollment_complete_dialog(
            img=canvas,
            student_name="Alex Morgan",
            student_id="23UAM888",
            samples_count=16,
            store_path="data/face_embeddings.npz",
        )
        self.assertGreater(canvas.sum(), 0, "Canvas should have drawn UI elements")

    def test_input_function_not_present_in_gui_flow(self):
        """Verify that built-in input() is never called in edge.demo."""
        import ast
        import inspect

        source = inspect.getsource(demo)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "input":
                    self.fail(f"Found forbidden blocking console input() call at line {node.lineno} in edge.demo")

    def test_in_gui_enrollment_state_constants(self):
        """Verify GUI state constants are defined and unique."""
        states = [
            demo.GUI_STATE_NORMAL,
            demo.GUI_STATE_ENROLL_ID,
            demo.GUI_STATE_ENROLL_NAME,
            demo.GUI_STATE_ENROLL_CONFIRM,
            demo.GUI_STATE_ENROLL_CAPTURE,
            demo.GUI_STATE_ENROLL_COMPLETE,
        ]
        self.assertEqual(len(states), len(set(states)), "GUI state constants must all be distinct")
        self.assertGreaterEqual(len(demo.SMART_POSES), 12, "Must define diverse enrollment poses")


if __name__ == "__main__":
    unittest.main()
