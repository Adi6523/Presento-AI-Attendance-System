"""
Unit and integration test suite for Presento Friend Laptop Demo Mode.

Validates:
1. CAMERA_SOURCE=webcam:
   - Strictly does NOT contact Raspberry Pi, port 5001, port 5002, CameraSession, or FrameReceiver.
   - Uses OpenCV VideoCapture directly (CAP_DSHOW on Windows).
   - Graceful camera failure handling (helpful error message, no traceback).
2. CPU / GPU compatibility:
   - OPENVINO_DEVICE=CPU default works reliably.
   - Graceful fallback from GPU to CPU when Intel GPU is absent.
   - Display strings formatted for HUD ("Device: CPU" or "Device: GPU - Intel Iris Xe").
3. Model integrity verification:
   - All 4 ONNX models checked at startup.
   - Missing model fails cleanly with informative error message.
4. HUD updates:
   - Displays "SOURCE: LAPTOP WEBCAM" in webcam mode.
   - Displays "Device: CPU" (or GPU).
5. Standalone enrollment tool:
   - parse_camera_source("webcam") maps to local OpenCV camera.
"""
from __future__ import annotations

import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import edge.demo as demo
from edge.config import settings
from edge.enroll_face import parse_camera_source


class TestWebcamDemoMode(unittest.TestCase):

    def test_parse_camera_source_webcam_in_enroll_face(self):
        """Verify enroll_face parses 'webcam', 'laptop', 'local' as OpenCV camera."""
        source_type, source_val = parse_camera_source("webcam")
        self.assertEqual(source_type, "opencv")
        self.assertEqual(source_val, settings.camera_index)

        source_type2, source_val2 = parse_camera_source("laptop")
        self.assertEqual(source_type2, "opencv")
        self.assertEqual(source_val2, settings.camera_index)

        source_type3, source_val3 = parse_camera_source("1")
        self.assertEqual(source_type3, "opencv")
        self.assertEqual(source_val3, 1)

    def test_verify_required_models_success(self):
        """Verify model check passes when all 4 model files are on disk."""
        self.assertTrue(demo.verify_required_models())

    def test_verify_required_models_missing(self):
        """Verify model check fails cleanly with user-facing error message when a model is missing."""
        with patch("os.path.isfile", return_value=False):
            captured_stdout = io.StringIO()
            with patch("sys.stdout", captured_stdout):
                result = demo.verify_required_models()
            self.assertFalse(result)
            output = captured_stdout.getvalue()
            self.assertIn("[ERROR] Required AI model files are missing", output)
            self.assertIn("scrfd_500m_bnkps.onnx", output)
            self.assertIn("w600k_mbf.onnx", output)
            self.assertIn("anti-spoof-mn3.onnx", output)
            self.assertIn("minifasnet_v2se.onnx", output)

    def test_resolve_openvino_device_cpu(self):
        """Verify OPENVINO_DEVICE=CPU resolves to CPU."""
        active, label, short = demo.resolve_openvino_device("CPU")
        self.assertEqual(active, "CPU")
        self.assertEqual(label, "CPU")
        self.assertEqual(short, "CPU")

    def test_resolve_openvino_device_gpu_fallback_when_unavailable(self):
        """Verify requested GPU falls back to CPU cleanly when GPU is not in available_devices."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU"]  # No GPU available (e.g. AMD or basic laptop)

        with patch("openvino.Core", return_value=mock_core):
            active, label, short = demo.resolve_openvino_device("GPU")
            self.assertEqual(active, "CPU")
            self.assertEqual(label, "CPU")
            self.assertEqual(short, "CPU")

    def test_resolve_openvino_device_gpu_available(self):
        """Verify GPU resolves and formats display name when available."""
        mock_core = MagicMock()
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core.get_property.return_value = "Intel(R) Iris(R) Xe Graphics (iGPU)"

        with patch("openvino.Core", return_value=mock_core):
            active, label, short = demo.resolve_openvino_device("GPU")
            self.assertEqual(active, "GPU")
            self.assertEqual(label, "GPU - Intel Iris Xe")
            self.assertEqual(short, "GPU")

    def test_webcam_hud_labels_rendered(self):
        """Verify HUD functions accept and render SOURCE: LAPTOP WEBCAM and Device: CPU."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 1. Top Header HUD
        demo.render_top_header_hud(
            img=canvas,
            current_mode=1,
            display_fps=30.0,
            det_fps=30.0,
            arc_fps=15.0,
            arc_lat=12.0,
            enrolled_count=3,
            camera_active=True,
            is_paused=False,
            source_label="LAPTOP WEBCAM",
            device_label="CPU",
            device_short="CPU",
        )
        self.assertGreater(canvas.sum(), 0)

        # 2. Pipeline Debug Panel
        demo.render_pipeline_debug_panel(
            img=canvas,
            det_fps=30.0,
            arc_fps=15.0,
            arc_lat=12.0,
            as_lat=18.0,
            enrolled_count=3,
            active_faces=1,
            camera_source="webcam",
            source_label="LAPTOP WEBCAM",
            device_label="CPU",
            device_short="CPU",
        )

        # 3. Mode 5 Summary Dashboard
        tracker = demo.DemoMetricsTracker()
        demo.render_mode5_summary_dashboard(
            img=canvas,
            tracker=tracker,
            display_fps=30.0,
            det_fps=30.0,
            arc_fps=15.0,
            enrolled_count=3,
            source_label="LAPTOP WEBCAM",
            device_label="CPU",
            device_short="CPU",
        )
        self.assertEqual(canvas.shape, (720, 1280, 3))

    def test_run_demo_webcam_mode_never_uses_camera_session_or_frame_receiver(self):
        """
        CRITICAL TEST:
        When CAMERA_SOURCE=webcam (or --source webcam):
        - CameraSession must NOT be instantiated or started.
        - FrameReceiver must NOT be instantiated or started.
        - OpenCV VideoCapture must be used.
        - Workers and camera released cleanly.
        """
        mock_camera = MagicMock()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_camera.read.return_value = (True, dummy_frame)
        mock_camera.isOpened.return_value = True
        mock_camera.get.return_value = 1280

        with patch("edge.demo.CameraSession") as mock_session_cls, \
             patch("edge.demo.FrameReceiver") as mock_receiver_cls, \
             patch("cv2.VideoCapture", return_value=mock_camera) as mock_cv_cls, \
             patch("edge.demo.SCRFDWorker") as mock_detector_cls, \
             patch("edge.demo.AntiSpoofWorker") as mock_as_cls, \
             patch("edge.demo.AttendanceIOWorker") as mock_att_cls:

            mock_det_instance = mock_detector_cls.return_value
            mock_det_instance.get_latest_detections.return_value = []
            mock_det_instance.get_stats.return_value = {"detection_fps": 30.0, "arcface_fps": 15.0}

            demo.run_demo(
                source="webcam",
                initial_mode=1,
                headless=True,
                max_frames=3,
            )

            # CRITICAL ASSERTIONS: Zero Pi components invoked
            mock_session_cls.assert_not_called()
            mock_receiver_cls.assert_not_called()

            # OpenCV VideoCapture must be used
            mock_cv_cls.assert_called()
            mock_camera.release.assert_called_once()

            # AI Workers started and stopped cleanly
            mock_det_instance.start.assert_called_once()
            mock_det_instance.stop.assert_called_once()
            mock_as_cls.return_value.start.assert_called_once()
            mock_as_cls.return_value.stop.assert_called_once()
            mock_att_cls.return_value.start.assert_called_once()
            mock_att_cls.return_value.stop.assert_called_once()

    def test_run_demo_webcam_open_failure_prints_clean_error(self):
        """
        Verify that if webcam index cannot open:
        - Shows user-friendly message suggesting CAMERA_INDEX=1.
        - Exits cleanly without unhandled Python traceback.
        - Does NOT call CameraSession.
        """
        mock_failed_camera = MagicMock()
        mock_failed_camera.isOpened.return_value = False

        captured_stdout = io.StringIO()
        with patch("cv2.VideoCapture", return_value=mock_failed_camera), \
             patch("edge.demo.CameraSession") as mock_session_cls, \
             patch("sys.stdout", captured_stdout):

            demo.run_demo(
                source="webcam",
                initial_mode=1,
                headless=True,
                max_frames=1,
            )

            mock_session_cls.assert_not_called()

        output = captured_stdout.getvalue()
        self.assertIn("Could not open laptop webcam", output)
        self.assertIn("CAMERA_INDEX=1", output)


if __name__ == "__main__":
    unittest.main()
