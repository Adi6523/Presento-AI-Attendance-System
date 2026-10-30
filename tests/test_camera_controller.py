"""
Tests for Presento Phase 16: Safe Dynamic Camera Controller.
Verifies measure-adjust-stabilize-lock sequence, dark/bright corrections,
cooldown periods, max adjustment ceilings, and graceful fallbacks.
"""
from unittest.mock import MagicMock
import numpy as np
import pytest
import cv2

from edge.camera_controller import SafeCameraController, ControlState


class TestCameraController:
    def test_unsupported_camera_fallback(self):
        controller = SafeCameraController()
        # Attach None / unsupported device
        supported = controller.attach_camera(None)
        assert supported is False
        assert controller.is_supported is False

        # Evaluation should safely return without crashing
        dummy_roi = np.full((50, 50, 3), 100, dtype=np.uint8)
        status = controller.evaluate_and_adjust(dummy_roi)
        assert status.is_supported is False
        assert "unavailable" in status.feedback.lower()

    def test_ideal_brightness_locks_immediately(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_EXPOSURE else -1.0

        controller = SafeCameraController(target_min_brightness=50.0, target_max_brightness=150.0)
        controller.attach_camera(mock_cap)

        # ROI with mean brightness 100 (in target range 50-150)
        good_roi = np.full((60, 60, 3), 100, dtype=np.uint8)
        status = controller.evaluate_and_adjust(good_roi, current_time=100.0)

        assert status.state == ControlState.LOCKED
        assert "locked" in status.feedback.lower()
        assert controller.adjustments_made == 0

    def test_dark_face_triggers_upward_adjustment_and_stabilization(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: 5.0 if prop == cv2.CAP_PROP_EXPOSURE else -1.0

        controller = SafeCameraController(
            target_min_brightness=60.0,
            target_max_brightness=180.0,
            stabilization_seconds=1.5,
            step_size=5.0,
        )
        controller.attach_camera(mock_cap)

        # 1. Dark face ROI (brightness 30 < 60)
        dark_roi = np.full((60, 60, 3), 30, dtype=np.uint8)
        status1 = controller.evaluate_and_adjust(dark_roi, current_time=10.0)

        assert status1.state == ControlState.STABILIZING
        assert controller.adjustments_made == 1
        # VideoCapture.set was called with increased exposure
        mock_cap.set.assert_called_with(cv2.CAP_PROP_EXPOSURE, 10.0)

        # 2. While stabilizing (10.5s < 10.0 + 1.5s), should NOT adjust again (no rapid oscillation)
        status2 = controller.evaluate_and_adjust(dark_roi, current_time=10.5)
        assert status2.state == ControlState.STABILIZING
        assert controller.adjustments_made == 1  # No new adjustment!

        # 3. After stabilization (12.0s >= 10.0 + 1.5s), if brightness is now optimal, lock!
        improved_roi = np.full((60, 60, 3), 100, dtype=np.uint8)
        status3 = controller.evaluate_and_adjust(improved_roi, current_time=12.0)
        assert status3.state == ControlState.LOCKED
        assert controller.adjustments_made == 1

    def test_max_adjustments_ceiling_enforces_lock(self):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_EXPOSURE else -1.0

        controller = SafeCameraController(
            target_min_brightness=60.0,
            target_max_brightness=180.0,
            stabilization_seconds=1.0,
            max_adjustments=2,
        )
        controller.attach_camera(mock_cap)

        dark_roi = np.full((60, 60, 3), 20, dtype=np.uint8)

        # Adjustment 1
        controller.evaluate_and_adjust(dark_roi, current_time=10.0)
        assert controller.adjustments_made == 1

        # Adjustment 2 (after stabilization)
        controller.evaluate_and_adjust(dark_roi, current_time=12.0)
        assert controller.adjustments_made == 2

        # 3rd measurement still dark, but max_adjustments reached -> MUST LOCK
        status = controller.evaluate_and_adjust(dark_roi, current_time=14.0)
        assert status.state == ControlState.LOCKED
        assert controller.adjustments_made == 2  # Hard ceiling respected
