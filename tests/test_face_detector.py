from __future__ import annotations

import time
import unittest
from unittest.mock import patch
import numpy as np

from edge.detector import FaceDetectorWorker


class TestFaceDetectorWorker(unittest.TestCase):
    def test_coordinate_scaling_and_persistence(self):
        """Verify downscaled detection coordinates are properly scaled back and persisted."""
        detector = FaceDetectorWorker(
            target_width=640,
            target_height=360,
            target_fps=20.0,
            persistence_seconds=0.4,
            smoothing_factor=0.5,
        )

        # Mock face_recognition.face_locations to return a fixed box on downscaled frame
        # (top, right, bottom, left) on 640x360: e.g. (50, 200, 150, 100)
        mock_downscaled_box = [(50, 200, 150, 100)]

        # Original frame is 1280x720 (scale factor 2.0x in x and y)
        test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

        with patch("edge.detector.face_recognition.face_locations", return_value=mock_downscaled_box):
            detector.start()
            try:
                # Submit frame
                detector.submit_frame(test_frame)

                # Wait briefly for worker to process
                start = time.time()
                boxes = []
                while time.time() - start < 1.0:
                    boxes = detector.get_latest_faces()
                    if boxes:
                        break
                    time.sleep(0.02)

                self.assertTrue(len(boxes) > 0, "Expected at least one detected face box")
                top, right, bottom, left = boxes[0]

                # Coordinates should be scaled by 2.0:
                # top: 50 * 2 = 100
                # right: 200 * 2 = 400
                # bottom: 150 * 2 = 300
                # left: 100 * 2 = 200
                self.assertEqual((top, right, bottom, left), (100, 400, 300, 200))

            finally:
                detector.stop()

    def test_ema_smoothing(self):
        """Verify EMA smoothing updates box coordinates smoothly without jumping."""
        detector = FaceDetectorWorker(
            target_width=640,
            target_height=360,
            target_fps=30.0,
            persistence_seconds=1.0,
            smoothing_factor=0.5,  # 50% new + 50% prev
        )

        # Initial detection
        now = time.monotonic()
        detector._update_tracks([(100, 400, 300, 200)], now)
        boxes = detector.get_latest_faces()
        self.assertEqual(boxes[0], (100, 400, 300, 200))

        # Second detection slightly shifted: (120, 420, 320, 220)
        # Expected smoothed: 0.5 * 120 + 0.5 * 100 = 110, etc.
        now += 0.05
        detector._update_tracks([(120, 420, 320, 220)], now)
        smoothed_boxes = detector.get_latest_faces()
        self.assertEqual(smoothed_boxes[0], (110, 410, 310, 210))

    def test_persistence_and_expiration(self):
        """Verify face boxes persist through missed detection frames and expire after persistence duration."""
        detector = FaceDetectorWorker(
            target_width=640,
            target_height=360,
            target_fps=30.0,
            persistence_seconds=0.2,
            smoothing_factor=0.5,
        )

        t0 = time.monotonic()
        # Cycle 1: face detected
        detector._update_tracks([(100, 400, 300, 200)], t0)
        self.assertEqual(len(detector.get_latest_faces()), 1)

        # Cycle 2: missed detection shortly after (0.05s < 0.2s persistence)
        t1 = t0 + 0.05
        detector._update_tracks([], t1)
        # Should still be retained
        self.assertEqual(len(detector.get_latest_faces()), 1)

        # Cycle 3: time passes past persistence threshold (0.25s > 0.2s)
        # Simulating checking faces later:
        with patch("edge.detector.time.monotonic", return_value=t0 + 0.3):
            self.assertEqual(len(detector.get_latest_faces()), 0)


if __name__ == "__main__":
    unittest.main()
