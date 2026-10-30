from __future__ import annotations

import time
import unittest
import numpy as np
import openvino as ov

from edge.scrfd_detector import SCRFDDetector, SCRFDWorker, SCRFDFace


class TestSCRFDDetector(unittest.TestCase):
    def test_openvino_device_verification(self):
        """Verify OpenVINO detects Intel GPU and CPU correctly and reports device descriptors."""
        core = ov.Core()
        devices = core.available_devices
        self.assertIn("GPU", devices, "Intel GPU should be detected by OpenVINO")
        self.assertIn("CPU", devices, "Intel CPU should be detected by OpenVINO")

        gpu_desc = core.get_property("GPU", "FULL_DEVICE_NAME")
        self.assertIn("Iris", gpu_desc, f"GPU should be Intel Iris Xe, got: {gpu_desc}")

    def test_scrfd_detector_initialization_and_inference(self):
        """Verify SCRFD detector initializes on GPU, runs inference, and returns SCRFDFace structures."""
        detector = SCRFDDetector(device="GPU", conf_thresh=0.3)
        self.assertEqual(detector.active_device, "GPU")
        self.assertIn("Iris", detector.full_device_name)

        # Test inference on a synthetic 1280x720 frame
        test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        cv2_circle = True
        import cv2
        cv2.circle(test_frame, (640, 360), 80, (220, 220, 220), -1)

        faces = detector.detect(test_frame)
        self.assertIsInstance(faces, list)

        # If faces are detected (or empty on uniform background), verify data structure
        for face in faces:
            self.assertIsInstance(face, SCRFDFace)
            self.assertEqual(len(face.bbox), 4)
            left, top, right, bottom = face.bbox
            self.assertTrue(0 <= left < right <= 1280)
            self.assertTrue(0 <= top < bottom <= 720)
            self.assertTrue(0.0 <= face.confidence <= 1.0)
            self.assertEqual(face.landmarks.shape, (5, 2))

    def test_scrfd_worker_lifecycle_and_non_blocking(self):
        """Verify SCRFDWorker runs in background, accepts frames, and shuts down cleanly."""
        worker = SCRFDWorker(device="GPU")
        worker.start()
        try:
            test_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
            worker.submit_frame(test_frame)

            time.sleep(0.1)
            detections = worker.get_latest_detections()
            self.assertIsInstance(detections, list)

            stats = worker.get_stats()
            self.assertIn("detection_fps", stats)
            self.assertEqual(stats["device"], "GPU")
            self.assertIn("Iris", stats["device_name"])
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
