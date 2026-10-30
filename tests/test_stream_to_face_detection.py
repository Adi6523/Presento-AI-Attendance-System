from __future__ import annotations

import socket
import struct
import threading
import time
import unittest
import cv2
import numpy as np
import face_recognition

from edge.frame_receiver import FrameReceiver
from edge.quality import assess_face_quality


class TestStreamToFaceDetection(unittest.TestCase):
    def test_stream_frame_passes_to_face_detection_and_quality(self):
        """Verify frames delivered by FrameReceiver are processed by the existing face detection and quality pipeline."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        # Generate a test image frame (1280x720) with a simple synthetic pattern
        test_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

        def _mock_sender():
            conn, _ = server_sock.accept()
            try:
                ok, encoded = cv2.imencode(".jpg", test_frame)
                if ok:
                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))
                    conn.sendall(header + payload)
            finally:
                conn.close()

        server_thread = threading.Thread(target=_mock_sender, daemon=True)
        server_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        try:
            ok, frame = receiver.read(timeout=2.0)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (720, 1280, 3))

            # 1. Existing face detection step (as in runner.py)
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            face_locations = face_recognition.face_locations(rgb, model="hog")
            self.assertIsInstance(face_locations, list)

            # 2. Existing quality check step (as in runner.py)
            dummy_loc = (100, 300, 300, 100)  # top, right, bottom, left
            quality = assess_face_quality(frame, dummy_loc)
            self.assertIsNotNone(quality)
            self.assertIsInstance(quality.score, float)
            self.assertTrue(hasattr(quality, "blur_ok"))
            self.assertTrue(hasattr(quality, "lighting_ok"))
        finally:
            receiver.stop()
            server_sock.close()
            server_thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
