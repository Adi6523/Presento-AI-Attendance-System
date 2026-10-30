from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from edge.anti_spoof_worker import AntiSpoofWorker, AntiSpoofJob, AntiSpoofResult
from edge.anti_spoof_fusion import FusionResult


class TestAntiSpoofWorker(unittest.TestCase):
    """Test suite for asynchronous AntiSpoofWorker."""

    def test_bounded_queue_latest_job_overwrites_stale(self):
        """Verify submitting a newer job for the same face_id overwrites older pending job."""
        worker = AntiSpoofWorker(
            mn3_model_path="dummy_mn3.onnx",
            minifas_model_path="dummy_mf.onnx",
        )
        # Do not start worker thread so jobs stay in pending queue
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        bbox = (10, 10, 80, 80)

        worker._running = True  # Allow submit_face without running worker loop

        # Submit frame 1 for face 1
        worker.submit_face(face_id=1, frame_id=101, frame=dummy_frame, bbox=bbox)
        with worker._lock:
            self.assertEqual(len(worker._pending_jobs), 1)
            self.assertEqual(worker._pending_jobs[1].frame_id, 101)
            self.assertEqual(worker.total_jobs_submitted, 1)
            self.assertEqual(worker.total_jobs_dropped, 0)

        # Submit frame 2 for face 1 (newer crop before worker processed frame 1)
        worker.submit_face(face_id=1, frame_id=102, frame=dummy_frame, bbox=bbox)
        with worker._lock:
            self.assertEqual(len(worker._pending_jobs), 1)
            self.assertEqual(worker._pending_jobs[1].frame_id, 102)
            self.assertEqual(worker.total_jobs_submitted, 2)
            self.assertEqual(worker.total_jobs_dropped, 1)  # Stale frame 101 dropped!

        # Submit frame for different face 2
        worker.submit_face(face_id=2, frame_id=201, frame=dummy_frame, bbox=bbox)
        with worker._lock:
            self.assertEqual(len(worker._pending_jobs), 2)
            self.assertIn(1, worker._pending_jobs)
            self.assertIn(2, worker._pending_jobs)

    def test_face_isolation_and_result_retrieval(self):
        """Verify results are strictly isolated per face_id with correct frame_id."""
        worker = AntiSpoofWorker("dummy_mn3.onnx", "dummy_mf.onnx")

        # Mock classifier predictions
        mock_mn3 = MagicMock()
        mock_mn3.predict.side_effect = [
            (0.95, 0.05, "REAL", 3.0),   # Face 1: REAL
            (0.05, 0.95, "SPOOF", 2.5),  # Face 2: SPOOF
        ]

        mock_mf = MagicMock()
        mock_mf.predict.side_effect = [
            (0.92, 0.08, "REAL", 6.0),   # Face 1: REAL
            (0.02, 0.98, "SPOOF", 7.0),  # Face 2: SPOOF
        ]

        with patch("edge.anti_spoof_worker.AntiSpoofClassifier", return_value=mock_mn3), \
             patch("edge.anti_spoof_worker.MiniFASNetV2SEClassifier", return_value=mock_mf):

            worker.start()
            dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)

            worker.submit_face(face_id=10, frame_id=501, frame=dummy_frame, bbox=(10, 10, 50, 50))
            worker.submit_face(face_id=20, frame_id=502, frame=dummy_frame, bbox=(60, 60, 100, 100))

            # Wait for worker to finish both jobs
            timeout = time.monotonic() + 2.0
            while time.monotonic() < timeout:
                r1 = worker.get_result(10)
                r2 = worker.get_result(20)
                if r1 is not None and r2 is not None:
                    break
                time.sleep(0.02)

            worker.stop()

            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)
            self.assertEqual(r1.face_id, 10)
            self.assertEqual(r1.frame_id, 501)
            self.assertEqual(r1.fusion_result.and_label, "REAL")

            self.assertEqual(r2.face_id, 20)
            self.assertEqual(r2.frame_id, 502)
            self.assertEqual(r2.fusion_result.and_label, "SPOOF")

    def test_prune_stale_faces(self):
        """Verify prune_stale removes fusion engines and results for vanished faces."""
        worker = AntiSpoofWorker("dummy_mn3.onnx", "dummy_mf.onnx")
        now = time.monotonic()

        with worker._lock:
            # Simulate face 1 active recently, face 2 inactive for 4 seconds
            worker._last_active[1] = now
            worker._last_active[2] = now - 4.0
            worker._results[1] = MagicMock(spec=AntiSpoofResult)
            worker._results[2] = MagicMock(spec=AntiSpoofResult)
            worker._fusion_engines[1] = MagicMock()
            worker._fusion_engines[2] = MagicMock()

        # Face 1 is active, face 2 is not
        worker.prune_stale(active_face_ids={1}, max_idle_seconds=3.0)

        with worker._lock:
            self.assertIn(1, worker._results)
            self.assertIn(1, worker._fusion_engines)
            self.assertNotIn(2, worker._results)
            self.assertNotIn(2, worker._fusion_engines)
            self.assertNotIn(2, worker._last_active)

    def test_clean_shutdown(self):
        """Verify worker stops cleanly when stop() is called."""
        with patch("edge.anti_spoof_worker.AntiSpoofClassifier"), \
             patch("edge.anti_spoof_worker.MiniFASNetV2SEClassifier"):
            worker = AntiSpoofWorker("dummy1.onnx", "dummy2.onnx")
            worker.start()
            self.assertTrue(worker._running)
            worker.stop()
            self.assertFalse(worker._running)
            self.assertIsNone(worker._thread)


if __name__ == "__main__":
    unittest.main()
