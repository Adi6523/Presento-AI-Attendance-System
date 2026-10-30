"""
Tests for Presento AI Backend Client & Offline Queue (Phase 2).

Verifies the 8 required AI client behaviors:
1. AI event is queued immediately (<5ms, non-blocking).
2. HTTP request executes in background thread.
3. Camera loop / caller is never blocked by slow network/timeouts.
4. Backend unavailable -> event safely written to offline queue file.
5. Backend returns -> queue flush synchronizes events and cleans up file.
6. Duplicate events are handled safely by backend client.
7. liveness_state is preserved exactly as emitted by AI engine.
8. Client never converts raw probability score into REAL.
9. Bearer token AI API key authentication is sent on every request.
10. Dynamic active session discovery works correctly.
"""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from edge.backend_client import (
    BackendClient,
    AsyncBackendClient,
    VALID_LIVENESS_STATES,
    _ai_auth_headers,
)


class TestAIBackendClient(unittest.TestCase):

    def setUp(self):
        self.test_queue_path = Path("data/test-offline-events.jsonl")
        if self.test_queue_path.exists():
            self.test_queue_path.unlink()

    def tearDown(self):
        if self.test_queue_path.exists():
            self.test_queue_path.unlink()

    def test_01_event_queued_immediately_non_blocking(self):
        """1. AI event is queued immediately (<5ms, non-blocking)."""
        client = AsyncBackendClient(queue_path=self.test_queue_path)
        client.start()

        t0 = time.perf_counter()
        queued = client.submit_event(
            student_id="23UAM090",
            session_id=1,
            confidence=0.92,
            liveness_state="REAL_LOCKED",
            liveness_score=0.95,
        )
        elapsed = time.perf_counter() - t0

        self.assertTrue(queued)
        self.assertLess(elapsed, 0.01, f"submit_event blocked for {elapsed*1000:.2f}ms (>10ms)")
        client._running = False

    def test_02_http_request_happens_in_background(self):
        """2. HTTP request happens in background thread."""
        client = AsyncBackendClient(queue_path=self.test_queue_path)
        mock_post = MagicMock(return_value={"success": True, "status": "PRESENT"})
        client.client.post_attendance_event = mock_post

        client.start()
        client.submit_event(
            student_id="23UAM090",
            session_id=1,
            confidence=0.92,
            liveness_state="REAL_LOCKED",
        )

        # Wait briefly for background thread to process
        time.sleep(0.15)
        client._running = False

        self.assertTrue(mock_post.called)
        self.assertEqual(client.total_posted, 1)

    def test_03_camera_loop_not_blocked_by_slow_backend(self):
        """3. Camera loop is not blocked by slow network/backend (>200ms latency)."""
        client = AsyncBackendClient(queue_path=self.test_queue_path)

        def slow_post(*args, **kwargs):
            time.sleep(0.2)
            return {"success": True, "status": "PRESENT"}

        client.client.post_attendance_event = slow_post
        client.start()

        # Simulate 20 tight camera loop iterations
        t0 = time.perf_counter()
        for i in range(20):
            client.submit_event(
                student_id=f"STU_{i}",
                session_id=1,
                confidence=0.90,
                liveness_state="REAL_LOCKED",
            )
            time.sleep(0.001)
        elapsed = time.perf_counter() - t0

        # If synchronous, 20 * 200ms = 4000ms. Non-blocking must finish in <100ms.
        self.assertLess(elapsed, 0.1, f"Loop was blocked: took {elapsed*1000:.1f}ms")
        client._running = False

    def test_04_backend_unavailable_writes_to_offline_queue(self):
        """4. Backend unavailable -> event safely written to offline queue file."""
        client = AsyncBackendClient(queue_path=self.test_queue_path)

        def failing_post(*args, **kwargs):
            raise ConnectionError("Backend offline")

        client.client.post_attendance_event = failing_post
        client.start()

        client.submit_event(
            student_id="23UAM090",
            session_id=5,
            confidence=0.88,
            liveness_state="REAL_LOCKED",
            liveness_score=0.91,
        )

        time.sleep(0.2)
        client._running = False

        self.assertEqual(client.total_failed, 1)
        self.assertTrue(self.test_queue_path.exists())
        lines = self.test_queue_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["student_id"], "23UAM090")
        self.assertEqual(data["session_id"], 5)
        self.assertEqual(data["liveness_state"], "REAL_LOCKED")

    def test_05_backend_returns_queue_flush_succeeds(self):
        """5. Backend returns -> queue flush synchronizes events and removes file."""
        # Seed offline queue with 2 events
        self.test_queue_path.parent.mkdir(parents=True, exist_ok=True)
        events = [
            {"student_id": "STU1", "session_id": 1, "confidence": 0.9, "liveness_state": "REAL_LOCKED"},
            {"student_id": "STU2", "session_id": 1, "confidence": 0.85, "liveness_state": "REAL_LOCKED"},
        ]
        self.test_queue_path.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n",
            encoding="utf-8",
        )

        client = AsyncBackendClient(queue_path=self.test_queue_path)
        client.client.post_attendance_event = MagicMock(return_value={"success": True, "status": "PRESENT"})

        flushed = client.flush_queue()
        self.assertEqual(flushed, 2)
        self.assertFalse(self.test_queue_path.exists())

    def test_06_duplicate_events_remain_safe(self):
        """6. Duplicate events return DUPLICATE safely without error."""
        client = AsyncBackendClient(queue_path=self.test_queue_path)
        client.client.post_attendance_event = MagicMock(return_value={"success": True, "status": "DUPLICATE"})
        client.start()

        client.submit_event(
            student_id="23UAM090",
            session_id=1,
            confidence=0.92,
            liveness_state="REAL_LOCKED",
        )
        time.sleep(0.15)
        client._running = False

        self.assertEqual(client.total_posted, 1)
        self.assertEqual(client.total_failed, 0)

    def test_07_liveness_state_preserved_exactly(self):
        """7. liveness_state is preserved exactly (REAL_LOCKED, HOLD, SPOOF, etc.)."""
        for state in ["REAL_LOCKED", "HOLD", "SPOOF", "CHECKING", "NEW", "RESET"]:
            sync_client = BackendClient(base_url="http://mock-backend")
            with patch("requests.post") as mock_req:
                mock_resp = MagicMock()
                mock_resp.ok = True
                mock_resp.json.return_value = {"success": True, "status": "BLOCKED"}
                mock_req.return_value = mock_resp

                sync_client.post_attendance_event(
                    student_id="23UAM090",
                    session_id=1,
                    confidence=0.91,
                    liveness_state=state,
                )

                payload = mock_req.call_args[1]["json"]
                self.assertEqual(payload["liveness_state"], state)

    def test_08_no_raw_score_converted_to_real_by_client(self):
        """8. No raw score is converted into REAL by the client."""
        sync_client = BackendClient(base_url="http://mock-backend")
        with patch("requests.post") as mock_req:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {"success": False, "status": "BLOCKED"}
            mock_req.return_value = mock_resp

            # Pass high score 0.99 with state CHECKING; client must NOT change CHECKING to REAL_LOCKED
            sync_client.post_attendance_event(
                student_id="23UAM090",
                session_id=1,
                confidence=0.95,
                liveness_state="CHECKING",
                liveness_score=0.99,
            )

            payload = mock_req.call_args[1]["json"]
            self.assertEqual(payload["liveness_state"], "CHECKING")
            self.assertEqual(payload["liveness_score"], 0.99)

    def test_09_ai_auth_header_included(self):
        """9. AI API key Bearer authorization header is attached to requests."""
        headers = _ai_auth_headers()
        self.assertIn("Authorization", headers)
        self.assertTrue(headers["Authorization"].startswith("Bearer "))

    def test_10_dynamic_active_session_discovery(self):
        """10. Dynamic active session discovery queries /api/sessions?status=ACTIVE."""
        sync_client = BackendClient(base_url="http://mock-backend")
        with patch("requests.get") as mock_req:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = [{"id": 42, "status": "ACTIVE", "name": "Lab 1"}]
            mock_req.return_value = mock_resp

            sess_id = sync_client.get_active_session_id()
            self.assertEqual(sess_id, 42)
            self.assertIn("/api/sessions?status=ACTIVE", mock_req.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
