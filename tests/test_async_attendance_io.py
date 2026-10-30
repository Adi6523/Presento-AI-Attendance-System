"""
Tests for asynchronous attendance network I/O and display loop decoupling.
Verifies:
1. Offline queue flush does not block display loop (<5 ms).
2. Failed backend does not reduce display FPS.
3. At most one offline queue flush runs concurrently.
4. Attendance POST failure is handled cleanly without blocking caller.
5. Attendance event is not duplicated by the async mechanism.
6. last_queue_flush uses time.monotonic() at dispatch time.
7. Camera read uses 35 ms timeout and respects FrameReceiver semantics.
"""
from __future__ import annotations

import inspect
import time
import unittest
from unittest.mock import MagicMock, patch

from edge.attendance_worker import AttendanceIOWorker
from edge.frame_receiver import FrameReceiver
import edge.runner


class TestAsyncAttendanceIO(unittest.TestCase):

    def test_offline_queue_flush_does_not_block_display_loop(self):
        """1. Offline queue flush does not block display loop."""
        worker = AttendanceIOWorker(flush_interval=0.0)
        worker.last_queue_flush = 0.0  # Force ready

        # Simulate flush_queue taking 0.3s (e.g. network timeout)
        def slow_flush():
            time.sleep(0.3)
            return 2

        with patch("edge.attendance_worker.flush_queue", side_effect=slow_flush):
            t0 = time.perf_counter()
            dispatched = worker.maybe_trigger_flush()
            t_call = time.perf_counter() - t0

            # Caller must return immediately (< 20 ms)
            self.assertTrue(dispatched)
            self.assertLess(t_call, 0.02, f"maybe_trigger_flush blocked for {t_call*1000:.1f}ms")
            self.assertTrue(worker.is_flushing())

            # Wait for background flush to finish
            time.sleep(0.35)
            self.assertFalse(worker.is_flushing())
            self.assertEqual(worker.total_flushed, 2)

    def test_failed_backend_does_not_reduce_display_fps(self):
        """2. Failed backend does not reduce display FPS."""
        worker = AttendanceIOWorker(flush_interval=10.0)
        worker.start()

        # Simulate failing backend with 0.1s network latency on each post
        def slow_failing_post(event):
            time.sleep(0.1)
            return False

        with patch("edge.attendance_worker.post_event", side_effect=slow_failing_post):
            # Run 30 simulated display frames
            t0 = time.perf_counter()
            for i in range(30):
                # Submit event periodically (every 5th frame)
                if i % 5 == 0:
                    event = {"event_id": f"evt-{i}", "student_id": "test"}
                    worker.submit_event(event)
                worker.maybe_trigger_flush()
                # Simulate small frame work (~1ms)
                time.sleep(0.001)

            elapsed = time.perf_counter() - t0
            # 30 frames with 6 events would take 6 * 100ms = 600ms if synchronous.
            # Asynchronously, it should finish in < 150ms total.
            self.assertLess(
                elapsed,
                0.15,
                f"Display loop took {elapsed*1000:.1f}ms (>150ms), showing main thread was blocked",
            )

        worker.stop()

    def test_only_one_flush_runs_concurrently(self):
        """3. Only one flush runs concurrently."""
        worker = AttendanceIOWorker(flush_interval=0.0)
        worker.last_queue_flush = 0.0

        flush_started = 0

        def slow_flush():
            nonlocal flush_started
            flush_started += 1
            time.sleep(0.2)
            return 1

        with patch("edge.attendance_worker.flush_queue", side_effect=slow_flush):
            # First flush starts
            first = worker.maybe_trigger_flush()
            self.assertTrue(first)
            self.assertTrue(worker.is_flushing())

            # Attempt multiple subsequent flushes while first is running
            worker.last_queue_flush = 0.0  # Reset timer to test lock guard
            second = worker.maybe_trigger_flush()
            third = worker.maybe_trigger_flush()

            self.assertFalse(second, "Second concurrent flush was dispatched")
            self.assertFalse(third, "Third concurrent flush was dispatched")

            # Wait for first flush to complete
            time.sleep(0.25)
            self.assertEqual(flush_started, 1)
            self.assertFalse(worker.is_flushing())

    def test_attendance_post_failure_handled_without_blocking(self):
        """4. Attendance POST failure is handled without blocking."""
        worker = AttendanceIOWorker()
        worker.start()

        def failing_post(event):
            time.sleep(0.15)
            raise ConnectionError("Backend down")

        with patch("edge.attendance_worker.post_event", side_effect=failing_post):
            t0 = time.perf_counter()
            queued = worker.submit_event({"event_id": "fail-1", "student_id": "s1"})
            t_submit = time.perf_counter() - t0

            self.assertTrue(queued)
            self.assertLess(t_submit, 0.01, f"submit_event took {t_submit*1000:.2f}ms")

            # Wait for background thread to encounter failure
            time.sleep(0.25)
            stats = worker.get_stats()
            self.assertEqual(stats["total_failed"], 1)
            self.assertEqual(stats["total_posted"], 0)

        worker.stop()

    def test_attendance_event_not_duplicated_by_async_mechanism(self):
        """5. Attendance event is not duplicated by the async mechanism."""
        worker = AttendanceIOWorker()
        worker.start()

        posted_events = []

        def record_post(event):
            posted_events.append(event["event_id"])
            return True

        with patch("edge.attendance_worker.post_event", side_effect=record_post):
            # Submit 5 distinct events
            for i in range(5):
                worker.submit_event({"event_id": f"evt-{i}", "student_id": f"stu-{i}"})

            # Wait for worker to finish processing
            time.sleep(0.3)
            self.assertEqual(len(posted_events), 5)
            # Ensure all IDs are unique (no duplicates generated)
            self.assertEqual(len(set(posted_events)), 5)
            self.assertEqual(posted_events, [f"evt-{i}" for i in range(5)])

        worker.stop()

    def test_last_queue_flush_uses_dispatch_time_timestamp(self):
        """6. last_queue_flush uses dispatch-time timestamp."""
        worker = AttendanceIOWorker(flush_interval=10.0)
        worker.last_queue_flush = 0.0  # Expired

        def delayed_flush():
            time.sleep(0.1)
            return 0

        with patch("edge.attendance_worker.flush_queue", side_effect=delayed_flush):
            t_dispatch = time.monotonic()
            worker.maybe_trigger_flush()
            t_after_dispatch = time.monotonic()

            # Verify last_queue_flush was updated to approximately t_dispatch
            self.assertGreaterEqual(worker.last_queue_flush, t_dispatch)
            self.assertLessEqual(worker.last_queue_flush, t_after_dispatch)

            # Wait for flush to end
            time.sleep(0.15)
            # Timestamp must still reflect the dispatch time, not post-flush time
            self.assertLessEqual(worker.last_queue_flush, t_after_dispatch)

    def test_camera_read_uses_35ms_timeout(self):
        """7. Camera read uses 35 ms timeout."""
        # 7a. Verify in edge/runner.py source code
        source = inspect.getsource(edge.runner.main)
        self.assertIn(
            "camera.read(timeout=0.035)",
            source,
            "edge.runner.main does not call camera.read(timeout=0.035)",
        )

        # 7b. Verify FrameReceiver.read(timeout=0.035) respects the timeout
        receiver = FrameReceiver(host="127.0.0.1", port=9999)
        receiver._running = True
        # Without receiving frames, read with timeout=0.035 should wait ~35ms
        t0 = time.perf_counter()
        ok, frame = receiver.read(timeout=0.035)
        dt = time.perf_counter() - t0

        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertGreaterEqual(dt, 0.025, f"Timeout returned too fast: {dt*1000:.1f}ms")
        self.assertLess(dt, 0.090, f"Timeout took too long: {dt*1000:.1f}ms (>90ms)")


if __name__ == "__main__":
    unittest.main()
