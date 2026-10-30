"""
Dedicated background worker for attendance event network I/O.
Decouples post_event() and flush_queue() from the main video display thread
so network latency, HTTP timeouts, or backend downtime cannot block or jitter
the live camera and OpenCV rendering loop.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .api import post_event, flush_queue

logger = logging.getLogger("presento.edge.attendance_worker")


class AttendanceIOWorker:
    """
    Dedicated background worker for all attendance HTTP API network I/O.
    Ensures that post_event() and flush_queue() NEVER block the main display thread,
    even when the backend API server is offline or experiencing network timeouts.
    """

    def __init__(self, flush_interval: float = 10.0):
        self.flush_interval = flush_interval
        self._event_queue: queue.Queue[dict] = queue.Queue(maxsize=200)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._flush_lock = threading.Lock()
        self._api_lock = threading.Lock()
        self._is_flushing = False
        self.last_queue_flush = time.monotonic()
        self.total_posted = 0
        self.total_failed = 0
        self.total_flushed = 0

    def start(self) -> "AttendanceIOWorker":
        """Start the background I/O thread."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AttendanceIOWorker",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit_event(self, event: dict) -> bool:
        """
        Submit an authorized attendance event for asynchronous background HTTP POST.
        Non-blocking (<0.001ms). Returns True if queued, False if queue full or stopped.
        """
        if not self._running:
            return False
        try:
            self._event_queue.put_nowait(event)
            return True
        except queue.Full:
            logger.warning("Attendance event queue full; dropping event %s", event.get("event_id"))
            return False

    def maybe_trigger_flush(self) -> bool:
        """
        Check if flush_interval has elapsed and trigger background flush if not already active.
        Returns True if a flush was dispatched, False otherwise.
        Non-blocking (<0.001ms).
        """
        now = time.monotonic()
        if (now - self.last_queue_flush) < self.flush_interval:
            return False

        with self._flush_lock:
            if self._is_flushing:
                return False
            self._is_flushing = True
            # Update last_queue_flush at dispatch time using monotonic clock
            self.last_queue_flush = time.monotonic()

        flush_thread = threading.Thread(
            target=self._do_flush,
            name="AttendanceQueueFlush",
            daemon=True,
        )
        flush_thread.start()
        return True

    def _do_flush(self) -> None:
        """Execute flush_queue() in dedicated thread."""
        logger.info("Background offline queue flush initiated (non-blocking)...")
        try:
            with self._api_lock:
                flushed = flush_queue()
            if flushed > 0:
                self.total_flushed += flushed
                logger.info("Background offline queue flush completed: %d synced", flushed)
            else:
                logger.info("Background offline queue flush completed: 0 synced (backend offline or queue empty)")
        except Exception as exc:
            logger.warning("Background flush_queue encountered error: %s", exc)
        finally:
            with self._flush_lock:
                self._is_flushing = False

    def _worker_loop(self) -> None:
        """Background loop executing HTTP POSTs for newly recorded events."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self._api_lock:
                    online = post_event(event)
                if online:
                    self.total_posted += 1
                else:
                    self.total_failed += 1
            except Exception as exc:
                logger.warning("Background post_event error: %s", exc)
                self.total_failed += 1
            finally:
                self._event_queue.task_done()

    def is_flushing(self) -> bool:
        """Return True if a background flush is currently active."""
        with self._flush_lock:
            return self._is_flushing

    def get_stats(self) -> dict:
        """Return current statistics dictionary."""
        return {
            "total_posted": self.total_posted,
            "total_failed": self.total_failed,
            "total_flushed": self.total_flushed,
            "is_flushing": self.is_flushing(),
            "queue_size": self._event_queue.qsize(),
        }

    def stop(self) -> None:
        """Stop background worker cleanly."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
