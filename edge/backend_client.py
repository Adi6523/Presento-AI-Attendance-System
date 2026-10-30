"""
Presento Edge-to-Backend Client Adapter.

This module acts as a non-blocking adapter connecting the AI Vision engine
to the clean FastAPI backend (`POST /api/attendance/events`).

Key Features:
- Does NOT block camera capture or OpenCV inference loop.
- Dynamic session discovery (`get_active_session_id`) - no hardcoded IDs.
- Thread-safe non-blocking submission queue.
- Safe offline queuing to `data/offline-events.jsonl` when network/backend is down.
- Automatic or manual synchronization (`flush_queue`) when reconnected.
- Authenticates via PRESENTO_AI_API_KEY (Bearer token), distinct from frontend JWT.

AI Liveness Contract:
    Only events with liveness_state == 'REAL_LOCKED' will be admitted by the backend.
    The AI engine is the exclusive authority for liveness decisions.
    Raw liveness_score is forwarded as optional diagnostic metadata only.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import requests

from .config import settings

logger = logging.getLogger("presento.edge.backend_client")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OFFLINE_QUEUE_PATH = _PROJECT_ROOT / "data" / "offline-events.jsonl"

# Valid liveness states the AI engine may report.
VALID_LIVENESS_STATES = {"NEW", "CHECKING", "REAL_LOCKED", "HOLD", "SPOOF", "RESET", "UNCERTAIN"}


def _ai_auth_headers() -> dict:
    """Return Bearer auth headers using the AI API key from config."""
    return {"Authorization": f"Bearer {settings.api_token}"}


class BackendClient:
    """Synchronous HTTP client communicating with Presento FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 5.0):
        self.base_url = (base_url or settings.api_url).rstrip("/")
        self.timeout = timeout

    def get_active_session_id(self, class_id: Optional[int] = None) -> Optional[int]:
        """
        Dynamically discovers the current ACTIVE session from the backend.
        Never hardcode session_id.
        """
        url = f"{self.base_url}/api/sessions?status=ACTIVE"
        if class_id is not None:
            url += f"&class_id={class_id}"

        try:
            resp = requests.get(url, headers=_ai_auth_headers(), timeout=self.timeout)
            if resp.ok:
                sessions = resp.json()
                if sessions and isinstance(sessions, list) and len(sessions) > 0:
                    return sessions[0]["id"]
            return None
        except requests.RequestException as exc:
            logger.debug("Failed to query active session from %s: %s", url, exc)
            return None

    def post_attendance_event(
        self,
        student_id: str,
        session_id: int,
        confidence: float,
        liveness_state: str,
        liveness_score: Optional[float] = None,
        source: str = "AI_CAMERA",
        timestamp: Optional[str] = None,
    ) -> dict:
        """
        Sends an attendance event according to the AI integration contract.

        liveness_state: Temporal state from the AI liveness engine.
                        Only 'REAL_LOCKED' will be admitted by the backend.
        liveness_score: Optional raw anti-spoof probability (metadata only).
        confidence:     Cosine similarity from identity engine.

        Returns the parsed backend decision: PRESENT, DUPLICATE, BLOCKED, or REJECTED.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        state_upper = liveness_state.upper()
        if state_upper not in VALID_LIVENESS_STATES:
            logger.warning(
                "Unknown liveness_state '%s' for student '%s'. Sending anyway; backend will reject.",
                liveness_state, student_id,
            )

        payload: dict = {
            "student_id": student_id,
            "session_id": session_id,
            "confidence": round(float(confidence), 4),
            "liveness_state": state_upper,
            "source": source,
            "timestamp": timestamp,
        }
        if liveness_score is not None:
            payload["liveness_score"] = round(float(liveness_score), 4)

        url = f"{self.base_url}/api/attendance/events"
        resp = requests.post(url, json=payload, headers=_ai_auth_headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class AsyncBackendClient:
    """
    Non-blocking background client adapter for attendance network I/O.
    Decoupled from camera frame rendering.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        queue_path: Optional[Path] = None,
        flush_interval: float = 10.0,
    ):
        self.client = BackendClient(base_url=base_url)
        self.queue_path = queue_path or DEFAULT_OFFLINE_QUEUE_PATH
        self.flush_interval = flush_interval

        self._event_queue: queue.Queue[dict] = queue.Queue(maxsize=300)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._is_flushing = False
        self.last_flush_time = time.monotonic()

        self.total_posted = 0
        self.total_failed = 0
        self.total_flushed = 0

    def start(self) -> "AsyncBackendClient":
        """Start background queue consumer thread."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncBackendClientWorker",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit_event(
        self,
        student_id: str,
        session_id: int,
        confidence: float,
        liveness_state: str,
        liveness_score: Optional[float] = None,
        source: str = "AI_CAMERA",
        timestamp: Optional[str] = None,
    ) -> bool:
        """
        Non-blocking enqueue (<0.001ms).
        liveness_state must be 'REAL_LOCKED' to be admitted — backend enforces this.
        Returns True if queued, False if queue is full or stopped.
        """
        if not self._running:
            return False

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        item = {
            "student_id": student_id,
            "session_id": session_id,
            "confidence": confidence,
            "liveness_state": liveness_state,
            "liveness_score": liveness_score,
            "source": source,
            "timestamp": timestamp,
        }

        try:
            self._event_queue.put_nowait(item)
            return True
        except queue.Full:
            logger.warning("Event queue full; persisting offline immediately.")
            self._write_offline(item)
            return False

    def _write_offline(self, item: dict) -> None:
        """Safely appends dropped/failed item to offline file."""
        try:
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
            with self.queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")
        except Exception as exc:
            logger.error("Failed to write offline event: %s", exc)

    def _worker_loop(self) -> None:
        """Background thread loop processing queued attendance events."""
        while self._running:
            try:
                item = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                with self._io_lock:
                    decision = self.client.post_attendance_event(
                        student_id=item["student_id"],
                        session_id=item["session_id"],
                        confidence=item["confidence"],
                        liveness_state=item["liveness_state"],
                        liveness_score=item.get("liveness_score"),
                        source=item.get("source", "AI_CAMERA"),
                        timestamp=item.get("timestamp"),
                    )
                self.total_posted += 1
                logger.info(
                    "Backend decision for %s in session %d: %s",
                    item["student_id"],
                    item["session_id"],
                    decision.get("status"),
                )
            except Exception as exc:
                self.total_failed += 1
                logger.warning("Backend unreachable (%s); writing to offline queue", exc)
                self._write_offline(item)
            finally:
                self._event_queue.task_done()

    def flush_queue(self) -> int:
        """Synchronizes events stored in offline queue."""
        if not self.queue_path.exists():
            return 0

        with self._flush_lock:
            try:
                raw_lines = self.queue_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return 0

            pending = [json.loads(line) for line in raw_lines if line.strip()]
            if not pending:
                return 0

            remaining, synced = [], 0
            for ev in pending:
                try:
                    with self._io_lock:
                        self.client.post_attendance_event(
                            student_id=ev["student_id"],
                            session_id=ev["session_id"],
                            confidence=ev["confidence"],
                            liveness_state=ev["liveness_state"],
                            liveness_score=ev.get("liveness_score"),
                            source=ev.get("source", "AI_CAMERA"),
                            timestamp=ev.get("timestamp"),
                        )
                    synced += 1
                except Exception:
                    remaining.append(ev)

            if remaining:
                self.queue_path.write_text(
                    "\n".join(json.dumps(r) for r in remaining) + "\n", encoding="utf-8"
                )
            else:
                self.queue_path.unlink(missing_ok=True)

            self.total_flushed += synced
            return synced

    def get_stats(self) -> dict:
        return {
            "total_posted": self.total_posted,
            "total_failed": self.total_failed,
            "total_flushed": self.total_flushed,
            "queue_size": self._event_queue.qsize(),
        }

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
