from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Dict, Optional, Tuple
import numpy as np

from .config import settings
from .anti_spoof import AntiSpoofClassifier, crop_face
from .minifas_antispoof import MiniFASNetV2SEClassifier, crop_face_minifas
from .anti_spoof_fusion import AntiSpoofFusionEngine, FusionResult
from .liveness_session import LivenessSession, LivenessDecision

logger = logging.getLogger("presento.edge.antispoof_worker")


@dataclass
class AntiSpoofJob:
    face_id: int
    frame_id: int
    crop_mn3: np.ndarray
    crop_minifas: np.ndarray
    submitted_at: float
    face_confidence: float = 0.90


@dataclass
class AntiSpoofResult:
    face_id: int
    frame_id: int
    mn3_p_real: float
    mn3_p_spoof: float
    mf_p_real: float
    mf_p_spoof: float
    fusion_result: FusionResult
    mn3_latency_ms: float
    mf_latency_ms: float
    updated_at: float
    liveness_decision: Optional[LivenessDecision] = None

    @property
    def total_latency_ms(self) -> float:
        return self.mn3_latency_ms + self.mf_latency_ms


class AntiSpoofWorker:
    """
    Dedicated background worker for asynchronous Anti-Spoofing inference.
    Executes Anti-Spoof-MN3 and MiniFASNet V2 SE inference off the main display thread.

    Key design principles:
    - Bounded single-slot pending job per face_id: If an older crop is pending when a newer
      crop arrives for the same face, the older crop is discarded immediately.
    - Zero lock contention during GPU/CPU neural network inference.
    - Per-face temporal fusion engine isolation (15-frame sliding window, 80% consensus).
    - Thread-safe non-blocking submission and latest-result retrieval.
    - Automatic pruning of inactive face engines.
    """

    def __init__(
        self,
        mn3_model_path: str,
        minifas_model_path: str,
        window_size: int = 15,
        real_ratio_threshold: float = 0.80,
        spoof_ratio_threshold: float = 0.80,
        device: str = settings.openvino_device,
    ):
        self.mn3_model_path = mn3_model_path
        self.minifas_model_path = minifas_model_path
        self.window_size = window_size
        self.real_ratio_threshold = real_ratio_threshold
        self.spoof_ratio_threshold = spoof_ratio_threshold
        self.device = device

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._job_event = threading.Event()

        # Bounded latest-job dictionary: face_id -> AntiSpoofJob (at most 1 pending per face)
        self._pending_jobs: Dict[int, AntiSpoofJob] = {}

        # Latest completed result per face_id: face_id -> AntiSpoofResult
        self._results: Dict[int, AntiSpoofResult] = {}

        # Per-face temporal fusion engines: face_id -> AntiSpoofFusionEngine
        self._fusion_engines: Dict[int, AntiSpoofFusionEngine] = {}

        # Per-face liveness state sessions: face_id -> LivenessSession
        self._liveness_sessions: Dict[int, LivenessSession] = {}

        # Last activity timestamp per face_id for cleanup
        self._last_active: Dict[int, float] = {}

        # Metrics
        self.total_jobs_submitted = 0
        self.total_jobs_completed = 0
        self.total_jobs_dropped = 0
        self.latest_latency_ms = 0.0

    def start(self) -> "AntiSpoofWorker":
        """Start the background worker thread."""
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="AntiSpoofWorker",
            daemon=True,
        )
        self._thread.start()
        return self

    def submit_face(
        self,
        face_id: int,
        frame_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        face_confidence: float = 0.90,
    ) -> None:
        """
        Submit a face for anti-spoof analysis.
        Extracts small crops immediately (sub-millisecond in OpenCV) and queues
        a bounded latest job. If a previous job for this face_id is still waiting,
        it is cleanly replaced with the latest crop.
        """
        if not self._running:
            return

        crop_mn3 = crop_face(frame, bbox)
        crop_mf = crop_face_minifas(frame, bbox)

        job = AntiSpoofJob(
            face_id=face_id,
            frame_id=frame_id,
            crop_mn3=crop_mn3,
            crop_minifas=crop_mf,
            submitted_at=time.monotonic(),
            face_confidence=face_confidence,
        )

        with self._lock:
            self.total_jobs_submitted += 1
            if face_id in self._pending_jobs:
                self.total_jobs_dropped += 1
            self._pending_jobs[face_id] = job
            self._last_active[face_id] = time.monotonic()
            self._job_event.set()

    def get_result(self, face_id: int) -> Optional[AntiSpoofResult]:
        """Fetch the latest completed anti-spoof result for face_id (non-blocking)."""
        with self._lock:
            return self._results.get(face_id)

    def prune_stale(self, active_face_ids: set[int], max_idle_seconds: float = 3.0) -> None:
        """Prune state and fusion engines for faces not seen for > max_idle_seconds."""
        now = time.monotonic()
        with self._lock:
            for fid in list(self._fusion_engines.keys()):
                last_seen = self._last_active.get(fid, 0.0)
                if fid not in active_face_ids and (now - last_seen) > max_idle_seconds:
                    self._fusion_engines.pop(fid, None)
                    self._liveness_sessions.pop(fid, None)
                    self._results.pop(fid, None)
                    self._pending_jobs.pop(fid, None)
                    self._last_active.pop(fid, None)

    def _worker_loop(self) -> None:
        """Background inference loop."""
        logger.info("AntiSpoofWorker background inference thread started (device=%s).", self.device)
        mn3 = AntiSpoofClassifier(self.mn3_model_path, device=self.device)
        minifas = MiniFASNetV2SEClassifier(self.minifas_model_path, device=self.device)

        while self._running:
            job: AntiSpoofJob | None = None
            with self._lock:
                if self._pending_jobs:
                    # Pop the next pending job
                    next_fid = next(iter(self._pending_jobs))
                    job = self._pending_jobs.pop(next_fid)
                else:
                    self._job_event.clear()

            if job is None:
                self._job_event.wait(timeout=0.05)
                continue

            # Run inference WITHOUT holding any lock!
            try:
                mn3_r, mn3_s, _, mn3_lat = mn3.predict(job.crop_mn3)
                mf_r, mf_s, _, mf_lat = minifas.predict(job.crop_minifas)
            except Exception as exc:
                logger.error("AntiSpoofWorker inference error on face %d: %s", job.face_id, exc)
                continue

            # Update per-face temporal fusion engine & liveness state machine
            with self._lock:
                if job.face_id not in self._fusion_engines:
                    self._fusion_engines[job.face_id] = AntiSpoofFusionEngine(
                        window_size=self.window_size,
                        real_ratio_threshold=self.real_ratio_threshold,
                        spoof_ratio_threshold=self.spoof_ratio_threshold,
                    )
                engine = self._fusion_engines[job.face_id]
                fus_res = engine.update(mn3_r, mn3_s, mf_r, mf_s)

                if job.face_id not in self._liveness_sessions:
                    self._liveness_sessions[job.face_id] = LivenessSession(
                        face_id=job.face_id,
                        window_size=settings.liveness_window_size,
                        real_entry_ratio=settings.liveness_real_entry_ratio,
                        spoof_entry_ratio=settings.liveness_spoof_entry_ratio,
                        real_exit_ratio=settings.liveness_real_exit_ratio,
                        required_consecutive_windows=settings.liveness_required_consecutive_windows,
                        ema_alpha=settings.liveness_ema_alpha,
                        lease_duration=settings.liveness_lease_duration,
                        hold_timeout=settings.liveness_hold_timeout,
                    )
                session = self._liveness_sessions[job.face_id]
                liv_decision = session.update(
                    frame_id=job.frame_id,
                    mn3_p_real=mn3_r,
                    mn3_p_spoof=mn3_s,
                    mf_p_real=mf_r,
                    mf_p_spoof=mf_s,
                    crop=job.crop_mn3,
                    face_confidence=getattr(job, "face_confidence", 0.90),
                )

                now = time.monotonic()
                self._results[job.face_id] = AntiSpoofResult(
                    face_id=job.face_id,
                    frame_id=job.frame_id,
                    mn3_p_real=mn3_r,
                    mn3_p_spoof=mn3_s,
                    mf_p_real=mf_r,
                    mf_p_spoof=mf_s,
                    fusion_result=fus_res,
                    mn3_latency_ms=mn3_lat,
                    mf_latency_ms=mf_lat,
                    updated_at=now,
                    liveness_decision=liv_decision,
                )
                self.total_jobs_completed += 1
                self.latest_latency_ms = mn3_lat + mf_lat

    def get_stats(self) -> dict[str, any]:
        """Return worker operational statistics."""
        with self._lock:
            return {
                "submitted": self.total_jobs_submitted,
                "completed": self.total_jobs_completed,
                "dropped": self.total_jobs_dropped,
                "pending": len(self._pending_jobs),
                "latest_latency_ms": self.latest_latency_ms,
            }

    def stop(self) -> None:
        """Stop worker cleanly."""
        self._running = False
        self._job_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("AntiSpoofWorker stopped.")
