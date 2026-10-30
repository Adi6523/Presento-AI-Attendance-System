"""
Presento Phase 9–15: Liveness Session State Machine & Temporal Aggregator.
Implements:
1. Strict session state machine: NEW -> CHECKING -> REAL_LOCKED -> HOLD -> SPOOF -> RESET.
2. Temporal evidence aggregation (~30 frames) with EMA smoothing and hysteresis.
3. Multi-model consensus (MN3 + MiniFASNet) with disagreement handling.
4. Face quality gating (bad frame != spoof).
5. Liveness lease with expiration and face disappearance handling.
6. Zero state-inheritance across different face IDs.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from .config import settings
from .quality import assess_face_crop_quality

logger = logging.getLogger("presento.edge.liveness_session")


class LivenessState(str, Enum):
    NEW = "NEW"
    CHECKING = "CHECKING"
    REAL_LOCKED = "REAL_LOCKED"
    HOLD = "HOLD"
    SPOOF = "SPOOF"
    RESET = "RESET"


@dataclass
class LivenessObservation:
    """Single-frame anti-spoof observation."""
    frame_id: int
    timestamp: float
    mn3_p_real: float
    mf_p_real: float
    avg_p_real: float
    both_real: bool
    both_spoof: bool
    disagreement: bool
    quality_ok: bool
    quality_reason: str = ""


@dataclass
class LivenessDecision:
    """Consolidated liveness decision output for UI and attendance gating."""
    state: LivenessState
    ui_status: str
    is_live: bool               # True strictly when REAL_LOCKED
    is_attack: bool             # True when SPOOF
    ema_p_real: float
    both_real_ratio: float
    both_spoof_ratio: float
    observations_count: int
    lease_remaining: float
    face_id: int
    diagnostics: Dict[str, any] = field(default_factory=dict)


class LivenessSession:
    """
    Stateful temporal liveness session for a specific tracked face.
    Operates off rolling anti-spoof observations and quality gates.
    """

    def __init__(
        self,
        face_id: int,
        window_size: int = settings.liveness_window_size,
        real_entry_ratio: float = settings.liveness_real_entry_ratio,
        spoof_entry_ratio: float = settings.liveness_spoof_entry_ratio,
        real_exit_ratio: float = settings.liveness_real_exit_ratio,
        required_consecutive_windows: int = settings.liveness_required_consecutive_windows,
        ema_alpha: float = settings.liveness_ema_alpha,
        lease_duration: float = settings.liveness_lease_duration,
        hold_timeout: float = settings.liveness_hold_timeout,
        min_observations: Optional[int] = None,
    ):
        self.face_id = face_id
        self.window_size = int(window_size)
        self.real_entry_ratio = float(real_entry_ratio)
        self.spoof_entry_ratio = float(spoof_entry_ratio)
        self.real_exit_ratio = float(real_exit_ratio)
        self.required_consecutive_windows = int(required_consecutive_windows)
        self.ema_alpha = float(ema_alpha)
        self.lease_duration = float(lease_duration)
        self.hold_timeout = float(hold_timeout)
        self.min_observations = int(min_observations) if min_observations is not None else self.window_size

        self.state: LivenessState = LivenessState.NEW
        self.observations: Deque[LivenessObservation] = deque(maxlen=self.window_size)
        self.ema_p_real: float = 0.50
        self.lease_expiry: float = 0.0
        self.last_seen: float = time.monotonic()
        self.session_started_at: float = time.monotonic()

        self._consecutive_real_windows: int = 0
        self._consecutive_spoof_windows: int = 0

    def update(
        self,
        frame_id: int,
        mn3_p_real: float,
        mn3_p_spoof: float,
        mf_p_real: float,
        mf_p_spoof: float,
        crop: Optional[np.ndarray] = None,
        face_confidence: float = 0.90,
    ) -> LivenessDecision:
        """
        Ingest a new frame observation from the background worker.
        Gated by face quality: low-quality frames are held/ignored, never marked as spoof.
        """
        now = time.monotonic()
        self.last_seen = now

        # 1. Quality Gating (Phase 15)
        # Check if the face crop is usable
        quality_ok = True
        quality_reason = "OK"
        if crop is not None and crop.size > 0:
            ok, reason, _ = assess_face_crop_quality(
                crop,
                min_size=55,
                blur_threshold=22.0,
                lighting_min=30.0,
                lighting_max=240.0,
                max_clipping_ratio=0.30,
            )
            if not ok or face_confidence < 0.50:
                quality_ok = False
                quality_reason = reason if not ok else f"Low detector confidence ({face_confidence:.2f})"

        # If quality failed: DO NOT evaluate as spoof; hold current state and ignore sample
        if not quality_ok:
            logger.debug("Liveness sample ignored for face %d: %s", self.face_id, quality_reason)
            return self._build_decision(now, quality_ok=False, quality_reason=quality_reason)

        # 2. Model Agreement Analysis (Phase 14)
        mn3_is_real = mn3_p_real >= mn3_p_spoof
        mf_is_real = mf_p_real >= mf_p_spoof
        both_real = mn3_is_real and mf_is_real
        both_spoof = (not mn3_is_real) and (not mf_is_real)
        disagreement = mn3_is_real != mf_is_real

        avg_real = (mn3_p_real + mf_p_real) / 2.0

        # Update EMA smoothing
        self.ema_p_real = self.ema_alpha * avg_real + (1.0 - self.ema_alpha) * self.ema_p_real

        # Record observation
        obs = LivenessObservation(
            frame_id=frame_id,
            timestamp=now,
            mn3_p_real=mn3_p_real,
            mf_p_real=mf_p_real,
            avg_p_real=avg_real,
            both_real=both_real,
            both_spoof=both_spoof,
            disagreement=disagreement,
            quality_ok=True,
            quality_reason="OK",
        )
        self.observations.append(obs)

        # 3. Temporal Sliding Window Statistics
        n = len(self.observations)
        both_real_count = sum(1 for o in self.observations if o.both_real)
        both_spoof_count = sum(1 for o in self.observations if o.both_spoof)
        both_real_ratio = (both_real_count / n) if n > 0 else 0.0
        both_spoof_ratio = (both_spoof_count / n) if n > 0 else 0.0

        # 4. State Machine Progression (Phase 10 & 13)
        if self.state in (LivenessState.NEW, LivenessState.CHECKING):
            self.state = LivenessState.CHECKING
            # Require minimum observations buffer before making confident transitions
            if n >= self.min_observations:
                # Check for REAL transition
                if both_real_ratio >= self.real_entry_ratio:
                    self._consecutive_real_windows += 1
                    if self._consecutive_real_windows >= self.required_consecutive_windows:
                        self.state = LivenessState.REAL_LOCKED
                        self.lease_expiry = now + self.lease_duration
                        self._consecutive_real_windows = 0
                        logger.info("Face %d passed liveness: entered REAL_LOCKED", self.face_id)
                else:
                    self._consecutive_real_windows = 0

                # Check for SPOOF transition
                if both_spoof_ratio >= self.spoof_entry_ratio:
                    self._consecutive_spoof_windows += 1
                    if self._consecutive_spoof_windows >= self.required_consecutive_windows:
                        self.state = LivenessState.SPOOF
                        self._consecutive_spoof_windows = 0
                        logger.warning("Face %d failed liveness: entered SPOOF", self.face_id)
                else:
                    self._consecutive_spoof_windows = 0

        elif self.state == LivenessState.REAL_LOCKED:
            # Active lease maintenance
            if both_real_ratio >= self.real_exit_ratio:
                # Renew lease while observations support REAL
                self.lease_expiry = max(self.lease_expiry, now + self.lease_duration)
            else:
                # Persistent contradictory evidence: exit lock
                if now > self.lease_expiry or both_spoof_ratio >= self.spoof_entry_ratio:
                    logger.info("Face %d exited REAL_LOCKED due to contradictory evidence", self.face_id)
                    self.state = LivenessState.CHECKING

        elif self.state == LivenessState.HOLD:
            # Face reappeared during hold window!
            self.state = LivenessState.REAL_LOCKED
            self.lease_expiry = now + self.lease_duration
            logger.info("Face %d restored from HOLD to REAL_LOCKED", self.face_id)

        elif self.state == LivenessState.SPOOF:
            # If spoofing persistent, remain in SPOOF unless overwhelming real evidence
            if both_real_ratio >= self.real_entry_ratio:
                self._consecutive_real_windows += 1
                if self._consecutive_real_windows >= (self.required_consecutive_windows + 1):
                    self.state = LivenessState.CHECKING
                    self._consecutive_real_windows = 0
            else:
                self._consecutive_real_windows = 0

        return self._build_decision(now)

    def on_face_absent(self, now: Optional[float] = None) -> LivenessDecision:
        """
        Handle frame where face was not detected.
        Manages brief disappearance (HOLD) and session expiration (RESET).
        """
        cur = now if now is not None else time.monotonic()
        dt = cur - self.last_seen

        if self.state == LivenessState.REAL_LOCKED:
            if dt <= self.hold_timeout:
                self.state = LivenessState.HOLD
            else:
                self.state = LivenessState.RESET
        elif self.state == LivenessState.HOLD:
            if dt > self.hold_timeout:
                self.state = LivenessState.RESET
        elif dt > self.hold_timeout * 2:
            self.state = LivenessState.RESET

        return self._build_decision(cur)

    def _build_decision(
        self,
        now: float,
        quality_ok: bool = True,
        quality_reason: str = "",
    ) -> LivenessDecision:
        """Construct user-friendly decision and diagnostics."""
        n = len(self.observations)
        both_real_count = sum(1 for o in self.observations if o.both_real)
        both_spoof_count = sum(1 for o in self.observations if o.both_spoof)
        both_real_ratio = (both_real_count / n) if n > 0 else 0.0
        both_spoof_ratio = (both_spoof_count / n) if n > 0 else 0.0

        lease_rem = max(0.0, self.lease_expiry - now) if self.state == LivenessState.REAL_LOCKED else 0.0

        # High-level clean status messages (Phase 21)
        if self.state == LivenessState.CHECKING:
            ui_status = "Verifying liveness..."
        elif self.state == LivenessState.REAL_LOCKED:
            ui_status = "Live person verified"
        elif self.state == LivenessState.HOLD:
            ui_status = "Maintaining verification..."
        elif self.state == LivenessState.SPOOF:
            ui_status = "Presentation attack detected"
        elif self.state == LivenessState.RESET:
            ui_status = "Session reset"
        else:
            ui_status = "Initializing..."

        return LivenessDecision(
            state=self.state,
            ui_status=ui_status,
            is_live=(self.state == LivenessState.REAL_LOCKED),
            is_attack=(self.state == LivenessState.SPOOF),
            ema_p_real=round(self.ema_p_real, 3),
            both_real_ratio=round(both_real_ratio, 3),
            both_spoof_ratio=round(both_spoof_ratio, 3),
            observations_count=n,
            lease_remaining=round(lease_rem, 1),
            face_id=self.face_id,
            diagnostics={
                "quality_ok": quality_ok,
                "quality_reason": quality_reason,
                "window_frames": n,
                "session_age_s": round(now - self.session_started_at, 1),
            },
        )
