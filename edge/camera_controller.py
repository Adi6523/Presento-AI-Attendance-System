"""
Presento Phase 16: Safe Dynamic Camera Controller Abstraction.
Provides conservative runtime exposure/gain control with stabilization delays
to prevent continuous oscillation and preserve the 30 FPS pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import platform
import subprocess
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("presento.edge.camera_controller")


class ControlState(str, Enum):
    IDLE = "IDLE"
    MEASURING = "MEASURING"
    ADJUSTING = "ADJUSTING"
    STABILIZING = "STABILIZING"
    LOCKED = "LOCKED"


@dataclass
class CameraControlStatus:
    state: ControlState
    is_supported: bool
    current_brightness: float
    target_brightness_range: Tuple[float, float]
    exposure_value: Optional[float]
    gain_value: Optional[float]
    adjustments_made: int
    feedback: str


class SafeCameraController:
    """
    Safe dynamic exposure, gain, and brightness controller for webcam / C270.
    Implements a strict:
        Measure -> Adjust once -> Wait for stabilization -> Measure again -> Lock
    sequence to avoid rapid oscillations, flickering, and frame-rate drops.
    """

    def __init__(
        self,
        target_min_brightness: float = 55.0,
        target_max_brightness: float = 195.0,
        stabilization_seconds: float = 1.5,
        max_adjustments: int = 3,
        step_size: float = 5.0,
    ):
        self.target_min = target_min_brightness
        self.target_max = target_max_brightness
        self.stabilization_seconds = stabilization_seconds
        self.max_adjustments = max_adjustments
        self.step_size = step_size

        self.state: ControlState = ControlState.IDLE
        self.adjustments_made: int = 0
        self.last_adjustment_time: float = 0.0
        self.is_supported: bool = True
        self._cv2_cap: Optional[cv2.VideoCapture] = None
        self._v4l2_device: Optional[str] = None

        # Probed capabilities
        self._supports_exposure: bool = False
        self._supports_gain: bool = False
        self._supports_brightness: bool = False

    def attach_camera(self, cap: cv2.VideoCapture, v4l2_device: Optional[str] = None) -> bool:
        """
        Probe camera device properties to determine control support.
        """
        self._cv2_cap = cap
        self._v4l2_device = v4l2_device
        self.state = ControlState.MEASURING
        self.adjustments_made = 0

        if cap is None:
            self.is_supported = False
            return False

        # Probe OpenCV properties
        try:
            exp = cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = cap.get(cv2.CAP_PROP_GAIN)
            bri = cap.get(cv2.CAP_PROP_BRIGHTNESS)

            self._supports_exposure = (exp != -1.0 and not np.isnan(exp))
            self._supports_gain = (gain != -1.0 and not np.isnan(gain))
            self._supports_brightness = (bri != -1.0 and not np.isnan(bri))

            self.is_supported = (self._supports_exposure or self._supports_gain or self._supports_brightness)
            logger.info(
                "CameraController probed (exposure=%s, gain=%s, brightness=%s)",
                self._supports_exposure,
                self._supports_gain,
                self._supports_brightness,
            )
        except Exception as exc:
            logger.debug("Camera control probe error: %s", exc)
            self.is_supported = False

        return self.is_supported

    def evaluate_and_adjust(
        self,
        face_roi: Optional[np.ndarray],
        current_time: Optional[float] = None,
    ) -> CameraControlStatus:
        """
        Evaluate face brightness and trigger conservative adjustment if needed.
        Non-blocking and adheres to stabilization timeouts.
        """
        now = current_time if current_time is not None else time.monotonic()

        if not self.is_supported or self._cv2_cap is None or face_roi is None or face_roi.size == 0:
            return CameraControlStatus(
                state=self.state,
                is_supported=self.is_supported,
                current_brightness=0.0,
                target_brightness_range=(self.target_min, self.target_max),
                exposure_value=None,
                gain_value=None,
                adjustments_made=self.adjustments_made,
                feedback="Control unavailable or no face ROI",
            )

        # Compute mean brightness on face ROI
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if face_roi.ndim == 3 else face_roi
        mean_brightness = float(np.mean(gray))

        # Check stabilization delay
        if self.state == ControlState.STABILIZING:
            if now - self.last_adjustment_time < self.stabilization_seconds:
                return CameraControlStatus(
                    state=ControlState.STABILIZING,
                    is_supported=True,
                    current_brightness=mean_brightness,
                    target_brightness_range=(self.target_min, self.target_max),
                    exposure_value=self._get_prop(cv2.CAP_PROP_EXPOSURE),
                    gain_value=self._get_prop(cv2.CAP_PROP_GAIN),
                    adjustments_made=self.adjustments_made,
                    feedback=f"Stabilizing camera sensor ({now - self.last_adjustment_time:.1f}s)",
                )
            else:
                self.state = ControlState.MEASURING

        # If already locked or maximum adjustments reached, maintain lock
        if self.state == ControlState.LOCKED or self.adjustments_made >= self.max_adjustments:
            self.state = ControlState.LOCKED
            return CameraControlStatus(
                state=ControlState.LOCKED,
                is_supported=True,
                current_brightness=mean_brightness,
                target_brightness_range=(self.target_min, self.target_max),
                exposure_value=self._get_prop(cv2.CAP_PROP_EXPOSURE),
                gain_value=self._get_prop(cv2.CAP_PROP_GAIN),
                adjustments_made=self.adjustments_made,
                feedback="Settings locked at acceptable quality",
            )

        # Measure quality
        if self.target_min <= mean_brightness <= self.target_max:
            # Ideal lighting achieved! Lock settings
            self.state = ControlState.LOCKED
            logger.info("Camera settings locked at brightness %.1f (adjustments=%d)", mean_brightness, self.adjustments_made)
            return CameraControlStatus(
                state=ControlState.LOCKED,
                is_supported=True,
                current_brightness=mean_brightness,
                target_brightness_range=(self.target_min, self.target_max),
                exposure_value=self._get_prop(cv2.CAP_PROP_EXPOSURE),
                gain_value=self._get_prop(cv2.CAP_PROP_GAIN),
                adjustments_made=self.adjustments_made,
                feedback="Optimal lighting reached, settings locked",
            )

        # Needs adjustment: too dark or too bright
        direction = 1.0 if mean_brightness < self.target_min else -1.0
        adjusted = self._apply_adjustment(direction)
        if adjusted:
            self.adjustments_made += 1
            self.last_adjustment_time = now
            self.state = ControlState.STABILIZING
            logger.info(
                "Adjusted camera (dir=%.0f, count=%d/%d, brightness=%.1f)",
                direction,
                self.adjustments_made,
                self.max_adjustments,
                mean_brightness,
            )
            feedback = f"Adjusted camera ({'increased' if direction > 0 else 'decreased'} exposure/gain)"
        else:
            self.state = ControlState.LOCKED
            feedback = "Hardware adjustment not accepted; locked"

        return CameraControlStatus(
            state=self.state,
            is_supported=True,
            current_brightness=mean_brightness,
            target_brightness_range=(self.target_min, self.target_max),
            exposure_value=self._get_prop(cv2.CAP_PROP_EXPOSURE),
            gain_value=self._get_prop(cv2.CAP_PROP_GAIN),
            adjustments_made=self.adjustments_made,
            feedback=feedback,
        )

    def _apply_adjustment(self, direction: float) -> bool:
        """Apply a conservative hardware adjustment via OpenCV or V4L2."""
        if self._cv2_cap is None:
            return False

        applied = False
        try:
            # 1. Try exposure
            if self._supports_exposure:
                curr_exp = self._cv2_cap.get(cv2.CAP_PROP_EXPOSURE)
                new_exp = curr_exp + (direction * self.step_size)
                self._cv2_cap.set(cv2.CAP_PROP_EXPOSURE, new_exp)
                applied = True

            # 2. Try brightness
            elif self._supports_brightness:
                curr_bri = self._cv2_cap.get(cv2.CAP_PROP_BRIGHTNESS)
                new_bri = curr_bri + (direction * (self.step_size / 100.0))
                self._cv2_cap.set(cv2.CAP_PROP_BRIGHTNESS, new_bri)
                applied = True

            # 3. Try gain
            elif self._supports_gain:
                curr_gain = self._cv2_cap.get(cv2.CAP_PROP_GAIN)
                new_gain = curr_gain + (direction * self.step_size)
                self._cv2_cap.set(cv2.CAP_PROP_GAIN, new_gain)
                applied = True

        except Exception as exc:
            logger.debug("Failed applying camera property adjustment: %s", exc)

        return applied

    def _get_prop(self, prop_id: int) -> Optional[float]:
        if self._cv2_cap is None:
            return None
        try:
            val = self._cv2_cap.get(prop_id)
            return float(val) if val != -1.0 and not np.isnan(val) else None
        except Exception:
            return None

    def reset(self) -> None:
        """Reset state machine for a new session."""
        self.state = ControlState.MEASURING
        self.adjustments_made = 0
        self.last_adjustment_time = 0.0
