from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class BlinkEvent:
    """Represents a single blink detection."""
    timestamp: float
    ear_value: float
    duration: float = 0.0  # Will be calculated when blink ends


@dataclass
class LivenessResult:
    """Result of liveness detection."""
    is_live: bool
    score: float  # Confidence score (0-1)
    blink_count: int
    ear_values: List[float]
    feedback: str
    timestamp: float = field(default_factory=time.time)


class LivenessDetector:
    """
    Advanced liveness detection using Eye Aspect Ratio (EAR) with temporal analysis.
    """

    def __init__(self,
                 blink_threshold: float = 0.20,
                 min_blink_interval: float = 0.1,
                 max_blink_duration: float = 0.5,
                 analysis_window: float = 5.0,
                 min_blinks_required: int = 2):
        """
        Initialize liveness detector.

        Args:
            blink_threshold: EAR threshold below which eye is considered closed
            min_blink_interval: Minimum time between blinks to prevent forced blinking
            max_blink_duration: Maximum duration for a valid blink
            analysis_window: Time window for liveness analysis (seconds)
            min_blinks_required: Minimum number of blinks required for liveness
        """
        self.blink_threshold = blink_threshold
        self.min_blink_interval = min_blink_interval
        self.max_blink_duration = max_blink_duration
        self.analysis_window = analysis_window
        self.min_blinks_required = min_blinks_required

        # State tracking
        self.blink_history: List[BlinkEvent] = []
        self.ear_baseline: Optional[float] = None
        self.is_eye_closed = False
        self.blink_start_time: Optional[float] = None
        self.last_blink_time: Optional[float] = None

    def calculate_ear(self, landmarks: dict) -> Optional[float]:
        """
        Calculate Eye Aspect Ratio from facial landmarks.

        Args:
            landmarks: Dictionary containing facial landmarks from face_recognition

        Returns:
            EAR value or None if landmarks insufficient
        """
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return None

        def _eye_aspect_ratio(eye_points):
            # Convert to numpy array
            points = np.array(eye_points, dtype=np.float64)

            # Compute vertical distances
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])

            # Compute horizontal distance
            horizontal = np.linalg.norm(points[0] - points[3])

            # Avoid division by zero
            if horizontal == 0:
                return 0.0

            return (vertical_1 + vertical_2) / (2.0 * horizontal)

        left_ear = _eye_aspect_ratio(landmarks["left_eye"])
        right_ear = _eye_aspect_ratio(landmarks["right_eye"])

        # Return the average of both eyes
        return (left_ear + right_ear) / 2.0

    def update_baseline(self, ear_values: List[float]):
        """
        Update the baseline EAR from initial measurements.

        Args:
            ear_values: List of EAR values to compute baseline from
        """
        if len(ear_values) >= 3:
            # Use median to be robust against outliers
            self.ear_baseline = np.median(ear_values)
            # Adaptive threshold based on baseline
            self.blink_threshold = max(0.15, min(0.25, self.ear_baseline * 0.7))

    def process_frame(self, ear_value: Optional[float]) -> LivenessResult:
        """
        Process a single frame for liveness detection.

        Args:
            ear_value: EAR value for current frame, None if face/landmarks not detected

        Returns:
            LivenessResult with current state
        """
        current_time = time.time()

        # Clean old blink history outside analysis window
        self._clean_old_blinks(current_time)

        # If no EAR value (no face/landmarks), return uncertain result
        if ear_value is None:
            return LivenessResult(
                is_live=False,
                score=0.0,
                blink_count=len(self.blink_history),
                ear_values=[],
                feedback="No face detected",
                timestamp=current_time
            )

        # Update baseline if we have enough samples
        if len(self.blink_history) < 10 and self.ear_baseline is None:
            # Collect initial samples for baseline
            pass  # Baseline will be updated externally

        # Blink detection state machine
        feedback = "Analyzing liveness..."

        if ear_value < self.blink_threshold and not self.is_eye_closed:
            # Eye just closed
            self.is_eye_closed = True
            self.blink_start_time = current_time
            feedback = "Eye closed - blink detected"

        elif ear_value >= self.blink_threshold and self.is_eye_closed:
            # Eye just opened - potential blink completion
            if self.blink_start_time is not None:
                blink_duration = current_time - self.blink_start_time

                # Validate blink duration
                if (blink_duration <= self.max_blink_duration and
                    (self.last_blink_time is None or
                     current_time - self.last_blink_time >= self.min_blink_interval)):

                # Valid blink detected
                    blink_event = BlinkEvent(
                        timestamp=current_time,
                        ear_value=ear_value,
                        duration=blink_duration
                    )
                    self.blink_history.append(blink_event)
                    self.last_blink_time = current_time
                    feedback = f"Blink detected ({len(self.blink_history)}/{self.min_blinks_required})"
                else:
                    feedback = f"Invalid blink duration ({blink_duration:.2f}s)"

            self.is_eye_closed = False
            self.blink_start_time = None

        elif self.is_eye_closed and self.blink_start_time is not None:
            # Eye still closed - check for excessive closure
            closure_duration = current_time - self.blink_start_time
            if closure_duration > self.max_blink_duration:
                feedback = "Eyes closed too long - possible spoof"
                # Force reset to avoid counting as blink
                self.is_eye_closed = False
                self.blink_start_time = None
            else:
                feedback = "Eye closed - waiting for opening"

        # Calculate liveness score
        liveness_score, is_live, feedback = self._calculate_liveness_score(current_time)

        # Get recent EAR values for display
        recent_ears = [blink.ear_value for blink in self.blink_history[-5:]] if self.blink_history else []

        return LivenessResult(
            is_live=is_live,
            score=liveness_score,
            blink_count=len(self.blink_history),
            ear_values=recent_ears,
            feedback=feedback,
            timestamp=current_time
        )

    def _clean_old_blinks(self, current_time: float):
        """Remove blink events outside the analysis window."""
        cutoff_time = current_time - self.analysis_window
        self.blink_history = [
            blink for blink in self.blink_history
            if blink.timestamp >= cutoff_time
        ]

    def _calculate_liveness_score(self, current_time: float) -> tuple[float, bool, str]:
        """
        Calculate liveness score based on blink history.

        Returns:
            Tuple of (score, is_live, feedback)
        """
        blink_count = len(self.blink_history)

        if blink_count < self.min_blinks_required:
            score = blink_count / self.min_blinks_required if self.min_blinks_required > 0 else 0.0
            feedback = f"Need {self.min_blinks_required - blink_count} more blinks"
            return score, False, feedback

        # Check blink regularity (variance in inter-blink intervals)
        if blink_count >= 2:
            intervals = []
            for i in range(1, blink_count):
                interval = self.blink_history[i].timestamp - self.blink_history[i-1].timestamp
                intervals.append(interval)

            if len(intervals) > 1:
                interval_variance = np.var(intervals)
                # Penalize too regular (spoof) or too irregular intervals
                regularity_penalty = min(1.0, interval_variance * 10)  # Adjust scaling as needed
            else:
                regularity_penalty = 0.0
        else:
            regularity_penalty = 0.0

        # Check for spoof indicators
        spoof_indicators = 0

        # Too many blinks in short time (possible replay attack)
        recent_blinks = [b for b in self.blink_history
                        if current_time - b.timestamp <= 2.0]  # Last 2 seconds
        if len(recent_blinks) > 3:  # More than 3 blinks in 2 seconds is suspicious
            spoof_indicators += 1

        # Blinks too periodic (spoof characteristic)
        if blink_count >= 3:
            intervals = []
            for i in range(1, blink_count):
                interval = self.blink_history[i].timestamp - self.blink_history[i-1].timestamp
                intervals.append(interval)

            if len(intervals) >= 2:
                # Check if intervals are too similar (coefficient of variation < 0.1)
                mean_interval = np.mean(intervals)
                if mean_interval > 0:
                    cv = np.std(intervals) / mean_interval
                    if cv < 0.1:  # Too regular
                        spoof_indicators += 1

        # Calculate base score
        base_score = min(1.0, blink_count / self.min_blinks_required)

        # Apply penalties
        penalty = min(0.5, regularity_penalty * 0.3 + spoof_indicators * 0.2)
        final_score = max(0.0, base_score - penalty)

        is_live = final_score >= 0.7 and blink_count >= self.min_blinks_required

        # Generate feedback
        if is_live:
            feedback = f"Liveness verified ({blink_count} blinks)"
        elif spoof_indicators > 0:
            feedback = f"Possible spoof detected ({blink_count} blinks)"
        elif regularity_penalty > 0.5:
            feedback = f"Blink pattern too regular ({blink_count} blinks)"
        else:
            feedback = f"Insufficient liveness ({blink_count}/{self.min_blinks_required} blinks)"

        return final_score, is_live, feedback

    def reset(self):
        """Reset the liveness detector state."""
        self.blink_history.clear()
        self.ear_baseline = None
        self.is_eye_closed = False
        self.blink_start_time = None
        self.last_blink_time = None


def create_liveness_detector(**kwargs) -> LivenessDetector:
    """Factory function to create a liveness detector with default or custom parameters."""
    return LivenessDetector(**kwargs)