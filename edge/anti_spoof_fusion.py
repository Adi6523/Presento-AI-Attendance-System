"""
Presento Hybrid Anti-Spoofing Fusion Module.

Benchmark-only fusion engine combining anti-spoof-mn3 and MiniFASNet V2 SE classifiers.
Evaluates four multi-model fusion strategies:
1. AND: Strict consensus (REAL only if both models classify REAL).
2. OR: Lenient consensus (REAL if either model classifies REAL).
3. AVERAGE: Score-level fusion averaging real probabilities with 0.50 decision boundary.
4. TEMPORAL: Windowed consistency tracking requiring >= 80% agreement over recent frames,
   yielding REAL, SPOOF, or UNCERTAIN.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class FrameEvaluation:
    """Individual frame classification results from both models."""
    mn3_p_real: float
    mn3_p_spoof: float
    mn3_label: str
    mf_p_real: float
    mf_p_spoof: float
    mf_label: str
    both_real: bool
    both_spoof: bool
    disagreement: bool


@dataclass
class FusionResult:
    """Consolidated decisions and scores across all four fusion strategies."""
    and_label: str          # "REAL" or "SPOOF"
    or_label: str           # "REAL" or "SPOOF"
    avg_p_real: float       # [0.0, 1.0]
    avg_p_spoof: float      # [0.0, 1.0]
    avg_label: str          # "REAL" or "SPOOF"
    temporal_label: str     # "REAL", "SPOOF", or "UNCERTAIN"
    both_real: bool
    both_spoof: bool
    disagreement: bool
    window_frames: int
    both_real_ratio: float
    both_spoof_ratio: float


def validate_probability(p: float, name: str = "probability") -> float:
    """Validate that a probability value is a finite number in [0.0, 1.0]."""
    if not isinstance(p, (int, float)):
        raise ValueError(f"{name} must be a numeric float or int, got {type(p).__name__}")
    if math.isnan(p) or math.isinf(p):
        raise ValueError(f"{name} must be finite, got {p}")
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {p}")
    return float(p)


def fuse_and(mn3_is_real: bool, mf_is_real: bool) -> str:
    """AND Strategy: REAL only when both models classify the frame as REAL."""
    return "REAL" if (mn3_is_real and mf_is_real) else "SPOOF"


def fuse_or(mn3_is_real: bool, mf_is_real: bool) -> str:
    """OR Strategy: REAL when either model classifies the frame as REAL."""
    return "REAL" if (mn3_is_real or mf_is_real) else "SPOOF"


def fuse_average(
    mn3_p_real: float,
    mf_p_real: float,
    threshold: float = 0.50,
) -> Tuple[float, float, str]:
    """
    AVERAGE Strategy: Arithmetic mean of real probabilities.
    Returns (combined_p_real, combined_p_spoof, label).
    """
    validate_probability(mn3_p_real, "mn3_p_real")
    validate_probability(mf_p_real, "mf_p_real")
    validate_probability(threshold, "threshold")

    avg_p_real = (mn3_p_real + mf_p_real) / 2.0
    avg_p_spoof = 1.0 - avg_p_real
    label = "REAL" if avg_p_real >= threshold else "SPOOF"
    return avg_p_real, avg_p_spoof, label


def fuse_temporal(
    history: List[FrameEvaluation],
    real_threshold: float = 0.80,
    spoof_threshold: float = 0.80,
) -> Tuple[str, float, float]:
    """
    TEMPORAL Strategy:
    Evaluates agreement across the sliding window:
    - REAL if >= 80% of frames are both_real
    - SPOOF if >= 80% of frames are both_spoof
    - Otherwise UNCERTAIN
    Returns (decision, both_real_ratio, both_spoof_ratio).
    """
    n = len(history)
    if n == 0:
        return "UNCERTAIN", 0.0, 0.0

    both_real_count = sum(1 for f in history if f.both_real)
    both_spoof_count = sum(1 for f in history if f.both_spoof)

    both_real_ratio = both_real_count / n
    both_spoof_ratio = both_spoof_count / n

    if both_real_ratio >= real_threshold:
        return "REAL", both_real_ratio, both_spoof_ratio
    elif both_spoof_ratio >= spoof_threshold:
        return "SPOOF", both_real_ratio, both_spoof_ratio
    else:
        return "UNCERTAIN", both_real_ratio, both_spoof_ratio


class AntiSpoofFusionEngine:
    """
    Sliding-window fusion engine combining anti-spoof-mn3 and MiniFASNet V2 SE outputs.
    Maintains a rolling queue of frame observations to provide instantaneous and temporal decisions.
    """

    def __init__(
        self,
        window_size: int = 15,
        real_ratio_threshold: float = 0.80,
        spoof_ratio_threshold: float = 0.80,
        average_boundary: float = 0.50,
    ):
        if window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        self.window_size = int(window_size)
        self.real_ratio_threshold = float(real_ratio_threshold)
        self.spoof_ratio_threshold = float(spoof_ratio_threshold)
        self.average_boundary = float(average_boundary)

        self.history: Deque[FrameEvaluation] = deque(maxlen=self.window_size)

    def reset(self):
        """Clear temporal history queue (e.g., between benchmark scenarios)."""
        self.history.clear()

    def update(
        self,
        mn3_p_real: float,
        mn3_p_spoof: float,
        mf_p_real: float,
        mf_p_spoof: float,
    ) -> FusionResult:
        """
        Ingest a new pair of frame predictions and compute all 4 fusion decisions.

        Args:
            mn3_p_real: float [0.0, 1.0] from anti-spoof-mn3
            mn3_p_spoof: float [0.0, 1.0] from anti-spoof-mn3
            mf_p_real: float [0.0, 1.0] from MiniFASNet V2 SE
            mf_p_spoof: float [0.0, 1.0] from MiniFASNet V2 SE

        Returns:
            FusionResult containing decisions for AND, OR, AVERAGE, and TEMPORAL.
        """
        mn3_r = validate_probability(mn3_p_real, "mn3_p_real")
        mn3_s = validate_probability(mn3_p_spoof, "mn3_p_spoof")
        mf_r = validate_probability(mf_p_real, "mf_p_real")
        mf_s = validate_probability(mf_p_spoof, "mf_p_spoof")

        mn3_is_real = mn3_r >= mn3_s
        mf_is_real = mf_r >= mf_s

        both_real = mn3_is_real and mf_is_real
        both_spoof = (not mn3_is_real) and (not mf_is_real)
        disagreement = mn3_is_real != mf_is_real

        frame_eval = FrameEvaluation(
            mn3_p_real=mn3_r,
            mn3_p_spoof=mn3_s,
            mn3_label="REAL" if mn3_is_real else "SPOOF",
            mf_p_real=mf_r,
            mf_p_spoof=mf_s,
            mf_label="REAL" if mf_is_real else "SPOOF",
            both_real=both_real,
            both_spoof=both_spoof,
            disagreement=disagreement,
        )
        self.history.append(frame_eval)

        # 1. AND strategy
        and_decision = fuse_and(mn3_is_real, mf_is_real)

        # 2. OR strategy
        or_decision = fuse_or(mn3_is_real, mf_is_real)

        # 3. AVERAGE strategy
        avg_r, avg_s, avg_decision = fuse_average(mn3_r, mf_r, self.average_boundary)

        # 4. TEMPORAL strategy
        temporal_decision, r_ratio, s_ratio = fuse_temporal(
            list(self.history),
            self.real_ratio_threshold,
            self.spoof_ratio_threshold,
        )

        return FusionResult(
            and_label=and_decision,
            or_label=or_decision,
            avg_p_real=avg_r,
            avg_p_spoof=avg_s,
            avg_label=avg_decision,
            temporal_label=temporal_decision,
            both_real=both_real,
            both_spoof=both_spoof,
            disagreement=disagreement,
            window_frames=len(self.history),
            both_real_ratio=r_ratio,
            both_spoof_ratio=s_ratio,
        )
