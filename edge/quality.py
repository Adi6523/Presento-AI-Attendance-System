from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FaceQuality:
    """Face quality assessment results."""
    score: float  # Overall quality score (0-1)
    size_ok: bool
    position_ok: bool
    blur_ok: bool
    lighting_ok: bool
    pose_ok: bool
    occlusion_ok: bool
    expression_ok: bool
    feedback: str  # Human-readable feedback


class FaceQualityAssessor:
    """Assess face quality for enrollment and recognition."""

    def __init__(self):
        # Load face cascade for additional validation (backup)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Quality thresholds (can be configured)
        self.min_face_size_ratio = 0.2  # Face should be at least 20% of frame
        self.max_face_size_ratio = 0.8  # But not more than 80%
        self.position_tolerance = 0.15  # Face center within 15% of frame center
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.lighting_min = 50  # Minimum mean pixel intensity
        self.lighting_max = 200  # Maximum mean pixel intensity
        self.pose_angle_threshold = 15.0  # Degrees for yaw/pitch/roll
        self.occlusion_threshold = 0.7  # Minimum visibility ratio for eyes/mouth

    def assess_quality(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> FaceQuality:
        """
        Assess the quality of a detected face.

        Args:
            frame: BGR image frame
            face_location: (top, right, bottom, bottom) face coordinates

        Returns:
            FaceQuality object with assessment results
        """
        top, right, bottom, left = face_location
        height, width = frame.shape[:2]

        # Extract face ROI
        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            return FaceQuality(
                score=0.0, size_ok=False, position_ok=False, blur_ok=False,
                lighting_ok=False, pose_ok=False, occlusion_ok=False,
                expression_ok=False, feedback="Invalid face region"
            )

        # Convert to grayscale for some calculations
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi

        # 1. Face Size Check
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        frame_area = width * height
        size_ratio = face_area / frame_area if frame_area > 0 else 0

        size_ok = (self.min_face_size_ratio <= size_ratio <= self.max_face_size_ratio)

        # 2. Face Position Check
        face_center_x = (left + right) / 2
        face_center_y = (top + bottom) / 2
        frame_center_x = width / 2
        frame_center_y = height / 2

        pos_error_x = abs(face_center_x - frame_center_x) / width
        pos_error_y = abs(face_center_y - frame_center_y) / height
        position_ok = (pos_error_x <= self.position_tolerance and
                      pos_error_y <= self.position_tolerance)

        # 3. Blur Detection (Laplacian variance)
        blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        blur_ok = blur_score >= self.blur_threshold

        # 4. Lighting Check
        mean_intensity = np.mean(gray_roi)
        lighting_ok = (self.lighting_min <= mean_intensity <= self.lighting_max)

        # 5-8. More advanced checks would require facial landmarks
        # For now, we'll use simplified versions or skip and return good scores
        # These would be enhanced with face_recognition landmarks in practice

        # Simplified pose check (based on aspect ratio as proxy)
        aspect_ratio = face_width / face_height if face_height > 0 else 0
        # Typical frontal face aspect ratio is around 0.7-0.8
        pose_ok = 0.6 <= aspect_ratio <= 0.9  # Simplified

        # Simplified occlusion check (based on ROI completeness)
        # In practice, this would use landmarks to check eye/mouth visibility
        occlusion_ok = True  # Placeholder - would be enhanced

        # Simplified expression check (based on ROI uniformity)
        # In practice, this would use landmarks or expression classifiers
        expression_ok = True  # Placeholder - would be enhanced

        # Calculate overall score
        checks = [size_ok, position_ok, blur_ok, lighting_ok,
                 pose_ok, occlusion_ok, expression_ok]
        passed_checks = sum(checks)
        overall_score = passed_checks / len(checks)

        # Generate feedback
        feedback_parts = []
        if not size_ok:
            if size_ratio < self.min_face_size_ratio:
                feedback_parts.append("Move closer")
            else:
                feedback_parts.append("Move farther")
        if not position_ok:
            feedback_parts.append("Center your face")
        if not blur_ok:
            feedback_parts.append("Hold still - image is blurry")
        if not lighting_ok:
            if mean_intensity < self.lighting_min:
                feedback_parts.append("Improve lighting")
            else:
                feedback_parts.append("Reduce lighting - too bright")
        if not pose_ok:
            feedback_parts.append("Adjust face angle")
        if not occlusion_ok:
            feedback_parts.append("Remove obstructions")
        if not expression_ok:
            feedback_parts.append("Maintain neutral expression")

        feedback = "Good quality" if overall_score >= 0.7 else ", ".join(feedback_parts) or "Unknown issue"

        return FaceQuality(
            score=overall_score,
            size_ok=size_ok,
            position_ok=position_ok,
            blur_ok=blur_ok,
            lighting_ok=lighting_ok,
            pose_ok=pose_ok,
            occlusion_ok=occlusion_ok,
            expression_ok=expression_ok,
            feedback=feedback
        )


def assess_face_quality(frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> FaceQuality:
    """
    Convenience function for face quality assessment.

    Args:
        frame: BGR image frame
        face_location: (top, right, bottom, left) face coordinates

    Returns:
        FaceQuality object
    """
    assessor = FaceQualityAssessor()
    return assessor.assess_quality(frame, face_location)


@dataclass
class NeuralFaceQuality:
    """Detailed quality metrics for SCRFD detections and ArcFace/Liveness gating."""
    is_valid: bool
    reason: str
    confidence: float
    width: int
    height: int
    blur_score: float
    mean_brightness: float
    clipping_ratio: float
    landmarks_valid: bool
    pose_symmetry: float
    metrics: dict


def assess_face_crop_quality(
    crop: np.ndarray,
    min_size: int = 60,
    blur_threshold: float = 30.0,
    lighting_min: float = 35.0,
    lighting_max: float = 230.0,
    max_clipping_ratio: float = 0.20,
) -> Tuple[bool, str, dict]:
    """
    Evaluate crop-level image quality (blur, lighting, saturation/clipping, dimensions).
    Used for enrollment frame validation and anti-spoof evidence gating.

    Returns:
        Tuple of (is_valid, reason, metrics_dict).
    """
    if crop is None or crop.size == 0:
        return False, "Empty or invalid face crop", {}

    h, w = crop.shape[:2]
    if w < min_size or h < min_size:
        return False, f"Face crop too small ({w}x{h} < {min_size}px)", {"width": w, "height": h}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # 1. Lighting / brightness check
    mean_brightness = float(np.mean(gray))
    if mean_brightness < lighting_min:
        return False, f"Face too dark ({mean_brightness:.1f} < {lighting_min:.1f}) - improve lighting", {
            "mean_brightness": mean_brightness,
        }
    if mean_brightness > lighting_max:
        return False, f"Face too bright ({mean_brightness:.1f} > {lighting_max:.1f}) - reduce glare", {
            "mean_brightness": mean_brightness,
        }

    # 2. Blur via Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < blur_threshold:
        return False, f"Face too blurry ({blur_score:.1f} < {blur_threshold:.1f}) - hold steady", {
            "blur_score": blur_score,
            "width": w,
            "height": h,
        }

    # 3. Saturation / overexposure / underexposure clipping
    clipped_pixels = int(np.sum((gray <= 5) | (gray >= 250)))
    clipping_ratio = float(clipped_pixels / gray.size)
    if clipping_ratio > max_clipping_ratio:
        return False, f"Severe lighting clipping ({clipping_ratio * 100:.1f}% > {max_clipping_ratio * 100:.1f}%)", {
            "clipping_ratio": clipping_ratio,
            "mean_brightness": mean_brightness,
            "blur_score": blur_score,
        }

    metrics = {
        "width": w,
        "height": h,
        "blur_score": round(blur_score, 2),
        "mean_brightness": round(mean_brightness, 2),
        "clipping_ratio": round(clipping_ratio, 4),
    }
    return True, "Quality acceptable", metrics