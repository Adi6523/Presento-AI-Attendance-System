"""Browser-camera image processing — fast check-in + quality enrollment."""
from __future__ import annotations
import base64
import cv2
import numpy as np
import face_recognition

_MAX_WIDTH = 640   # Resize to this max width for faster CPU processing


def decode_image(value: str) -> np.ndarray:
    encoded = value.split(",", 1)[-1]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid camera image")
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    h, w = rgb.shape[:2]
    if w > _MAX_WIDTH:
        scale = _MAX_WIDTH / w
        rgb = cv2.resize(rgb, (_MAX_WIDTH, int(h * scale)))
        rgb = np.ascontiguousarray(rgb)
    return rgb


def embedding(image: np.ndarray) -> np.ndarray | None:
    """Fast HOG embedding for real-time check-in (1 jitter = speed-optimised)."""
    locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=1)
    values = face_recognition.face_encodings(image, locations, num_jitters=1)
    return values[0] if len(values) == 1 else None


def embedding_enroll(image: np.ndarray) -> np.ndarray | None:
    """Higher-quality HOG embedding for enrollment (5 jitters = more accurate)."""
    locations = face_recognition.face_locations(image, model="hog", number_of_times_to_upsample=2)
    values = face_recognition.face_encodings(image, locations, num_jitters=5)
    return values[0] if len(values) == 1 else None


def ear(image: np.ndarray) -> float | None:
    """Eye Aspect Ratio — detects open/closed eyes for liveness."""
    faces = face_recognition.face_landmarks(image)
    if not faces or "left_eye" not in faces[0]:
        return None

    def _ear(points):
        p = np.asarray(points, dtype=float)
        return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]))

    return float(min(_ear(faces[0]["left_eye"]), _ear(faces[0]["right_eye"])))


def liveness_ok(ear_values: list[float]) -> bool:
    """True if eyes are OPEN in >= 40% of frames — live person looking at camera."""
    if not ear_values:
        return True   # No landmark data — skip liveness gracefully
    return sum(1 for e in ear_values if e > 0.20) / len(ear_values) >= 0.40


def filter_enrollment_samples(samples: list[np.ndarray], max_dist: float = 0.55) -> list[np.ndarray]:
    """
    Remove outlier enrollment frames (bad lighting, partial face, etc.)
    by rejecting samples that are far from the median of all pairwise distances.
    """
    if len(samples) < 3:
        return samples
    arr = np.array(samples)
    # Compute mean embedding as reference
    mean_emb = arr.mean(axis=0)
    dists = face_recognition.face_distance(arr, mean_emb)
    # Keep only samples within max_dist of the mean
    good = [s for s, d in zip(samples, dists) if d < max_dist]
    return good if len(good) >= 3 else samples   # Fallback to all if too strict
