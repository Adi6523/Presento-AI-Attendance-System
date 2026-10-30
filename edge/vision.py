from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import face_recognition
import cv2
from .quality import FaceQualityAssessor, FaceQuality
from .storage import PreprocessingPipeline

@dataclass
class Candidate:
    student_id: str
    name: str
    confidence: float
    location: tuple[int, int, int, int]

def _parse_embeddings(people: list[dict]) -> tuple[list[dict], np.ndarray]:
    valid_people = []
    known_list = []
    import json
    for p in people:
        emb = p.get("embedding")
        if emb is None or emb == "[]" or emb == []:
            continue
        try:
            if isinstance(emb, str):
                vec = np.array(json.loads(emb), dtype=np.float64)
            else:
                vec = np.array(emb, dtype=np.float64)
            if len(vec) == 128:
                valid_people.append(p)
                known_list.append(vec)
        except Exception:
            continue
    if not known_list:
        return [], np.empty((0, 128), dtype=np.float64)
    return valid_people, np.asarray(known_list, dtype=np.float64)

def match_face_encoding(encoding: np.ndarray, location: tuple[int, int, int, int], people: list[dict], tolerance: float) -> Candidate | None:
    """Matches an already extracted face encoding against enrolled people without running HOG detection."""
    valid_people, known = _parse_embeddings(people)
    if len(valid_people) == 0 or len(encoding) == 0:
        return None
    distances = face_recognition.face_distance(known, encoding)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    if distance <= tolerance:
        confidence = max(0.0, min(1.0, 1.0 - distance / tolerance))
        person = valid_people[index]
        return Candidate(person["id"], person["name"], confidence, location)
    return None

def find_candidate(frame: np.ndarray, people: list[dict], tolerance: float) -> Candidate | None:
    # Apply preprocessing for better recognition
    processed_frame = PreprocessingPipeline.preprocess_for_recognition(
        frame,
        align=True,
        enhance_lighting=True,
        reduce_noise=False
    )

    rgb = np.ascontiguousarray(processed_frame[:, :, ::-1])
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings or not people: return None
    best: Candidate | None = None
    for encoding, location in zip(encodings, locations):
        candidate = match_face_encoding(encoding, location, people, tolerance)
        if candidate and (best is None or candidate.confidence > best.confidence):
            best = candidate
    return best

def eye_aspect_ratio(frame: np.ndarray) -> float | None:
    """Returns the smaller EAR of both eyes from 6-point facial landmarks.
    Uses the FIRST face in the frame — kept for backward compatibility."""
    landmarks = face_recognition.face_landmarks(np.ascontiguousarray(frame[:, :, ::-1]))
    if not landmarks or "left_eye" not in landmarks[0]: return None
    def ear(points):
        p = np.asarray(points, dtype=float)
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*np.linalg.norm(p[0]-p[3]))
    return float(min(ear(landmarks[0]["left_eye"]), ear(landmarks[0]["right_eye"])))


def eye_aspect_ratio_for_face(frame: np.ndarray, face_location: tuple[int, int, int, int]) -> float | None:
    """Returns the EAR for a SPECIFIC face location.

    This solves the multi-face liveness issue: when multiple faces are visible,
    each face gets its own EAR computation instead of always using the first face's landmarks.

    Args:
        frame: BGR image frame
        face_location: (top, right, bottom, left) face coordinates from face_recognition

    Returns:
        Minimum EAR of both eyes, or None if landmarks are not detected for this face
    """
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    # face_landmarks accepts face_locations to compute landmarks for specific faces
    landmarks_list = face_recognition.face_landmarks(rgb, face_locations=[face_location])
    if not landmarks_list or "left_eye" not in landmarks_list[0]:
        return None
    def ear(points):
        p = np.asarray(points, dtype=float)
        return (np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*np.linalg.norm(p[0]-p[3]))
    return float(min(ear(landmarks_list[0]["left_eye"]), ear(landmarks_list[0]["right_eye"])))


def assess_face_quality(frame: np.ndarray, face_location: tuple[int, int, int, int]) -> FaceQuality:
    """
    Assess the quality of a detected face.

    Args:
        frame: BGR image frame
        face_location: (top, right, bottom, left) face coordinates

    Returns:
        FaceQuality object with assessment results
    """
    assessor = FaceQualityAssessor()
    return assessor.assess_quality(frame, face_location)

