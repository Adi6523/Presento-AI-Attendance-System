from __future__ import annotations

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from dataclasses import dataclass
import hashlib


@dataclass
class CachedEmbedding:
    """Cached embedding with metadata."""
    student_id: str
    embedding: np.ndarray
    timestamp: float
    access_count: int = 0
    ttl: float = 300.0  # 5 minutes default TTL


class EmbeddingCache:
    """LRU cache for face embeddings to reduce database queries."""

    def __init__(self, max_size: int = 100, default_ttl: float = 300.0):
        """
        Initialize embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CachedEmbedding] = {}
        self._access_order: List[str] = []  # For LRU eviction

    def get(self, student_id: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.

        Args:
            student_id: Student identifier

        Returns:
            Embedding array or None if not found/expired
        """
        if student_id not in self._cache:
            return None

        cached = self._cache[student_id]

        # Check if expired
        if time.time() - cached.timestamp > cached.ttl:
            self._remove(student_id)
            return None

        # Update access info
        cached.access_count += 1
        self._update_access_order(student_id)

        return cached.embedding.copy()  # Return copy to prevent modification

    def put(self, student_id: str, embedding: np.ndarray, ttl: Optional[float] = None):
        """
        Put embedding in cache.

        Args:
            student_id: Student identifier
            embedding: Embedding array to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl

        # If already exists, update it
        if student_id in self._cache:
            self._cache[student_id].embedding = embedding.copy()
            self._cache[student_id].timestamp = time.time()
            self._cache[student_id].ttl = ttl
            self._update_access_order(student_id)
            return

        # If cache is full, evict LRU item
        if len(self._cache) >= self.max_size:
            self._evict_lru()

        # Add new entry
        cached_embedding = CachedEmbedding(
            student_id=student_id,
            embedding=embedding.copy(),
            timestamp=time.time(),
            ttl=ttl
        )
        self._cache[student_id] = cached_embedding
        self._access_order.append(student_id)

    def _remove(self, student_id: str):
        """Remove item from cache."""
        if student_id in self._cache:
            del self._cache[student_id]
            if student_id in self._access_order:
                self._access_order.remove(student_id)

    def _evict_lru(self):
        """Evict least recently used item."""
        if self._access_order:
            lru_student_id = self._access_order.pop(0)
            self._remove(lru_student_id)

    def _update_access_order(self, student_id: str):
        """Update access order for LRU."""
        if student_id in self._access_order:
            self._access_order.remove(student_id)
        self._access_order.append(student_id)

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_ids = [
            student_id for student_id, cached in self._cache.items()
            if current_time - cached.timestamp > cached.ttl
        ]
        for student_id in expired_ids:
            self._remove(student_id)


class EmbeddingStorage:
    """Handles embedding storage and retrieval with optimization."""

    def __init__(self, use_binary: bool = False):
        """
        Initialize embedding storage.

        Args:
            use_binary: Whether to use binary storage (future enhancement)
        """
        self.use_binary = use_binary
        self.cache = EmbeddingCache()

    def embedding_to_json(self, embedding: np.ndarray) -> str:
        """
        Convert embedding to JSON string for storage.

        Args:
            embedding: Numpy array embedding

        Returns:
            JSON string representation
        """
        return json.dumps(embedding.tolist())

    def json_to_embedding(self, json_str: str) -> np.ndarray:
        """
        Convert JSON string to embedding array.

        Args:
            json_str: JSON string representation

        Returns:
            Numpy array embedding
        """
        try:
            return np.array(json.loads(json_str), dtype=np.float64)
        except (json.JSONDecodeError, ValueError):
            # Return empty array on error
            return np.array([])

    def hash_embedding(self, embedding: np.ndarray) -> str:
        """
        Create hash of embedding for quick comparison.

        Args:
            embedding: Numpy array embedding

        Returns:
            MD5 hash string
        """
        return hashlib.md5(embedding.tobytes()).hexdigest()

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length.

        Args:
            embedding: Input embedding array

        Returns:
            Normalized embedding array
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Cosine similarity (-1 to 1)
        """
        # Normalize embeddings
        emb1_norm = self.normalize_embedding(emb1)
        emb2_norm = self.normalize_embedding(emb2)

        # Calculate dot product
        return np.dot(emb1_norm, emb2_norm)

    def distance_to_confidence(self, distance: float, tolerance: float) -> float:
        """
        Convert face distance to confidence score.

        Args:
            distance: Face distance (lower is better)
            tolerance: Tolerance threshold

        Returns:
            Confidence score (0-1)
        """
        if distance >= tolerance:
            return 0.0
        return max(0.0, 1.0 - (distance / tolerance))


class PreprocessingPipeline:
    """Handles image preprocessing for face recognition."""

    @staticmethod
    def align_face(frame: np.ndarray, landmarks: dict) -> Optional[np.ndarray]:
        """
        Align face using facial landmarks (affine transformation).

        Args:
            frame: Input BGR frame
            landmarks: Facial landmarks from face_recognition

        Returns:
            Aligned face image or None if alignment fails
        """
        try:
            # Extract key points for alignment
            # Using eyes and nose for basic alignment
            if ("left_eye" not in landmarks or
                "right_eye" not in landmarks or
                "nose_tip" not in landmarks):
                return frame  # Return original if insufficient landmarks

            left_eye = np.mean(landmarks["left_eye"], axis=0)
            right_eye = np.mean(landmarks["right_eye"], axis=0)
            nose_tip = np.mean(landmarks["nose_tip"], axis=0)

            # Calculate angle between eyes
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Calculate center point between eyes
            eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                          (left_eye[1] + right_eye[1]) / 2)

            # Get rotation matrix
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

            # Apply rotation
            aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            # Optional: crop to face region based on landmarks
            # For simplicity, we'll return the aligned full frame
            return aligned

        except Exception:
            # Return original frame if alignment fails
            return frame

    @staticmethod
    def enhance_lighting(frame: np.ndarray) -> np.ndarray:
        """
        Enhance lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            frame: Input BGR frame

        Returns:
            Enhanced frame
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)

            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced
        except Exception:
            return frame  # Return original if enhancement fails

    @staticmethod
    def reduce_noise(frame: np.ndarray) -> np.ndarray:
        """
        Reduce noise using bilateral filter.

        Args:
            frame: Input BGR frame

        Returns:
            Denoised frame
        """
        try:
            return cv2.bilateralFilter(frame, 9, 75, 75)
        except Exception:
            return frame  # Return original if denoising fails

    @staticmethod
    def preprocess_for_recognition(frame: np.ndarray,
                                  landmarks: Optional[dict] = None,
                                  align: bool = True,
                                  enhance_lighting: bool = True,
                                  reduce_noise: bool = False) -> np.ndarray:
        """
        Apply preprocessing pipeline for face recognition.

        Args:
            frame: Input BGR frame
            landmarks: Facial landmarks (optional, for alignment)
            align: Whether to apply face alignment
            enhance_lighting: Whether to enhance lighting
            reduce_noise: Whether to reduce noise

        Returns:
            Preprocessed frame
        """
        processed = frame.copy()

        # Apply preprocessing steps
        if reduce_noise:
            processed = PreprocessingPipeline.reduce_noise(processed)

        if enhance_lighting:
            processed = PreprocessingPipeline.enhance_lighting(processed)

        if align and landmarks is not None:
            processed = PreprocessingPipeline.align_face(processed, landmarks) or processed

        return processed