"""
Presento Phase 2B: Multi-Embedding Face Matcher & Diagnostic Identity Engine.
Ranks live 512-D ArcFace embeddings against enrolled students' multiple reference vectors,
calculates best similarity, second-best similarity, margin, and multi-frame stability.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Optional
import numpy as np

from .config import settings
from .face_enrollment import FaceEnrollmentStore, validate_embedding

logger = logging.getLogger("presento.edge.matcher")


@dataclass
class CandidateMatch:
    """Represents an enrolled student candidate match with similarity score."""
    student_id: str
    name: str
    similarity: float
    best_sample_index: int


@dataclass
class MatchResult:
    """Comprehensive match diagnostic for a live detected face."""
    best_match: Optional[CandidateMatch]
    second_best_match: Optional[CandidateMatch]
    margin: float
    is_stable: bool
    stability_label: str
    all_candidates: List[CandidateMatch]
    face_id: int
    frame_id: int
    is_recognized: bool = False
    tolerance: float = 0.48

    @property
    def candidate_name(self) -> str:
        if self.best_match is not None:
            return self.best_match.name
        return "Unknown"

    @property
    def candidate_id(self) -> str:
        if self.best_match is not None:
            return self.best_match.student_id
        return "None"

    @property
    def recognized_identity(self) -> Optional[CandidateMatch]:
        """Returns the recognized enrolled student candidate if acceptance passes, else None."""
        if self.is_recognized and self.best_match is not None:
            return self.best_match
        return None

    @property
    def recognized_name(self) -> str:
        """Enrolled-only recognized name, or 'UNKNOWN' if below acceptance criteria."""
        if self.is_recognized and self.best_match is not None:
            return self.best_match.name
        return "UNKNOWN"

    @property
    def recognized_id(self) -> str:
        """Enrolled-only recognized student_id, or 'UNKNOWN' if below acceptance criteria."""
        if self.is_recognized and self.best_match is not None:
            return self.best_match.student_id
        return "UNKNOWN"

    @property
    def best_similarity(self) -> float:
        if self.best_match is not None:
            return self.best_match.similarity
        return 0.0

    @property
    def second_similarity(self) -> float:
        if self.second_best_match is not None:
            return self.second_best_match.similarity
        return 0.0


class FaceMatcher:
    """
    Evaluates live 512-D ArcFace embeddings against enrolled multi-embedding references.
    Decoupled from face_id: face_id is used only for temporary multi-frame stability tracking,
    while student identity is derived purely from embedding comparison.
    """

    def __init__(
        self,
        enrollment_store: FaceEnrollmentStore,
        stability_window: int = 5,
        min_consecutive_stable: int = 4,
        tolerance: Optional[float] = None,
    ):
        self.store = enrollment_store
        self.stability_window = stability_window
        self.min_consecutive_stable = min_consecutive_stable
        self.tolerance = float(tolerance if tolerance is not None else settings.tolerance)

        # Tracking history for multi-frame stability:
        # face_id -> deque of recent candidate student_ids (maxlen=stability_window)
        self._track_history: Dict[int, deque[str]] = {}
        self._track_last_seen: Dict[int, float] = {}

    def match(
        self,
        live_embedding: np.ndarray,
        face_id: int = -1,
        frame_id: int = 0,
    ) -> MatchResult:
        """
        Compare live embedding against all enrolled students.
        For each student, compares against ALL stored reference embeddings and uses the best score.

        Args:
            live_embedding: (512,) normalized float32 embedding.
            face_id: Temporary detection track ID from SCRFD.
            frame_id: Current frame counter.

        Returns:
            MatchResult containing best match, second-best, margin, and multi-frame stability.
        """
        now = time.monotonic()
        self._prune_stale_tracks(now)

        if live_embedding is None or live_embedding.size == 0:
            return MatchResult(
                best_match=None,
                second_best_match=None,
                margin=0.0,
                is_stable=False,
                stability_label="No embedding",
                all_candidates=[],
                face_id=face_id,
                frame_id=frame_id,
            )

        try:
            val_live = validate_embedding(live_embedding)
        except Exception:
            # Fallback: normalize if valid shape
            val_live = live_embedding.flatten().astype(np.float32)
            norm = float(np.linalg.norm(val_live))
            if norm > 1e-10:
                val_live = val_live / norm

        students = self.store.get_all_students()
        if not students:
            return MatchResult(
                best_match=None,
                second_best_match=None,
                margin=0.0,
                is_stable=False,
                stability_label="No enrolled students",
                all_candidates=[],
                face_id=face_id,
                frame_id=frame_id,
            )

        candidates: List[CandidateMatch] = []

        for st in students:
            embs = st.get("embeddings", [])
            if not embs:
                continue

            # Stack student embeddings: shape (K, 512)
            ref_matrix = np.vstack(embs).astype(np.float32)
            # Dot product against normalized live vector yields cosine similarities: shape (K,)
            sims = np.dot(ref_matrix, val_live)

            best_idx = int(np.argmax(sims))
            best_sim = float(np.clip(sims[best_idx], -1.0, 1.0))

            candidates.append(
                CandidateMatch(
                    student_id=st["student_id"],
                    name=st["name"],
                    similarity=best_sim,
                    best_sample_index=best_idx,
                )
            )

        if not candidates:
            return MatchResult(
                best_match=None,
                second_best_match=None,
                margin=0.0,
                is_stable=False,
                stability_label="No candidates",
                all_candidates=[],
                face_id=face_id,
                frame_id=frame_id,
            )

        # Sort descending by similarity
        candidates.sort(key=lambda c: c.similarity, reverse=True)

        best_match = candidates[0]
        second_best_match = candidates[1] if len(candidates) > 1 else None

        if second_best_match is not None:
            margin = float(best_match.similarity - second_best_match.similarity)
        else:
            margin = float(best_match.similarity)

        # Enrolled-only recognition acceptance gate
        is_recognized = bool(best_match.similarity >= self.tolerance)

        # Multi-frame stability tracking per temporary face_id
        # Note: An identity is only stable if it passes recognition criteria AND is consistently tracked
        is_stable = False
        stability_label = "Candidate" if is_recognized else f"Unknown ({best_match.similarity:.2f} < {self.tolerance:.2f})"

        if face_id >= 0:
            if face_id not in self._track_history:
                self._track_history[face_id] = deque(maxlen=self.stability_window)

            history = self._track_history[face_id]
            tracked_id = best_match.student_id if is_recognized else "__UNKNOWN__"
            history.append(tracked_id)
            self._track_last_seen[face_id] = now

            if is_recognized:
                consecutive_matches = sum(1 for sid in history if sid == best_match.student_id)
                if len(history) >= self.min_consecutive_stable and consecutive_matches >= self.min_consecutive_stable:
                    is_stable = True
                    stability_label = f"Candidate stable ({consecutive_matches}/{len(history)})"
                else:
                    stability_label = f"Candidate ({consecutive_matches}/{len(history)})"
            else:
                is_stable = False
                stability_label = f"Unknown ({len(history)} frames)"

        return MatchResult(
            best_match=best_match,
            second_best_match=second_best_match,
            margin=margin,
            is_stable=is_stable,
            stability_label=stability_label,
            all_candidates=candidates,
            face_id=face_id,
            frame_id=frame_id,
            is_recognized=is_recognized,
            tolerance=self.tolerance,
        )

    def _prune_stale_tracks(self, now: float, ttl: float = 3.0) -> None:
        """Clean up stability history for face_ids that haven't been seen recently."""
        stale_ids = [fid for fid, last in self._track_last_seen.items() if (now - last) > ttl]
        for fid in stale_ids:
            self._track_history.pop(fid, None)
            self._track_last_seen.pop(fid, None)

    def reset_history(self) -> None:
        """Clear all active face tracking and stability histories."""
        self._track_history.clear()
        self._track_last_seen.clear()

    def reload(self, path: Optional[str] = None) -> int:
        """Reload student enrollments from store and flush history."""
        self.store.load(path)
        self.reset_history()
        return len(self.store.get_all_students())

    @property
    def enrolled_students(self) -> List[str]:
        """List of currently enrolled student IDs."""
        return [st["student_id"] for st in self.store.get_all_students()]
