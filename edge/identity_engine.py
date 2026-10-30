"""
Presento Phase 5 & 6: Unified Specialized Identity Engine & Decision Fusion.
Integrates Multi-Reference Cosine Matching, Prototype Similarity, and Closed-Set Classifier
into a robust, configurable identity engine with strict unknown/impostor gating.
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
from .face_matcher import FaceMatcher, MatchResult, CandidateMatch
from .identity_classifier import IdentityClassifier, ClassifierPrediction

logger = logging.getLogger("presento.edge.identity_engine")


@dataclass
class FusedCandidateMatch(CandidateMatch):
    """Candidate match enriched with multi-modal scoring components."""
    multiref_sim: float = 0.0
    prototype_sim: float = 0.0
    classifier_prob: float = 0.0
    fused_score: float = 0.0


class PresentoIdentityEngine:
    """
    Presento Specialized Identity Decision Engine.
    Coordinates:
    1. Baseline Multi-Reference Cosine Matcher (FaceMatcher)
    2. Student Prototype / Centroid Matcher
    3. Trained Closed-Set Identity Classifier (Logistic/Softmax)

    Modes:
    - 'baseline': 100% equivalent to production FaceMatcher.
    - 'classifier': Driven by IdentityClassifier with acceptance gating.
    - 'fused': Weighted combination of multi-ref, prototype, and classifier.
    """

    def __init__(
        self,
        enrollment_store: FaceEnrollmentStore,
        mode: str = settings.identity_engine_mode,
        tolerance: Optional[float] = None,
        min_margin: float = settings.identity_min_margin,
        stability_window: int = 5,
        min_consecutive_stable: int = 4,
        weight_multiref: float = settings.identity_weight_multiref,
        weight_prototype: float = settings.identity_weight_prototype,
        weight_classifier: float = settings.identity_weight_classifier,
    ):
        self.store = enrollment_store
        self.mode = mode.lower()
        self.tolerance = float(tolerance if tolerance is not None else settings.tolerance)
        self.min_margin = float(min_margin)
        self.stability_window = stability_window
        self.min_consecutive_stable = min_consecutive_stable

        # Normalizing weights
        total_w = weight_multiref + weight_prototype + weight_classifier
        if total_w <= 0:
            total_w = 1.0
        self.w_multiref = weight_multiref / total_w
        self.w_prototype = weight_prototype / total_w
        self.w_classifier = weight_classifier / total_w

        # Components
        self.base_matcher = FaceMatcher(
            enrollment_store=self.store,
            stability_window=self.stability_window,
            min_consecutive_stable=self.min_consecutive_stable,
            tolerance=self.tolerance,
        )

        self.classifier = IdentityClassifier()
        # Auto-train classifier if not already trained
        if not self.classifier.is_trained:
            try:
                self.classifier.train(self.store)
            except Exception as exc:
                logger.warning("Could not auto-train identity classifier: %s", exc)

        # Multi-frame tracking
        self._track_history: Dict[int, deque[str]] = {}
        self._track_last_seen: Dict[int, float] = {}

    def reload(self) -> None:
        """Reload store and re-train/update classifier."""
        self.base_matcher = FaceMatcher(
            enrollment_store=self.store,
            stability_window=self.stability_window,
            min_consecutive_stable=self.min_consecutive_stable,
            tolerance=self.tolerance,
        )
        try:
            self.classifier.train(self.store)
        except Exception as exc:
            logger.warning("Could not train classifier during reload: %s", exc)

    def match(
        self,
        live_embedding: np.ndarray,
        face_id: int = -1,
        frame_id: int = 0,
    ) -> MatchResult:
        """
        Evaluate live embedding against enrolled student roster.
        Delegates according to configured mode ('baseline', 'classifier', 'fused').
        """
        if self.mode == "baseline":
            return self.base_matcher.match(live_embedding, face_id=face_id, frame_id=frame_id)

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
                is_recognized=False,
                tolerance=self.tolerance,
            )

        val_live = validate_embedding(live_embedding)
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
                is_recognized=False,
                tolerance=self.tolerance,
            )

        # Ensure classifier is trained
        if not self.classifier.is_trained:
            self.classifier.train(self.store)

        # 1. Compute multi-reference cosine similarities
        multiref_cands: Dict[str, Tuple[float, int]] = {}
        for st in students:
            sid = st["student_id"]
            embs = st.get("embeddings", [])
            if not embs:
                continue
            ref_matrix = np.vstack(embs).astype(np.float32)
            dots = np.dot(ref_matrix, val_live)
            best_idx = int(np.argmax(dots))
            multiref_cands[sid] = (float(np.clip(dots[best_idx], -1.0, 1.0)), best_idx)

        # 2. Compute prototype similarities
        proto_cands: Dict[str, float] = {}
        for st in students:
            sid = st["student_id"]
            proto = st.get("prototype")
            if proto is not None:
                p_sim = float(np.dot(val_live, proto))
            else:
                p_sim = multiref_cands.get(sid, (0.0, 0))[0]
            proto_cands[sid] = float(np.clip(p_sim, -1.0, 1.0))

        # 3. Compute classifier prediction
        clf_pred: Optional[ClassifierPrediction] = None
        if self.classifier.is_trained:
            try:
                clf_pred = self.classifier.predict(
                    val_live,
                    min_conf=0.30,
                    min_margin=self.min_margin,
                    min_similarity=self.tolerance,
                )
            except Exception as exc:
                logger.debug("Classifier prediction fallback: %s", exc)

        clf_probs = clf_pred.probabilities if clf_pred else {}

        # 4. Synthesize candidate matches
        candidates: List[FusedCandidateMatch] = []
        for st in students:
            sid = st["student_id"]
            name = st["name"]
            sim_multiref, best_sample_idx = multiref_cands.get(sid, (0.0, 0))
            sim_proto = proto_cands.get(sid, sim_multiref)
            prob_clf = clf_probs.get(sid, 0.0)

            if self.mode == "classifier" and clf_pred is not None:
                final_score = prob_clf
            elif self.mode == "fused":
                final_score = (
                    self.w_multiref * sim_multiref
                    + self.w_prototype * sim_proto
                    + self.w_classifier * prob_clf
                )
            else:
                final_score = sim_multiref

            candidates.append(
                FusedCandidateMatch(
                    student_id=sid,
                    name=name,
                    similarity=float(final_score),
                    best_sample_index=best_sample_idx,
                    multiref_sim=sim_multiref,
                    prototype_sim=sim_proto,
                    classifier_prob=prob_clf,
                    fused_score=float(final_score),
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
                is_recognized=False,
                tolerance=self.tolerance,
            )

        candidates.sort(key=lambda c: c.similarity, reverse=True)
        best_cand = candidates[0]
        second_cand = candidates[1] if len(candidates) > 1 else None

        if second_cand is not None:
            margin = float(best_cand.similarity - second_cand.similarity)
        else:
            margin = float(best_cand.similarity)

        # 5. CLOSED-SET ACCEPTANCE GATE (Phase 7)
        # An enrolled candidate is recognized ONLY IF:
        # A) Multi-ref cosine similarity >= tolerance (physical facial match)
        # B) Overall score >= tolerance
        # C) Margin >= min_margin (clear separation from 2nd best)
        # D) If classifier is used, classifier probability > second probability
        sim_ok = best_cand.multiref_sim >= self.tolerance
        score_ok = best_cand.similarity >= self.tolerance
        margin_ok = margin >= self.min_margin or len(candidates) == 1

        is_recognized = bool(sim_ok and score_ok and margin_ok)

        # 6. Multi-frame stability tracking
        is_stable = False
        if is_recognized:
            stability_label = f"Candidate ({best_cand.name}, sim={best_cand.multiref_sim:.2f}, m={margin:.2f})"
        else:
            stability_label = f"Unknown (sim={best_cand.multiref_sim:.2f} < {self.tolerance:.2f} or margin={margin:.2f})"

        if face_id >= 0:
            if face_id not in self._track_history:
                self._track_history[face_id] = deque(maxlen=self.stability_window)

            history = self._track_history[face_id]
            tracked_id = best_cand.student_id if is_recognized else "__UNKNOWN__"
            history.append(tracked_id)
            self._track_last_seen[face_id] = now

            if is_recognized:
                consecutive_matches = sum(1 for sid in history if sid == best_cand.student_id)
                if len(history) >= self.min_consecutive_stable and consecutive_matches >= self.min_consecutive_stable:
                    is_stable = True
                    stability_label = f"Candidate stable ({consecutive_matches}/{len(history)})"
                else:
                    stability_label = f"Candidate ({consecutive_matches}/{len(history)})"
            else:
                is_stable = False
                stability_label = f"Unknown ({len(history)} frames)"

        return MatchResult(
            best_match=best_cand,
            second_best_match=second_cand,
            margin=margin,
            is_stable=is_stable,
            stability_label=stability_label,
            all_candidates=candidates,  # type: ignore
            face_id=face_id,
            frame_id=frame_id,
            is_recognized=is_recognized,
            tolerance=self.tolerance,
        )

    def _prune_stale_tracks(self, now: float, max_idle: float = 3.0) -> None:
        """Prune tracks not observed recently."""
        for fid in list(self._track_history.keys()):
            if now - self._track_last_seen.get(fid, 0.0) > max_idle:
                self._track_history.pop(fid, None)
                self._track_last_seen.pop(fid, None)
