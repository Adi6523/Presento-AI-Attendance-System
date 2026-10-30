"""
Presento Phase 5 & 7: Specialized Closed-Set Identity Classifier.
Trains on enrolled ArcFace 512-D embeddings and provides student-specific probabilities,
confidence margins, and unknown rejection gating.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import settings
from .face_enrollment import FaceEnrollmentStore, validate_embedding

logger = logging.getLogger("presento.edge.identity_classifier")


@dataclass
class ClassifierPrediction:
    """Diagnostic prediction from the specialized identity classifier."""
    student_id: str
    name: str
    confidence: float
    second_student_id: Optional[str]
    second_confidence: float
    margin: float
    prototype_similarity: float
    multiref_similarity: float
    probabilities: Dict[str, float]
    is_accepted: bool = False


class IdentityClassifier:
    """
    Closed-set identity classifier trained on top of frozen ArcFace 512-D embeddings.
    Combines:
    1. Regularized multinomial logistic regression on unit hypersphere.
    2. Student prototype (centroid) cosine distances.
    3. Multi-reference nearest-neighbor similarity.
    """

    def __init__(self, model_path: str = settings.identity_classifier_path):
        self.model_path = model_path
        self.classifier: Optional[LogisticRegression] = None
        self.classes: List[str] = []                     # student_ids
        self.student_names: Dict[str, str] = {}           # student_id -> name
        self.prototypes: Dict[str, np.ndarray] = {}       # student_id -> (512,)
        self.reference_matrices: Dict[str, np.ndarray] = {}  # student_id -> (K, 512)
        self.is_trained: bool = False
        self.trained_at: str = ""

        # Auto-load if model file exists
        if os.path.exists(self.model_path):
            try:
                self.load(self.model_path)
            except Exception as exc:
                logger.warning("Could not auto-load identity classifier from %s: %s", self.model_path, exc)

    def train(self, store: FaceEnrollmentStore) -> bool:
        """
        Train the classifier using all enrolled students in the FaceEnrollmentStore.
        Supports 1 student (fallback to prototype matching) and >= 2 students (multinomial logistic).

        Returns:
            True if training succeeded, False if store has no enrolled students.
        """
        students = store.get_all_students()
        if not students:
            logger.warning("Cannot train identity classifier: store has 0 enrolled students")
            self.is_trained = False
            return False

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        class_ids: List[str] = []
        names: Dict[str, str] = {}
        protos: Dict[str, np.ndarray] = {}
        ref_mats: Dict[str, np.ndarray] = {}

        for idx, st in enumerate(students):
            sid = st["student_id"]
            name = st["name"]
            embs = st["embeddings"]
            if not embs:
                continue

            class_ids.append(sid)
            names[sid] = name
            ref_mat = np.vstack(embs).astype(np.float32)
            ref_mats[sid] = ref_mat

            # Compute prototype
            mean_vec = np.mean(ref_mat, axis=0)
            norm = float(np.linalg.norm(mean_vec))
            protos[sid] = (mean_vec / max(norm, 1e-10)).astype(np.float32)

            for e in embs:
                X_list.append(e)
                y_list.append(idx)

        if not class_ids or not X_list:
            self.is_trained = False
            return False

        self.classes = class_ids
        self.student_names = names
        self.prototypes = protos
        self.reference_matrices = ref_mats

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)

        if len(self.classes) >= 2:
            # Fit regularized logistic regression with L2 penalty on normalized vectors
            # Since vectors are on unit sphere, dot product is cosine similarity
            clf = LogisticRegression(
                C=1.0,
                max_iter=300,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X, y)
            self.classifier = clf
        else:
            # Single student enrolled - linear classifier cannot form a boundary; use centroid prototype
            self.classifier = None

        self.is_trained = True
        logger.info(
            "Trained Presento Identity Classifier: %d students, %d total reference samples",
            len(self.classes),
            len(X_list),
        )
        return True

    def predict(
        self,
        embedding: np.ndarray,
        min_conf: float = 0.45,
        min_margin: float = settings.identity_min_margin,
        min_similarity: float = settings.tolerance,
    ) -> ClassifierPrediction:
        """
        Evaluate live embedding against closed-set student population.
        Calculates:
        - Class probabilities
        - Best and second-best candidate
        - Margin
        - Prototype similarity
        - Multi-ref similarity
        - Acceptance criteria
        """
        if not self.is_trained or not self.classes:
            return ClassifierPrediction(
                student_id="UNKNOWN",
                name="UNKNOWN",
                confidence=0.0,
                second_student_id=None,
                second_confidence=0.0,
                margin=0.0,
                prototype_similarity=0.0,
                multiref_similarity=0.0,
                probabilities={},
                is_accepted=False,
            )

        val_emb = validate_embedding(embedding)

        # 1. Compute prototype cosine similarities and multi-ref similarities
        proto_sims: Dict[str, float] = {}
        multiref_sims: Dict[str, float] = {}
        for sid in self.classes:
            proto = self.prototypes[sid]
            proto_sims[sid] = float(np.dot(val_emb, proto))

            ref_mat = self.reference_matrices[sid]
            dots = np.dot(ref_mat, val_emb)
            multiref_sims[sid] = float(np.max(dots))

        # 2. Compute probabilities
        if self.classifier is not None and len(self.classes) >= 2:
            probas = self.classifier.predict_proba(val_emb.reshape(1, -1))[0]
            prob_dict = {sid: float(probas[i]) for i, sid in enumerate(self.classes)}
        else:
            # Single class or prototype-based: softmax of prototype similarities scaled by temperature
            tau = 0.10
            raw_scores = np.array([proto_sims[sid] for sid in self.classes])
            exp_scores = np.exp((raw_scores - np.max(raw_scores)) / tau)
            softmax_scores = exp_scores / np.sum(exp_scores)
            prob_dict = {sid: float(softmax_scores[i]) for i, sid in enumerate(self.classes)}

        # 3. Rank candidates by probability
        ranked_sids = sorted(self.classes, key=lambda sid: prob_dict[sid], reverse=True)
        best_sid = ranked_sids[0]
        best_prob = prob_dict[best_sid]
        best_name = self.student_names.get(best_sid, "UNKNOWN")

        second_sid: Optional[str] = None
        second_prob = 0.0
        if len(ranked_sids) > 1:
            second_sid = ranked_sids[1]
            second_prob = prob_dict[second_sid]
            margin = float(best_prob - second_prob)
        else:
            margin = float(best_prob)

        best_proto_sim = proto_sims[best_sid]
        best_multiref_sim = multiref_sims[best_sid]

        # 4. Unknown / Impostor Acceptance Gate
        # To be accepted as an enrolled student:
        # A) Multi-ref cosine similarity >= min_similarity
        # B) Margin >= min_margin
        # C) Classifier confidence >= min_conf
        is_accepted = bool(
            best_multiref_sim >= min_similarity
            and margin >= min_margin
            and best_prob >= min_conf
        )

        return ClassifierPrediction(
            student_id=best_sid if is_accepted else "UNKNOWN",
            name=best_name if is_accepted else "UNKNOWN",
            confidence=best_prob,
            second_student_id=second_sid,
            second_confidence=second_prob,
            margin=margin,
            prototype_similarity=best_proto_sim,
            multiref_similarity=best_multiref_sim,
            probabilities=prob_dict,
            is_accepted=is_accepted,
        )

    def save(self, path: Optional[str] = None) -> str:
        """Save classifier state and prototype matrices to .npz file."""
        target_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        for sid, proto in self.prototypes.items():
            arrays[f"proto_{sid}"] = proto
        for sid, ref in self.reference_matrices.items():
            arrays[f"refs_{sid}"] = ref

        meta = {
            "classes": self.classes,
            "student_names": self.student_names,
            "is_trained": self.is_trained,
        }
        if self.classifier is not None:
            arrays["clf_coef"] = self.classifier.coef_
            arrays["clf_intercept"] = self.classifier.intercept_
            arrays["clf_classes"] = self.classifier.classes_
            meta["has_linear_clf"] = True
        else:
            meta["has_linear_clf"] = False

        arrays["__meta__"] = np.array([json.dumps(meta)])
        np.savez_compressed(target_path, **arrays)
        logger.info("Saved IdentityClassifier model to %s", target_path)
        return target_path

    def load(self, path: Optional[str] = None) -> None:
        """Load classifier state from .npz file."""
        source_path = path or self.model_path
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Model file not found at: {source_path}")

        loaded = np.load(source_path, allow_pickle=False)
        if "__meta__" not in loaded:
            raise ValueError(f"Invalid model archive: missing '__meta__' in {source_path}")

        meta = json.loads(str(loaded["__meta__"][0]))
        self.classes = meta["classes"]
        self.student_names = meta["student_names"]
        self.is_trained = meta.get("is_trained", True)

        self.prototypes = {}
        self.reference_matrices = {}
        for sid in self.classes:
            p_key = f"proto_{sid}"
            r_key = f"refs_{sid}"
            if p_key in loaded:
                self.prototypes[sid] = loaded[p_key].astype(np.float32)
            if r_key in loaded:
                self.reference_matrices[sid] = loaded[r_key].astype(np.float32)

        if meta.get("has_linear_clf") and "clf_coef" in loaded:
            clf = LogisticRegression()
            clf.coef_ = loaded["clf_coef"]
            clf.intercept_ = loaded["clf_intercept"]
            clf.classes_ = loaded["clf_classes"]
            self.classifier = clf
        else:
            self.classifier = None

        logger.info("Loaded IdentityClassifier from %s (%d classes)", source_path, len(self.classes))
