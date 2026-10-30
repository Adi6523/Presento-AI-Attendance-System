"""
Tests for Presento Phase 5 & 7: IdentityClassifier.
Verifies training, probabilities, margins, unknown rejection, and persistence.
"""
from pathlib import Path
import numpy as np
import pytest

from edge.face_enrollment import FaceEnrollmentStore
from edge.identity_classifier import IdentityClassifier, ClassifierPrediction


@pytest.fixture
def multi_student_store(tmp_path):
    """Create enrollment store with 3 distinct students."""
    npz_path = tmp_path / "test_store_classifier.npz"
    store = FaceEnrollmentStore(storage_path=str(npz_path))

    rng = np.random.default_rng(1000)

    # 3 distinct identity directions
    base1 = rng.standard_normal(512).astype(np.float32)
    base1 /= np.linalg.norm(base1)

    base2 = rng.standard_normal(512).astype(np.float32)
    base2 /= np.linalg.norm(base2)

    base3 = rng.standard_normal(512).astype(np.float32)
    base3 /= np.linalg.norm(base3)

    # 10 samples each with slight variation
    def make_samples(base, n=10):
        samples = []
        for _ in range(n):
            noise = rng.normal(0, 0.008, 512).astype(np.float32)
            v = base + noise
            v /= np.linalg.norm(v)
            samples.append(v)
        return samples

    store.add_student("23UAM090", "Aditya", make_samples(base1))
    store.add_student("23UAM092", "Piyush", make_samples(base2))
    store.add_student("23UAM084", "Paryusha", make_samples(base3))
    store.save()

    return store, base1, base2, base3


class TestIdentityClassifier:
    def test_train_and_predict_known_students(self, multi_student_store, tmp_path):
        store, base_aditya, base_piyush, _ = multi_student_store
        model_path = tmp_path / "identity_clf.npz"
        clf = IdentityClassifier(model_path=str(model_path))

        assert clf.train(store) is True
        assert clf.is_trained is True
        assert len(clf.classes) == 3

        # Query with sample of Aditya
        pred_aditya = clf.predict(base_aditya)
        assert pred_aditya.is_accepted is True
        assert pred_aditya.student_id == "23UAM090"
        assert pred_aditya.name == "Aditya"
        assert pred_aditya.confidence >= 0.70
        assert pred_aditya.margin >= 0.40
        assert pred_aditya.multiref_similarity >= 0.90

        # Query with sample of Piyush
        pred_piyush = clf.predict(base_piyush)
        assert pred_piyush.is_accepted is True
        assert pred_piyush.student_id == "23UAM092"
        assert pred_piyush.name == "Piyush"
        assert pred_piyush.confidence >= 0.70

    def test_unknown_probe_rejected_by_acceptance_gate(self, multi_student_store, tmp_path):
        store, _, _, _ = multi_student_store
        model_path = tmp_path / "identity_clf.npz"
        clf = IdentityClassifier(model_path=str(model_path))
        clf.train(store)

        # Orthogonal random vector represents an unknown person not enrolled in the college
        rng = np.random.default_rng(9999)
        unknown_vec = rng.standard_normal(512).astype(np.float32)
        unknown_vec /= np.linalg.norm(unknown_vec)

        pred_unknown = clf.predict(unknown_vec, min_similarity=0.48)
        # MUST BE REJECTED (CLASSIFIER PREDICTION != AUTOMATIC ACCEPTANCE)
        assert pred_unknown.is_accepted is False
        assert pred_unknown.student_id == "UNKNOWN"
        assert pred_unknown.name == "UNKNOWN"
        assert pred_unknown.multiref_similarity < 0.48

    def test_single_student_training_fallback(self, tmp_path):
        single_path = tmp_path / "single_store.npz"
        store = FaceEnrollmentStore(storage_path=str(single_path))

        rng = np.random.default_rng(42)
        vec = rng.standard_normal(512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store.add_student("SOLO01", "Solo Student", [vec])

        clf = IdentityClassifier(model_path=str(tmp_path / "solo_clf.npz"))
        assert clf.train(store) is True
        assert clf.is_trained is True

        pred = clf.predict(vec)
        assert pred.is_accepted is True
        assert pred.student_id == "SOLO01"

    def test_model_persistence_save_load(self, multi_student_store, tmp_path):
        store, base_aditya, _, _ = multi_student_store
        model_path = tmp_path / "saved_clf.npz"
        clf1 = IdentityClassifier(model_path=str(model_path))
        clf1.train(store)
        clf1.save()

        # Load in new instance
        clf2 = IdentityClassifier(model_path=str(model_path))
        assert clf2.is_trained is True
        assert clf2.classes == clf1.classes
        assert clf2.student_names == clf1.student_names

        pred2 = clf2.predict(base_aditya)
        assert pred2.is_accepted is True
        assert pred2.student_id == "23UAM090"
