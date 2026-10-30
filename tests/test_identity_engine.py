"""
Tests for Presento Phase 5, 6 & 7: PresentoIdentityEngine & Decision Fusion.
Verifies baseline mode, fused mode, classifier mode, margin gating, unknown rejection,
and multi-frame identity stability tracking.
"""
import numpy as np
import pytest

from edge.face_enrollment import FaceEnrollmentStore
from edge.identity_engine import PresentoIdentityEngine


@pytest.fixture
def identity_store(tmp_path):
    archive_path = tmp_path / "engine_store.npz"
    store = FaceEnrollmentStore(storage_path=str(archive_path))

    rng = np.random.default_rng(2026)
    v1 = rng.standard_normal((5, 512)).astype(np.float32)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)

    v2 = rng.standard_normal((5, 512)).astype(np.float32)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)

    store.add_student("23UAM090", "Aditya Patil", list(v1))
    store.add_student("23UAM092", "Piyush", list(v2))
    store.save()

    return store, v1[0], v2[0]


class TestPresentoIdentityEngine:
    def test_baseline_mode_matches_face_matcher(self, identity_store):
        store, v_aditya, _ = identity_store
        engine = PresentoIdentityEngine(enrollment_store=store, mode="baseline", tolerance=0.48)

        res = engine.match(v_aditya, face_id=1, frame_id=1)
        assert res.is_recognized is True
        assert res.recognized_name == "Aditya Patil"
        assert res.recognized_id == "23UAM090"
        assert res.best_match.similarity >= 0.99

    def test_fused_mode_scoring_and_recognition(self, identity_store):
        store, v_aditya, _ = identity_store
        engine = PresentoIdentityEngine(
            enrollment_store=store,
            mode="fused",
            tolerance=0.48,
            min_margin=0.10,
            weight_multiref=0.50,
            weight_prototype=0.25,
            weight_classifier=0.25,
        )

        res = engine.match(v_aditya, face_id=2, frame_id=1)
        assert res.is_recognized is True
        assert res.recognized_name == "Aditya Patil"
        assert res.recognized_id == "23UAM090"
        assert res.margin >= 0.10

    def test_classifier_mode_recognition(self, identity_store):
        store, v_aditya, _ = identity_store
        engine = PresentoIdentityEngine(
            enrollment_store=store,
            mode="classifier",
            tolerance=0.48,
        )

        res = engine.match(v_aditya, face_id=3, frame_id=1)
        assert res.is_recognized is True
        assert res.recognized_name == "Aditya Patil"

    def test_unknown_probe_rejected_in_fused_mode(self, identity_store):
        store, _, _ = identity_store
        engine = PresentoIdentityEngine(
            enrollment_store=store,
            mode="fused",
            tolerance=0.48,
            min_margin=0.10,
        )

        rng = np.random.default_rng(8888)
        unknown_probe = rng.standard_normal(512).astype(np.float32)
        unknown_probe /= np.linalg.norm(unknown_probe)

        res = engine.match(unknown_probe, face_id=4, frame_id=1)
        assert res.is_recognized is False
        assert res.recognized_name == "UNKNOWN"
        assert res.recognized_id == "UNKNOWN"

    def test_identity_stability_tracking(self, identity_store):
        store, v_aditya, _ = identity_store
        engine = PresentoIdentityEngine(
            enrollment_store=store,
            mode="fused",
            tolerance=0.48,
            stability_window=5,
            min_consecutive_stable=3,
        )

        # Frame 1: Candidate, not yet stable
        r1 = engine.match(v_aditya, face_id=10, frame_id=1)
        assert r1.is_recognized is True
        assert r1.is_stable is False

        # Frame 2: Still not reaching 3 consecutive
        r2 = engine.match(v_aditya, face_id=10, frame_id=2)
        assert r2.is_stable is False

        # Frame 3: Reaches 3 consecutive -> Stable!
        r3 = engine.match(v_aditya, face_id=10, frame_id=3)
        assert r3.is_stable is True
        assert "stable" in r3.stability_label.lower()

    def test_unknown_never_achieves_stability(self, identity_store):
        store, _, _ = identity_store
        engine = PresentoIdentityEngine(
            enrollment_store=store,
            mode="fused",
            tolerance=0.48,
            stability_window=5,
            min_consecutive_stable=3,
        )

        rng = np.random.default_rng(7777)
        unknown_probe = rng.standard_normal(512).astype(np.float32)
        unknown_probe /= np.linalg.norm(unknown_probe)

        for f_idx in range(10):
            res = engine.match(unknown_probe, face_id=20, frame_id=f_idx)
            assert res.is_recognized is False
            assert res.is_stable is False
            assert "unknown" in res.stability_label.lower()
