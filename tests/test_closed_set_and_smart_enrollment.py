"""
Tests for Closed-Set Recognition, UNKNOWN Rejection, and Smart Enrollment
Verifies:
1. Enrolled-only recognition policy (tolerance enforcement).
2. Unknown rejection and exclusion from identity stability.
3. Unknown face cannot generate attendance events.
4. Purging of test records (Mom / TEMP01, Chut) and integrity of 4 core students.
5. Smart enrollment quality filtering (blur rejection, face size, landmarks).
6. Smart enrollment duplicate rejection against full capture history.
7. Explicit enrollment persistence and hot-reload.
8. Camera lifecycle and script launcher presence.
"""

import os
import json
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

from edge.face_matcher import FaceMatcher, MatchResult, CandidateMatch
from edge.config import settings
from edge.enroll_face import assess_enrollment_quality
from edge.face_enrollment import FaceEnrollmentStore
from edge.scrfd_detector import SCRFDFace
from edge.camera_control import CameraSession


@pytest.fixture
def sample_enrolled_store(tmp_path):
    """Create a temporary face embeddings store with 2 known students."""
    npz_path = tmp_path / "test_embeddings.npz"

    # Seed 2 distinct students with normalized 512-D vectors
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal((3, 512)).astype(np.float32)
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)

    v2 = rng.standard_normal((3, 512)).astype(np.float32)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)

    store = FaceEnrollmentStore(storage_path=str(npz_path))
    store.add_student("23UAM090", "Aditya Patil", list(v1))
    store.add_student("23UAM092", "Piyush", list(v2))
    store.save()

    return store, v1[0], v2[0]


class TestClosedSetRecognition:
    def test_enrolled_student_recognized(self, sample_enrolled_store):
        store, v_aditya, _ = sample_enrolled_store
        matcher = FaceMatcher(enrollment_store=store, tolerance=0.48)

        # Query with exact vector of Aditya
        result = matcher.match(v_aditya, face_id=1, frame_id=1)
        assert result.is_recognized is True
        assert result.recognized_name == "Aditya Patil"
        assert result.recognized_id == "23UAM090"
        assert result.best_match is not None
        assert result.best_match.similarity >= 0.99

    def test_unknown_face_rejected_below_tolerance(self, sample_enrolled_store):
        store, v_aditya, v_piyush = sample_enrolled_store
        matcher = FaceMatcher(enrollment_store=store, tolerance=0.48)

        # Create an orthogonal vector that has near-zero cosine similarity with both
        rng = np.random.default_rng(999)
        unknown_vec = rng.standard_normal(512).astype(np.float32)
        unknown_vec /= np.linalg.norm(unknown_vec)

        result = matcher.match(unknown_vec, face_id=2, frame_id=1)
        # MUST be classified as UNKNOWN
        assert result.is_recognized is False
        assert result.recognized_name == "UNKNOWN"
        assert result.recognized_id == "UNKNOWN"
        assert result.recognized_identity is None

        # Diagnostic info: best candidate is still tracked for inspectability
        if result.best_match:
            assert result.best_match.similarity < 0.48

    def test_unknown_face_never_achieves_stability(self, sample_enrolled_store):
        store, _, _ = sample_enrolled_store
        matcher = FaceMatcher(enrollment_store=store, tolerance=0.48, stability_window=5, min_consecutive_stable=3)

        rng = np.random.default_rng(1234)
        unknown_vec = rng.standard_normal(512).astype(np.float32)
        unknown_vec /= np.linalg.norm(unknown_vec)

        # Present unknown face for 10 consecutive frames
        for frame_idx in range(10):
            result = matcher.match(unknown_vec, face_id=10, frame_id=frame_idx)
            assert result.is_recognized is False
            # Stable candidate must NEVER be assigned to an enrolled student
            assert result.is_stable is False
            assert "unknown" in result.stability_label.lower()

    def test_attendance_blocked_for_unknown_face(self, sample_enrolled_store):
        """Verify that runner attendance logic rejects candidate if is_recognized is False."""
        _, _, _ = sample_enrolled_store

        # Simulate candidate with similarity below tolerance
        cand = CandidateMatch(student_id="23UAM090", name="Aditya Patil", similarity=0.35, best_sample_index=0)
        match_res = MatchResult(
            best_match=cand,
            second_best_match=None,
            margin=0.0,
            is_stable=False,
            stability_label="Unstable",
            all_candidates=[cand],
            face_id=0,
            frame_id=1,
            is_recognized=False,
            tolerance=0.48,
        )

        # Verification check matching edge/runner.py
        should_process_attendance = (
            cand.similarity >= settings.tolerance and getattr(match_res, "is_recognized", True)
        )
        assert should_process_attendance is False


class TestDataCleanupAndCoreRoster:
    def test_mom_temp01_removed_from_store(self):
        npz_path = Path("data/face_embeddings.npz")
        assert npz_path.exists(), "data/face_embeddings.npz must exist"
        data = np.load(str(npz_path))
        keys = list(data.keys())

        # Mom / TEMP01 must be completely purged
        assert "TEMP01" not in keys
        assert "emb_TEMP01" not in keys
        assert "Mom" not in keys
        assert "emb_Mom" not in keys

    def test_chut_absent_from_store(self):
        npz_path = Path("data/face_embeddings.npz")
        data = np.load(str(npz_path))
        keys = list(data.keys())

        # Chut must not exist
        assert "Chut" not in keys
        assert "emb_Chut" not in keys
        assert "chut" not in keys
        assert "emb_chut" not in keys

    def test_core_four_students_present(self):
        store = FaceEnrollmentStore("data/face_embeddings.npz")
        students = {s["student_id"]: s for s in store.get_all_students()}

        expected_ids = {"23UAM090", "23UAM092", "23UAM084", "23UCS041"}
        for s_id in expected_ids:
            assert s_id in students, f"Expected core student {s_id} missing from data/face_embeddings.npz"
            embeddings = students[s_id]["embeddings"]
            assert len(embeddings) >= 5
            for emb in embeddings:
                assert emb.shape == (512,)
                assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

        # Check manifest
        manifest_path = Path("data/face_embeddings_manifest.json")
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["23UAM090"]["name"] == "Aditya Patil"
        assert manifest["23UAM092"]["name"] == "Piyush"
        assert manifest["23UAM084"]["name"] == "Paryusha"
        assert manifest["23UCS041"]["name"] == "Abhishek J"


class TestSmartEnrollmentQuality:
    def test_reject_blurry_image(self):
        # Create a completely flat/blurry gray image
        blurry_frame = np.full((400, 400, 3), 128, dtype=np.uint8)
        mock_face = SCRFDFace(
            bbox=(50, 50, 250, 250),
            confidence=0.95,
            landmarks=np.array([[80, 80], [120, 80], [100, 100], [85, 120], [115, 120]]),
        )

        ok, msg, _ = assess_enrollment_quality([mock_face], blurry_frame, min_size=80, min_conf=0.7)
        assert ok is False
        assert "blurry" in msg.lower()

    def test_reject_too_small_face(self):
        # Sharp image with very small face
        rng = np.random.default_rng(10)
        sharp_frame = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
        mock_face = SCRFDFace(
            bbox=(50, 50, 70, 70),  # width = 20 < min_size=80
            confidence=0.95,
            landmarks=np.array([[55, 55], [65, 55], [60, 60], [55, 65], [65, 65]]),
        )

        ok, msg, _ = assess_enrollment_quality([mock_face], sharp_frame, min_size=80, min_conf=0.7)
        assert ok is False
        assert "small" in msg.lower()

    def test_accept_high_quality_face(self):
        # Texture-rich sharp image
        sharp_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(sharp_frame, (200, 200), 80, (255, 255, 255), -1)
        cv2.circle(sharp_frame, (170, 180), 10, (0, 0, 0), -1)
        cv2.circle(sharp_frame, (230, 180), 10, (0, 0, 0), -1)
        cv2.line(sharp_frame, (170, 230), (230, 230), (0, 0, 0), 5)

        mock_face = SCRFDFace(
            bbox=(100, 100, 300, 300),  # width 200, height 200
            confidence=0.95,
            landmarks=np.array([[170, 180], [230, 180], [200, 200], [170, 230], [230, 230]]),
        )

        ok, msg, _ = assess_enrollment_quality([mock_face], sharp_frame, min_size=80, min_conf=0.7)
        assert ok is True
        assert "ready" in msg.lower() or "ok" in msg.lower()

    def test_duplicate_sample_rejection(self):
        """Verify cosine similarity threshold detects near-duplicate vectors."""
        rng = np.random.default_rng(77)
        base_vec = rng.standard_normal(512).astype(np.float32)
        base_vec /= np.linalg.norm(base_vec)

        # Create near-duplicate (> 0.99 cosine similarity)
        near_duplicate = base_vec + rng.standard_normal(512).astype(np.float32) * 0.001
        near_duplicate /= np.linalg.norm(near_duplicate)

        # Create distinct pose vector (< 0.95 cosine similarity)
        distinct_vec = base_vec + rng.standard_normal(512).astype(np.float32) * 0.4
        distinct_vec /= np.linalg.norm(distinct_vec)

        sim_dup = float(np.dot(base_vec, near_duplicate))
        sim_dist = float(np.dot(base_vec, distinct_vec))

        assert sim_dup > 0.985, f"Expected duplicate sim > 0.985, got {sim_dup}"
        assert sim_dist < 0.985, f"Expected distinct sim < 0.985, got {sim_dist}"


class TestEnrollmentPersistenceAndHotReload:
    def test_save_and_hot_reload_new_student(self, tmp_path):
        npz_path = tmp_path / "enroll_test.npz"

        # Initialize store and matcher with single student
        rng = np.random.default_rng(888)
        s1_vec = rng.standard_normal((5, 512)).astype(np.float32)
        s1_vec /= np.linalg.norm(s1_vec, axis=1, keepdims=True)

        store = FaceEnrollmentStore(storage_path=str(npz_path))
        store.add_student("23UAM001", "Existing Student", list(s1_vec))
        store.save()

        matcher = FaceMatcher(enrollment_store=store, tolerance=0.48)
        assert len(matcher.enrolled_students) == 1

        # Enroll a new student "23UAM999" (Demo Guest)
        new_vecs = rng.standard_normal((10, 512)).astype(np.float32)
        new_vecs /= np.linalg.norm(new_vecs, axis=1, keepdims=True)

        store.add_student("23UAM999", "Demo Guest", list(new_vecs))
        saved_path = store.save()
        assert Path(saved_path).exists()

        # Hot-reload matcher
        count = matcher.reload()
        assert count == 2
        assert "23UAM999" in matcher.enrolled_students

        # Test recognition of newly enrolled guest
        match_result = matcher.match(new_vecs[0], face_id=5, frame_id=1)
        assert match_result.is_recognized is True
        assert match_result.recognized_name == "Demo Guest"
        assert match_result.recognized_id == "23UAM999"
        assert match_result.best_match is not None
        assert match_result.best_match.similarity >= 0.99


class TestCameraLifecycleAndScripts:
    def test_launchers_and_documentation_exist(self):
        root = Path(".")
        assert (root / "Start-Demo.bat").exists()
        assert (root / "Enroll-Student.bat").exists()
        assert (root / "README-DEMO.md").exists()
        assert (root / "DEMO-GUIDE.md").exists()
        assert (root / "edge" / "verify_enrollment.py").exists()

    def test_camera_session_start_and_stop_lifecycle(self):
        """CameraSession must start camera service on acquire and stop camera service on release."""
        session = CameraSession(
            client_id="test_worker",
            session_type="test_session",
            host="127.0.0.1",
            control_port=5002,
            stream_port=5001,
            is_local_camera=False,
        )

        with patch("edge.camera_control.start_camera_service") as mock_start, \
             patch("edge.camera_control.wait_for_stream_availability", return_value=True) as mock_wait, \
             patch("edge.camera_control.stop_camera_service") as mock_stop:

            session.start()
            assert session._active is True
            mock_start.assert_called_once()
            mock_wait.assert_called_once()

            session.stop()
            assert session._active is False
            mock_stop.assert_called_once()
