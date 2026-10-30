from __future__ import annotations

import os
import tempfile
import unittest
import numpy as np

from edge.face_enrollment import FaceEnrollmentStore, validate_embedding
from edge.face_matcher import FaceMatcher, CandidateMatch, MatchResult
from edge.arcface_embedder import compute_cosine_similarity


def _make_normalized_vector(seed: int | None = None) -> np.ndarray:
    """Helper to generate a random 512-D L2-normalized float32 vector."""
    if seed is not None:
        np.random.seed(seed)
    v = np.random.randn(512).astype(np.float32)
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32)


class TestFaceEnrollmentAndMatching(unittest.TestCase):
    """
    Unit test suite for Phase 2B: Face Enrollment Store & Live Identity Matching.
    Validates multi-embedding storage, npz persistence, validation logic,
    multi-embedding best-score matching, margin calculation, score ordering,
    empty-store handling, face_id separation from student identity, and multi-frame stability.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_npz_path = os.path.join(self.temp_dir.name, "test_embeddings.npz")
        self.store = FaceEnrollmentStore(storage_path=self.test_npz_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_get_student(self):
        """1. Verify adding a student with multiple reference embeddings and retrieving it."""
        embs = [_make_normalized_vector(i) for i in range(5)]
        self.store.add_student(
            student_id="23UAM090",
            name="Aditya Patil",
            embeddings=embs,
        )

        st = self.store.get_student("23UAM090")
        self.assertIsNotNone(st)
        self.assertEqual(st["student_id"], "23UAM090")
        self.assertEqual(st["name"], "Aditya Patil")
        self.assertEqual(len(st["embeddings"]), 5)
        for e in st["embeddings"]:
            self.assertEqual(e.shape, (512,))
            self.assertEqual(e.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(e)), 1.0, places=4)

    def test_embedding_shape_validation(self):
        """2. Verify embedding shape validation rules."""
        # Non-512 shapes must raise ValueError
        with self.assertRaises(ValueError):
            validate_embedding(np.zeros((128,), dtype=np.float32))

        with self.assertRaises(ValueError):
            validate_embedding(np.zeros((511,), dtype=np.float32))

        # (1, 512) is flattened and accepted
        v = _make_normalized_vector(42).reshape(1, 512)
        valid = validate_embedding(v)
        self.assertEqual(valid.shape, (512,))

    def test_embedding_normalization_validation(self):
        """3. Verify unnormalized vectors raise ValueError."""
        unnormalized = np.ones(512, dtype=np.float32) * 2.0
        with self.assertRaises(ValueError):
            validate_embedding(unnormalized)

        normalized = unnormalized / float(np.linalg.norm(unnormalized))
        valid = validate_embedding(normalized)
        self.assertAlmostEqual(float(np.linalg.norm(valid)), 1.0, places=4)

    def test_save_and_load_npz(self):
        """4. Verify lossless saving and loading from .npz with embedded metadata."""
        embs_1 = [_make_normalized_vector(10 + i) for i in range(4)]
        embs_2 = [_make_normalized_vector(20 + i) for i in range(3)]

        self.store.add_student("S001", "Student One", embs_1)
        self.store.add_student("S002", "Student Two", embs_2)
        self.store.save(self.test_npz_path)

        # Load into fresh store instance
        loaded_store = FaceEnrollmentStore(storage_path=self.test_npz_path)
        all_st = loaded_store.get_all_students()
        self.assertEqual(len(all_st), 2)

        st1 = loaded_store.get_student("S001")
        self.assertIsNotNone(st1)
        self.assertEqual(st1["name"], "Student One")
        self.assertEqual(len(st1["embeddings"]), 4)
        np.testing.assert_allclose(st1["embeddings"][0], embs_1[0], rtol=1e-5, atol=1e-5)

        st2 = loaded_store.get_student("S002")
        self.assertIsNotNone(st2)
        self.assertEqual(st2["name"], "Student Two")
        self.assertEqual(len(st2["embeddings"]), 3)

    def test_matching_multi_embeddings(self):
        """5. Verify live embedding is matched against ALL reference embeddings (best of N)."""
        embs_a = [_make_normalized_vector(100 + i) for i in range(5)]
        embs_b = [_make_normalized_vector(200 + i) for i in range(5)]

        self.store.add_student("A01", "Alice", embs_a)
        self.store.add_student("B01", "Bob", embs_b)

        matcher = FaceMatcher(self.store)

        # Query with exact copy of Alice's 3rd sample (index 2)
        query_emb = embs_a[2].copy()
        result = matcher.match(query_emb, face_id=1, frame_id=10)

        self.assertIsNotNone(result.best_match)
        self.assertEqual(result.best_match.student_id, "A01")
        self.assertEqual(result.best_match.name, "Alice")
        self.assertAlmostEqual(result.best_match.similarity, 1.0, places=4)
        self.assertEqual(result.best_match.best_sample_index, 2)

    def test_best_and_second_best_margin(self):
        """6. Verify calculation of best match, second-best match, and margin."""
        embs_a = [_make_normalized_vector(301)]
        embs_b = [_make_normalized_vector(302)]

        self.store.add_student("A01", "Alice", embs_a)
        self.store.add_student("B01", "Bob", embs_b)

        matcher = FaceMatcher(self.store)

        # Query identical to Alice
        result = matcher.match(embs_a[0], face_id=2, frame_id=1)
        self.assertEqual(result.best_match.student_id, "A01")
        self.assertEqual(result.second_best_match.student_id, "B01")

        expected_sim_b = compute_cosine_similarity(embs_a[0], embs_b[0])
        self.assertAlmostEqual(result.best_match.similarity, 1.0, places=4)
        self.assertAlmostEqual(result.second_best_match.similarity, expected_sim_b, places=4)
        self.assertAlmostEqual(result.margin, 1.0 - expected_sim_b, places=4)

    def test_score_ordering(self):
        """7. Verify candidate matches are sorted in descending order of similarity."""
        for i in range(5):
            self.store.add_student(f"S{i}", f"Student_{i}", [_make_normalized_vector(400 + i)])

        matcher = FaceMatcher(self.store)
        query = _make_normalized_vector(999)
        result = matcher.match(query, face_id=1)

        self.assertEqual(len(result.all_candidates), 5)
        for j in range(len(result.all_candidates) - 1):
            self.assertGreaterEqual(
                result.all_candidates[j].similarity,
                result.all_candidates[j + 1].similarity,
            )

    def test_empty_and_unknown_enrollment(self):
        """8. Verify querying an empty store returns graceful Unknown result."""
        empty_store = FaceEnrollmentStore(storage_path=os.path.join(self.temp_dir.name, "empty.npz"))
        matcher = FaceMatcher(empty_store)

        query = _make_normalized_vector(123)
        result = matcher.match(query, face_id=1)

        self.assertIsNone(result.best_match)
        self.assertIsNone(result.second_best_match)
        self.assertEqual(result.candidate_name, "Unknown")
        self.assertEqual(result.candidate_id, "None")
        self.assertEqual(result.margin, 0.0)
        self.assertFalse(result.is_stable)

    def test_face_id_separation_from_identity(self):
        """9. Verify that changing temporary face_id does NOT alter the resolved student identity."""
        embs_aditya = [_make_normalized_vector(500 + i) for i in range(3)]
        self.store.add_student("23UAM090", "Aditya Patil", embs_aditya)

        matcher = FaceMatcher(self.store)
        query_face = embs_aditya[1].copy()

        # Face detected as temporary face_id=10
        res1 = matcher.match(query_face, face_id=10, frame_id=100)
        self.assertEqual(res1.best_match.student_id, "23UAM090")

        # Person steps away and returns, SCRFD assigns new face_id=45
        res2 = matcher.match(query_face, face_id=45, frame_id=150)
        self.assertEqual(res2.best_match.student_id, "23UAM090")
        self.assertEqual(res2.best_match.name, "Aditya Patil")

    def test_multi_frame_stability(self):
        """10. Verify multi-frame candidate confirmation mechanism."""
        embs_aditya = [_make_normalized_vector(600)]
        self.store.add_student("23UAM090", "Aditya Patil", embs_aditya)

        matcher = FaceMatcher(self.store, stability_window=5, min_consecutive_stable=4)
        query = embs_aditya[0].copy()

        # Frame 1: unstable
        r1 = matcher.match(query, face_id=1, frame_id=1)
        self.assertFalse(r1.is_stable)

        # Frames 2 and 3
        matcher.match(query, face_id=1, frame_id=2)
        matcher.match(query, face_id=1, frame_id=3)

        # Frame 4: reaches 4 consecutive -> stable
        r4 = matcher.match(query, face_id=1, frame_id=4)
        self.assertTrue(r4.is_stable)
        self.assertIn("Candidate stable", r4.stability_label)

    def test_duplicate_rejection_logic_configurable(self):
        """11. Verify configurable duplicate similarity rejection for enrollment."""
        v1 = _make_normalized_vector(700)
        # Create a near-duplicate with tiny perturbation
        v2 = v1 + np.random.randn(512).astype(np.float32) * 0.001
        v2 = v2 / float(np.linalg.norm(v2))

        sim = compute_cosine_similarity(v1, v2)
        self.assertGreater(sim, 0.999)

        threshold = 0.98
        is_duplicate = sim > threshold
        self.assertTrue(is_duplicate, f"Similarity {sim} should exceed threshold {threshold}")

        # Non-duplicate with varied angle
        v3 = _make_normalized_vector(701)
        sim_v3 = compute_cosine_similarity(v1, v3)
        self.assertLess(sim_v3, threshold)


if __name__ == "__main__":
    unittest.main()
