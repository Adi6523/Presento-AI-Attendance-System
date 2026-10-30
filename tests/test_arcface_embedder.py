from __future__ import annotations

import os
import unittest
import numpy as np
import openvino as ov

from edge.config import settings
from edge.arcface_embedder import (
    ArcFaceEmbedder,
    ArcFaceWorker,
    align_face,
    transform_mirrored_bbox,
    transform_mirrored_landmarks,
    compute_cosine_similarity,
    ARCFACE_REFERENCE_POINTS,
)


class TestArcFaceEmbedder(unittest.TestCase):
    """
    Unit test suite for Phase 2A ArcFace embedding integration.
    Validates model loading, GPU availability, 112x112 alignment,
    512-D output, L2 normalization, cosine similarity math,
    mirror coordinate transformation, and strict face_id association.
    """

    @classmethod
    def setUpClass(cls):
        cls.core = ov.Core()
        cls.available_devices = [d.upper() for d in cls.core.available_devices]
        cls.model_path = settings.arcface_model_path
        cls.embedder = ArcFaceEmbedder(model_path=cls.model_path, device="GPU")

    def test_model_loads(self):
        """1. Verify that ArcFace ONNX model exists and compiles successfully."""
        self.assertTrue(os.path.exists(self.model_path), f"Model not found: {self.model_path}")
        self.assertIsNotNone(self.embedder.compiled_model)
        self.assertEqual(len(self.embedder.compiled_model.outputs), 1)

    def test_gpu_is_available(self):
        """2. Verify that Intel GPU is available and actively utilized (no silent CPU fallback)."""
        self.assertIn("GPU", self.available_devices, "OpenVINO GPU is not available on this machine")
        self.assertEqual(self.embedder.active_device, "GPU")
        self.assertIn("GPU", self.embedder.full_device_name.upper())

    def test_alignment_produces_112x112x3(self):
        """3. Verify facial alignment yields exactly (112, 112, 3) uint8 image."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Sample detected landmarks
        landmarks = np.array(
            [
                [600.0, 320.0],  # left eye
                [680.0, 322.0],  # right eye
                [640.0, 360.0],  # nose
                [610.0, 410.0],  # left mouth
                [670.0, 408.0],  # right mouth
            ],
            dtype=np.float32,
        )
        aligned = align_face(frame, landmarks)
        self.assertEqual(aligned.shape, (112, 112, 3))
        self.assertEqual(aligned.dtype, np.uint8)

    def test_embedding_has_512_dimensions(self):
        """4. Verify embedding output has exactly 512 dimensions."""
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.get_embedding(dummy_face)
        self.assertEqual(emb.shape, (512,))
        self.assertEqual(emb.dtype, np.float32)

    def test_embedding_l2_norm_approx_1(self):
        """5. Verify embedding L2 norm is normalized to ~1.0."""
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        emb = self.embedder.get_embedding(dummy_face)
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=4, msg=f"L2 norm {norm} is not approximately 1.0")

    def test_cosine_similarity_function(self):
        """6. Verify cosine similarity mathematical properties."""
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        # Identical vectors -> 1.0
        self.assertAlmostEqual(compute_cosine_similarity(v1, v1), 1.0, places=5)

        # Opposite vectors -> -1.0
        self.assertAlmostEqual(compute_cosine_similarity(v1, -v1), -1.0, places=5)

        # Orthogonal vectors -> 0.0
        v2 = np.zeros(512, dtype=np.float32)
        v2[0] = 1.0
        v3 = np.zeros(512, dtype=np.float32)
        v3[1] = 1.0
        self.assertAlmostEqual(compute_cosine_similarity(v2, v3), 0.0, places=5)

    def test_mirror_coordinate_transformation(self):
        """7. Verify horizontal mirror coordinate transformation logic."""
        width = 1280
        bbox = (200, 100, 500, 400)  # (left, top, right, bottom)
        landmarks = np.array([[250.0, 150.0], [450.0, 155.0]], dtype=np.float32)

        # Mirrored coordinates
        m_bbox = transform_mirrored_bbox(bbox, width)
        m_landmarks = transform_mirrored_landmarks(landmarks, width)

        # Expected: m_left = 1280 - 1 - 500 = 779, m_right = 1280 - 1 - 200 = 1079
        self.assertEqual(m_bbox, (779, 100, 1079, 400))
        self.assertEqual(m_bbox[2] - m_bbox[0], bbox[2] - bbox[0], "Bbox width should be invariant")
        self.assertEqual(m_bbox[3] - m_bbox[1], bbox[3] - bbox[1], "Bbox height should be invariant")

        self.assertAlmostEqual(m_landmarks[0, 0], 1280 - 1 - 250.0, places=2)
        self.assertAlmostEqual(m_landmarks[0, 1], 150.0, places=2)

        # Applying mirror transformation twice must restore original coordinates
        restored_bbox = transform_mirrored_bbox(m_bbox, width)
        self.assertEqual(restored_bbox, bbox)

    def test_arcface_worker_face_id_association(self):
        """8. Verify ArcFace worker strictly associates embeddings with (face_id, frame_id)."""
        worker = ArcFaceWorker(model_path=self.model_path, device="GPU", target_fps=15.0).start()
        try:
            dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            queued = worker.submit_job(face_id=42, frame_id=100, aligned_face=dummy_face)
            self.assertTrue(queued)

            # Wait briefly for asynchronous inference
            import time
            result = None
            for _ in range(50):
                result = worker.get_result(face_id=42)
                if result is not None:
                    break
                time.sleep(0.02)

            self.assertIsNotNone(result, "Asynchronous worker did not return result")
            self.assertEqual(result.face_id, 42)
            self.assertEqual(result.frame_id, 100)
            self.assertEqual(result.embedding.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(result.embedding)), 1.0, places=4)
            self.assertGreater(result.latency_ms, 0.0)

            # Querying an unmatched face_id returns None
            self.assertIsNone(worker.get_result(face_id=999))

            # Pruning face removes it from result cache
            worker.prune_face(42)
            self.assertIsNone(worker.get_result(face_id=42))
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
