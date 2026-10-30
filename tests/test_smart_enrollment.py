"""
Tests for Presento Phase 1–4: Smart Enrollment 2.0, Quality Filtering, Diversity Selection,
and Rich Student Representation with Backward Compatibility.
"""
import json
from pathlib import Path
import numpy as np
import pytest

from edge.face_enrollment import (
    FaceEnrollmentStore,
    validate_embedding,
    compute_prototype_embedding,
    select_diverse_embeddings,
    upgrade_enrollment_store,
)
from edge.quality import assess_face_crop_quality
from edge.enroll_face import assess_enrollment_quality
from edge.scrfd_detector import SCRFDFace


class TestQualityFiltering:
    """Verifies Phase 2: Frame and Crop Quality Filtering."""

    def test_reject_dark_frame(self):
        dark_crop = np.full((100, 100, 3), 15, dtype=np.uint8)  # mean brightness 15 < 35
        ok, msg, metrics = assess_face_crop_quality(dark_crop, lighting_min=35.0)
        assert ok is False
        assert "dark" in msg.lower()
        assert metrics["mean_brightness"] < 35.0

    def test_reject_bright_frame(self):
        bright_crop = np.full((100, 100, 3), 245, dtype=np.uint8)  # mean brightness 245 > 230
        ok, msg, metrics = assess_face_crop_quality(bright_crop, lighting_max=230.0)
        assert ok is False
        assert "bright" in msg.lower()
        assert metrics["mean_brightness"] > 230.0

    def test_reject_small_crop(self):
        small_crop = np.full((40, 40, 3), 120, dtype=np.uint8)  # 40x40 < 60px
        ok, msg, _ = assess_face_crop_quality(small_crop, min_size=60)
        assert ok is False
        assert "small" in msg.lower()

    def test_reject_blurry_crop(self):
        flat_crop = np.full((100, 100, 3), 128, dtype=np.uint8)
        ok, msg, _ = assess_face_crop_quality(flat_crop, blur_threshold=30.0)
        assert ok is False
        assert "blurry" in msg.lower()

    def test_reject_clipped_crop(self):
        # 50% overexposed white pixels, 50% normal
        crop = np.full((100, 100, 3), 120, dtype=np.uint8)
        crop[:50, :] = 255  # 50% clipped > 20% limit
        ok, msg, metrics = assess_face_crop_quality(crop, max_clipping_ratio=0.20)
        assert ok is False
        assert "clipping" in msg.lower()

    def test_accept_good_quality_crop(self):
        rng = np.random.default_rng(42)
        # Texture with good variance and normal brightness
        good_crop = rng.integers(60, 180, (120, 120, 3), dtype=np.uint8)
        ok, msg, metrics = assess_face_crop_quality(good_crop, min_size=60, blur_threshold=20.0)
        assert ok is True
        assert "acceptable" in msg.lower()
        assert metrics["blur_score"] > 20.0
        assert 50.0 <= metrics["mean_brightness"] <= 200.0


class TestEmbeddingDiversitySelection:
    """Verifies Phase 3: Greedy Max-Min Diversity Selection Algorithm."""

    def test_deduplication_removes_near_identical_samples(self):
        rng = np.random.default_rng(100)
        base = rng.standard_normal(512).astype(np.float32)
        base /= np.linalg.norm(base)

        # Create 10 nearly identical clones (cos sim > 0.999)
        clones = []
        for _ in range(10):
            noise = rng.normal(0, 0.001, 512).astype(np.float32)
            c = base + noise
            c /= np.linalg.norm(c)
            clones.append(c)

        diverse = select_diverse_embeddings(clones, max_references=10, duplicate_threshold=0.985)
        # All duplicate clones should collapse into a single representative embedding
        assert len(diverse) == 1

    def test_greedy_selection_maximizes_angular_coverage(self):
        rng = np.random.default_rng(200)
        # Generate 50 embeddings: cluster of 40 near vector A, and 10 spread out
        v_a = rng.standard_normal(512).astype(np.float32)
        v_a /= np.linalg.norm(v_a)

        raw_embeddings = []
        # Dense cluster
        for _ in range(40):
            noise = rng.normal(0, 0.05, 512).astype(np.float32)
            v = v_a + noise
            raw_embeddings.append(v / np.linalg.norm(v))

        # Diverse outliers (different poses)
        for _ in range(10):
            outlier = rng.standard_normal(512).astype(np.float32)
            raw_embeddings.append(outlier / np.linalg.norm(outlier))

        selected = select_diverse_embeddings(
            raw_embeddings,
            max_references=15,
            duplicate_threshold=0.985,
        )
        assert len(selected) <= 15
        assert len(selected) >= 5

        # Check all are normalized 512-D vectors
        for emb in selected:
            assert emb.shape == (512,)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-3

    def test_compute_prototype_centroid(self):
        rng = np.random.default_rng(300)
        v1 = rng.standard_normal(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = rng.standard_normal(512).astype(np.float32)
        v2 /= np.linalg.norm(v2)

        proto = compute_prototype_embedding([v1, v2])
        assert proto.shape == (512,)
        assert abs(np.linalg.norm(proto) - 1.0) < 1e-3

        # Prototype should have equal dot product with v1 and v2 if they are symmetric
        sim1 = float(np.dot(proto, v1))
        sim2 = float(np.dot(proto, v2))
        assert abs(sim1 - sim2) < 1e-4


class TestStudentRepresentationAndStorage:
    """Verifies Phase 4: Student Metadata and Backward Compatibility."""

    def test_store_saves_and_loads_v2_metadata(self, tmp_path):
        archive_path = tmp_path / "test_store_v2.npz"
        store = FaceEnrollmentStore(storage_path=str(archive_path))

        rng = np.random.default_rng(400)
        embs = [rng.standard_normal(512).astype(np.float32) for _ in range(5)]
        embs = [e / np.linalg.norm(e) for e in embs]

        meta = {
            "raw_count": 50,
            "accepted_count": 5,
            "rejected_count": 45,
            "conditions_represented": ["frontal", "left_15", "right_15", "smile", "distance_near"],
            "v2_compatible": True,
        }

        store.add_student("23UAM999", "Test Student V2", embs, metadata=meta)
        store.save()

        # Reload in a clean store instance
        loaded_store = FaceEnrollmentStore(storage_path=str(archive_path))
        st = loaded_store.get_student("23UAM999")

        assert st is not None
        assert st["name"] == "Test Student V2"
        assert st["num_embeddings"] == 5
        assert st["prototype"] is not None
        assert st["prototype"].shape == (512,)
        assert st["metadata"]["raw_count"] == 50
        assert st["metadata"]["v2_compatible"] is True
        assert "smile" in st["metadata"]["conditions_represented"]

    def test_upgrade_legacy_store_adds_prototypes_and_v2_meta(self, tmp_path):
        # Create a legacy-style store without prototypes
        legacy_path = tmp_path / "legacy_store.npz"
        store = FaceEnrollmentStore(storage_path=str(legacy_path))

        rng = np.random.default_rng(500)
        embs = [rng.standard_normal(512).astype(np.float32) for _ in range(3)]
        embs = [e / np.linalg.norm(e) for e in embs]

        # Add student and simulate legacy store state without prototype or v2 metadata
        store.add_student("LEGACY01", "Old Student", embs)
        store._students["LEGACY01"]["prototype"] = None
        store._students["LEGACY01"].pop("metadata", None)
        store.save()

        # Upgrade store
        upgraded = upgrade_enrollment_store(store)
        assert upgraded is True

        reloaded = FaceEnrollmentStore(storage_path=str(legacy_path))
        st = reloaded.get_student("LEGACY01")
        assert st["prototype"] is not None
        assert st["metadata"]["v2_compatible"] is True
