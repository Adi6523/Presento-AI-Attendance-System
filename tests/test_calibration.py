"""
Tests for Presento Phase 8 & 17: Identity Calibration & Liveness Dataset Infrastructure.
Verifies distribution calculations, threshold sweeps, markdown reporting,
and zero-leakage session-level partitioning.
"""
from pathlib import Path
import numpy as np
import pytest

from edge.calibration import (
    IdentityCalibrator,
    compute_distribution_stats,
    CalibrationReport,
)
from edge.face_enrollment import FaceEnrollmentStore
from edge.liveness_dataset import LivenessDatasetManager


@pytest.fixture
def populated_store(tmp_path):
    archive_path = tmp_path / "calib_store.npz"
    store = FaceEnrollmentStore(storage_path=str(archive_path))

    rng = np.random.default_rng(777)
    # Student 1
    b1 = rng.standard_normal(512).astype(np.float32)
    b1 /= np.linalg.norm(b1)
    s1 = [b1 + rng.normal(0, 0.005, 512).astype(np.float32) for _ in range(5)]
    s1 = [v / np.linalg.norm(v) for v in s1]

    # Student 2 (distinct direction)
    b2 = rng.standard_normal(512).astype(np.float32)
    b2 /= np.linalg.norm(b2)
    s2 = [b2 + rng.normal(0, 0.005, 512).astype(np.float32) for _ in range(5)]
    s2 = [v / np.linalg.norm(v) for v in s2]

    store.add_student("23UAM001", "Student One", s1)
    store.add_student("23UAM002", "Student Two", s2)
    store.save()

    return store


class TestIdentityCalibration:
    def test_compute_distribution_stats(self):
        values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        stats = compute_distribution_stats(values)

        assert stats.count == 10
        assert abs(stats.mean - 0.55) < 1e-3
        assert abs(stats.median - 0.55) < 1e-3
        assert stats.min == 0.10
        assert stats.max == 1.00
        assert stats.p5 <= stats.median <= stats.p95

    def test_calibration_sweep_and_recommendations(self, populated_store):
        calibrator = IdentityCalibrator(store=populated_store)
        report = calibrator.run_calibration(sweep_start=0.35, sweep_end=0.75, sweep_step=0.05)

        assert isinstance(report, CalibrationReport)
        assert report.num_students == 2
        assert report.known_same_person.mean > report.known_cross_person.mean
        assert report.known_same_person.mean > report.unknown_best_match.mean
        assert len(report.threshold_sweep) > 5

        # Check threshold sweep monotonic trend: as threshold increases, FAR decreases
        fars = [pt.far_unknown for pt in report.threshold_sweep]
        for i in range(len(fars) - 1):
            assert fars[i] >= fars[i + 1]

        # Verify Markdown report generation
        md = calibrator.generate_markdown_report(report)
        assert "# Presento Identity Engine: Evidence-Based Calibration Report" in md
        assert "Known Same Person" in md
        assert "Unknown Impostor Best Match" in md
        assert f"{report.recommended_tolerance}" in md


class TestLivenessDatasetInfrastructure:
    def test_dataset_manager_session_partitioning_zero_leakage(self, tmp_path):
        mgr = LivenessDatasetManager(root_dir=str(tmp_path / "liveness_data"))

        # Create dummy crops across 4 sessions
        rng = np.random.default_rng(888)
        sessions = ["sess_001", "sess_002", "sess_003", "sess_004"]

        for sess in sessions:
            for i in range(5):
                frame = rng.integers(0, 255, (60, 60, 3), dtype=np.uint8)
                mgr.add_sample(
                    frame_or_crop=frame,
                    session_id=sess,
                    subject_id="SUBJ_01",
                    label="real" if i % 2 == 0 else "spoof",
                    attack_type=None if i % 2 == 0 else "phone_photo",
                )

        assert len(mgr.samples) == 20

        # Partition sessions: 2 train, 1 val, 1 test
        partition_stats = mgr.partition_sessions(train_ratio=0.50, val_ratio=0.25, test_ratio=0.25)
        assert partition_stats["train"] > 0

        # Validate integrity
        is_valid, issues = mgr.validate_integrity()
        assert is_valid is True, f"Integrity failed: {issues}"

        # Verify summary
        summary = mgr.get_summary()
        assert summary["total_samples"] == 20
        assert summary["total_sessions"] == 4
        assert "real" in summary["by_label"]
        assert "spoof" in summary["by_label"]
