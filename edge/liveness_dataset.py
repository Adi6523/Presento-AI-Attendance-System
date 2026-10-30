"""
Presento Phase 17: Liveness Dataset Infrastructure & Session-Level Partitioning.
Prepares structured real and spoof dataset captures with strict session-level isolation
to prevent data leakage across train, validation, and test splits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("presento.edge.liveness_dataset")

DATASET_ROOT = "data/liveness_dataset"


@dataclass
class DatasetSample:
    sample_id: str
    session_id: str
    subject_id: str
    label: str               # "real" or "spoof"
    attack_type: Optional[str]  # None for real; "printed_photo", "phone_photo", "phone_video", "replay_video" for spoof
    split: str               # "train", "val", "test", or "unassigned"
    file_path: str
    lighting: str            # "normal", "dim", "bright"
    distance: str            # "near", "normal", "far"
    device_type: str         # "laptop_webcam", "c270", "phone_oled", "laptop_ips", "paper"
    timestamp: str


class LivenessDatasetManager:
    """
    Manages collection, session-level grouping, and validation of real and spoof data.
    Enforces strict zero-leakage session partitioning.
    """

    def __init__(self, root_dir: str = DATASET_ROOT):
        self.root_dir = Path(root_dir)
        self.manifest_path = self.root_dir / "dataset_manifest.json"
        self.samples: Dict[str, DatasetSample] = {}
        self._init_directories()
        self.load_manifest()

    def _init_directories(self) -> None:
        """Create standard dataset split directory hierarchy."""
        for split in ("train", "val", "test"):
            for label in ("real", "spoof"):
                (self.root_dir / split / label).mkdir(parents=True, exist_ok=True)

    def load_manifest(self) -> None:
        """Load manifest from disk if present."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for sid, item in data.get("samples", {}).items():
                        self.samples[sid] = DatasetSample(**item)
            except Exception as exc:
                logger.warning("Could not read liveness dataset manifest: %s", exc)

    def save_manifest(self) -> None:
        """Persist manifest to JSON."""
        data = {
            "version": "2.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(self.samples),
            "samples": {sid: asdict(s) for sid, s in self.samples.items()},
        }
        self.root_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_sample(
        self,
        frame_or_crop: np.ndarray,
        session_id: str,
        subject_id: str,
        label: str,
        attack_type: Optional[str] = None,
        split: str = "train",
        lighting: str = "normal",
        distance: str = "normal",
        device_type: str = "webcam",
    ) -> DatasetSample:
        """
        Save a face frame or crop into the appropriate split folder and register in manifest.
        """
        label = label.lower()
        if label not in ("real", "spoof"):
            raise ValueError(f"Label must be 'real' or 'spoof', got '{label}'")
        if label == "spoof" and not attack_type:
            attack_type = "unknown_spoof"

        split = split.lower()
        if split not in ("train", "val", "test"):
            split = "train"

        sample_id = f"{session_id}_{label}_{int(datetime.now(timezone.utc).timestamp() * 1000)}_{len(self.samples)}"
        rel_path = f"{split}/{label}/{sample_id}.jpg"
        abs_path = self.root_dir / rel_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(abs_path), frame_or_crop)

        sample = DatasetSample(
            sample_id=sample_id,
            session_id=session_id,
            subject_id=subject_id,
            label=label,
            attack_type=attack_type,
            split=split,
            file_path=rel_path,
            lighting=lighting,
            distance=distance,
            device_type=device_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.samples[sample_id] = sample
        self.save_manifest()
        return sample

    def partition_sessions(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Dict[str, int]:
        """
        Partition dataset across train, val, and test splits at the SESSION level.
        Prevents frames from the same recording/session from leaking across splits.
        Moves files on disk to match their new split paths.
        """
        # Group samples by session_id
        session_map: Dict[str, List[DatasetSample]] = {}
        for s in self.samples.values():
            session_map.setdefault(s.session_id, []).append(s)

        sessions = sorted(session_map.keys())
        rng = random.Random(random_seed)
        rng.shuffle(sessions)

        n = len(sessions)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0

        train_sessions = set(sessions[:n_train])
        val_sessions = set(sessions[n_train:n_train + n_val])
        test_sessions = set(sessions[n_train + n_val:])

        stats = {"train": 0, "val": 0, "test": 0}

        for sid, sample in list(self.samples.items()):
            old_split = sample.split
            if sample.session_id in train_sessions:
                new_split = "train"
            elif sample.session_id in val_sessions:
                new_split = "val"
            else:
                new_split = "test"

            if old_split != new_split:
                old_path = self.root_dir / sample.file_path
                new_rel = f"{new_split}/{sample.label}/{sample.sample_id}.jpg"
                new_path = self.root_dir / new_rel

                if old_path.exists():
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(old_path), str(new_path))
                    sample.file_path = new_rel

                sample.split = new_split

            stats[new_split] += 1

        self.save_manifest()
        logger.info("Session-level partitioning completed: %s", stats)
        return stats

    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset integrity:
        1. All manifest files exist on disk.
        2. No session appears in more than one split (ZERO DATA LEAKAGE).
        """
        issues = []
        session_splits: Dict[str, set[str]] = {}

        for sid, s in self.samples.items():
            full_p = self.root_dir / s.file_path
            if not full_p.exists():
                issues.append(f"Missing file on disk: {s.file_path}")

            session_splits.setdefault(s.session_id, set()).add(s.split)

        # Check session leakages
        for session_id, splits in session_splits.items():
            if len(splits) > 1:
                issues.append(f"LEAKAGE DETECTED: Session '{session_id}' found in multiple splits: {splits}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_summary(self) -> Dict[str, any]:
        """Return dataset distribution metrics."""
        summary = {
            "total_samples": len(self.samples),
            "total_sessions": len(set(s.session_id for s in self.samples.values())),
            "by_split": {},
            "by_label": {"real": 0, "spoof": 0},
            "by_attack_type": {},
        }
        for s in self.samples.values():
            summary["by_split"][s.split] = summary["by_split"].get(s.split, 0) + 1
            summary["by_label"][s.label] = summary["by_label"].get(s.label, 0) + 1
            if s.attack_type:
                summary["by_attack_type"][s.attack_type] = summary["by_attack_type"].get(s.attack_type, 0) + 1

        return summary
