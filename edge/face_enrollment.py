"""
Presento Phase 2B: Local Face Enrollment Store.
Stores and validates multiple normalized 512-D ArcFace reference embeddings per student
in a persistent compressed numpy archive (data/face_embeddings.npz).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from typing import Dict, List, Optional
import numpy as np

from .config import settings

logger = logging.getLogger("presento.edge.enrollment")


def validate_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Validate that an embedding is a valid 512-D float32 normalized vector.

    Rules:
    - Must be a numpy array.
    - Must have shape (512,). If shape is (1, 512), it will be flattened.
    - Dtype must be float32 (or convertible).
    - L2 norm must be approximately 1.0 (tolerance 1e-3).

    Returns:
        Flattened float32 numpy array of shape (512,).

    Raises:
        ValueError: If validation fails.
    """
    if embedding is None:
        raise ValueError("Embedding cannot be None")

    if not isinstance(embedding, np.ndarray):
        try:
            embedding = np.array(embedding, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Cannot convert embedding to numpy array: {exc}")

    if embedding.ndim == 2 and embedding.shape[0] == 1 and embedding.shape[1] == 512:
        embedding = embedding.flatten()

    if embedding.shape != (512,):
        raise ValueError(f"Invalid embedding shape {embedding.shape}; expected (512,)")

    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)

    norm = float(np.linalg.norm(embedding))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(
            f"Embedding is not L2-normalized: ||v|| = {norm:.6f} (expected ~1.0)"
        )

    return embedding


def compute_prototype_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute the student prototype (centroid) embedding.
    Calculates arithmetic mean across reference embeddings and L2-normalizes the result.
    Returns (512,) float32 unit vector.
    """
    if not embeddings:
        raise ValueError("Cannot compute prototype from empty embedding list")
    val_embs = [validate_embedding(e) for e in embeddings]
    stacked = np.vstack(val_embs).astype(np.float32)
    mean_vec = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(mean_vec))
    if norm < 1e-10:
        raise ValueError("Degenerate embedding centroid with zero norm")
    return (mean_vec / norm).astype(np.float32)


def select_diverse_embeddings(
    embeddings: List[np.ndarray],
    max_references: int = settings.enrollment_max_references,
    duplicate_threshold: float = settings.enrollment_duplicate_threshold,
    diversity_threshold: float = settings.enrollment_diversity_threshold,
) -> List[np.ndarray]:
    """
    Greedy diversity selection algorithm for ArcFace embeddings.
    1. Validates and L2-normalizes input embeddings.
    2. Filters out near-duplicates (pairwise cosine similarity >= duplicate_threshold).
    3. Greedily selects representative embeddings maximizing angular coverage
       of the student's embedding space up to max_references.

    Returns:
        List of representative float32 (512,) embeddings.
    """
    if not embeddings:
        return []

    val_embs = [validate_embedding(e) for e in embeddings]
    if len(val_embs) <= 1:
        return val_embs

    # 1. Deduplication pass
    deduped: List[np.ndarray] = []
    for emb in val_embs:
        is_dup = False
        for kept in deduped:
            sim = float(np.dot(emb, kept))
            if sim >= duplicate_threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(emb)

    if len(deduped) <= max_references:
        return deduped

    # 2. Greedy Max-Min Angular Coverage Selection
    matrix = np.vstack(deduped).astype(np.float32)  # (M, 512)
    sim_matrix = np.dot(matrix, matrix.T)           # (M, M)

    # Step A: Choose the medoid (candidate with highest mean similarity) as seed
    mean_sims = np.mean(sim_matrix, axis=1)
    seed_idx = int(np.argmax(mean_sims))

    selected_indices = [seed_idx]
    remaining_indices = [i for i in range(len(deduped)) if i != seed_idx]

    # Step B: Greedily pick candidate that has the minimum maximum-similarity to already selected
    while len(selected_indices) < max_references and remaining_indices:
        best_candidate = None
        min_max_sim = float("inf")

        for idx in remaining_indices:
            max_sim_to_sel = float(np.max(sim_matrix[idx, selected_indices]))
            if max_sim_to_sel < min_max_sim:
                min_max_sim = max_sim_to_sel
                best_candidate = idx

        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
        else:
            break

    return [deduped[i] for i in selected_indices]


class FaceEnrollmentStore:
    """
    Persistent local repository for enrolled student face embeddings.
    Allows multiple 512-D ArcFace reference embeddings per student.
    Persists data in data/face_embeddings.npz with embedded JSON metadata.
    """

    def __init__(self, storage_path: str = settings.face_enrollment_path):
        self.storage_path = storage_path
        # In-memory student map:
        # student_id -> {
        #     "student_id": str,
        #     "name": str,
        #     "embeddings": list[np.ndarray],  # list of (512,) float32
        #     "enrolled_at": str,
        # }
        self._students: Dict[str, dict] = {}

        # Auto-load if file exists
        if os.path.exists(self.storage_path):
            try:
                self.load(self.storage_path)
            except Exception as exc:
                logger.warning(
                    "Could not load existing face enrollment store from %s: %s",
                    self.storage_path,
                    exc,
                )

    def add_student(
        self,
        student_id: str,
        name: str,
        embeddings: List[np.ndarray],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Register a student with multiple validated reference embeddings.
        Supports rich Phase 4 metadata (quality stats, condition coverage).
        """
        student_id = str(student_id).strip()
        name = str(name).strip()
        if not student_id:
            raise ValueError("student_id cannot be empty")
        if not name:
            raise ValueError("name cannot be empty")
        if not embeddings or len(embeddings) == 0:
            raise ValueError("At least one embedding is required to enroll a student")

        validated_embs = [validate_embedding(e) for e in embeddings]
        prototype = compute_prototype_embedding(validated_embs)

        student_meta = {
            "raw_count": len(validated_embs),
            "accepted_count": len(validated_embs),
            "rejected_count": 0,
            "average_quality": 1.0,
            "conditions_represented": ["general"],
            "v2_compatible": True,
        }
        if metadata:
            student_meta.update(metadata)

        self._students[student_id] = {
            "student_id": student_id,
            "name": name,
            "embeddings": validated_embs,
            "prototype": prototype,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "metadata": student_meta,
        }
        logger.info(
            "Enrolled student %s (%s) with %d reference embeddings (v2)",
            student_id,
            name,
            len(validated_embs),
        )

    def add_embedding(self, student_id: str, embedding: np.ndarray) -> None:
        """
        Append a new reference embedding to an already registered student.
        """
        student_id = str(student_id).strip()
        if student_id not in self._students:
            raise KeyError(f"Student '{student_id}' is not enrolled in store")

        val_emb = validate_embedding(embedding)
        self._students[student_id]["embeddings"].append(val_emb)
        # Update prototype with new embedding
        self._students[student_id]["prototype"] = compute_prototype_embedding(
            self._students[student_id]["embeddings"]
        )
        logger.debug(
            "Added reference embedding to student %s (total: %d)",
            student_id,
            len(self._students[student_id]["embeddings"]),
        )

    def get_student(self, student_id: str) -> Optional[dict]:
        """
        Retrieve student record with all reference embeddings.
        Returns None if student does not exist.
        """
        student = self._students.get(str(student_id).strip())
        if student is None:
            return None
        proto = student.get("prototype")
        return {
            "student_id": student["student_id"],
            "name": student["name"],
            "embeddings": [e.copy() for e in student["embeddings"]],
            "prototype": proto.copy() if proto is not None else None,
            "enrolled_at": student.get("enrolled_at", ""),
            "num_embeddings": len(student["embeddings"]),
            "metadata": dict(student.get("metadata", {})),
        }

    def get_all_students(self) -> List[dict]:
        """
        Retrieve all enrolled students.
        """
        results = []
        for sid in sorted(self._students.keys()):
            st = self._students[sid]
            proto = st.get("prototype")
            results.append(
                {
                    "student_id": st["student_id"],
                    "name": st["name"],
                    "embeddings": [e.copy() for e in st["embeddings"]],
                    "prototype": proto.copy() if proto is not None else None,
                    "enrolled_at": st.get("enrolled_at", ""),
                    "num_embeddings": len(st["embeddings"]),
                    "metadata": dict(st.get("metadata", {})),
                }
            )
        return results

    def remove_student(self, student_id: str) -> bool:
        """
        Remove a student and all their reference embeddings.
        Returns True if removed, False if student was not found.
        """
        student_id = str(student_id).strip()
        if student_id in self._students:
            del self._students[student_id]
            logger.info("Removed student %s from enrollment store", student_id)
            return True
        return False

    def clear(self) -> None:
        """Clear all in-memory student records."""
        self._students.clear()

    def save(self, path: Optional[str] = None) -> str:
        """
        Persist in-memory enrollments to a compressed .npz file with embedded metadata.
        Also creates a companion JSON manifest for human inspection.
        Maintains 100% backward compatibility with existing readers.
        """
        target_path = path or self.storage_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

        arrays_to_save: dict[str, np.ndarray] = {}
        metadata: dict[str, dict] = {}

        for student_id, st in self._students.items():
            if st["embeddings"]:
                stacked = np.vstack(st["embeddings"]).astype(np.float32)
                # Store array keyed by sanitized identifier
                key = f"emb_{student_id}"
                arrays_to_save[key] = stacked

                proto = st.get("prototype")
                if proto is not None:
                    arrays_to_save[f"proto_{student_id}"] = np.asarray(proto, dtype=np.float32)

                metadata[student_id] = {
                    "student_id": st["student_id"],
                    "name": st["name"],
                    "count": len(st["embeddings"]),
                    "enrolled_at": st.get("enrolled_at", ""),
                    "key": key,
                    "metadata": st.get("metadata", {}),
                    "v2_compatible": True,
                }

        # Embed metadata directly into the npz archive as a JSON string
        meta_json_str = json.dumps(metadata, indent=2)
        arrays_to_save["__metadata__"] = np.array([meta_json_str])

        np.savez_compressed(target_path, **arrays_to_save)

        # Write human-readable companion manifest
        manifest_path = os.path.splitext(target_path)[0] + "_manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(meta_json_str)
        except Exception as exc:
            logger.warning("Could not write manifest file: %s", exc)

        logger.info(
            "Saved %d enrolled students to %s",
            len(self._students),
            target_path,
        )
        return target_path

    def load(self, path: Optional[str] = None) -> None:
        """
        Load student enrollments from a compressed .npz file.
        Maintains backward compatibility with V1 stores.
        """
        source_path = path or self.storage_path
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Enrollment file not found at: {source_path}")

        loaded = np.load(source_path, allow_pickle=False)
        if "__metadata__" not in loaded:
            raise ValueError(f"Invalid enrollment archive: missing '__metadata__' in {source_path}")

        meta_json_str = str(loaded["__metadata__"][0])
        metadata = json.loads(meta_json_str)

        new_students: dict[str, dict] = {}
        for student_id, info in metadata.items():
            key = info.get("key", f"emb_{student_id}")
            if key in loaded:
                emb_array = loaded[key]
                # emb_array has shape (N, 512)
                emb_list = [emb_array[i].astype(np.float32) for i in range(emb_array.shape[0])]
                validated_list = [validate_embedding(e) for e in emb_list]

                # Check for cached prototype or compute on load
                proto_key = f"proto_{student_id}"
                if proto_key in loaded:
                    prototype = validate_embedding(loaded[proto_key])
                else:
                    prototype = compute_prototype_embedding(validated_list)

                student_meta = info.get("metadata", {
                    "raw_count": len(validated_list),
                    "accepted_count": len(validated_list),
                    "rejected_count": 0,
                    "average_quality": 1.0,
                    "conditions_represented": ["legacy_v1"],
                    "v2_compatible": True,
                })

                new_students[student_id] = {
                    "student_id": info["student_id"],
                    "name": info["name"],
                    "embeddings": validated_list,
                    "prototype": prototype,
                    "enrolled_at": info.get("enrolled_at", ""),
                    "metadata": student_meta,
                }

        self._students = new_students
        logger.info(
            "Loaded %d enrolled students from %s",
            len(self._students),
            source_path,
        )


def upgrade_enrollment_store(store: FaceEnrollmentStore) -> bool:
    """
    Upgrade a legacy enrollment store to V2 format.
    Computes prototype embeddings, ensures rich metadata dictionaries,
    and writes back without modifying existing reference vectors.
    """
    students = store.get_all_students()
    if not students:
        return False

    upgraded_any = False
    for st in students:
        sid = st["student_id"]
        record = store._students.get(sid)
        if record is None:
            continue
        if "prototype" not in record or record["prototype"] is None:
            record["prototype"] = compute_prototype_embedding(record["embeddings"])
            upgraded_any = True
        meta = record.setdefault("metadata", {})
        if not meta.get("v2_compatible"):
            meta["v2_compatible"] = True
            meta.setdefault("conditions_represented", ["upgraded_v1"])
            upgraded_any = True

    if upgraded_any:
        store.save()
        logger.info("Successfully upgraded %d student records to V2 format", len(students))
    return upgraded_any
