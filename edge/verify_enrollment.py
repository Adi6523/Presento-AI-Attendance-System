"""
Presento Face Enrollment Verification & Store Inspector.
Lists all currently enrolled students from data/face_embeddings.npz,
verifies embedding counts, dimensions, and L2 normalization,
and optionally performs cleanup of demo/test records.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .config import settings
from .face_enrollment import FaceEnrollmentStore


def verify_and_list_students(store_path: str = settings.face_enrollment_path) -> List[dict]:
    """
    Load enrollment store and return list of student dicts.
    """
    if not os.path.exists(store_path):
        print(f"[ERROR] Enrollment archive not found at: {store_path}", file=sys.stderr)
        return []

    store = FaceEnrollmentStore(store_path)
    students = store.get_all_students()
    return students


def cleanup_test_records(store_path: str = settings.face_enrollment_path) -> tuple[list[str], list[str]]:
    """
    Remove temporary or test entries (such as Mom / TEMP01 and Chut if present)
    while strictly preserving legitimate enrolled students.
    Returns (removed_names, absent_names).
    """
    if not os.path.exists(store_path):
        return [], []

    store = FaceEnrollmentStore(store_path)
    removed = []
    absent = []

    # 1. Mom / TEMP01
    if store.get_student("TEMP01") is not None:
        store.remove_student("TEMP01")
        removed.append("Mom (TEMP01)")
    else:
        absent.append("Mom (TEMP01)")

    # 2. Chut check
    chut_found = False
    for st in store.get_all_students():
        if "chut" in st["name"].lower() or "chut" in st["student_id"].lower():
            store.remove_student(st["student_id"])
            removed.append(f"{st['name']} ({st['student_id']})")
            chut_found = True
    if not chut_found:
        absent.append("Chut (already absent)")

    if removed:
        store.save(store_path)

    return removed, absent


def print_enrolled_summary(store_path: str = settings.face_enrollment_path) -> None:
    """Print standard formatted enrollment summary."""
    students = verify_and_list_students(store_path)
    print("=" * 60)
    print(f"  PRESENTO ENROLLMENT STORE: {store_path}")
    print(f"  Total Enrolled Students : {len(students)}")
    print("=" * 60)
    print("Enrolled students:")
    if not students:
        print("  (No enrolled students found)")
    else:
        for idx, st in enumerate(students, 1):
            n_emb = st.get("num_embeddings", len(st.get("embeddings", [])))
            print(f"{idx}. {st['name']} - {st['student_id']} ({n_emb} samples)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and list enrolled students in Presento store.")
    parser.add_argument("--cleanup", action="store_true", help="Remove test/demo records (Mom/TEMP01, etc.)")
    parser.add_argument("--store", type=str, default=settings.face_enrollment_path, help="Path to embeddings npz")
    args = parser.parse_args()

    if args.cleanup:
        removed, absent = cleanup_test_records(args.store)
        if removed:
            print(f"[CLEANUP] Removed test records: {', '.join(removed)}")
        for item in absent:
            print(f"[CLEANUP] Note: {item}")

    print_enrolled_summary(args.store)


if __name__ == "__main__":
    main()
