r"""
Presento Phase 2A: ArcFace Embedding Diagnostic Utility.
Performs same-person and different-person cosine similarity diagnostics
using camera-captured face crops or synthetic verification vectors.

Usage:
  .venv\Scripts\python.exe -m edge.arcface_diagnostic
"""
from __future__ import annotations

import argparse
import logging
import sys
import numpy as np

from .config import settings
from .arcface_embedder import ArcFaceEmbedder, compute_cosine_similarity, align_face
from .scrfd_detector import SCRFDDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("presento.diagnostic")


def run_synthetic_diagnostic():
    """
    Run diagnostic with synthetic face crops to verify:
    - Embedding extraction
    - High similarity on slight variations of same face
    - Low similarity on different faces
    - Detailed similarity matrix
    """
    print("=" * 70)
    print("       Presento Phase 2A: ArcFace Cosine Similarity Diagnostic")
    print("=" * 70)

    embedder = ArcFaceEmbedder(device=settings.openvino_device)
    print(f"[Hardware] Device: {embedder.active_device} ({embedder.full_device_name})")

    # Generate synthetic face-like patterns
    np.random.seed(42)
    base_face_a = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    # Variations of A (same subject with minor perturbations/lighting)
    noise_1 = np.random.randint(-10, 10, (112, 112, 3), dtype=np.int16)
    noise_2 = np.random.randint(-15, 15, (112, 112, 3), dtype=np.int16)
    face_a1 = np.clip(base_face_a.astype(np.int16) + noise_1, 0, 255).astype(np.uint8)
    face_a2 = np.clip(base_face_a.astype(np.int16) + noise_2, 0, 255).astype(np.uint8)
    face_a3 = np.clip(base_face_a * 0.9 + 15, 0, 255).astype(np.uint8)

    # Completely different face B
    base_face_b = np.random.randint(30, 220, (112, 112, 3), dtype=np.uint8)
    face_b1 = base_face_b
    face_b2 = np.clip(base_face_b * 1.1, 0, 255).astype(np.uint8)

    samples = {
        "A1 (Person A, sample 1)": embedder.get_embedding(face_a1),
        "A2 (Person A, sample 2)": embedder.get_embedding(face_a2),
        "A3 (Person A, sample 3)": embedder.get_embedding(face_a3),
        "B1 (Person B, sample 1)": embedder.get_embedding(face_b1),
        "B2 (Person B, sample 2)": embedder.get_embedding(face_b2),
    }

    print("\n[Embedding Verification]")
    for name, emb in samples.items():
        norm = float(np.linalg.norm(emb))
        print(f"  {name:<25}: shape={emb.shape}, L2 norm={norm:.6f}")

    print("\n[Pairwise Cosine Similarities]")
    same_person_pairs = [
        ("A1", "A2", samples["A1 (Person A, sample 1)"], samples["A2 (Person A, sample 2)"]),
        ("A1", "A3", samples["A1 (Person A, sample 1)"], samples["A3 (Person A, sample 3)"]),
        ("A2", "A3", samples["A2 (Person A, sample 2)"], samples["A3 (Person A, sample 3)"]),
    ]

    diff_person_pairs = [
        ("A1", "B1", samples["A1 (Person A, sample 1)"], samples["B1 (Person B, sample 1)"]),
        ("A1", "B2", samples["A1 (Person A, sample 1)"], samples["B2 (Person B, sample 2)"]),
        ("A2", "B1", samples["A2 (Person A, sample 2)"], samples["B1 (Person B, sample 1)"]),
        ("B1", "B2", samples["B1 (Person B, sample 1)"], samples["B2 (Person B, sample 2)"]),
    ]

    print("\n--- Same Person Pairs ---")
    same_sims = []
    for n1, n2, e1, e2 in same_person_pairs:
        sim = compute_cosine_similarity(e1, e2)
        same_sims.append(sim)
        print(f"  {n1} <-> {n2}: {sim:+.4f}")

    print("\n--- Different People Pairs ---")
    diff_sims = []
    for n1, n2, e1, e2 in diff_person_pairs:
        sim = compute_cosine_similarity(e1, e2)
        diff_sims.append(sim)
        print(f"  {n1} <-> {n2}: {sim:+.4f}")

    mean_same = np.mean(same_sims)
    mean_diff = np.mean(diff_sims)
    margin = mean_same - mean_diff

    print("-" * 70)
    print(f"Mean Same-Person Similarity     : {mean_same:+.4f}")
    print(f"Mean Different-Person Similarity: {mean_diff:+.4f}")
    print(f"Separation Margin               : {margin:+.4f}")
    print("=" * 70)
    print("NOTE: As per project guidelines, final recognition threshold will be established")
    print("      after gathering real classroom camera samples.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Presento ArcFace Diagnostic")
    parser.parse_args()
    run_synthetic_diagnostic()


if __name__ == "__main__":
    main()
