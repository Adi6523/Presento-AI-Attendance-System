# Presento Identity Engine: Evidence-Based Calibration Report

- **Generated At**: `2026-09-29T05:45:01.576051+00:00`
- **Enrolled Roster**: 4 students, 20 total reference embeddings
- **Recommended Tolerance**: `0.35`
- **Recommended Min Margin**: `0.2`

## 1. Similarity Distributions

| Population | Count | Mean | Median | P5 (Worst 5%) | P95 (Best 5%) | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Known Same Person (Intra-Class)** | 20 | 0.851 | 0.865 | `0.755` | 0.931 | 0.749 | 0.931 |
| **Known Cross Person (Inter-Class)** | 60 | 0.122 | 0.120 | -0.017 | `0.344` | -0.038 | 0.357 |
| **Unknown Impostor Best Match** | 50 | 0.059 | 0.061 | 0.017 | `0.097` | -0.017 | 0.130 |
| **Identity Margins (Top1 - Top2)** | 20 | 0.612 | 0.662 | `0.411` | 0.734 | 0.409 | 0.750 |

## 2. Empirical Threshold Sweep Analysis

| Threshold | TAR (Genuine Accept) | FRR (False Reject) | FAR Cross (Student Confusion) | FAR Unknown (Impostor Leak) | Unknown Rejection Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.35` **(Recommended)** | 100.0% | 0.0% | 3.3% | 0.0% | `100.0%` |
| `0.40` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.45` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.50` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.55` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.60` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.65` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.70` | 100.0% | 0.0% | 0.0% | 0.0% | `100.0%` |
| `0.75` | 95.0% | 5.0% | 0.0% | 0.0% | `100.0%` |

## 3. Conclusions & Acceptance Rationale

1. **Tolerance `0.35`** ensures unknown impostor rejection of `0.0%` while preserving `0.851` average genuine similarity.
2. **Margin threshold `0.2`** enforces strict separation against ambiguous border cases.
3. **Zero ArcFace Retraining**: Frozen representations provide clean, non-overfit feature separability.