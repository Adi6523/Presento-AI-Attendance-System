from __future__ import annotations
import argparse
import time
import cv2
import face_recognition
import numpy as np
import requests
from .api import headers
from .config import settings
from .quality import FaceQualityAssessor, FaceQuality

# Import filter_enrollment_samples from backend web_vision
try:
    from ..backend.web_vision import filter_enrollment_samples
except ImportError:
    # Fallback implementation if import fails
    def filter_enrollment_samples(samples: list[np.ndarray], max_dist: float = 0.55) -> list[np.ndarray]:
        """
        Remove outlier enrollment frames (bad lighting, partial face, etc.)
        by rejecting samples that are far from the median of all pairwise distances.
        """
        if len(samples) < 3:
            return samples
        arr = np.array(samples)
        # Compute mean embedding as reference
        mean_emb = arr.mean(axis=0)
        dists = face_recognition.face_distance(arr, mean_emb)
        # Keep only samples within max_dist of the mean
        good = [s for s, d in zip(samples, dists) if d < max_dist]
        return good if len(good) >= 3 else samples   # Fallback to all if too strict

def main():
    parser = argparse.ArgumentParser(description="Capture face embeddings with quality validation; no images are retained.")
    parser.add_argument("--student-id", required=True); parser.add_argument("--name", required=True); parser.add_argument("--email")
    args = parser.parse_args()
    camera = cv2.VideoCapture(settings.camera_index)
    if not camera.isOpened(): raise SystemExit("Camera could not be opened")

    # Initialize quality assessor
    quality_assessor = FaceQualityAssessor()

    samples = []
    quality_scores = []
    deadline = time.monotonic() + 30  # 30 second timeout

    print(f"Starting enrollment for {args.name} ({args.student_id})")
    print(f"Target: {settings.enrollment_samples_target} quality samples (min {settings.enrollment_samples_min})")
    print("Follow the on-screen prompts for best results")
    print("Press 'q' to cancel at any time")

    while len(samples) < settings.enrollment_samples_target and time.monotonic() < deadline:
        ok, frame = camera.read()
        if not ok:
            print("Warning: Failed to capture frame")
            continue

        # Convert to RGB for face_recognition
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        locations = face_recognition.face_locations(rgb, model="hog")

        # Validate exactly one face
        if len(locations) != 1:
            if len(locations) == 0:
                feedback = "No face detected - please face the camera"
            else:
                feedback = f"Multiple faces detected ({len(locations)}) - please ensure only one person is visible"

            # Show feedback on frame
            cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Presento enrollment - q to cancel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
            continue

        # Get face location and assess quality
        face_location = locations[0]
        quality = quality_assessor.assess_quality(frame, face_location)

        # Draw face rectangle and quality feedback
        top, right, bottom, left = face_location
        color = (0, 255, 0) if quality.score >= settings.enrollment_quality_threshold else (0, 165, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display quality feedback
        feedback_y = bottom + 25 if bottom + 25 < frame.shape[0] - 10 else top - 10
        cv2.putText(frame, quality.feedback, (left, feedback_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display sample count and quality score
        cv2.putText(frame, f"Samples: {len(samples)}/{settings.enrollment_samples_target}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Quality: {quality.score:.2f}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Accept sample if quality meets threshold
        if len(enc := face_recognition.face_encodings(rgb, locations)) == 1 and quality.score >= settings.enrollment_quality_threshold:
            samples.append(enc[0])
            quality_scores.append(quality.score)
            print(f"Captured {len(samples)}/{settings.enrollment_samples_target} - Quality: {quality.score:.2f}")

            # Provide specific guidance based on quality issues
            if quality.score >= 0.9:
                print("Excellent quality!")
            elif quality.score >= 0.7:
                print("Good quality - continue varying angle slightly")
            else:
                print("Acceptable quality - try to improve lighting/pose")

            time.sleep(0.3)  # Reduced delay for better user experience
        elif len(enc) == 1:
            print(f"Sample rejected - Quality too low: {quality.score:.2f} ({quality.feedback})")

        # Show the frame
        cv2.imshow("Presento enrollment - q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Enrollment cancelled by user")
            break

    camera.release()
    cv2.destroyAllWindows()

    # Validate we have enough samples
    if len(samples) < settings.enrollment_samples_min:
        raise SystemExit(f"Enrollment failed: Only {len(samples)} valid samples captured (minimum {settings.enrollment_samples_min} required)")

    # Remove outlier frames (bad angle, partial face, shadow) before averaging
    import json
    clean_samples = filter_enrollment_samples(samples, max_dist=0.55)

    if len(clean_samples) < 3:
        raise SystemExit("Enrollment failed: Too many outlier samples removed - please try again with better conditions")

    # Calculate final embedding
    vector = np.mean(clean_samples, axis=0).tolist()
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0

    print(f"Enrollment summary:")
    print(f"  - Raw samples: {len(samples)}")
    print(f"  - Clean samples: {len(clean_samples)}")
    print(f"  - Average quality: {avg_quality:.2f}")

    # Prepare payload with enhanced metadata
    payload = {
        "id": args.student_id,
        "name": args.name,
        "email": args.email,
        "embedding": vector,  # Already a Python list; requests serializes correctly
        "enrollment_quality": float(avg_quality),
        "enrollment_samples": len(clean_samples)
    }

    response = requests.post(f"{settings.api_url}/v1/students", headers=headers(), json=payload, timeout=15)
    response.raise_for_status();
    print(f"Enrolled {args.name} ({args.student_id}) successfully")

if __name__ == "__main__": main()
