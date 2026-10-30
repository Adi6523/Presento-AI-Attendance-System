r"""
Presento Phase 2B: Interactive Guided Face Enrollment Tool.
Captures multiple diverse ArcFace reference embeddings per student with quality validation
and configurable duplicate rejection, saving them to data/face_embeddings.npz.

Usage:
  .venv\Scripts\python.exe -m edge.enroll_face --id 23UAM090 --name "Aditya Patil"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from .config import settings
from .camera_control import (
    CameraSession,
    CameraInUseError,
    CameraAuthError,
    CameraControlError,
)
from .face_enrollment import FaceEnrollmentStore, select_diverse_embeddings
from .frame_receiver import FrameReceiver
from .scrfd_detector import SCRFDDetector, SCRFDFace
from .arcface_embedder import (
    ArcFaceEmbedder,
    align_face,
    transform_mirrored_bbox,
    transform_mirrored_landmarks,
    compute_cosine_similarity,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("presento.edge.enroll")

POSE_PROMPTS = [
    "Pose 1/12: Look straight / front-facing at camera",
    "Pose 2/12: Turn head slightly LEFT (~15 deg)",
    "Pose 3/12: Turn head slightly RIGHT (~15 deg)",
    "Pose 4/12: Tilt head slightly UP (~10 deg)",
    "Pose 5/12: Tilt head slightly DOWN (~10 deg)",
    "Pose 6/12: Step back slightly / farther from camera",
    "Pose 7/12: Step closer / natural frontal view",
    "Pose 8/12: Slight smile / natural expression variation",
    "Pose 9/12: Neutral expression / front-facing",
    "Pose 10/12: Turn head slight UP-LEFT (~10 deg)",
    "Pose 11/12: Turn head slight UP-RIGHT (~10 deg)",
    "Pose 12/12: Final natural frontal gaze",
]

SMART_ENROLLMENT_CONDITIONS = [
    {"prompt": "Look straight at camera (neutral expression)", "condition": "frontal_neutral"},
    {"prompt": "Look straight at camera (smile / natural expression)", "condition": "frontal_smile"},
    {"prompt": "Turn head SLIGHTLY LEFT (~10-15 deg)", "condition": "yaw_slight_left"},
    {"prompt": "Turn head LEFT (~25-30 deg)", "condition": "yaw_left"},
    {"prompt": "Turn head SLIGHTLY RIGHT (~10-15 deg)", "condition": "yaw_slight_right"},
    {"prompt": "Turn head RIGHT (~25-30 deg)", "condition": "yaw_right"},
    {"prompt": "Tilt head SLIGHTLY UP (~10-15 deg)", "condition": "pitch_slight_up"},
    {"prompt": "Tilt head SLIGHTLY DOWN (~10-15 deg)", "condition": "pitch_slight_down"},
    {"prompt": "Step closer to camera (near view)", "condition": "distance_near"},
    {"prompt": "Step back slightly (classroom distance)", "condition": "distance_normal"},
    {"prompt": "Step farther back slightly", "condition": "distance_far"},
    {"prompt": "Natural subtle head movement / daily appearance", "condition": "natural_movement"},
    {"prompt": "Final natural frontal gaze", "condition": "frontal_final"},
]


def parse_camera_source(
    source: int | str | None,
    default_host: str = settings.pi_ip,
    default_port: int = settings.stream_port,
) -> tuple[str, int | str] | tuple[str, str, int]:
    """
    Parse a camera source into either:
      ('opencv', int | str): Local camera index (e.g. 0) or file path.
      ('tcp', host_str, port_int): TCP camera stream (e.g. '192.168.1.100', 5001).

    Raises:
        ValueError: If the source string is an invalid HOST:PORT format.
    """
    if source is None:
        source = settings.camera_source

    if isinstance(source, int):
        return ("opencv", source)

    source_str = str(source).strip()
    if not source_str:
        return ("opencv", 0)

    # 1. Numeric camera index (e.g. "0", "1")
    if source_str.isdigit():
        return ("opencv", int(source_str))

    # 2. Named stream keywords
    if source_str.lower() in ("stream", "pi", "network"):
        return ("tcp", default_host, int(default_port))

    # 2b. Named webcam keywords
    if source_str.lower() in ("webcam", "laptop", "local"):
        return ("opencv", settings.camera_index)

    # 3. HOST:PORT stream address (e.g. "192.168.1.100:5001", "localhost:5001")
    if ":" in source_str:
        # Avoid treating Windows drive letter paths (e.g. "C:\video.mp4") as HOST:PORT
        if len(source_str) >= 2 and source_str[1] == ":" and source_str[0].isalpha():
            if len(source_str) == 2 or source_str[2] in ("\\", "/"):
                return ("opencv", source_str)

        if source_str.startswith("[") and "]:" in source_str:
            host = source_str[1:source_str.index("]:")].strip()
            port_str = source_str[source_str.index("]:") + 2:].strip()
        else:
            parts = source_str.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid camera stream address '{source}': expected HOST:PORT")
            host = parts[0].strip()
            port_str = parts[1].strip()

        if not host:
            raise ValueError(f"Invalid camera stream address '{source}': host cannot be empty")
        if not port_str:
            raise ValueError(f"Invalid camera stream address '{source}': port cannot be empty")

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid camera stream address '{source}': port must be an integer, got '{port_str}'")

        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid camera stream address '{source}': port {port} out of range (1-65535)")

        return ("tcp", host, port)

    # 4. Fallback for other paths/strings (e.g. video file "video.mp4")
    return ("opencv", source_str)


def open_camera_source(
    source: int | str | None,
    connect_timeout: float = 3.0,
) -> cv2.VideoCapture | FrameReceiver | None:
    """
    Open the camera source, either via OpenCV VideoCapture or Presento FrameReceiver.

    Returns the opened camera/receiver or None if opening/connection failed.
    """
    try:
        parsed = parse_camera_source(source)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return None

    source_type = parsed[0]

    if source_type == "opencv":
        cam_src = parsed[1]
        camera = cv2.VideoCapture(cam_src)

        if not camera.isOpened():
            print(f"[ERROR] Camera source '{cam_src}' could not be opened.", file=sys.stderr)
            if isinstance(cam_src, int):
                print(f"[ERROR] Could not open laptop webcam (index {cam_src}).", file=sys.stderr)
                print("Try: CAMERA_INDEX=1 (or check camera permissions / active apps)", file=sys.stderr)
            return None

        if isinstance(cam_src, int):
            cam_w = os.getenv("CAMERA_WIDTH")
            cam_h = os.getenv("CAMERA_HEIGHT")
            if cam_w and cam_h:
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_w))
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_h))
                except Exception:
                    pass
            try:
                camera.set(cv2.CAP_PROP_FPS, 30.0)
            except Exception:
                pass
        logger.info("Camera opened: source=%s", cam_src)
        return camera

    elif source_type == "tcp":
        host = str(parsed[1])
        port = int(parsed[2])
        logger.info("Connecting to Presento camera stream at %s:%d...", host, port)
        receiver = FrameReceiver(host=host, port=port)
        receiver.start()

        # Wait up to connect_timeout for the stream to deliver the first frame
        ok, frame = receiver.read(timeout=connect_timeout)
        if not ok or frame is None:
            receiver.stop()
            print(f"[ERROR] Could not connect to Presento camera stream at {host}:{port}", file=sys.stderr)
            return None

        logger.info("Connected to Presento camera stream at %s:%d", host, port)
        return receiver

    return None


def assess_enrollment_quality(
    faces: List[SCRFDFace],
    frame_or_shape: Union[np.ndarray, Tuple[int, ...]],
    min_conf: float = settings.enrollment_min_conf,
    min_size: int = settings.enrollment_min_face_size,
    blur_threshold: float = 35.0,
) -> Tuple[bool, str, Optional[SCRFDFace]]:
    """
    Validate that exactly one clear, sufficiently sized, non-blurry face with valid landmarks is present.
    """
    if len(faces) == 0:
        return False, "No face detected in frame", None
    if len(faces) > 1:
        return (
            False,
            f"Multiple faces ({len(faces)}) detected! Only 1 person allowed during enrollment",
            None,
        )

    face = faces[0]
    if face.confidence < min_conf:
        return (
            False,
            f"Face confidence too low ({face.confidence:.2f} < {min_conf:.2f})",
            None,
        )

    w = face.right - face.left
    h = face.bottom - face.top
    if w < min_size or h < min_size:
        return (
            False,
            f"Face too small ({w}x{h} < {min_size}px) - move closer to camera",
            None,
        )

    frame_shape = frame_or_shape.shape if isinstance(frame_or_shape, np.ndarray) else frame_or_shape
    frame_h, frame_w = frame_shape[:2]
    for pt in face.landmarks:
        if pt[0] < 0 or pt[0] >= frame_w or pt[1] < 0 or pt[1] >= frame_h:
            return False, "Facial landmarks out of camera frame boundaries", None

    # Blur quality check if actual frame array is provided
    if isinstance(frame_or_shape, np.ndarray) and frame_or_shape.ndim >= 2:
        top = max(0, min(frame_h, face.top))
        bottom = max(0, min(frame_h, face.bottom))
        left = max(0, min(frame_w, face.left))
        right = max(0, min(frame_w, face.right))
        crop = frame_or_shape[top:bottom, left:right]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if blur_var < blur_threshold:
                return (
                    False,
                    f"Face too blurry ({blur_var:.1f} < {blur_threshold:.1f}) - hold steady",
                    None,
                )

    return True, "Ready to capture", face


def run_enrollment(
    student_id: str,
    name: str,
    target_samples: int = settings.enrollment_samples_target,
    min_samples: int = settings.enrollment_samples_min,
    duplicate_threshold: float = settings.enrollment_duplicate_threshold,
    source_override: Optional[str] = None,
    connect_timeout: float = 3.0,
) -> bool:
    """
    Run the interactive camera capture workflow for face enrollment.
    """
    print("=" * 70)
    print("         Presento Phase 2B: Guided Face Enrollment Capture")
    print("=" * 70)
    print(f"  Student ID           : {student_id}")
    print(f"  Student Name         : {name}")
    print(f"  Target Samples       : {target_samples} (min required: {min_samples})")
    print(f"  Duplicate Threshold  : {duplicate_threshold:.4f} (enrollment diversity check)")
    print(f"  Target Storage       : {settings.face_enrollment_path}")
    print("=" * 70)

    # 1. Initialize camera source and on-demand session
    cam_source = source_override or settings.camera_source
    try:
        parsed = parse_camera_source(cam_source)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return False

    is_tcp = parsed[0] == "tcp"
    if is_tcp:
        host = str(parsed[1])
        stream_port = int(parsed[2])
        cam_session = CameraSession(
            host=host,
            stream_port=stream_port,
            session_type="enrollment",
        )
    else:
        cam_session = CameraSession(is_local_camera=True, session_type="enrollment")

    try:
        cam_session.start()
    except CameraInUseError as exc:
        print(f"\n[ERROR] Camera currently in use: {exc}", file=sys.stderr)
        print("        Another operation is using the camera. Please wait for it to finish.\n", file=sys.stderr)
        return False
    except CameraAuthError as exc:
        print(f"\n[ERROR] Camera authentication failed: {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"\n[ERROR] Could not activate camera session: {exc}", file=sys.stderr)
        return False

    camera = open_camera_source(cam_source, connect_timeout=connect_timeout)
    if camera is None:
        cam_session.stop()
        return False

    # 2. Initialize SCRFD detector & ArcFace embedder on OpenVINO GPU
    print("\n[Hardware Initialization]")
    detector = SCRFDDetector(
        model_path=settings.scrfd_model_path,
        device=settings.openvino_device,
        input_size=settings.scrfd_input_size,
        conf_thresh=settings.scrfd_conf_thresh,
        nms_thresh=settings.scrfd_nms_thresh,
    )
    embedder = ArcFaceEmbedder(
        model_path=settings.arcface_model_path,
        device=settings.openvino_device,
    )

    store = FaceEnrollmentStore(settings.face_enrollment_path)

    captured_embeddings: List[np.ndarray] = []
    captured_crops: List[np.ndarray] = []
    captured_conditions: List[str] = []

    feedback_msg = "Look at the camera and follow the instructions below."
    feedback_color = (0, 255, 0)
    feedback_time = time.monotonic()
    auto_capture = False
    last_auto_capture = 0.0

    print("\n[Controls]")
    print("  Press 'A'          : Toggle Auto-Capture mode (recommended)")
    print("  Press 'C' or SPACE : Manual single sample capture")
    print("  Press 'S'          : Save and finish early (if min samples met)")
    print("  Press 'Q' or ESC   : Cancel without saving")
    print("-" * 70)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            current_sample_idx = len(captured_embeddings)
            if current_sample_idx >= target_samples:
                print("\n[SUCCESS] All target samples captured successfully!")
                break

            # SCRFD detection on unmirrored original frame
            faces = detector.detect(frame)

            # Quality assessment
            is_quality_ok, quality_msg, valid_face = assess_enrollment_quality(faces, frame)

            # Mirrored display frame for user preview
            display_frame = cv2.flip(frame, 1)
            disp_h, disp_w = display_frame.shape[:2]

            # Draw face annotations if face detected
            if faces:
                for face in faces:
                    m_bbox = transform_mirrored_bbox(face.bbox, disp_w)
                    m_kps = transform_mirrored_landmarks(face.landmarks, disp_w)

                    box_color = (0, 255, 0) if is_quality_ok else (0, 0, 255)
                    cv2.rectangle(
                        display_frame,
                        (m_bbox[0], m_bbox[1]),
                        (m_bbox[2], m_bbox[3]),
                        box_color,
                        2,
                    )
                    for pt in m_kps:
                        cv2.circle(
                            display_frame,
                            (int(round(pt[0])), int(round(pt[1]))),
                            3,
                            (0, 0, 255),
                            -1,
                        )

            # Overlay information
            # 1. Header Banner
            cv2.rectangle(display_frame, (0, 0), (disp_w, 90), (25, 25, 25), -1)
            cv2.putText(
                display_frame,
                f"Presento Smart Enrollment 2.0 - {name} ({student_id})",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            mode_str = "Auto-Capture: ON" if auto_capture else "Manual Capture (Press 'A' for Auto)"
            mode_col = (0, 255, 255) if auto_capture else (180, 180, 180)
            cv2.putText(
                display_frame,
                f"Samples: {current_sample_idx}/{target_samples} | {mode_str}",
                (15, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                mode_col,
                1,
            )

            # Progress bar visualization
            pct = int((current_sample_idx / max(1, target_samples)) * 100)
            bar_w = 260
            bar_h = 12
            bx = 15
            by = 68
            cv2.rectangle(display_frame, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 60), -1)
            fill_w = int((current_sample_idx / max(1, target_samples)) * bar_w)
            cv2.rectangle(display_frame, (bx, by), (bx + fill_w, by + bar_h), (0, 220, 100), -1)
            cv2.rectangle(display_frame, (bx, by), (bx + bar_w, by + bar_h), (180, 180, 180), 1)
            cv2.putText(
                display_frame,
                f"Progress: {pct}% ({current_sample_idx}/{target_samples})",
                (bx + bar_w + 12, by + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # 2. Pose & Category Instruction Banner
            cond_idx = min(current_sample_idx, len(SMART_ENROLLMENT_CONDITIONS) - 1)
            current_cond = SMART_ENROLLMENT_CONDITIONS[cond_idx]
            pose_text = f"Phase {cond_idx + 1}/{len(SMART_ENROLLMENT_CONDITIONS)}: {current_cond['prompt']}"
            cv2.rectangle(display_frame, (0, disp_h - 90), (disp_w, disp_h), (25, 25, 25), -1)
            cv2.putText(
                display_frame,
                pose_text,
                (15, disp_h - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            # 3. Status & Feedback text
            if time.monotonic() - feedback_time > 2.5:
                feedback_msg = quality_msg
                feedback_color = (0, 255, 0) if is_quality_ok else (0, 165, 255)

            cv2.putText(
                display_frame,
                feedback_msg,
                (15, disp_h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                feedback_color,
                2,
            )

            # 4. Key instruction hint
            hint_text = "Press 'A' Auto | 'C' Capture | 'S' Save | 'Q' Quit"
            cv2.putText(
                display_frame,
                hint_text,
                (disp_w - 460, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Presento - Face Enrollment", display_frame)

            key = cv2.waitKey(1) & 0xFF
            now_m = time.monotonic()

            if key in (ord("q"), 27):  # 'q' or ESC
                print("\n[CANCELLED] Enrollment cancelled by user. No data saved.")
                return False

            elif key == ord("a"):  # toggle auto-capture
                auto_capture = not auto_capture
                feedback_msg = f"Auto-Capture {'ENABLED' if auto_capture else 'PAUSED'}"
                feedback_color = (0, 255, 255)
                feedback_time = now_m

            elif key == ord("s"):  # 's' to save early
                if len(captured_embeddings) >= min_samples:
                    print(
                        f"\n[INFO] Saving early with {len(captured_embeddings)} samples (>= min {min_samples})."
                    )
                    break
                else:
                    feedback_msg = f"Need at least {min_samples} samples to save! (Have {len(captured_embeddings)})"
                    feedback_color = (0, 0, 255)
                    feedback_time = now_m
                    print(f"  [!] {feedback_msg}")

            trigger_capture = False
            if key in (ord("c"), 32):  # 'c' or SPACE
                trigger_capture = True
            elif auto_capture and is_quality_ok and valid_face is not None and (now_m - last_auto_capture >= 0.18):
                trigger_capture = True

            if trigger_capture:
                if not is_quality_ok or valid_face is None:
                    if not auto_capture:
                        feedback_msg = f"Capture rejected: {quality_msg}"
                        feedback_color = (0, 0, 255)
                        feedback_time = now_m
                        print(f"  [X] {feedback_msg}")
                    continue

                # Crop and align face to standard 112x112 geometry
                aligned = align_face(frame, valid_face.landmarks)
                try:
                    emb = embedder.get_embedding(aligned)
                except Exception as exc:
                    feedback_msg = f"ArcFace extraction failed: {exc}"
                    feedback_color = (0, 0, 255)
                    feedback_time = now_m
                    print(f"  [X] {feedback_msg}")
                    continue

                # Full-history duplicate check against prior samples
                if captured_embeddings:
                    sims = [compute_cosine_similarity(emb, prev) for prev in captured_embeddings]
                    max_sim = max(sims)
                    if max_sim > duplicate_threshold:
                        if not auto_capture:
                            feedback_msg = f"Sample too similar to prior capture ({max_sim:.3f} > {duplicate_threshold:.2f}) - adjust pose"
                            feedback_color = (0, 165, 255)
                            feedback_time = now_m
                            print(f"  [!] {feedback_msg}")
                        continue

                # Accept sample
                captured_embeddings.append(emb)
                captured_crops.append(aligned)
                captured_conditions.append(current_cond["condition"])
                last_auto_capture = now_m
                sample_num = len(captured_embeddings)
                feedback_msg = f"Sample {sample_num}/{target_samples} Accepted!"
                feedback_color = (0, 255, 0)
                feedback_time = now_m
                if sample_num % 5 == 0 or not auto_capture:
                    print(f"  [OK] Sample {sample_num}/{target_samples} captured ({current_cond['condition']}).")

    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        if cam_session is not None:
            cam_session.stop(force=True)

    if len(captured_embeddings) < min_samples:
        print(f"\n[ERROR] Only {len(captured_embeddings)} samples captured (minimum {min_samples} required).")
        return False

    # Phase 3: Embedding Diversity Selection
    print("\n[Phase 3: Greedy Embedding Diversity Selection]")
    raw_count = len(captured_embeddings)
    representative_embeddings = select_diverse_embeddings(
        captured_embeddings,
        max_references=settings.enrollment_max_references,
        duplicate_threshold=duplicate_threshold,
        diversity_threshold=settings.enrollment_diversity_threshold,
    )
    print(f"  Raw captures collected   : {raw_count}")
    print(f"  Representative references: {len(representative_embeddings)} (selected via greedy angular coverage)")

    metadata = {
        "raw_count": raw_count,
        "accepted_count": len(representative_embeddings),
        "rejected_count": raw_count - len(representative_embeddings),
        "conditions_represented": list(set(captured_conditions)) if captured_conditions else ["diverse_conditions"],
        "v2_compatible": True,
    }

    # Save student in FaceEnrollmentStore
    print("\n[Saving Enriched Enrollment]")
    store.add_student(
        student_id=student_id,
        name=name,
        embeddings=representative_embeddings,
        metadata=metadata,
    )
    saved_path = store.save()
    print(f"  Enrolled {name} ({student_id}) with {len(representative_embeddings)} representative embeddings into {saved_path}")

    # Phase 5: Train/Update Presento Identity Classifier
    try:
        from .identity_classifier import IdentityClassifier
        clf = IdentityClassifier()
        clf.train(store)
        clf.save()
        print("  [OK] Specialized closed-set Identity Classifier updated and saved.")
    except Exception as exc:
        logger.warning("Could not auto-train identity classifier: %s", exc)

    # Compute and display sample diversity matrix
    print("\n[Enrollment Sample Diversity Analysis]")
    n_samples = len(representative_embeddings)
    sim_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_samples):
            sim_matrix[i, j] = compute_cosine_similarity(
                representative_embeddings[i], representative_embeddings[j]
            )

    print("      " + "".join([f"S{k+1:>7}" for k in range(n_samples)]))
    for i in range(n_samples):
        row_str = f"S{i+1:<4}: " + "".join([f"{sim_matrix[i, j]:7.4f}" for j in range(n_samples)])
        print(row_str)

    # Calculate average inter-sample similarity
    triu_indices = np.triu_indices(n_samples, k=1)
    if len(triu_indices[0]) > 0:
        avg_inter_sim = float(np.mean(sim_matrix[triu_indices]))
        min_inter_sim = float(np.min(sim_matrix[triu_indices]))
        max_inter_sim = float(np.max(sim_matrix[triu_indices]))
        print(f"\n  Average Sample Pairwise Similarity: {avg_inter_sim:.4f}")
        print(f"  Minimum Pairwise Similarity       : {min_inter_sim:.4f}")
        print(f"  Maximum Pairwise Similarity       : {max_inter_sim:.4f}")

    print("=" * 70)
    print(f"[DONE] Student {name} ({student_id}) is successfully enrolled with V2 smart references!")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Presento Face Enrollment")
    parser.add_argument("--id", type=str, default=None, help="Student ID (e.g. DEMO001)")
    parser.add_argument("--name", type=str, default=None, help="Student Name (e.g. Demo Student)")
    parser.add_argument("--samples", type=int, default=settings.enrollment_samples_target, help="Target samples (default: 5)")
    parser.add_argument("--dup-threshold", type=float, default=settings.enrollment_duplicate_threshold, help="Duplicate similarity rejection threshold")
    parser.add_argument("--source", type=str, default=None, help="Camera source override (e.g. 0, stream, or HOST:PORT such as 192.168.1.100:5001)")
    args = parser.parse_args()

    student_id = args.id
    name = args.name

    if not student_id:
        try:
            student_id = input("Enter Student ID (e.g. DEMO001): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)

    if not name:
        try:
            name = input("Enter Student Name (e.g. Demo Student): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            sys.exit(0)

    if not student_id or not name:
        print("[ERROR] Both Student ID and Name are required.", file=sys.stderr)
        sys.exit(1)

    success = run_enrollment(
        student_id=student_id,
        name=name,
        target_samples=args.samples,
        duplicate_threshold=args.dup_threshold,
        source_override=args.source,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
