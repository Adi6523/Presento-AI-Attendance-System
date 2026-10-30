from __future__ import annotations
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
import cv2
import numpy as np
from .api import flush_queue, get_active_session, get_people, get_people_for_class, post_event
from .config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("presento.edge")

EAR_CLOSED = 0.20

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    logger.info("Shutdown signal received (signal %d) — finishing current frame...", signum)
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def motion_sensor():
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM); GPIO.setup(settings.pir_pin, GPIO.IN)
        logger.info("PIR motion sensor initialized on GPIO %d", settings.pir_pin)
        return lambda: bool(GPIO.input(settings.pir_pin))
    except (ImportError, RuntimeError):
        logger.info("PIR unavailable; using continuous camera scanning")
        return lambda: True

def beep():
    try:
        import RPi.GPIO as GPIO
        GPIO.setup(settings.buzzer_pin, GPIO.OUT); GPIO.output(settings.buzzer_pin, 1); time.sleep(.12); GPIO.output(settings.buzzer_pin, 0)
    except (ImportError, RuntimeError): pass


def _load_people() -> list[dict]:
    """Load enrolled students from API (class-scoped if configured)."""
    try:
        if settings.class_id:
            logger.info("Loading students for class %s...", settings.class_id)
            people = get_people_for_class(settings.class_id)
        else:
            logger.info("Loading all enrolled students...")
            people = get_people()
        logger.info("Loaded %d enrolled students", len(people))
        return people
    except Exception as exc:
        logger.warning("Could not load enrolled students from API (%s). Proceeding with empty student list for face detection.", exc)
        return []


def _run_phase3_recognition_pipeline(camera):
    """
    Preserved Phase 3+ recognition pipeline (embeddings, liveness, attendance).
    Kept intact for when Phase 3 is started.
    """
    import face_recognition
    from .vision import eye_aspect_ratio_for_face, match_face_encoding, assess_face_quality
    from .liveness import LivenessDetector
    from .storage import EmbeddingCache
    import json as _json

    people = _load_people()
    embedding_cache = EmbeddingCache(max_size=settings.cache_size, default_ttl=settings.cache_ttl)
    for person in people:
        if person.get("embedding") and person["embedding"] != "[]":
            try:
                embedding_data = person["embedding"]
                if isinstance(embedding_data, str):
                    embedding = np.array(_json.loads(embedding_data), dtype=np.float64)
                else:
                    embedding = np.array(embedding_data, dtype=np.float64)
                embedding_cache.put(person["id"], embedding)
            except (_json.JSONDecodeError, ValueError, TypeError):
                pass

    face_tracker = None
    use_tracking = settings.use_tracking
    if use_tracking:
        try:
            face_tracker = cv2.TrackerKCF_create()
            tracking_initialized = False
            tracked_bbox = None
        except AttributeError:
            logger.info("KCF tracker not available, falling back to detection-only mode")
            use_tracking = False

    has_motion, seen_at, cooldown = motion_sensor(), {}, {}
    blink_state, blink_count, candidate_id = False, 0, None
    liveness_detectors: dict[str, LivenessDetector] = {}

    def _get_liveness_detector(student_id: str) -> LivenessDetector:
        if student_id not in liveness_detectors:
            liveness_detectors[student_id] = LivenessDetector(
                blink_threshold=settings.blink_threshold,
                min_blinks_required=settings.min_blinks_required,
                analysis_window=settings.liveness_seconds
            )
        return liveness_detectors[student_id]

    frame_counter = 0
    last_recognition_start = time.monotonic()
    last_embedding_refresh = time.monotonic()
    last_session_check = 0.0
    current_session = None

    logger.info("Phase 3 recognition pipeline initialized.")
    while not _shutdown_requested:
        flush_queue()
        now_mono = time.monotonic()
        if now_mono - last_session_check > 60:
            try:
                current_session = get_active_session()
            except Exception:
                pass
            last_session_check = now_mono

        if not has_motion():
            time.sleep(.1)
            continue

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1
        if frame_counter % settings.frame_skip != 0:
            if not settings.headless:
                cv2.imshow("Presento - Press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb, model="hog")
        now = time.monotonic()

        for face_location in face_locations:
            last_recognition_start = time.monotonic()
            top, right, bottom, left = face_location
            quality = assess_face_quality(frame, face_location)
            if quality.score < 0.3:
                if not settings.headless:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
                    cv2.putText(frame, "Low Quality", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            encodings = face_recognition.face_encodings(rgb, [face_location])
            if not encodings:
                continue
            encoding = encodings[0]
            candidate = match_face_encoding(encoding, face_location, people, settings.tolerance)
            if not candidate or cooldown.get(candidate.student_id, 0) > now:
                if not settings.headless:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 1)
                    cv2.putText(frame, "No Match/Cooldown", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                continue

            if candidate_id != candidate.student_id:
                candidate_id = candidate.student_id
                blink_count = 0
                blink_state = False
                seen_at[candidate.student_id] = now

            liveness = _get_liveness_detector(candidate.student_id)
            ear_value = eye_aspect_ratio_for_face(frame, face_location)
            liveness_result = liveness.process_frame(ear_value)

            if not settings.headless:
                color = (0, 255, 0) if liveness_result.is_live else (0, 165, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, candidate.name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if (liveness_result.is_live and
                liveness_result.blink_count >= settings.min_blinks_required and
                now - seen_at[candidate.student_id] <= settings.liveness_seconds):
                event = {
                    "event_id": str(uuid.uuid4()),
                    "student_id": candidate.student_id,
                    "device_id": settings.device_id,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "confidence": round(candidate.confidence, 3),
                    "liveness_score": liveness_result.score,
                    "face_quality_score": quality.score,
                    "room": settings.room,
                }
                online = post_event(event)
                logger.info("%s: recorded (%s)", candidate.name, "synced" if online else "queued")
                beep()
                cooldown[candidate.student_id] = now + settings.cooldown_seconds
                blink_count = 0
                liveness.reset()

        if not settings.headless:
            cv2.imshow("Presento - Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main():
    """
    Presento Phase 2A: Decoupled Live Video + SCRFD Face Detection + ArcFace Embedding Pipeline.
    Runs the 1280x720 mirrored video display at camera rate (~30 FPS),
    SCRFD face detection at ~30 FPS on unmirrored frames, and ArcFace feature extraction
    at ~10-15 FPS per face on native OpenVINO GPU (Intel Iris Xe).
    """
    global _shutdown_requested

    from .scrfd_detector import SCRFDWorker, SCRFDFace
    from .arcface_embedder import (
        transform_mirrored_bbox,
        transform_mirrored_landmarks,
        compute_cosine_similarity,
    )
    from .face_enrollment import FaceEnrollmentStore
    from .face_matcher import FaceMatcher
    from .identity_engine import PresentoIdentityEngine
    from .anti_spoof import AntiSpoofClassifier, crop_face
    from .minifas_antispoof import MiniFASNetV2SEClassifier, crop_face_minifas
    from .anti_spoof_fusion import AntiSpoofFusionEngine, FusionResult
    from .anti_spoof_worker import AntiSpoofWorker, AntiSpoofResult
    from .attendance_worker import AttendanceIOWorker
    from .api import get_active_session
    from .camera_control import (
        CameraSession,
        CameraInUseError,
        CameraAuthError,
        CameraControlError,
    )
    from collections import deque

    # 1. Initialize camera source and on-demand session
    is_stream = str(settings.camera_source).lower() in ("stream", "pi", "network")
    cam_session = None
    if is_stream:
        cam_session = CameraSession(
            host=settings.pi_ip,
            stream_port=settings.stream_port,
            session_type="attendance",
        )
    else:
        cam_session = CameraSession(is_local_camera=True, session_type="attendance")

    try:
        cam_session.start()
    except CameraInUseError as exc:
        logger.error("Camera currently in use: %s. Attendance runner cannot start.", exc)
        return
    except CameraAuthError as exc:
        logger.error("Camera control authentication failed: %s. Check PRESENTO_CONTROL_TOKEN.", exc)
        return
    except Exception as exc:
        logger.error("Could not activate camera session: %s", exc)
        return

    camera = None
    if is_stream:
        from .frame_receiver import FrameReceiver
        logger.info("Connecting to Raspberry Pi camera stream at %s:%d...", settings.pi_ip, settings.stream_port)
        camera = FrameReceiver(host=settings.pi_ip, port=settings.stream_port)
        camera.start()
    else:
        try:
            cam_src: int | str = int(settings.camera_source)
        except ValueError:
            cam_src = settings.camera_source
        camera = cv2.VideoCapture(cam_src)
        if not camera.isOpened():
            if cam_session is not None:
                cam_session.stop()
            raise SystemExit(f"Camera source '{cam_src}' could not be opened")
        logger.info("Camera opened: source=%s", cam_src)

    # 2. Initialize modern SCRFD detector & ArcFace embedder on OpenVINO GPU (Intel Iris Xe)
    detector = SCRFDWorker(
        model_path=settings.scrfd_model_path,
        device=settings.openvino_device,
        input_size=settings.scrfd_input_size,
        conf_thresh=settings.scrfd_conf_thresh,
        nms_thresh=settings.scrfd_nms_thresh,
        target_fps=settings.scrfd_target_fps,
        persistence_seconds=settings.box_persistence_seconds,
        smoothing_factor=settings.box_smoothing_factor,
        arcface_enabled=settings.arcface_enabled,
        arcface_target_fps=settings.arcface_target_fps,
    )
    detector.start()

    # 3. Initialize face enrollment store & face matcher / specialized identity engine
    enrollment_store = FaceEnrollmentStore(settings.face_enrollment_path)
    if settings.identity_engine_mode != "baseline":
        matcher = PresentoIdentityEngine(enrollment_store, mode=settings.identity_engine_mode)
    else:
        matcher = FaceMatcher(enrollment_store)
    enrolled_count = len(enrollment_store.get_all_students())

    # 4. Initialize asynchronous dual anti-spoof worker (Anti-Spoof-MN3 + MiniFASNet V2 SE)
    # Offloads all anti-spoof inference to a background thread to prevent display loop jitter
    as_worker = AntiSpoofWorker(
        mn3_model_path=settings.anti_spoof_model_path,
        minifas_model_path=settings.minifas_model_path,
        window_size=15,
        real_ratio_threshold=0.80,
        spoof_ratio_threshold=0.80,
    )
    as_worker.start()

    # 5. Initialize asynchronous attendance I/O worker
    # Completely offloads post_event() and flush_queue() network I/O to background threads
    attendance_worker = AttendanceIOWorker(flush_interval=10.0)
    attendance_worker.start()

    # Attendance cooldown tracker: student_id -> next allowed monotonic timestamp
    cooldown: dict[str, float] = {}
    active_session_id: Optional[int] = None
    last_session_query_time = 0.0

    has_motion = motion_sensor()

    frame_count = 0
    frame_timestamps: deque[float] = deque(maxlen=30)
    display_fps = 0.0

    stats = detector.get_stats()
    logger.info(
        "Presento Active Runner (Phase 2B + Async Dual Anti-Spoof) started (device=%s, enrolled=%d, headless=%s)",
        stats.get("device_name"),
        enrolled_count,
        settings.headless,
    )

    try:
        while not _shutdown_requested:
            if not has_motion():
                time.sleep(0.1)
                continue

            try:
                ok, frame = camera.read(timeout=0.035)
            except TypeError:
                ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            # Submit unmirrored original frame to the AI pipeline (SCRFD + ArcFace)
            detector.submit_frame(frame)

            # Retrieve latest smoothed face detections with strictly associated embeddings
            faces: list[SCRFDFace] = detector.get_latest_detections()

            # Continuous rolling Display FPS calculation (monotonic clock over last 30 frames)
            frame_count += 1
            now = time.monotonic()
            frame_timestamps.append(now)
            if len(frame_timestamps) >= 2:
                dt = frame_timestamps[-1] - frame_timestamps[0]
                if dt > 0:
                    display_fps = round((len(frame_timestamps) - 1) / dt, 1)

            stats = detector.get_stats()
            det_fps = stats.get("detection_fps", 0.0)
            arc_fps = stats.get("arcface_fps", 0.0)
            arc_lat = stats.get("arcface_latency_ms", 0.0)
            device_str = f"OpenVINO {stats.get('device', 'GPU')}"

            # 1. Match candidates for active faces
            face_matches: dict[int, any] = {}
            for face in faces:
                if face.embedding is not None:
                    match_res = matcher.match(
                        face.embedding,
                        face_id=face.face_id,
                        frame_id=frame_count,
                    )
                    face_matches[face.face_id] = match_res

            # 2. Asynchronously submit face crops to AntiSpoofWorker (non-blocking)
            active_fids = {f.face_id for f in faces}
            for face in faces:
                as_worker.submit_face(face.face_id, frame_count, frame, face.bbox)
            as_worker.prune_stale(active_fids, max_idle_seconds=3.0)

            # Retrieve latest completed anti-spoof results per face (non-blocking)
            face_anti_spoof: dict[int, AntiSpoofResult] = {}
            for face in faces:
                as_res = as_worker.get_result(face.face_id)
                if as_res is not None:
                    face_anti_spoof[face.face_id] = as_res

            # 3. Attendance Gating:
            # Attendance requires: valid matched identity + stability + anti-spoof temporal_label == REAL + freshness + cooldown
            # SPOOF and UNCERTAIN must never create attendance
            for face in faces:
                fid = face.face_id
                m = face_matches.get(fid)
                as_res = face_anti_spoof.get(fid)
                if m is not None and m.best_match is not None and as_res is not None:
                    fus_res = as_res.fusion_result
                    cand = m.best_match
                    is_fresh = (now - as_res.updated_at) <= 2.0

                    # CLOSED-SET / ENROLLED-ONLY GATE:
                    # Face must pass configured recognition acceptance criteria (similarity >= settings.tolerance)
                    if cand.similarity < settings.tolerance or not getattr(m, "is_recognized", True):
                        continue  # UNKNOWN face - BLOCKED from attendance

                    is_live = (fus_res.temporal_label == "REAL")
                    if getattr(as_res, "liveness_decision", None) is not None:
                        is_live = as_res.liveness_decision.is_live

                    if m.is_stable and is_live and is_fresh:
                        if cooldown.get(cand.student_id, 0.0) <= now:
                            # Dynamically discover active session from backend
                            if active_session_id is None or (now - last_session_query_time) > 30.0:
                                last_session_query_time = now
                                sess_info = get_active_session()
                                if sess_info:
                                    active_session_id = sess_info.get("id")

                            # Explicit temporal liveness state
                            l_state = "REAL_LOCKED"
                            if getattr(as_res, "liveness_decision", None) is not None:
                                l_state = as_res.liveness_decision.state.value

                            event = {
                                "event_id": str(uuid.uuid4()),
                                "student_id": cand.student_id,
                                "session_id": active_session_id or 1,
                                "confidence": round(float(cand.similarity), 3),
                                "liveness_state": l_state,
                                "liveness_score": round(float(fus_res.avg_p_real), 3),
                                "source": "AI_CAMERA",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "device_id": settings.device_id,
                                "observed_at": datetime.now(timezone.utc).isoformat(),
                                "face_quality_score": round(float(face.confidence), 3),
                                "room": settings.room,
                            }
                            cooldown[cand.student_id] = now + settings.cooldown_seconds
                            queued = attendance_worker.submit_event(event)
                            if queued:
                                logger.info(
                                    "Attendance recorded: %s (%s) [Sim: %.3f, Liveness: %s (%.0f%%), Session: %s, Status: queued for background post]",
                                    cand.name,
                                    cand.student_id,
                                    cand.similarity,
                                    l_state,
                                    fus_res.both_real_ratio * 100.0,
                                    active_session_id or "default",
                                )
                                beep()
                            else:
                                logger.warning("Attendance event queue full or stopped for %s", cand.student_id)

            # Periodically flush queued offline attendance events (non-blocking background thread)
            attendance_worker.maybe_trigger_flush()

            # Display annotations on horizontally mirrored frame (if not headless)
            if not settings.headless:
                # 1. Flip frame horizontally for natural mirror behavior
                display_frame = cv2.flip(frame, 1)
                disp_h, disp_w = display_frame.shape[:2]

                # 2. Draw faces with transformed mirrored coordinates and candidate + anti-spoof labels
                for face in faces:
                    m_left, m_top, m_right, m_bottom = transform_mirrored_bbox(face.bbox, disp_w)
                    m_landmarks = transform_mirrored_landmarks(face.landmarks, disp_w)

                    as_res = face_anti_spoof.get(face.face_id)
                    match_res = face_matches.get(face.face_id)

                    if as_res is not None:
                        fus_res = as_res.fusion_result
                        if fus_res.temporal_label == "REAL":
                            as_text = f"AntiSpoof: REAL ({fus_res.both_real_ratio * 100:.0f}%)"
                            as_color = (0, 255, 0)
                        elif fus_res.temporal_label == "SPOOF":
                            as_text = f"AntiSpoof: SPOOF ({fus_res.both_spoof_ratio * 100:.0f}%)"
                            as_color = (0, 0, 255)
                        else:
                            as_text = f"AntiSpoof: CHECKING ({fus_res.window_frames}/15)"
                            as_color = (0, 215, 255)
                        scores_text = f"MN3: {as_res.mn3_p_real:.2f} | MiniFAS: {as_res.mf_p_real:.2f}"
                    else:
                        as_text = "AntiSpoof: CHECKING"
                        as_color = (0, 215, 255)
                        scores_text = "MN3: N/A | MiniFAS: N/A"

                    if match_res is not None and match_res.best_match is not None:
                        cand = match_res.best_match
                        is_recognized = cand.similarity >= settings.tolerance and getattr(match_res, "is_recognized", True)
                        is_verified_real = (as_res is not None and as_res.fusion_result.temporal_label == "REAL")
                        is_spoof = (as_res is not None and as_res.fusion_result.temporal_label == "SPOOF")

                        if is_spoof:
                            box_color = (0, 0, 255)
                        elif is_recognized and match_res.is_stable and is_verified_real:
                            box_color = (0, 255, 0)
                        elif is_recognized:
                            box_color = (255, 255, 0)
                        else:
                            box_color = (130, 130, 150)  # Neutral gray for UNKNOWN

                        if is_recognized:
                            label_line1 = f"Face #{face.face_id} | Identity: {cand.name}"
                            label_line2 = (
                                f"ID: {cand.student_id} | "
                                f"Sim: {cand.similarity:.3f} | "
                                f"Margin: {match_res.margin:.3f}"
                            )
                            status_text = f"[{match_res.stability_label}]"
                        else:
                            label_line1 = f"Face #{face.face_id} | Identity: UNKNOWN"
                            label_line2 = (
                                f"Best: {cand.name} | "
                                f"Sim: {cand.similarity:.2f} < {settings.tolerance:.2f} (FAILED)"
                            )
                            status_text = "[UNKNOWN - No enrolled match]"
                    elif match_res is not None:
                        # No enrolled students in store
                        box_color = (0, 215, 255)
                        label_line1 = f"Face #{face.face_id} | Candidate: Unknown"
                        label_line2 = "[No enrolled students in store]"
                        status_text = ""
                    else:
                        # Embedding still extracting
                        box_color = (0, 215, 255)
                        label_line1 = f"Face #{face.face_id} ({face.confidence:.2f})"
                        label_line2 = "[Extracting 512-D embedding...]"
                        status_text = ""

                    # Bounding box
                    cv2.rectangle(display_frame, (m_left, m_top), (m_right, m_bottom), box_color, 2)

                    # 5 facial landmarks
                    for pt in m_landmarks:
                        px, py = int(round(pt[0])), int(round(pt[1]))
                        cv2.circle(display_frame, (px, py), 3, (0, 0, 255), -1)

                    # Text overlays above box
                    top_y = max(40, m_top - 24)
                    cv2.putText(
                        display_frame,
                        label_line1,
                        (m_left, top_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        label_line2,
                        (m_left, top_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

                    # Text overlays below box
                    y_sub = m_bottom + 18
                    if status_text:
                        cv2.putText(
                            display_frame,
                            status_text,
                            (m_left, y_sub),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.50,
                            box_color,
                            2,
                        )
                        y_sub += 18

                    cv2.putText(
                        display_frame,
                        as_text,
                        (m_left, y_sub),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        as_color,
                        2,
                    )
                    y_sub += 16
                    cv2.putText(
                        display_frame,
                        scores_text,
                        (m_left, y_sub),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                    )

                # 3. Diagnostic HUD overlay
                enrolled_current = len(enrollment_store.get_all_students())
                as_stats = as_worker.get_stats()
                as_lat = as_stats.get("latest_latency_ms", 0.0)
                hud_line1 = (
                    f"Pi Stream: 1280x720 | Display: {display_fps:.1f} FPS | SCRFD: {det_fps:.1f} FPS"
                )
                hud_line2 = (
                    f"ArcFace: {arc_fps:.1f} FPS ({arc_lat:.1f}ms) | AntiSpoof: MN3+MiniFAS GPU ({as_lat:.1f}ms) | Enrolled: {enrolled_current}"
                )
                if frame_count % 60 == 0:
                    logger.info("Live HUD (Frame #%d): Display=%.1f FPS | SCRFD=%.1f FPS | ArcFace=%.1f FPS | Enrolled=%d", frame_count, display_fps, det_fps, arc_fps, enrolled_current)

                cv2.putText(
                    display_frame,
                    hud_line1,
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    display_frame,
                    hud_line2,
                    (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                hint_msg = "Press 'D' to log match & anti-spoof diagnostic | 'Q' to quit"
                cv2.putText(
                    display_frame,
                    hint_msg,
                    (15, disp_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow("Presento - Live Face Detection, Identity & Anti-Spoof", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

                max_frames = int(os.getenv("PRESENTO_MAX_FRAMES", "0"))
                if max_frames > 0 and frame_count >= max_frames:
                    logger.info("Reached PRESENTO_MAX_FRAMES (%d) — exiting cleanly.", max_frames)
                    break
                elif key in (ord("d"), ord("c")):
                    # Log structured match & anti-spoof diagnostic to console
                    print("\n" + "=" * 65)
                    print(f"       Presento Identity & Anti-Spoof Diagnostic (Frame #{frame_count})")
                    print("=" * 65)
                    if not faces:
                        print("  [!] No faces detected in current frame.")
                    else:
                        for f in faces:
                            fid = f.face_id
                            m = face_matches.get(fid)
                            as_res = face_anti_spoof.get(fid)
                            print(f"\n  [Face #{fid}]")
                            if m is not None and m.best_match is not None:
                                print(f"    Candidate     : {m.best_match.name} ({m.best_match.student_id})")
                                print(f"    Best Sim      : {m.best_match.similarity:.4f} (sample #{m.best_match.best_sample_index + 1})")
                                if m.second_best_match is not None:
                                    print(f"    Second Best   : {m.second_best_match.name} ({m.second_best_match.student_id}) -> {m.second_best_match.similarity:.4f}")
                                    print(f"    Margin        : {m.margin:.4f}")
                                else:
                                    print("    Second Best   : None (only 1 student enrolled)")
                                    print(f"    Margin        : {m.margin:.4f}")
                                print(f"    Status        : {m.stability_label}")
                                print("    All Candidates: " + ", ".join([f"{c.name}: {c.similarity:.3f}" for c in m.all_candidates]))
                            elif m is not None:
                                print("    Candidate     : Unknown (No enrolled students)")
                            else:
                                print("    Candidate     : [Extracting 512-D embedding...]")

                            # Anti-Spoof diagnostics
                            if as_res is not None:
                                fus_res = as_res.fusion_result
                                print(f"    AntiSpoof     : {fus_res.temporal_label} (Consensus: {fus_res.both_real_ratio * 100:.1f}% REAL, {fus_res.both_spoof_ratio * 100:.1f}% SPOOF)")
                                print(f"    MN3 (CelebA)  : p_real={as_res.mn3_p_real:.3f}, p_spoof={as_res.mn3_p_spoof:.3f} [{as_res.mn3_latency_ms:.1f} ms]")
                                print(f"    MiniFASNet V2 : p_real={as_res.mf_p_real:.3f}, p_spoof={as_res.mf_p_spoof:.3f} [{as_res.mf_latency_ms:.1f} ms]")
                                print(f"    Fusion Window : {fus_res.window_frames}/15 frames | Avg p_real: {fus_res.avg_p_real:.3f}")
                    print("=" * 65 + "\n")

    finally:
        logger.info("Shutting down edge runner...")
        if "attendance_worker" in locals() and attendance_worker is not None:
            stats = attendance_worker.get_stats()
            logger.info("Attendance I/O summary: posted=%d, failed=%d, flushed=%d, active_flush=%s", stats["total_posted"], stats["total_failed"], stats["total_flushed"], stats["is_flushing"])
            attendance_worker.stop()
        if "as_worker" in locals() and as_worker is not None:
            as_worker.stop()
        if "detector" in locals() and detector is not None:
            detector.stop()
        if camera is not None:
            camera.release()
        if not settings.headless:
            cv2.destroyAllWindows()
        if cam_session is not None:
            cam_session.stop()
        logger.info("Edge runner stopped.")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        from .demo import main as demo_main
        demo_main()
    else:
        main()
