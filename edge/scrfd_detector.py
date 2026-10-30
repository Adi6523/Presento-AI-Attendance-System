from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
import time
from typing import List, Tuple
import cv2
import numpy as np
import openvino as ov

from .config import settings
from .arcface_embedder import ArcFaceWorker, align_face

logger = logging.getLogger("presento.edge.scrfd")


@dataclass
class SCRFDFace:
    """Structure for a detected face with bounding box, confidence, landmarks, and ArcFace embedding."""
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)
    confidence: float
    landmarks: np.ndarray  # shape (5, 2), float coordinates on original frame
    face_id: int = -1  # persistent track ID
    frame_id: int = 0  # frame counter when detected
    embedding: np.ndarray | None = None  # 512-D L2-normalized ArcFace embedding
    embedding_frame_id: int = 0  # frame_id when this embedding was computed
    embedding_time: float = 0.0  # monotonic timestamp when embedding was computed

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def left(self) -> int:
        return self.bbox[0]


class SCRFDDetector:
    """
    Native OpenVINO SCRFD face detector.
    Extracts high-accuracy bounding boxes and 5 facial landmarks on Intel GPU/CPU.
    """

    def __init__(
        self,
        model_path: str = settings.scrfd_model_path,
        device: str = settings.openvino_device,
        input_size: int = settings.scrfd_input_size,
        conf_thresh: float = settings.scrfd_conf_thresh,
        nms_thresh: float = settings.scrfd_nms_thresh,
    ):
        self.model_path = model_path
        self.requested_device = device.upper()
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.strides = (8, 16, 32)
        self.num_anchors = 2
        self._anchor_cache: dict[tuple[int, int], dict[int, np.ndarray]] = {}

        # Initialize OpenVINO Core and verify hardware
        self.core = ov.Core()
        available_devices = [d.upper() for d in self.core.available_devices]
        logger.info("OpenVINO Core initialized. Available hardware devices: %s", available_devices)

        # Explicit verification: Do NOT silently claim GPU while running CPU
        if self.requested_device == "GPU":
            if "GPU" not in available_devices:
                raise RuntimeError(
                    f"OpenVINO GPU device requested but 'GPU' not found in available devices: {available_devices}"
                )
            self.active_device = "GPU"
        elif self.requested_device == "CPU":
            self.active_device = "CPU"
        else:
            if self.requested_device not in available_devices:
                raise RuntimeError(
                    f"Requested OpenVINO device '{self.requested_device}' is unavailable: {available_devices}"
                )
            self.active_device = self.requested_device

        # Get full device descriptor name
        try:
            self.full_device_name = self.core.get_property(self.active_device, "FULL_DEVICE_NAME")
        except Exception:
            self.full_device_name = self.active_device

        logger.info(
            "Compiling SCRFD model '%s' on verified OpenVINO device: %s (%s)...",
            self.model_path,
            self.active_device,
            self.full_device_name,
        )
        t0 = time.perf_counter()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SCRFD model file not found at: {self.model_path}")

        ov_model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(ov_model, self.active_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = self.compiled_model.inputs[0].any_name
        self.output_names = [out.any_name for out in self.compiled_model.outputs]
        load_ms = (time.perf_counter() - t0) * 1000.0

        logger.info("SCRFD compiled successfully on %s in %.1f ms.", self.full_device_name, load_ms)
        print(f"[SCRFD] Active OpenVINO Device: {self.active_device} -> {self.full_device_name}", flush=True)

    def _get_anchors(self, height: int, width: int) -> dict[int, np.ndarray]:
        key = (height, width)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        anchors_dict = {}
        for stride in self.strides:
            feat_h = height // stride
            feat_w = width // stride
            grid_y, grid_x = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
            anchor_centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) * stride
            anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0).astype(np.float32)
            anchors_dict[stride] = anchor_centers

        self._anchor_cache[key] = anchors_dict
        return anchors_dict

    def detect(self, frame: np.ndarray) -> list[SCRFDFace]:
        """
        Run SCRFD detection on a BGR image frame.
        Returns a list of SCRFDFace objects with bounding box and 5 landmarks scaled to the input frame.
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.input_size, self.input_size

        # Preprocess: resize to 640x640, normalize (img - 127.5) / 128.0, transpose to NCHW
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale_x = orig_w / float(target_w)
        scale_y = orig_h / float(target_h)

        blob = (resized.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # Inference
        res = self.infer_request.infer({self.input_tensor_name: blob})
        outputs = [res[out] for out in self.compiled_model.outputs]

        # SCRFD 500M 9 outputs:
        # 0, 1, 2: scores for strides 8, 16, 32
        # 3, 4, 5: bboxes for strides 8, 16, 32
        # 6, 7, 8: 5 landmarks for strides 8, 16, 32
        scores_list = [outputs[0], outputs[1], outputs[2]]
        bboxes_list = [outputs[3], outputs[4], outputs[5]]
        kpss_list = [outputs[6], outputs[7], outputs[8]]

        anchors = self._get_anchors(target_h, target_w)
        all_boxes, all_scores, all_kpss = [], [], []

        for idx, stride in enumerate(self.strides):
            scores = scores_list[idx].reshape(-1)
            bboxes = bboxes_list[idx].reshape(-1, 4)
            kpss = kpss_list[idx].reshape(-1, 10)
            anchor_pts = anchors[stride]

            valid_mask = scores >= self.conf_thresh
            if not np.any(valid_mask):
                continue

            valid_scores = scores[valid_mask]
            valid_bboxes = bboxes[valid_mask] * stride
            valid_kpss = kpss[valid_mask] * stride
            valid_anchors = anchor_pts[valid_mask]

            x1 = valid_anchors[:, 0] - valid_bboxes[:, 0]
            y1 = valid_anchors[:, 1] - valid_bboxes[:, 1]
            x2 = valid_anchors[:, 0] + valid_bboxes[:, 2]
            y2 = valid_anchors[:, 1] + valid_bboxes[:, 3]
            decoded_boxes = np.column_stack([x1, y1, x2, y2])

            decoded_kps = np.zeros_like(valid_kpss)
            for k in range(5):
                decoded_kps[:, 2 * k] = valid_anchors[:, 0] + valid_kpss[:, 2 * k]
                decoded_kps[:, 2 * k + 1] = valid_anchors[:, 1] + valid_kpss[:, 2 * k + 1]

            all_boxes.append(decoded_boxes)
            all_scores.append(valid_scores)
            all_kpss.append(decoded_kps)

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        kpss = np.vstack(all_kpss).reshape(-1, 5, 2)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores=[float(s) for s in scores],
            score_threshold=self.conf_thresh,
            nms_threshold=self.nms_thresh,
        )

        if len(indices) == 0:
            return []

        keep = indices.flatten()
        final_faces: list[SCRFDFace] = []

        for k in keep:
            b = boxes[k]
            # Scale coordinates back to original frame
            x1 = max(0, min(orig_w, int(round(b[0] * scale_x))))
            y1 = max(0, min(orig_h, int(round(b[1] * scale_y))))
            x2 = max(0, min(orig_w, int(round(b[2] * scale_x))))
            y2 = max(0, min(orig_h, int(round(b[3] * scale_y))))

            scaled_kps = kpss[k].copy()
            scaled_kps[:, 0] = np.clip(scaled_kps[:, 0] * scale_x, 0, orig_w)
            scaled_kps[:, 1] = np.clip(scaled_kps[:, 1] * scale_y, 0, orig_h)

            final_faces.append(
                SCRFDFace(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(scores[k]),
                    landmarks=scaled_kps,
                )
            )

        return final_faces


class SCRFDWorker:
    """
    Decoupled background worker executing SCRFD face detection (~30 FPS)
    and asynchronous ArcFace embedding extraction (~10-15 FPS per face).
    Ensures the live display loop runs at camera rate (~25-30 FPS) without blocking,
    retaining latest detections with temporal smoothing, persistence, and strict
    (face_id, frame_id) embedding association.
    """

    def __init__(
        self,
        model_path: str = settings.scrfd_model_path,
        device: str = settings.openvino_device,
        input_size: int = settings.scrfd_input_size,
        conf_thresh: float = settings.scrfd_conf_thresh,
        nms_thresh: float = settings.scrfd_nms_thresh,
        target_fps: float = settings.scrfd_target_fps,
        persistence_seconds: float = settings.box_persistence_seconds,
        smoothing_factor: float = settings.box_smoothing_factor,
        arcface_enabled: bool = settings.arcface_enabled,
        arcface_target_fps: float = settings.arcface_target_fps,
    ):
        self.detector = SCRFDDetector(
            model_path=model_path,
            device=device,
            input_size=input_size,
            conf_thresh=conf_thresh,
            nms_thresh=nms_thresh,
        )
        self.target_fps = target_fps
        self.persistence_seconds = persistence_seconds
        self.smoothing_factor = smoothing_factor
        self.arcface_target_fps = arcface_target_fps

        # Initialize asynchronous ArcFace worker if enabled
        self.arcface_worker: ArcFaceWorker | None = None
        if arcface_enabled:
            try:
                self.arcface_worker = ArcFaceWorker(
                    model_path=settings.arcface_model_path,
                    device=device,
                    target_fps=arcface_target_fps,
                )
            except Exception as exc:
                logger.warning("Could not initialize ArcFaceWorker: %s. Continuing with detection only.", exc)

        # Single-slot input buffer (zero unbounded queue)
        self._pending_frame: np.ndarray | None = None
        self._input_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # Output state
        self._output_lock = threading.Lock()
        # Track structure:
        # {
        #   "face_id": int,
        #   "face": SCRFDFace,
        #   "last_seen": float,
        #   "last_arcface_submit": float,
        #   "last_embedding_time": float,
        # }
        self._active_tracks: list[dict] = []
        self._next_face_id = 1
        self._frame_counter = 0

        self._detection_fps = 0.0
        self._detection_count = 0
        self._last_fps_calc_time = time.monotonic()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "SCRFDWorker":
        """Start the background worker threads."""
        if self._running:
            return self

        self._running = True

        if self.arcface_worker is not None:
            self.arcface_worker.start()

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="PresentoSCRFDWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SCRFDWorker started (device=%s, input=%dx%d, target_fps=%.1f, arcface_enabled=%s)",
            self.detector.full_device_name,
            self.detector.input_size,
            self.detector.input_size,
            self.target_fps,
            self.arcface_worker is not None,
        )
        return self

    def submit_frame(self, frame: np.ndarray) -> None:
        """Submit latest unmirrored frame to background detector (non-blocking)."""
        if not self._running or frame is None or frame.size == 0:
            return

        with self._input_lock:
            self._pending_frame = frame
        self._new_frame_event.set()

    def get_latest_detections(self) -> list[SCRFDFace]:
        """
        Fetch latest smoothed face detections with associated ArcFace embeddings (non-blocking, <0.1ms).
        """
        now = time.monotonic()
        with self._output_lock:
            # Poll ArcFace results and strictly associate by face_id
            if self.arcface_worker is not None:
                for track in self._active_tracks:
                    fid = track["face_id"]
                    res = self.arcface_worker.get_result(fid)
                    if res is not None and res.completed_time > track["last_embedding_time"]:
                        # Strict association: only apply embedding if face_id matches
                        track["face"].embedding = res.embedding
                        track["face"].embedding_frame_id = res.frame_id
                        track["face"].embedding_time = res.completed_time
                        track["last_embedding_time"] = res.completed_time

            # Prune tracks past persistence duration
            pruned_tracks = []
            retained_tracks = []
            for t in self._active_tracks:
                if (now - t["last_seen"]) <= self.persistence_seconds:
                    retained_tracks.append(t)
                else:
                    pruned_tracks.append(t)

            # Clean up pruned tracks from ArcFace cache
            if self.arcface_worker is not None:
                for t in pruned_tracks:
                    self.arcface_worker.prune_face(t["face_id"])

            self._active_tracks = retained_tracks
            return [t["face"] for t in self._active_tracks]

    def get_stats(self) -> dict:
        """Diagnostic stats for HUD overlay."""
        with self._output_lock:
            stats = {
                "detection_fps": self._detection_fps,
                "face_count": len(self._active_tracks),
                "device_name": self.detector.full_device_name,
                "device": self.detector.active_device,
                "arcface_fps": 0.0,
                "arcface_latency_ms": 0.0,
            }

        if self.arcface_worker is not None:
            arc_stats = self.arcface_worker.get_stats()
            stats["arcface_fps"] = arc_stats.get("arcface_fps", 0.0)
            stats["arcface_latency_ms"] = arc_stats.get("arcface_latency_ms", 0.0)

        return stats

    def _worker_loop(self) -> None:
        """Decoupled background loop executing neural detection & ArcFace scheduling."""
        min_interval = 1.0 / max(1.0, self.target_fps)
        last_run_time = 0.0

        while self._running:
            if not self._new_frame_event.wait(timeout=0.1):
                continue

            if not self._running:
                break

            with self._input_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._new_frame_event.clear()

            if frame is None:
                continue

            now = time.monotonic()
            elapsed_since_last = now - last_run_time
            if elapsed_since_last < min_interval:
                sleep_time = min_interval - elapsed_since_last
                time.sleep(sleep_time)
                now = time.monotonic()

            last_run_time = now
            self._frame_counter += 1
            current_frame_id = self._frame_counter

            try:
                faces = self.detector.detect(frame)
                self._update_tracks_and_schedule_arcface(faces, frame, current_frame_id, now)
                self._update_fps(now)
            except Exception as exc:
                logger.warning("Error in SCRFD detection loop: %s", exc, exc_info=True)

    def _update_tracks_and_schedule_arcface(
        self,
        detected_faces: list[SCRFDFace],
        frame: np.ndarray,
        frame_id: int,
        now: float,
    ) -> None:
        """
        1. Smooth bounding boxes & landmarks.
        2. Assign persistent face_ids.
        3. Asynchronously trigger ArcFace inference at ~10-15 FPS per face or when a fresh embedding is needed.
        """
        with self._output_lock:
            # First, check for any newly completed ArcFace results
            if self.arcface_worker is not None:
                for track in self._active_tracks:
                    fid = track["face_id"]
                    res = self.arcface_worker.get_result(fid)
                    if res is not None and res.completed_time > track["last_embedding_time"]:
                        track["face"].embedding = res.embedding
                        track["face"].embedding_frame_id = res.frame_id
                        track["face"].embedding_time = res.completed_time
                        track["last_embedding_time"] = res.completed_time

            if not detected_faces:
                self._active_tracks = [
                    t for t in self._active_tracks
                    if (now - t["last_seen"]) <= self.persistence_seconds
                ]
                return

            updated_tracks: list[dict] = []
            used_detected = set()

            # Match incoming detections to active tracks by spatial proximity
            for track in self._active_tracks:
                if (now - track["last_seen"]) > self.persistence_seconds:
                    continue

                prev_face = track["face"]
                pl, pt, pr, pb = prev_face.bbox
                prev_cx = (pl + pr) / 2.0
                prev_cy = (pt + pb) / 2.0

                best_idx = -1
                best_dist = float("inf")

                for idx, face in enumerate(detected_faces):
                    if idx in used_detected:
                        continue
                    nl, nt, nr, nb = face.bbox
                    ncx = (nl + nr) / 2.0
                    ncy = (nt + nb) / 2.0
                    dist = ((prev_cx - ncx) ** 2 + (prev_cy - ncy) ** 2) ** 0.5

                    if dist < 180.0 and dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx != -1:
                    used_detected.add(best_idx)
                    new_face = detected_faces[best_idx]
                    nl, nt, nr, nb = new_face.bbox
                    alpha = self.smoothing_factor

                    sm_l = int(round(alpha * nl + (1.0 - alpha) * pl))
                    sm_t = int(round(alpha * nt + (1.0 - alpha) * pt))
                    sm_r = int(round(alpha * nr + (1.0 - alpha) * pr))
                    sm_b = int(round(alpha * nb + (1.0 - alpha) * pb))

                    sm_kps = alpha * new_face.landmarks + (1.0 - alpha) * prev_face.landmarks

                    # Retain current embedding and metadata
                    smoothed_face = SCRFDFace(
                        bbox=(sm_l, sm_t, sm_r, sm_b),
                        confidence=new_face.confidence,
                        landmarks=sm_kps,
                        face_id=track["face_id"],
                        frame_id=frame_id,
                        embedding=prev_face.embedding,
                        embedding_frame_id=prev_face.embedding_frame_id,
                        embedding_time=prev_face.embedding_time,
                    )
                    track["face"] = smoothed_face
                    track["last_seen"] = now
                    updated_tracks.append(track)
                else:
                    if (now - track["last_seen"]) <= self.persistence_seconds:
                        updated_tracks.append(track)

            # Assign new persistent face_id to unmatched detections
            for idx, face in enumerate(detected_faces):
                if idx not in used_detected:
                    assigned_fid = self._next_face_id
                    self._next_face_id += 1

                    new_track_face = SCRFDFace(
                        bbox=face.bbox,
                        confidence=face.confidence,
                        landmarks=face.landmarks,
                        face_id=assigned_fid,
                        frame_id=frame_id,
                        embedding=None,
                        embedding_frame_id=0,
                        embedding_time=0.0,
                    )
                    updated_tracks.append(
                        {
                            "face_id": assigned_fid,
                            "face": new_track_face,
                            "last_seen": now,
                            "last_arcface_submit": 0.0,
                            "last_embedding_time": 0.0,
                        }
                    )

            self._active_tracks = updated_tracks

            # Asynchronous ArcFace job scheduling
            # Throttle ArcFace to ~10-15 FPS per face or trigger immediately if no embedding exists
            if self.arcface_worker is not None and frame is not None:
                arc_min_interval = 1.0 / max(1.0, self.arcface_target_fps)
                for track in self._active_tracks:
                    fid = track["face_id"]
                    needs_embedding = (
                        track["last_arcface_submit"] == 0.0
                        or (now - track["last_arcface_submit"]) >= arc_min_interval
                    )
                    if needs_embedding:
                        track["last_arcface_submit"] = now
                        aligned = align_face(frame, track["face"].landmarks)
                        self.arcface_worker.submit_job(
                            face_id=fid,
                            frame_id=frame_id,
                            aligned_face=aligned,
                        )

    def _update_fps(self, now: float) -> None:
        self._detection_count += 1
        elapsed = now - self._last_fps_calc_time
        if elapsed >= 1.0:
            self._detection_fps = round(self._detection_count / elapsed, 1)
            self._detection_count = 0
            self._last_fps_calc_time = now

    def stop(self) -> None:
        """Stop worker threads cleanly."""
        self._running = False
        self._new_frame_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.arcface_worker is not None:
            self.arcface_worker.stop()
            self.arcface_worker = None
        logger.info("SCRFDWorker stopped.")
