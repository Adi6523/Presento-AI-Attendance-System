from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import queue
import threading
import time
from typing import Tuple

import cv2
import numpy as np
import openvino as ov

from .config import settings

logger = logging.getLogger("presento.edge.arcface")

# Standard 5 facial reference landmarks on a 112x112 aligned image (InsightFace ArcFace standard):
# Order: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
ARCFACE_REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def align_face(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face crop to standard ArcFace 112x112 geometry using partial affine similarity warp.
    Preserves facial proportions, removes in-plane head tilt, and centers facial features.

    Args:
        frame: Original camera frame (BGR uint8, shape HxWx3).
        landmarks: 5 facial landmarks (shape 5x2, float32 or int coordinates on frame).

    Returns:
        Aligned face image of shape (112, 112, 3) uint8 BGR.
    """
    if frame is None or frame.size == 0 or landmarks is None or len(landmarks) < 5:
        return np.zeros((112, 112, 3), dtype=np.uint8)

    src_pts = landmarks[:5].astype(np.float32)
    dst_pts = ARCFACE_REFERENCE_POINTS

    # Estimate similarity transform (rotation + uniform scale + translation, no shear)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)

    if M is None or not np.all(np.isfinite(M)):
        # Fallback to standard 3-point affine transform (left eye, right eye, nose)
        try:
            M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
        except Exception:
            # Fallback to simple center crop if affine estimation fails
            logger.warning("Affine alignment failed; using fallback crop")
            min_x = max(0, int(np.min(src_pts[:, 0])))
            min_y = max(0, int(np.min(src_pts[:, 1])))
            max_x = min(frame.shape[1], int(np.max(src_pts[:, 0])))
            max_y = min(frame.shape[0], int(np.max(src_pts[:, 1])))
            crop = frame[min_y:max_y, min_x:max_x]
            if crop.size > 0:
                return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            return np.zeros((112, 112, 3), dtype=np.uint8)

    aligned = cv2.warpAffine(frame, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0.0)
    return aligned


def transform_mirrored_bbox(bbox: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    """
    Convert an unmirrored bounding box (left, top, right, bottom)
    to horizontally mirrored coordinates for a display frame of given width.

    In a horizontally flipped frame (cv2.flip(frame, 1)):
        mirrored_x = width - 1 - original_x
    Therefore:
        mirrored_left = width - 1 - original_right
        mirrored_right = width - 1 - original_left
        top and bottom remain unchanged.
    """
    left, top, right, bottom = bbox
    m_left = max(0, min(width - 1, width - 1 - right))
    m_right = max(0, min(width - 1, width - 1 - left))
    return (m_left, top, m_right, bottom)


def transform_mirrored_landmarks(landmarks: np.ndarray, width: int) -> np.ndarray:
    """
    Convert unmirrored facial landmarks (shape Nx2)
    to horizontally mirrored coordinates for a display frame of given width.
    """
    if landmarks is None or len(landmarks) == 0:
        return landmarks
    mirrored = landmarks.copy().astype(np.float32)
    mirrored[:, 0] = (width - 1) - mirrored[:, 0]
    mirrored[:, 0] = np.clip(mirrored[:, 0], 0, width - 1)
    return mirrored


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 512-D face embeddings.
    Cosine similarity = dot(emb1, emb2) / (||emb1|| * ||emb2||).
    Range: [-1.0, 1.0].
    """
    if emb1 is None or emb2 is None:
        return 0.0
    v1 = emb1.flatten().astype(np.float32)
    v2 = emb2.flatten().astype(np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return float(np.clip(sim, -1.0, 1.0))


class ArcFaceEmbedder:
    """
    Native OpenVINO ArcFace (MobileFaceNet w600k_mbf) feature extractor.
    Extracts 512-D L2-normalized face embeddings on Intel Iris Xe GPU / CPU.
    """

    def __init__(
        self,
        model_path: str = settings.arcface_model_path,
        device: str = settings.openvino_device,
    ):
        self.model_path = model_path
        self.requested_device = device.upper()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ArcFace model file not found at: {self.model_path}")

        # Initialize OpenVINO Core and verify hardware
        self.core = ov.Core()
        available_devices = [d.upper() for d in self.core.available_devices]
        logger.info("OpenVINO Core initialized for ArcFace. Available devices: %s", available_devices)

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

        try:
            self.full_device_name = self.core.get_property(self.active_device, "FULL_DEVICE_NAME")
        except Exception:
            self.full_device_name = self.active_device

        logger.info(
            "Compiling ArcFace model '%s' on verified OpenVINO device: %s (%s)...",
            self.model_path,
            self.active_device,
            self.full_device_name,
        )

        t0 = time.perf_counter()
        ov_model = self.core.read_model(self.model_path)
        self.compiled_model = self.core.compile_model(ov_model, self.active_device)
        self.infer_request = self.compiled_model.create_infer_request()
        self.input_tensor_name = self.compiled_model.inputs[0].any_name
        self.output_tensor = self.compiled_model.outputs[0]

        # Warm up model with a dummy inference
        dummy_input = np.zeros((1, 3, 112, 112), dtype=np.float32)
        _ = self.infer_request.infer({self.input_tensor_name: dummy_input})
        warmup_ms = (time.perf_counter() - t0) * 1000.0

        logger.info("ArcFace compiled and warmed up on %s in %.1f ms.", self.full_device_name, warmup_ms)
        print(f"[ArcFace] Active OpenVINO Device: {self.active_device} -> {self.full_device_name}", flush=True)

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract 512-D L2-normalized embedding from an aligned 112x112 BGR face crop.

        Args:
            aligned_face: (112, 112, 3) BGR uint8 image.

        Returns:
            (512,) float32 embedding vector with L2 norm == 1.0.
        """
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("Input aligned_face is empty")

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Preprocessing:
        # 1. Convert BGR to RGB
        # 2. Normalize to [-1.0, 1.0]: (pixel - 127.5) / 127.5
        # 3. Transpose from HWC to CHW: (3, 112, 112)
        # 4. Expand batch dimension: (1, 3, 112, 112), float32
        rgb = aligned_face[:, :, ::-1]
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        # Run inference
        res = self.infer_request.infer({self.input_tensor_name: blob})
        raw_embedding = res[self.output_tensor].flatten().astype(np.float32)

        # L2 normalization: vector / ||vector||
        norm = float(np.linalg.norm(raw_embedding))
        if norm > 1e-10:
            embedding = raw_embedding / norm
        else:
            embedding = raw_embedding

        return embedding


@dataclass
class ArcFaceJob:
    """Job container sent to the asynchronous ArcFace worker."""
    face_id: int
    frame_id: int
    aligned_face: np.ndarray
    submit_time: float


@dataclass
class ArcFaceResult:
    """Result container produced by the asynchronous ArcFace worker."""
    face_id: int
    frame_id: int
    embedding: np.ndarray
    latency_ms: float
    completed_time: float


class ArcFaceWorker:
    """
    Decoupled background worker executing ArcFace embedding extraction asynchronously.
    Runs at roughly 10-15 FPS per face without blocking the SCRFD detection thread (~30 FPS).
    Strictly tags every 512-D embedding to its source (face_id, frame_id).
    """

    def __init__(
        self,
        model_path: str = settings.arcface_model_path,
        device: str = settings.openvino_device,
        target_fps: float = settings.arcface_target_fps,
    ):
        self.embedder = ArcFaceEmbedder(model_path=model_path, device=device)
        self.target_fps = target_fps

        # Job queue (bounded to avoid backlog)
        self._job_queue: queue.Queue[ArcFaceJob] = queue.Queue(maxsize=16)

        # Thread-safe results dictionary: face_id -> ArcFaceResult
        self._results_lock = threading.Lock()
        self._latest_results: dict[int, ArcFaceResult] = {}

        # Performance metrics
        self._metrics_lock = threading.Lock()
        self._arcface_fps = 0.0
        self._arcface_latency_ms = 0.0
        self._inferences_count = 0
        self._last_fps_time = time.monotonic()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "ArcFaceWorker":
        """Start the asynchronous ArcFace worker thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="PresentoArcFaceWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ArcFaceWorker started (device=%s, target_fps=%.1f)",
            self.embedder.full_device_name,
            self.target_fps,
        )
        return self

    def submit_job(self, face_id: int, frame_id: int, aligned_face: np.ndarray) -> bool:
        """
        Submit a face alignment crop for embedding extraction (non-blocking).
        Returns True if queued, False if queue is full or worker stopped.
        """
        if not self._running or aligned_face is None or aligned_face.size == 0:
            return False

        job = ArcFaceJob(
            face_id=face_id,
            frame_id=frame_id,
            aligned_face=aligned_face.copy(),
            submit_time=time.monotonic(),
        )
        try:
            self._job_queue.put_nowait(job)
            return True
        except queue.Full:
            # Drop oldest job if full to guarantee fresh embedding processing
            try:
                _ = self._job_queue.get_nowait()
                self._job_queue.put_nowait(job)
                return True
            except (queue.Empty, queue.Full):
                return False

    def get_result(self, face_id: int) -> ArcFaceResult | None:
        """Retrieve the latest embedding result for a given face_id (non-blocking)."""
        with self._results_lock:
            return self._latest_results.get(face_id)

    def prune_face(self, face_id: int) -> None:
        """Clean up embedding state when a face track is pruned/lost."""
        with self._results_lock:
            self._latest_results.pop(face_id, None)

    def get_stats(self) -> dict:
        """Diagnostic stats for HUD overlay."""
        with self._metrics_lock:
            return {
                "arcface_fps": self._arcface_fps,
                "arcface_latency_ms": self._arcface_latency_ms,
                "device_name": self.embedder.full_device_name,
                "device": self.embedder.active_device,
            }

    def _worker_loop(self) -> None:
        """Decoupled background loop executing ArcFace inference."""
        while self._running:
            try:
                job = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._running:
                break

            try:
                t0 = time.perf_counter()
                embedding = self.embedder.get_embedding(job.aligned_face)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                now = time.monotonic()

                result = ArcFaceResult(
                    face_id=job.face_id,
                    frame_id=job.frame_id,
                    embedding=embedding,
                    latency_ms=latency_ms,
                    completed_time=now,
                )

                with self._results_lock:
                    self._latest_results[job.face_id] = result

                self._update_metrics(latency_ms, now)

            except Exception as exc:
                logger.warning("Error in ArcFace inference loop: %s", exc, exc_info=True)
            finally:
                self._job_queue.task_done()

    def _update_metrics(self, latency_ms: float, now: float) -> None:
        with self._metrics_lock:
            self._inferences_count += 1
            # Exponential moving average of latency
            if self._arcface_latency_ms == 0.0:
                self._arcface_latency_ms = latency_ms
            else:
                self._arcface_latency_ms = 0.2 * latency_ms + 0.8 * self._arcface_latency_ms

            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self._arcface_fps = round(self._inferences_count / elapsed, 1)
                self._inferences_count = 0
                self._last_fps_time = now

    def stop(self) -> None:
        """Stop worker thread cleanly."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("ArcFaceWorker stopped.")
