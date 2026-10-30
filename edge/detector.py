from __future__ import annotations

import logging
import threading
import time
from typing import List, Tuple
import cv2
import face_recognition
import numpy as np

from .config import settings

logger = logging.getLogger("presento.edge.detector")


class FaceDetectorWorker:
    """
    Decoupled face detection worker that runs HOG detection in a background thread
    on a downscaled copy (~640x360) at approximately 8-10 FPS.

    Applies coordinate scaling back to the original resolution, simple EMA smoothing,
    and latest-box persistence to prevent flickering.
    """

    def __init__(
        self,
        target_width: int = settings.detection_width,
        target_height: int = settings.detection_height,
        target_fps: float = settings.detection_fps,
        persistence_seconds: float = settings.box_persistence_seconds,
        smoothing_factor: float = settings.box_smoothing_factor,
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.persistence_seconds = persistence_seconds
        self.smoothing_factor = smoothing_factor

        # Single-slot input buffer (zero unbounded queue)
        self._pending_frame: np.ndarray | None = None
        self._input_lock = threading.Lock()
        self._new_frame_event = threading.Event()

        # Output state
        self._output_lock = threading.Lock()
        # Each active track: dict with "box": (t, r, b, l), "last_seen": float
        self._active_tracks: list[dict] = []
        self._detection_fps = 0.0
        self._detection_count = 0
        self._last_fps_calc_time = time.monotonic()

        # Thread control
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "FaceDetectorWorker":
        """Start the background detector thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop,
            name="PresentoDetectionWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "FaceDetectorWorker started (target=%dx%d @ %.1f FPS, persistence=%.2fs, alpha=%.2f)",
            self.target_width,
            self.target_height,
            self.target_fps,
            self.persistence_seconds,
            self.smoothing_factor,
        )
        return self

    def submit_frame(self, frame: np.ndarray) -> None:
        """
        Submit a new frame to be detected.
        Always retains only the latest frame (single-slot buffer).
        Non-blocking and thread-safe.
        """
        if not self._running or frame is None or frame.size == 0:
            return

        with self._input_lock:
            self._pending_frame = frame
        self._new_frame_event.set()

    def get_latest_faces(self) -> list[tuple[int, int, int, int]]:
        """
        Get the most recent valid face bounding boxes scaled to the original frame.
        Prunes expired boxes if they exceed the persistence duration.
        Non-blocking and thread-safe.
        """
        now = time.monotonic()
        with self._output_lock:
            # Prune expired boxes
            self._active_tracks = [
                track for track in self._active_tracks
                if (now - track["last_seen"]) <= self.persistence_seconds
            ]
            return [track["box"] for track in self._active_tracks]

    def get_stats(self) -> dict:
        """Return diagnostic metrics for overlay display."""
        with self._output_lock:
            return {
                "detection_fps": self._detection_fps,
                "face_count": len(self._active_tracks),
            }

    def _detection_loop(self) -> None:
        """Background thread loop running HOG detection on downscaled frames."""
        min_interval = 1.0 / max(1.0, self.target_fps)
        last_run_time = 0.0

        while self._running:
            # Wait for a new frame or shutdown signal
            if not self._new_frame_event.wait(timeout=0.1):
                continue

            if not self._running:
                break

            # Fetch the newest pending frame
            with self._input_lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._new_frame_event.clear()

            if frame is None:
                continue

            # Enforce target detection FPS throttling (~8-10 FPS)
            now = time.monotonic()
            elapsed_since_last = now - last_run_time
            if elapsed_since_last < min_interval:
                sleep_time = min_interval - elapsed_since_last
                time.sleep(sleep_time)
                now = time.monotonic()

            last_run_time = now

            try:
                orig_h, orig_w = frame.shape[:2]
                if orig_h == 0 or orig_w == 0:
                    continue

                # 1. Downscale frame to ~640x360
                scale_x = orig_w / float(self.target_width)
                scale_y = orig_h / float(self.target_height)

                small_frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR,
                )

                # 2. Convert to RGB for face_recognition
                rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])

                # 3. Run HOG detection on downscaled image
                raw_locations = face_recognition.face_locations(rgb_small, model="hog")

                # 4. Scale detected coordinates back to original resolution
                scaled_boxes = []
                for (top, right, bottom, left) in raw_locations:
                    st = int(round(top * scale_y))
                    sr = int(round(right * scale_x))
                    sb = int(round(bottom * scale_y))
                    sl = int(round(left * scale_x))

                    # Clamp to bounds
                    st = max(0, min(orig_h, st))
                    sr = max(0, min(orig_w, sr))
                    sb = max(0, min(orig_h, sb))
                    sl = max(0, min(orig_w, sl))
                    scaled_boxes.append((st, sr, sb, sl))

                # 5. Temporal smoothing & persistence update
                self._update_tracks(scaled_boxes, now)

                # 6. Update detection FPS measurement
                self._update_fps(now)

            except Exception as exc:
                logger.warning("Error in face detection worker loop: %s", exc, exc_info=True)

    def _update_tracks(self, detected_boxes: list[tuple[int, int, int, int]], now: float) -> None:
        """
        Match detected boxes with existing tracks, apply EMA smoothing,
        and handle persistence for missed cycles.
        """
        with self._output_lock:
            if not detected_boxes:
                # No faces detected in this cycle: keep existing valid tracks within persistence
                self._active_tracks = [
                    track for track in self._active_tracks
                    if (now - track["last_seen"]) <= self.persistence_seconds
                ]
                return

            updated_tracks: list[dict] = []
            used_detected = set()

            # Lightweight match: find closest active track by center distance / overlap
            for track in self._active_tracks:
                if (now - track["last_seen"]) > self.persistence_seconds:
                    continue

                prev_t, prev_r, prev_b, prev_l = track["box"]
                prev_cx = (prev_l + prev_r) / 2.0
                prev_cy = (prev_t + prev_b) / 2.0

                best_idx = -1
                best_dist = float("inf")

                for idx, box in enumerate(detected_boxes):
                    if idx in used_detected:
                        continue
                    bt, br, bb, bl = box
                    bcx = (bl + br) / 2.0
                    bcy = (bt + bb) / 2.0
                    dist = ((prev_cx - bcx) ** 2 + (prev_cy - bcy) ** 2) ** 0.5

                    # Maximum acceptable pixel displacement for matching (e.g., 180px in 1280x720)
                    if dist < 180.0 and dist < best_dist:
                        best_dist = dist
                        best_idx = idx

                if best_idx != -1:
                    # Match found: apply Exponential Moving Average (EMA) smoothing
                    used_detected.add(best_idx)
                    new_t, new_r, new_b, new_l = detected_boxes[best_idx]
                    alpha = self.smoothing_factor

                    sm_t = int(round(alpha * new_t + (1.0 - alpha) * prev_t))
                    sm_r = int(round(alpha * new_r + (1.0 - alpha) * prev_r))
                    sm_b = int(round(alpha * new_b + (1.0 - alpha) * prev_b))
                    sm_l = int(round(alpha * new_l + (1.0 - alpha) * prev_l))

                    updated_tracks.append({
                        "box": (sm_t, sm_r, sm_b, sm_l),
                        "last_seen": now,
                    })
                else:
                    # Track had no match in this cycle: retain it if within persistence
                    if (now - track["last_seen"]) <= self.persistence_seconds:
                        updated_tracks.append(track)

            # Any newly detected boxes without matches become new tracks
            for idx, box in enumerate(detected_boxes):
                if idx not in used_detected:
                    updated_tracks.append({
                        "box": box,
                        "last_seen": now,
                    })

            self._active_tracks = updated_tracks

    def _update_fps(self, now: float) -> None:
        """Update detection FPS rate."""
        self._detection_count += 1
        elapsed = now - self._last_fps_calc_time
        if elapsed >= 1.0:
            self._detection_fps = round(self._detection_count / elapsed, 1)
            self._detection_count = 0
            self._last_fps_calc_time = now

    def stop(self) -> None:
        """Stop worker thread and wait for clean exit."""
        self._running = False
        self._new_frame_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("FaceDetectorWorker stopped.")
