"""
Presento AI Vision Demo — Temporary Demonstration Mode (September 30, 2026).

Interactive visual showcase demonstrating the complete biometric pipeline:
Camera -> SCRFD -> ArcFace -> Matcher -> Identity Stability -> Anti-Spoof Fusion -> Attendance

Key Features:
- 5 Demonstration Modes (Face Detection, Face Recognition, Anti-Spoof, Attendance, Summary)
- Multi-face support with independent state tracking and callout overlays
- Honest reporting of UNCERTAIN states (no fabricated confidence)
- Interactive keyboard shortcuts: [1-5] Modes, [P] Pipeline HUD, [Space] Pause/Freeze, [D] Diag, [Q] Quit
- Strict on-demand camera start/stop lifecycle ensuring C270 hardware is completely released upon exit.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
import uuid

import cv2
import numpy as np

from .config import settings
from .camera_control import (
    CameraSession,
    CameraInUseError,
    CameraAuthError,
    CameraControlError,
    stop_camera_service,
)
from .frame_receiver import FrameReceiver
from .scrfd_detector import SCRFDWorker, SCRFDFace
from .arcface_embedder import (
    ArcFaceEmbedder,
    align_face,
    compute_cosine_similarity,
    transform_mirrored_bbox,
    transform_mirrored_landmarks,
)
from .enroll_face import assess_enrollment_quality
from .face_enrollment import FaceEnrollmentStore
from .face_matcher import FaceMatcher, MatchResult, CandidateMatch
from .anti_spoof_worker import AntiSpoofWorker, AntiSpoofResult
from .anti_spoof_fusion import FusionResult
from .attendance_worker import AttendanceIOWorker
from .api import get_active_session
from .hardware_detector import detect_system_hardware
from .inference_backend import select_inference_device, InferenceDeviceDecision


def beep() -> None:
    """Trigger physical piezo buzzer if available on Raspberry Pi GPIO."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setup(settings.buzzer_pin, GPIO.OUT)
        GPIO.output(settings.buzzer_pin, 1)
        time.sleep(0.12)
        GPIO.output(settings.buzzer_pin, 0)
    except (ImportError, RuntimeError):
        pass


_stdin_key_queue: queue.Queue = queue.Queue()
_stdin_listener_started = False


def _ensure_stdin_listener() -> None:
    global _stdin_listener_started
    if _stdin_listener_started:
        return
    _stdin_listener_started = True

    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                for ch in line:
                    if ch not in ('\r', '\n'):
                        _stdin_key_queue.put(ord(ch))
            except Exception:
                break

    t = threading.Thread(target=_reader, daemon=True, name="DemoStdinListener")
    t.start()


# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("presento.edge.demo")


# ── Color Palette (BGR for OpenCV) ────────────────────────────────────────────
COLOR_BG_DARK = (18, 16, 26)          # Deep midnight slate
COLOR_BG_CARD = (30, 26, 42)          # Card background
COLOR_CYAN = (255, 220, 0)            # Electric Cyan (B, G, R)
COLOR_EMERALD = (80, 235, 70)         # Present / Verified Real
COLOR_GREEN = (50, 210, 50)           # Recognized
COLOR_AMBER = (0, 190, 255)           # Checking / In-Progress
COLOR_GOLD = (0, 140, 255)            # Uncertain
COLOR_RED = (50, 50, 235)             # Spoof Attack Detected
COLOR_GRAY = (180, 180, 190)          # Unknown / Muted
COLOR_WHITE = (255, 255, 255)         # Pure White
COLOR_TEXT_DIM = (160, 165, 180)      # Secondary text
COLOR_BLUE = (245, 130, 50)           # Accent Blue


# ── Face State Definitions ───────────────────────────────────────────────────
STATE_UNKNOWN = "UNKNOWN"
STATE_RECOGNIZED = "RECOGNIZED"
STATE_CHECKING = "CHECKING"
STATE_SPOOF = "SPOOF"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_PRESENT = "PRESENT"

STATE_COLORS = {
    STATE_UNKNOWN: COLOR_GRAY,
    STATE_RECOGNIZED: COLOR_GREEN,
    STATE_CHECKING: COLOR_AMBER,
    STATE_SPOOF: COLOR_RED,
    STATE_UNCERTAIN: COLOR_GOLD,
    STATE_PRESENT: COLOR_EMERALD,
}


@dataclass
class AttendanceLogEntry:
    timestamp: str
    name: str
    student_id: str
    similarity: float
    anti_spoof_label: str
    status: str = "RECORDED"


class LatestFrameCamera:
    """
    Decoupled, non-blocking camera capture wrapper for OpenCV VideoCapture.
    Runs a lightweight background thread to continuously fetch the newest frame
    from hardware into a single-slot buffer, preventing physical camera I/O
    from blocking the main display or AI inference loop.
    Compatible with cv2.VideoCapture interface (read, isOpened, release, get, set).
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="PresentoWebcamCapture",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            if not self._cap.isOpened():
                time.sleep(0.01)
                continue
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
                # Yield/pacing guard to prevent thread spinning if read() returns instantly (e.g. in tests/mocks)
                time.sleep(0.001)
            else:
                time.sleep(0.005)

    def read(self, timeout: float = 0.035) -> tuple[bool, np.ndarray | None]:
        """
        Fetch the newest frame. Compatible with cv2.VideoCapture.read() and FrameReceiver.read().
        Waits up to timeout seconds for a genuinely new frame to arrive.
        """
        if not self._running:
            return False, None

        if timeout is not None and timeout > 0:
            if not self._new_frame_event.wait(timeout=timeout):
                with self._lock:
                    if self._latest_frame is not None:
                        return True, self._latest_frame.copy()
                    return False, None

        with self._lock:
            self._new_frame_event.clear()
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def isOpened(self) -> bool:
        return self._running and self._cap.isOpened()

    def release(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._cap.release()

    def get(self, propId: int):
        return self._cap.get(propId)

    def set(self, propId: int, value):
        return self._cap.set(propId, value)


class DemoMetricsTracker:
    """Tracks session-wide cumulative analytics for Demo Mode 5."""

    def __init__(self):
        self.start_time = time.monotonic()
        self.total_detected_faces = 0
        self.seen_face_ids: set[int] = set()
        self.recognized_encounters = 0
        self.unknown_encounters = 0
        self.spoof_attempts = 0
        self.uncertain_attempts = 0
        self.attendance_events = 0
        self.recent_logs: deque[AttendanceLogEntry] = deque(maxlen=8)

    def register_faces(
        self,
        faces: List[SCRFDFace],
        states: Dict[int, str],
        matches: Dict[int, Optional[MatchResult]],
        anti_spoof: Dict[int, Optional[AntiSpoofResult]],
    ):
        for face in faces:
            fid = face.face_id
            if fid not in self.seen_face_ids:
                self.seen_face_ids.add(fid)
                self.total_detected_faces += 1

            st = states.get(fid, STATE_CHECKING)
            if st == STATE_SPOOF:
                self.spoof_attempts += 1
            elif st == STATE_UNCERTAIN:
                self.uncertain_attempts += 1
            elif st == STATE_UNKNOWN:
                self.unknown_encounters += 1
            elif st in (STATE_RECOGNIZED, STATE_PRESENT):
                self.recognized_encounters += 1

    def log_attendance(self, name: str, student_id: str, similarity: float, anti_spoof_label: str):
        self.attendance_events += 1
        now_str = datetime.now().strftime("%H:%M:%S")
        entry = AttendanceLogEntry(
            timestamp=now_str,
            name=name,
            student_id=student_id,
            similarity=similarity,
            anti_spoof_label=anti_spoof_label,
            status="RECORDED",
        )
        self.recent_logs.appendleft(entry)


# ── OpenCV Drawing & HUD Utilities ────────────────────────────────────────────

def draw_glass_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    bg_color: Tuple[int, int, int] = COLOR_BG_DARK,
    alpha: float = 0.78,
    border_color: Optional[Tuple[int, int, int]] = None,
    border_thickness: int = 1,
) -> None:
    """Draw a semi-transparent glassmorphic panel with optional border."""
    ih, iw = img.shape[:2]
    x1 = max(0, min(iw - 1, x))
    y1 = max(0, min(ih - 1, y))
    x2 = max(0, min(iw, x + w))
    y2 = max(0, min(ih, y + h))

    if x2 <= x1 or y2 <= y1:
        return

    sub = img[y1:y2, x1:x2]
    rect = np.full_like(sub, bg_color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

    if border_color is not None:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), border_color, border_thickness)


def draw_tech_corners(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    corner_len: int = 14,
    thickness: int = 2,
) -> None:
    """Draw sleek sci-fi corner reticle brackets around a bounding box."""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def draw_pill_badge(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = COLOR_WHITE,
    font_scale: float = 0.44,
    padding: int = 5,
) -> Tuple[int, int]:
    """Draw a compact colored status pill badge. Returns (width, height)."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bw = tw + padding * 2
    bh = th + padding * 2
    ih, iw = img.shape[:2]
    x = max(2, min(iw - bw - 2, x))
    y = max(2, min(ih - bh - 2, y))

    cv2.rectangle(img, (x, y), (x + bw, y + bh), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), COLOR_WHITE, 1)
    cv2.putText(
        img,
        text,
        (x + padding, y + bh - padding),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return (bw, bh)


def classify_face_state(
    face: SCRFDFace,
    match_res: Optional[MatchResult],
    as_res: Optional[AntiSpoofResult],
    cooldown_rem: float,
    tolerance: float,
) -> Tuple[str, Tuple[int, int, int], str]:
    """
    Evaluates face attributes into one of the 6 formal states:
    UNKNOWN, RECOGNIZED, CHECKING, SPOOF, UNCERTAIN, PRESENT.
    Returns (state_label, state_color, reason_text).
    """
    now = time.monotonic()

    # 1. Check if embedding has been extracted
    if face.embedding is None:
        return (STATE_CHECKING, COLOR_AMBER, "Extracting 512-D embedding...")

    # 2. Check enrolled match
    if match_res is None or match_res.best_match is None or match_res.best_match.similarity < tolerance:
        # Candidate not found or below threshold
        sim = match_res.best_match.similarity if (match_res and match_res.best_match) else 0.0
        return (STATE_UNKNOWN, COLOR_GRAY, f"No enrolled match (best sim: {sim:.2f})")

    # Candidate enrolled match found
    cand = match_res.best_match

    # 3. Check anti-spoof consensus
    if as_res is None or as_res.fusion_result.window_frames < 5:
        frames_filled = as_res.fusion_result.window_frames if as_res else 0
        return (STATE_CHECKING, COLOR_AMBER, f"Checking liveness ({frames_filled}/15)")

    fus = as_res.fusion_result
    if fus.temporal_label == "SPOOF":
        return (STATE_SPOOF, COLOR_RED, f"Spoof attack detected ({fus.both_spoof_ratio * 100:.0f}%)")
    elif fus.temporal_label == "UNCERTAIN":
        return (STATE_UNCERTAIN, COLOR_GOLD, "Liveness uncertain (no consensus)")

    # fus.temporal_label == "REAL"
    # 4. Check attendance cooldown / confirmation
    if cooldown_rem > 0:
        return (STATE_PRESENT, COLOR_EMERALD, f"Attendance recorded (cooldown {cooldown_rem:.0f}s)")

    if match_res.is_stable:
        return (STATE_PRESENT, COLOR_EMERALD, "Confirmed: Ready for attendance")
    else:
        return (STATE_RECOGNIZED, COLOR_GREEN, f"Recognized ({match_res.stability_label})")


# ── Render Mode Overlays ──────────────────────────────────────────────────────

def render_top_header_hud(
    img: np.ndarray,
    current_mode: int,
    display_fps: float,
    det_fps: float,
    arc_fps: float,
    arc_lat: float,
    enrolled_count: int,
    camera_active: bool,
    is_paused: bool,
    source_label: str = "LAPTOP WEBCAM",
    device_label: str = "CPU",
    device_short: str = "CPU",
    cpu_label: Optional[str] = None,
    gpu_label: Optional[str] = None,
    ai_label: Optional[str] = None,
) -> None:
    """Render top glassmorphism HUD bar with system vitals, hardware devices, and active mode."""
    disp_h, disp_w = img.shape[:2]
    header_h = 58
    draw_glass_rect(img, 0, 0, disp_w, header_h, bg_color=COLOR_BG_DARK, alpha=0.82, border_color=(60, 60, 80))

    # Brand Title
    cv2.putText(img, "PRESENTO", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "AI VISION", (124, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)

    # Mode Name
    mode_names = {
        1: "MODE 1: FACE DETECTION (SCRFD + 5 LANDMARKS)",
        2: "MODE 2: FACE RECOGNITION (ARCFACE + MATCHER)",
        3: "MODE 3: ANTI-SPOOF (DUAL MODEL + FUSION)",
        4: "MODE 4: ATTENDANCE VERIFICATION (GATING)",
        5: "MODE 5: SUMMARY DASHBOARD & ANALYTICS",
    }
    mode_str = mode_names.get(current_mode, f"MODE {current_mode}")
    draw_pill_badge(img, mode_str, 245, 10, bg_color=(60, 40, 100), text_color=COLOR_WHITE, font_scale=0.46)

    # Status Indicators on Right (dynamically placed right-to-left to prevent overlap)
    x_curr = disp_w - 16

    # 1. Enrolled count
    enroll_str = f"Enrolled: {enrolled_count}"
    enroll_w = cv2.getTextSize(enroll_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    x_curr -= enroll_w
    cv2.putText(img, enroll_str, (x_curr, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
    x_curr -= 14

    # 2. FPS
    fps_str = f"FPS: {display_fps:.1f}"
    fps_w = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    x_curr -= fps_w
    cv2.putText(img, fps_str, (x_curr, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)
    x_curr -= 14

    # 3. AI Runtime (e.g. "AI: OpenVINO GPU" or "AI: OpenVINO CPU")
    ai_str = f"AI: {ai_label}" if ai_label else f"Device: {device_label}"
    ai_w = cv2.getTextSize(ai_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    x_curr -= ai_w
    cv2.putText(img, ai_str, (x_curr, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 240, 255), 1, cv2.LINE_AA)
    x_curr -= 14

    # 4. GPU info (if provided and screen width allows)
    if gpu_label and disp_w >= 1050:
        gpu_str = f"GPU: {gpu_label}"
        gpu_w = cv2.getTextSize(gpu_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        if x_curr - gpu_w > 450:
            x_curr -= gpu_w
            cv2.putText(img, gpu_str, (x_curr, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 210, 140), 1, cv2.LINE_AA)
            x_curr -= 14

    # 5. CPU info (if provided and screen width allows)
    if cpu_label and disp_w >= 1150:
        cpu_str = f"CPU: {cpu_label}"
        cpu_w = cv2.getTextSize(cpu_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        if x_curr - cpu_w > 400:
            x_curr -= cpu_w
            cv2.putText(img, cpu_str, (x_curr, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 220, 255), 1, cv2.LINE_AA)
            x_curr -= 14

    # 6. Source with indicator dot
    source_str = f"SOURCE: {source_label}"
    source_w = cv2.getTextSize(source_str, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
    x_curr -= (source_w + 14)
    src_dot_col = COLOR_EMERALD if camera_active else COLOR_RED
    cv2.circle(img, (x_curr + 4, 20), 4, src_dot_col, -1, cv2.LINE_AA)
    cv2.putText(img, source_str, (x_curr + 14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_EMERALD if camera_active else COLOR_RED, 1, cv2.LINE_AA)

    # Secondary line: Pipeline vitals & Keybindings
    sub_text = (
        f"SCRFD: {det_fps:.1f} FPS | ArcFace: {arc_fps:.1f} FPS ({arc_lat:.1f}ms) | AntiSpoof: {device_short} | "
        f"[1-5] Modes | [P] Pipeline | [D] Diag | [E] Enroll Demo | [R] Retry | [Space] Pause | [Q] Quit"
    )
    if is_paused:
        sub_text = ">>> [PAUSED / FROZEN] Press SPACE to resume live feed <<<  " + sub_text
    cv2.putText(img, sub_text, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def render_bottom_mode_banner(img: np.ndarray, current_mode: int) -> None:
    """Render explanatory banner for the active mode at the bottom."""
    disp_h, disp_w = img.shape[:2]
    banner_h = 36
    draw_glass_rect(img, 0, disp_h - banner_h, disp_w, banner_h, bg_color=COLOR_BG_DARK, alpha=0.82, border_color=(60, 60, 80))

    mode_explanations = {
        1: "MODE 1 - SCRFD Detector locates facial bounding boxes and 5 facial landmarks in real time (~30 FPS).",
        2: "MODE 2 - ArcFace MobileFaceNet extracts 512-D normalized vector; FaceMatcher evaluates cosine similarity & margin.",
        3: "MODE 3 - Asynchronous Dual Anti-Spoof (MN3 + MiniFASNet V2 SE). Demonstrates photo/screen attack rejection.",
        4: "MODE 4 - Attendance requires: Enrolled Identity + Stable Tracking + Anti-Spoof REAL consensus. Gated before logging.",
        5: "MODE 5 - Cumulative session statistics, face encounter metrics, blocked attacks, and live attendance log.",
    }
    exp_text = mode_explanations.get(current_mode, "")
    cv2.putText(img, exp_text, (16, disp_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)


def render_face_callout_card(
    img: np.ndarray,
    face: SCRFDFace,
    m_bbox: Tuple[int, int, int, int],
    m_landmarks: np.ndarray,
    match_res: Optional[MatchResult],
    as_res: Optional[AntiSpoofResult],
    cooldown_rem: float,
    current_mode: int,
    tolerance: float,
) -> None:
    """
    Renders the rich Presento AI Vision per-face overlay:
    - Bounding box with tech corners & state badge
    - Standard callout card with Identity, ID, Similarity, Margin, Stability, Anti-Spoof, Attendance
    - Mode-specific visual augmentations (Mode 1 landmarks, Mode 2 similarity bar, Mode 3 liveness, Mode 4 gates)
    """
    disp_h, disp_w = img.shape[:2]
    m_left, m_top, m_right, m_bottom = m_bbox
    box_w = m_right - m_left
    box_h = m_bottom - m_top

    # Classify state
    state, state_col, reason_text = classify_face_state(face, match_res, as_res, cooldown_rem, tolerance)
    is_unknown = (state == STATE_UNKNOWN)

    # 1. Draw Tech Corner Bounding Box
    cv2.rectangle(img, (m_left, m_top), (m_right, m_bottom), state_col, 1)
    draw_tech_corners(img, m_left, m_top, m_right, m_bottom, state_col, corner_len=min(18, box_w // 4), thickness=2)

    # 2. Draw State Pill Badge directly on top edge of bounding box
    badge_x = m_left
    badge_y = max(60, m_top - 20)
    draw_pill_badge(img, state, badge_x, badge_y, bg_color=state_col, text_color=COLOR_WHITE, font_scale=0.42)

    # 3. Mode 1 Specific: Draw 5 Landmarks & Pose Alignment Geometry
    if current_mode == 1:
        landmark_labels = ["L-Eye", "R-Eye", "Nose", "L-Mouth", "R-Mouth"]
        landmark_colors = [
            (255, 200, 0),   # Left Eye: Cyan
            (255, 200, 0),   # Right Eye: Cyan
            (0, 230, 255),   # Nose: Yellow
            (80, 100, 255),  # Left Mouth: Coral
            (80, 100, 255),  # Right Mouth: Coral
        ]
        if m_landmarks is not None and len(m_landmarks) >= 5:
            # Draw eye-to-eye alignment line
            pt_le = (int(round(m_landmarks[0][0])), int(round(m_landmarks[0][1])))
            pt_re = (int(round(m_landmarks[1][0])), int(round(m_landmarks[1][1])))
            cv2.line(img, pt_le, pt_re, (180, 180, 255), 1, cv2.LINE_AA)

            # Draw landmark dots
            for idx, pt in enumerate(m_landmarks[:5]):
                px, py = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(img, (px, py), 4, landmark_colors[idx], -1, cv2.LINE_AA)
                cv2.circle(img, (px, py), 6, COLOR_WHITE, 1, cv2.LINE_AA)

            # Head roll / tilt angle calculation:
            # Uses physical unmirrored face.landmarks for reliable angle normalized to [-180, +180]
            dx_orig = float(face.landmarks[1][0] - face.landmarks[0][0])
            dy_orig = float(face.landmarks[1][1] - face.landmarks[0][1])
            raw_angle = math.degrees(math.atan2(dy_orig, dx_orig))
            angle_deg = (raw_angle + 180.0) % 360.0 - 180.0
            cv2.putText(
                img,
                f"Tilt: {angle_deg:+.1f} deg",
                (m_left, m_bottom + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_CYAN,
                1,
                cv2.LINE_AA,
            )

    # 4. Standard Callout Information Card
    card_w = 236
    card_h = 180 if (current_mode in (3, 4) or is_unknown) else 165

    # Position card intelligently
    if m_right + card_w + 10 < disp_w:
        card_x = m_right + 8
        card_y = max(60, min(disp_h - card_h - 40, m_top))
    elif m_left - card_w - 8 > 0:
        card_x = m_left - card_w - 8
        card_y = max(60, min(disp_h - card_h - 40, m_top))
    else:
        card_x = max(8, min(disp_w - card_w - 8, m_left))
        card_y = min(disp_h - card_h - 40, m_bottom + 10)

    draw_glass_rect(img, card_x, card_y, card_w, card_h, bg_color=COLOR_BG_DARK, alpha=0.85, border_color=state_col)

    # Populate candidate data fields
    cand_name = "Unknown"
    cand_id = "None"
    sim_val = 0.0
    margin_val = 0.0
    stability_val = "UNSTABLE"
    second_info = "None"

    if match_res is not None and match_res.best_match is not None:
        cand_name = match_res.best_match.name
        cand_id = match_res.best_match.student_id
        sim_val = match_res.best_match.similarity
        margin_val = match_res.margin
        stability_val = "STABLE" if match_res.is_stable else match_res.stability_label
        if match_res.second_best_match is not None:
            second_info = f"{match_res.second_best_match.name} ({match_res.second_best_match.similarity:.2f})"

    as_label = "CHECKING"
    p_real_avg = 0.0
    if as_res is not None:
        as_label = as_res.fusion_result.temporal_label
        p_real_avg = as_res.fusion_result.avg_p_real

    att_status = "PRESENT" if (state == STATE_PRESENT) else ("PENDING" if state == STATE_RECOGNIZED else "NONE")

    # Render card lines
    cy = card_y + 18
    # Header: Face #ID | Conf
    cv2.putText(img, f"Face #{face.face_id}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(img, f"Det: {face.confidence:.2f}", (card_x + card_w - 70, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    cy += 18

    as_col = COLOR_EMERALD if as_label == "REAL" else (COLOR_RED if as_label == "SPOOF" else COLOR_GOLD)
    sim_col = COLOR_EMERALD if sim_val >= tolerance else COLOR_GRAY

    if is_unknown:
        # Honest UNKNOWN representation: Never present candidate as identity when below acceptance threshold
        cv2.putText(img, "Identity: UNKNOWN", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 220), 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, "No enrolled identity accepted", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cy += 16
        best_cand_str = cand_name if cand_name != "Unknown" else "None"
        cv2.putText(img, f"Best: {best_cand_str}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, f"Similarity: {sim_val:.2f} (Req: {tolerance:.2f})", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.39, COLOR_AMBER, 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, "Acceptance: FAILED", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.39, COLOR_RED, 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, f"Anti-Spoof: {as_label}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, as_col, 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, "Attendance: BLOCKED", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_RED, 1, cv2.LINE_AA)
        cy += 16
        cv2.putText(img, "[E] Enroll Demo | [R] Retry", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)
    else:
        # Recognized enrolled student
        cv2.putText(img, f"Identity: {cand_name}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_WHITE, 1, cv2.LINE_AA)
        cy += 17
        cv2.putText(img, f"ID: {cand_id}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cy += 17
        cv2.putText(img, f"Similarity: {sim_val:.2f}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sim_col, 1, cv2.LINE_AA)
        cv2.putText(img, f"Margin: {margin_val:.2f}", (card_x + 130, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
        cy += 17
        stab_col = COLOR_EMERALD if "STABLE" in stability_val else COLOR_AMBER
        cv2.putText(img, f"Identity: {stability_val}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, stab_col, 1, cv2.LINE_AA)
        cy += 17
        cv2.putText(img, f"Anti-Spoof: {as_label}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, as_col, 1, cv2.LINE_AA)
        cy += 17
        att_col = COLOR_EMERALD if att_status == "PRESENT" else COLOR_TEXT_DIM
        cv2.putText(img, f"Attendance: {att_status}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, att_col, 1, cv2.LINE_AA)
        cy += 16

    # 5. Mode 2 Specific: 2nd Candidate & Similarity Bar
    if current_mode == 2:
        cv2.putText(img, f"2nd Best: {second_info}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cy += 14
        bar_w = card_w - 20
        bar_x = card_x + 8
        bar_y = cy
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (40, 40, 50), -1)
        fill_w = int(max(0.0, min(1.0, sim_val)) * bar_w)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), sim_col if not is_unknown else (90, 90, 110), -1)
        thresh_x = bar_x + int(tolerance * bar_w)
        cv2.line(img, (thresh_x, bar_y - 2), (thresh_x, bar_y + 10), COLOR_WHITE, 1)

    # 6. Mode 3 Specific: Dual Model Breakdown (MN3 + MiniFASNet)
    elif current_mode == 3 and as_res is not None:
        cv2.line(img, (card_x + 8, cy), (card_x + card_w - 8, cy), (60, 60, 80), 1)
        cy += 13
        p_mn3 = as_res.mn3_p_real
        p_mf = as_res.mf_p_real
        cv2.putText(img, f"MN3: p_real={p_mn3:.2f} ({(1-p_mn3):.2f})", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_WHITE, 1, cv2.LINE_AA)
        cy += 13
        cv2.putText(img, f"MiniFAS: p_real={p_mf:.2f} ({(1-p_mf):.2f})", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_WHITE, 1, cv2.LINE_AA)
        cy += 13
        fus = as_res.fusion_result
        cv2.putText(img, f"Window: {fus.window_frames}/15 | Agree: {fus.both_real_ratio*100:.0f}%", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.37, COLOR_CYAN, 1, cv2.LINE_AA)

    # 7. Mode 4 Specific: Attendance Gates
    elif current_mode == 4:
        cv2.line(img, (card_x + 8, cy), (card_x + card_w - 8, cy), (60, 60, 80), 1)
        cy += 13
        g1 = "[OK] Sim Match" if (sim_val >= tolerance and not is_unknown) else "[X] Sim Match"
        g2 = "[OK] Anti-Spoof" if as_label == "REAL" else "[X] Anti-Spoof"
        g3 = "[OK] Stable" if "STABLE" in stability_val else "[X] Stable"
        cv2.putText(img, f"G1: {g1} | G2: {g2}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)
        cy += 13
        gate_res = att_status if (not is_unknown and att_status == "PRESENT") else "BLOCKED"
        gate_col = COLOR_EMERALD if gate_res == "PRESENT" else COLOR_RED
        cv2.putText(img, f"G3: {g3} -> {gate_res}", (card_x + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_col, 1, cv2.LINE_AA)


def render_pipeline_debug_panel(
    img: np.ndarray,
    det_fps: float,
    arc_fps: float,
    arc_lat: float,
    as_lat: float,
    enrolled_count: int,
    active_faces: int,
    camera_source: str,
    source_label: str = "LAPTOP WEBCAM",
    device_label: str = "CPU",
    device_short: str = "CPU",
) -> None:
    """
    Renders the AI Pipeline Architecture Debug Panel:
    Camera -> SCRFD -> ArcFace -> Matcher -> Identity Stability -> Anti-Spoof Fusion -> Attendance
    """
    disp_h, disp_w = img.shape[:2]
    panel_w = 260
    panel_h = 370
    panel_x = disp_w - panel_w - 14
    panel_y = 68

    draw_glass_rect(img, panel_x, panel_y, panel_w, panel_h, bg_color=COLOR_BG_DARK, alpha=0.88, border_color=COLOR_CYAN)

    # Title
    cv2.putText(img, "AI PIPELINE ARCHITECTURE", (panel_x + 10, panel_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(img, (panel_x + 8, panel_y + 30), (panel_x + panel_w - 8, panel_y + 30), (70, 70, 90), 1)

    stages = [
        ("1. Camera Stream", f"SOURCE: {source_label[:14]}", COLOR_EMERALD),
        ("2. SCRFD Detector", f"OpenVINO {device_short} ({det_fps:.1f} FPS)", COLOR_EMERALD),
        ("3. ArcFace Embedder", f"512-D ({arc_lat:.1f}ms, {arc_fps:.1f} FPS)", COLOR_EMERALD),
        ("4. Face Matcher", f"Cosine Multi-Ref ({enrolled_count} Enrolled)", COLOR_EMERALD),
        ("5. Identity Stability", f"5-Frame Temporal Window", COLOR_EMERALD if active_faces > 0 else COLOR_GRAY),
        ("6. Anti-Spoof Fusion", f"Dual MN3+MiniFAS ({device_short})", COLOR_EMERALD if active_faces > 0 else COLOR_GRAY),
        ("7. Attendance Gating", "Async Background Queue", COLOR_EMERALD),
    ]

    py = panel_y + 48
    for idx, (name, detail, color) in enumerate(stages):
        # Stage indicator dot
        cv2.circle(img, (panel_x + 16, py - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(img, name, (panel_x + 28, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
        py += 15
        cv2.putText(img, detail, (panel_x + 28, py), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        py += 17

        # Down arrow between stages
        if idx < len(stages) - 1:
            cv2.putText(img, "|", (panel_x + 15, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 140), 1, cv2.LINE_AA)
            cv2.putText(img, "v", (panel_x + 14, py), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_CYAN, 1, cv2.LINE_AA)
            py += 12


def render_mode5_summary_dashboard(
    img: np.ndarray,
    tracker: DemoMetricsTracker,
    display_fps: float,
    det_fps: float,
    arc_fps: float,
    enrolled_count: int,
    source_label: str = "LAPTOP WEBCAM",
    device_label: str = "CPU",
    device_short: str = "CPU",
) -> None:
    """Renders the comprehensive Mode 5 Analytics & Session Summary Dashboard."""
    disp_h, disp_w = img.shape[:2]
    dash_w = min(780, disp_w - 60)
    dash_h = min(460, disp_h - 120)
    dash_x = (disp_w - dash_w) // 2
    dash_y = (disp_h - dash_h) // 2

    draw_glass_rect(img, dash_x, dash_y, dash_w, dash_h, bg_color=COLOR_BG_DARK, alpha=0.92, border_color=COLOR_CYAN, border_thickness=2)

    # Header
    cv2.putText(img, "PRESENTO AI VISION - SESSION SUMMARY DASHBOARD", (dash_x + 20, dash_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_CYAN, 2, cv2.LINE_AA)
    cv2.line(img, (dash_x + 16, dash_y + 44), (dash_x + dash_w - 16, dash_y + 44), (70, 70, 95), 1)

    uptime = time.monotonic() - tracker.start_time
    m, s = divmod(int(uptime), 60)

    # Left Column: Session & Pipeline Performance
    col1_x = dash_x + 24
    col1_y = dash_y + 70
    cv2.putText(img, "PIPELINE PERFORMANCE", (col1_x, col1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (col1_x, col1_y + 6), (col1_x + 300, col1_y + 6), (50, 50, 70), 1)

    metrics_col1 = [
        ("Session Uptime", f"{m:02d}m {s:02d}s"),
        ("Live Display Rate", f"{display_fps:.1f} FPS"),
        ("Camera Feed Source", f"SOURCE: {source_label}"),
        ("SCRFD Face Detector", f"{det_fps:.1f} FPS ({device_short})"),
        ("ArcFace 512-D Embedder", f"{arc_fps:.1f} FPS ({device_short})"),
        ("Inference Device", f"Device: {device_label}"),
        ("Enrolled Student Database", f"{enrolled_count} Students"),
    ]

    py = col1_y + 28
    for label, val in metrics_col1:
        cv2.putText(img, label, (col1_x, py), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, val, (col1_x + 180, py), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_WHITE, 1, cv2.LINE_AA)
        py += 22

    # Right Column: Biometric Encounter Totals
    col2_x = dash_x + 380
    col2_y = dash_y + 70
    cv2.putText(img, "BIOMETRIC ENCOUNTER METRICS", (col2_x, col2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (col2_x, col2_y + 6), (col2_x + 320, col2_y + 6), (50, 50, 70), 1)

    metrics_col2 = [
        ("Total Unique Faces Seen", str(len(tracker.seen_face_ids)), COLOR_CYAN),
        ("Recognized Presentations", str(tracker.recognized_encounters), COLOR_EMERALD),
        ("Unknown Presentations", str(tracker.unknown_encounters), COLOR_GRAY),
        ("Spoof Attacks Blocked", str(tracker.spoof_attempts), COLOR_RED),
        ("Uncertain Liveness Events", str(tracker.uncertain_attempts), COLOR_GOLD),
        ("Attendance Records Logged", str(tracker.attendance_events), COLOR_EMERALD),
    ]

    py = col2_y + 28
    for label, val, col in metrics_col2:
        cv2.putText(img, label, (col2_x, py), cv2.FONT_HERSHEY_SIMPLEX, 0.41, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, val, (col2_x + 220, py), cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)
        py += 22

    # Bottom Half: Recent Attendance Activity Log
    log_y = dash_y + 235
    cv2.line(img, (dash_x + 16, log_y), (dash_x + dash_w - 16, log_y), (70, 70, 95), 1)
    cv2.putText(img, "RECENT ATTENDANCE ACTIVITY LOG", (dash_x + 24, log_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_WHITE, 1, cv2.LINE_AA)

    py = log_y + 44
    if not tracker.recent_logs:
        cv2.putText(img, "[No attendance events recorded in this session yet]", (dash_x + 24, py + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    else:
        for entry in list(tracker.recent_logs)[:4]:
            log_line = (
                f"[{entry.timestamp}]  {entry.name} ({entry.student_id})  |  "
                f"Sim: {entry.similarity:.3f}  |  Anti-Spoof: {entry.anti_spoof_label}  |  Status: {entry.status}"
            )
            cv2.putText(img, log_line, (dash_x + 24, py), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_EMERALD, 1, cv2.LINE_AA)
            py += 20


# ── GUI States for Interactive Demo & In-Window Enrollment ─────────────────────
GUI_STATE_NORMAL = "NORMAL"
GUI_STATE_ENROLL_ID = "ENROLL_ID"
GUI_STATE_ENROLL_NAME = "ENROLL_NAME"
GUI_STATE_ENROLL_CONFIRM = "ENROLL_CONFIRM"
GUI_STATE_ENROLL_CAPTURE = "ENROLL_CAPTURE"
GUI_STATE_ENROLL_COMPLETE = "ENROLL_COMPLETE"

SMART_POSES = [
    "Pose 1/16: Look straight / front-facing at camera",
    "Pose 2/16: Turn head slightly LEFT (~15 deg)",
    "Pose 3/16: Turn head slightly RIGHT (~15 deg)",
    "Pose 4/16: Tilt head slightly UP (~10 deg)",
    "Pose 5/16: Tilt head slightly DOWN (~10 deg)",
    "Pose 6/16: Step back slightly / medium distance",
    "Pose 7/16: Step closer / natural frontal view",
    "Pose 8/16: Natural smile / expression variation",
    "Pose 9/16: Turn head slight UP-LEFT (~10 deg)",
    "Pose 10/16: Turn head slight UP-RIGHT (~10 deg)",
    "Pose 11/16: Turn head slight DOWN-LEFT (~10 deg)",
    "Pose 12/16: Turn head slight DOWN-RIGHT (~10 deg)",
    "Pose 13/16: Neutral expression / front-facing",
    "Pose 14/16: Slight head tilt / roll (~10 deg)",
    "Pose 15/16: Eyes open / relaxed frontal gaze",
    "Pose 16/16: Final natural frontal verification",
]


def render_enrollment_input_dialog(
    img: np.ndarray,
    step_title: str,
    field_label: str,
    value: str,
    hint_text: str,
    extra_info: str = "",
    cursor_visible: bool = True,
) -> None:
    """
    Renders non-blocking in-GUI text input dialog with glassmorphism card and blinking cursor.
    """
    disp_h, disp_w = img.shape[:2]
    card_w = 640
    card_h = 240
    card_x = (disp_w - card_w) // 2
    card_y = (disp_h - card_h) // 2

    # Dim background overlay
    draw_glass_rect(img, 0, 0, disp_w, disp_h, bg_color=(0, 0, 0), alpha=0.45)
    # Centered modal card
    draw_glass_rect(img, card_x, card_y, card_w, card_h, bg_color=COLOR_BG_DARK, alpha=0.94, border_color=COLOR_CYAN)
    draw_tech_corners(img, card_x, card_y, card_x + card_w, card_y + card_h, color=COLOR_CYAN, corner_len=18, thickness=2)

    # Header
    cv2.putText(img, step_title, (card_x + 24, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_CYAN, 2, cv2.LINE_AA)
    cv2.line(img, (card_x + 20, card_y + 48), (card_x + card_w - 20, card_y + 48), (70, 70, 95), 1)

    cy = card_y + 76
    if extra_info:
        cv2.putText(img, extra_info, (card_x + 24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_EMERALD, 1, cv2.LINE_AA)
        cy += 24

    # Field Label
    cv2.putText(img, field_label, (card_x + 24, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_WHITE, 1, cv2.LINE_AA)
    cy += 14

    # Input Box
    box_x = card_x + 24
    box_y = cy
    box_w = card_w - 48
    box_h = 42
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (22, 24, 38), -1)
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), COLOR_CYAN, 1)

    # Render typed text with flashing cursor
    disp_text = value + ("_" if cursor_visible else "")
    cv2.putText(img, disp_text, (box_x + 14, box_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

    # Footer Hint
    cv2.putText(img, hint_text, (card_x + 24, card_y + card_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def render_enrollment_confirm_dialog(
    img: np.ndarray,
    student_id: str,
    student_name: str,
    target_samples: int = 16,
) -> None:
    """
    Renders non-blocking in-GUI confirmation modal before capture starts.
    """
    disp_h, disp_w = img.shape[:2]
    card_w = 640
    card_h = 260
    card_x = (disp_w - card_w) // 2
    card_y = (disp_h - card_h) // 2

    draw_glass_rect(img, 0, 0, disp_w, disp_h, bg_color=(0, 0, 0), alpha=0.45)
    draw_glass_rect(img, card_x, card_y, card_w, card_h, bg_color=COLOR_BG_DARK, alpha=0.94, border_color=COLOR_GOLD)
    draw_tech_corners(img, card_x, card_y, card_x + card_w, card_y + card_h, color=COLOR_GOLD, corner_len=18, thickness=2)

    cv2.putText(img, "CONFIRM DEMO ENROLLMENT (Step 3 of 3)", (card_x + 24, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_GOLD, 2, cv2.LINE_AA)
    cv2.line(img, (card_x + 20, card_y + 48), (card_x + card_w - 20, card_y + 48), (70, 70, 95), 1)

    cv2.putText(img, f"Student ID   :  {student_id}", (card_x + 28, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Student Name :  {student_name}", (card_x + 28, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Capture Plan :  {target_samples} diverse real poses (Frontal, Left, Right, Up, Down, Expressions)", (card_x + 28, card_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(img, f"Quality Gate :  Laplacian blur check, face size, landmarks, and duplicate filter (>0.985)", (card_x + 28, card_y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

    # Actions banner
    draw_glass_rect(img, card_x + 12, card_y + card_h - 48, card_w - 24, 38, bg_color=(25, 30, 45), alpha=0.9)
    cv2.putText(
        img,
        "[Y / ENTER] Confirm and Start Capture     |     [N / ESC] Cancel",
        (card_x + 36, card_y + card_h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def render_enrollment_capture_overlay(
    img: np.ndarray,
    student_name: str,
    student_id: str,
    current_count: int,
    target_samples: int,
    min_samples: int,
    current_pose_instruction: str,
    feedback_msg: str,
    feedback_color: Tuple[int, int, int],
) -> None:
    """
    Renders top progress bar, pose instructions, and quality feedback during guided capture.
    """
    disp_h, disp_w = img.shape[:2]

    # Top Header Banner
    draw_glass_rect(img, 0, 0, disp_w, 95, bg_color=COLOR_BG_DARK, alpha=0.92, border_color=COLOR_CYAN)
    cv2.putText(
        img,
        f"SMART ENROLLMENT (DEMO MODE) - {student_name} ({student_id})",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )

    # Progress bar
    pct = int((current_count / max(1, target_samples)) * 100)
    bx, by, bw, bh = 16, 42, 340, 16
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (35, 38, 50), -1)
    fill_w = int((current_count / max(1, target_samples)) * bw)
    cv2.rectangle(img, (bx, by), (bx + fill_w, by + bh), COLOR_EMERALD, -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)
    cv2.putText(
        img,
        f"Progress: {pct}% [{current_count}/{target_samples}] (Min: {min_samples})",
        (bx + bw + 14, by + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )

    # Control instructions
    cv2.putText(
        img,
        "[C / Space] Capture Sample  |  [S] Save Early (>=10)  |  [ESC] Cancel",
        (disp_w - 490, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLOR_TEXT_DIM,
        1,
        cv2.LINE_AA,
    )

    # Bottom Banner with Pose Instruction
    draw_glass_rect(img, 0, disp_h - 75, disp_w, 75, bg_color=COLOR_BG_DARK, alpha=0.92, border_color=COLOR_CYAN)
    cv2.putText(
        img,
        current_pose_instruction,
        (16, disp_h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        COLOR_CYAN,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        feedback_msg,
        (16, disp_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        feedback_color,
        1,
        cv2.LINE_AA,
    )


def render_enrollment_complete_dialog(
    img: np.ndarray,
    student_name: str,
    student_id: str,
    samples_count: int,
    store_path: str,
) -> None:
    """
    Renders non-blocking in-GUI completion card showing enrolled student details.
    """
    disp_h, disp_w = img.shape[:2]
    card_w = 640
    card_h = 270
    card_x = (disp_w - card_w) // 2
    card_y = (disp_h - card_h) // 2

    draw_glass_rect(img, 0, 0, disp_w, disp_h, bg_color=(0, 0, 0), alpha=0.45)
    draw_glass_rect(img, card_x, card_y, card_w, card_h, bg_color=COLOR_BG_DARK, alpha=0.95, border_color=COLOR_EMERALD)
    draw_tech_corners(img, card_x, card_y, card_x + card_w, card_y + card_h, color=COLOR_EMERALD, corner_len=20, thickness=2)

    cv2.putText(img, "ENROLLMENT COMPLETE - STORE UPDATED", (card_x + 24, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_EMERALD, 2, cv2.LINE_AA)
    cv2.line(img, (card_x + 20, card_y + 48), (card_x + card_w - 20, card_y + 48), (70, 70, 95), 1)

    cv2.putText(img, f"Student Name : {student_name}", (card_x + 28, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Student ID   : {student_id}", (card_x + 28, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"Samples      : {samples_count} real diverse reference embeddings", (card_x + 28, card_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.putText(img, f"Embeddings   : 512-D L2-Normalized ArcFace feature vectors", (card_x + 28, card_y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"Store Status : Saved to {store_path} (Hot-Reloaded)", (card_x + 28, card_y + 198), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_EMERALD, 1, cv2.LINE_AA)

    # Hint
    draw_glass_rect(img, card_x + 12, card_y + card_h - 44, card_w - 24, 34, bg_color=(20, 35, 30), alpha=0.9)
    cv2.putText(
        img,
        "Press any key to return to live recognition (or wait 3s)...",
        (card_x + 36, card_y + card_h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )


def handle_demo_unknown_enrollment(
    camera,
    detector,
    embedder: ArcFaceEmbedder,
    enrollment_store: FaceEnrollmentStore,
    target_samples: int = settings.enrollment_samples_target,
    min_samples: int = settings.enrollment_samples_min,
    duplicate_threshold: float = settings.enrollment_duplicate_threshold,
    student_id: str = "23UAM999",
    name: str = "Demo Guest",
) -> Tuple[bool, str, str]:
    """
    Non-blocking programmatic unknown enrollment helper (never calls console input()).
    Main interactive enrollment is handled directly inside the run_demo GUI state machine.
    """
    logger.info("Starting enrollment for %s (%s)...", name, student_id)

    captured_embeddings: List[np.ndarray] = []
    feedback_msg = "Look at camera to begin."
    feedback_color = COLOR_EMERALD
    feedback_time = time.monotonic()

    SMART_POSES = [
        "Pose 1/12: Look straight / front-facing at camera",
        "Pose 2/12: Turn head slightly LEFT (~15 deg)",
        "Pose 3/12: Turn head slightly RIGHT (~15 deg)",
        "Pose 4/12: Tilt head slightly UP (~10 deg)",
        "Pose 5/12: Tilt head slightly DOWN (~10 deg)",
        "Pose 6/12: Step back slightly / farther from camera",
        "Pose 7/12: Step closer / natural frontal view",
        "Pose 8/12: Slight smile / natural expression",
        "Pose 9/12: Neutral expression / front-facing",
        "Pose 10/12: Turn head slight UP-LEFT (~10 deg)",
        "Pose 11/12: Turn head slight UP-RIGHT (~10 deg)",
        "Pose 12/12: Final natural frontal gaze",
    ]

    while len(captured_embeddings) < target_samples:
        try:
            ok, frame = camera.read(timeout=0.04)
        except TypeError:
            ok, frame = camera.read()

        if not ok or frame is None:
            time.sleep(0.005)
            continue

        faces = detector.detect(frame)
        is_quality_ok, quality_msg, valid_face = assess_enrollment_quality(faces, frame)

        display_frame = cv2.flip(frame, 1)
        disp_h, disp_w = display_frame.shape[:2]

        if faces:
            for face in faces:
                m_bbox = transform_mirrored_bbox(face.bbox, disp_w)
                m_kps = transform_mirrored_landmarks(face.landmarks, disp_w)
                box_color = COLOR_EMERALD if is_quality_ok else COLOR_RED
                cv2.rectangle(display_frame, (m_bbox[0], m_bbox[1]), (m_bbox[2], m_bbox[3]), box_color, 2)
                for pt in m_kps:
                    cv2.circle(display_frame, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 255), -1)

        # Header banner
        draw_glass_rect(display_frame, 0, 0, disp_w, 95, bg_color=COLOR_BG_DARK, alpha=0.9)
        cv2.putText(display_frame, f"SMART ENROLLMENT (DEMO MODE) - {name} ({student_id})", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_WHITE, 2, cv2.LINE_AA)

        # Progress bar
        curr_cnt = len(captured_embeddings)
        pct = int((curr_cnt / max(1, target_samples)) * 100)
        bx, by, bw, bh = 16, 42, 300, 16
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (40, 40, 50), -1)
        fill_w = int((curr_cnt / max(1, target_samples)) * bw)
        cv2.rectangle(display_frame, (bx, by), (bx + fill_w, by + bh), COLOR_EMERALD, -1)
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)
        cv2.putText(display_frame, f"Progress: {pct}% [{curr_cnt}/{target_samples}]", (bx + bw + 12, by + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)

        # Controls hint
        cv2.putText(display_frame, "[C / Space] Capture | [S] Save Early | [Q / ESC] Cancel", (disp_w - 420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Bottom banner with pose instruction
        curr_pose = SMART_POSES[min(curr_cnt, len(SMART_POSES) - 1)]
        draw_glass_rect(display_frame, 0, disp_h - 75, disp_w, 75, bg_color=COLOR_BG_DARK, alpha=0.9)
        cv2.putText(display_frame, curr_pose, (16, disp_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_CYAN, 2, cv2.LINE_AA)

        if time.monotonic() - feedback_time > 2.5:
            feedback_msg = quality_msg
            feedback_color = COLOR_EMERALD if is_quality_ok else COLOR_AMBER

        cv2.putText(display_frame, feedback_msg, (16, disp_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, feedback_color, 1, cv2.LINE_AA)

        cv2.imshow("Presento AI Vision Demo", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("\n  [CANCELLED] Enrollment aborted by user.\n")
            return False, name, student_id

        elif key in (ord("s"), ord("S")):
            if len(captured_embeddings) >= min_samples:
                print(f"\n  [INFO] Saving enrollment early with {len(captured_embeddings)} samples.")
                break
            else:
                feedback_msg = f"Need at least {min_samples} samples! (Have {len(captured_embeddings)})"
                feedback_color = COLOR_RED
                feedback_time = time.monotonic()

        elif key in (ord("c"), ord("C"), 32):  # C or SPACE
            if not is_quality_ok or valid_face is None:
                feedback_msg = f"Rejected: {quality_msg}"
                feedback_color = COLOR_RED
                feedback_time = time.monotonic()
                continue

            aligned = align_face(frame, valid_face.landmarks)
            try:
                emb = embedder.get_embedding(aligned)
            except Exception as exc:
                feedback_msg = f"Extraction failed: {exc}"
                feedback_color = COLOR_RED
                feedback_time = time.monotonic()
                continue

            # Check full duplicate rejection
            if captured_embeddings:
                sims = [compute_cosine_similarity(emb, prev) for prev in captured_embeddings]
                max_sim = max(sims)
                if max_sim > duplicate_threshold:
                    feedback_msg = f"Sample too similar to prior ({max_sim:.3f} > {duplicate_threshold:.2f}) - adjust pose"
                    feedback_color = COLOR_AMBER
                    feedback_time = time.monotonic()
                    continue

            captured_embeddings.append(emb)
            sample_num = len(captured_embeddings)
            feedback_msg = f"Sample {sample_num}/{target_samples} Captured Successfully!"
            feedback_color = COLOR_EMERALD
            feedback_time = time.monotonic()
            print(f"  [OK] Sample {sample_num}/{target_samples} accepted.")

    if len(captured_embeddings) < min_samples:
        print(f"  [ERROR] Only {len(captured_embeddings)} samples captured. Enrollment failed.")
        return False, name, student_id

    # Store enrollment
    enrollment_store.add_student(
        student_id=student_id,
        name=name,
        embeddings=captured_embeddings,
    )
    saved_path = enrollment_store.save()

    print("\n" + "=" * 65)
    print("  ENROLLMENT SUCCESSFUL")
    print(f"  Name        : {name}")
    print(f"  Student ID  : {student_id}")
    print(f"  Samples     : {len(captured_embeddings)} reference embeddings")
    print(f"  Saved Store : {saved_path}")
    print("  Returning to live recognition...")
    print("=" * 65 + "\n")

    # Render a 1.5-second success screen
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        try:
            ok, frame = camera.read(timeout=0.035)
        except (TypeError, cv2.error, Exception):
            ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        display_frame = cv2.flip(frame, 1)
        dh, dw = display_frame.shape[:2]
        draw_glass_rect(display_frame, dw // 2 - 220, dh // 2 - 70, 440, 140, bg_color=(20, 50, 30), alpha=0.92, border_color=COLOR_EMERALD)
        cv2.putText(display_frame, "ENROLLMENT SUCCESSFUL", (dw // 2 - 170, dh // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, COLOR_EMERALD, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"{name} ({student_id})", (dw // 2 - 170, dh // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "Returning to live recognition...", (dw // 2 - 170, dh // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.imshow("Presento AI Vision Demo", display_frame)
        cv2.waitKey(20)

    return True, name, student_id


# ── OpenVINO Device Resolution & Model Integrity Checks ──────────────────────

def resolve_openvino_device(requested_device: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Checks OpenVINO device availability and resolves the execution device.
    Delegates to edge.inference_backend.select_inference_device.
    Returns:
        (active_device, hud_label, short_label)
        e.g. ("CPU", "CPU", "CPU") or ("GPU", "GPU - Intel Iris Xe", "GPU")
    """
    decision = select_inference_device(preferred_device=requested_device)
    hud_label = f"GPU - {decision.short_device_name}" if decision.selected_device == "GPU" else "CPU"
    return decision.selected_device, hud_label, decision.selected_device


def verify_required_models() -> bool:
    """
    Verifies that all 4 required ONNX models exist on disk before starting the demo.
    Returns True if all exist, False otherwise.
    """
    required_models = [
        ("SCRFD Face Detector", settings.scrfd_model_path),
        ("ArcFace Embedder", settings.arcface_model_path),
        ("Anti-Spoof MobileNetV3", settings.anti_spoof_model_path),
        ("Anti-Spoof MiniFASNet V2 SE", settings.minifas_model_path),
    ]
    missing = [(name, path) for name, path in required_models if not os.path.isfile(path)]
    if missing:
        msg = (
            "\n" + "=" * 65 + "\n"
            "[ERROR] Required AI model files are missing from data/models/:\n\n"
        )
        for name, path in missing:
            msg += f"  - Missing: {name} -> {path}\n"
        msg += (
            "\nPlease ensure all 4 ONNX models are present in data/models/:\n"
            "  1. data/models/scrfd_500m_bnkps.onnx\n"
            "  2. data/models/w600k_mbf.onnx\n"
            "  3. data/models/anti-spoof-mn3.onnx\n"
            "  4. data/models/minifasnet_v2se.onnx\n"
            + "=" * 65 + "\n"
        )
        logger.error("Demo cannot start: %d model file(s) missing.", len(missing))
        print(msg)
        return False
    return True


# ── Main Demo Execution Loop ──────────────────────────────────────────────────

def run_demo(
    source: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: Optional[float] = None,
    initial_mode: int = 1,
    show_pipeline: bool = False,
    headless: bool = False,
    max_frames: int = 0,
    duration: float = 0.0,
) -> None:
    """
    Executes the Presento AI Vision Demo with guaranteed camera acquisition,
    zero-buffering display loop, and guaranteed cleanup upon exit.
    """
    _ensure_stdin_listener()
    logger.info("Initializing Presento AI Vision Demo (Mode %d)...", initial_mode)

    # 0. Model existence verification (fail clearly if any model is missing)
    if not verify_required_models():
        return

    # 1. Hardware Detection & Intelligent Inference Backend Selection
    device_decision = select_inference_device()
    active_device = device_decision.selected_device
    device_label = device_decision.display_name
    device_short = device_decision.short_device_name
    cpu_hud_str, gpu_hud_str = (
        device_decision.hardware_report.get_compact_hud_labels()
        if device_decision.hardware_report
        else ("CPU", "None")
    )
    ai_hud_str = device_decision.get_hud_label()

    # Print polished startup hardware report to console
    print(device_decision.format_startup_box())
    logger.info(
        "Inference hardware selected: %s (%s, Backend: %s, Status: %s)",
        active_device,
        device_label,
        device_decision.backend_name,
        device_decision.status,
    )

    # 2. Camera source determination
    cam_src_setting = source or settings.camera_source
    src_str = str(cam_src_setting).strip()
    src_lower = src_str.lower()

    # Determine camera type: stream vs webcam vs fallback (mock/file)
    is_stream = src_lower in ("stream", "pi", "network") or bool(host) or (":" in src_str and not src_str.startswith("http"))
    is_webcam = src_lower in ("webcam", "laptop", "local") or (src_str.isdigit() and not is_stream)

    # Resolve target host and stream port for remote stream
    target_host = host or settings.pi_ip
    target_stream_port = port or settings.stream_port
    target_control_port = settings.control_port

    if is_stream and src_lower not in ("stream", "pi", "network"):
        if ":" in src_str:
            parts = src_str.split(":", 1)
            target_host = parts[0]
            try:
                target_stream_port = int(parts[1])
            except ValueError:
                pass
        elif "." in src_str or src_str != "stream":
            target_host = src_str

    cam_session = None
    camera = None

    if is_stream:
        # Raspberry Pi hardware session & TCP FrameReceiver
        source_label = "RASPBERRY PI STREAM"
        cam_session = CameraSession(
            host=target_host,
            control_port=target_control_port,
            stream_port=target_stream_port,
            session_type="demo",
        )
        try:
            cam_session.start()
        except CameraInUseError as exc:
            logger.error("Camera currently in use by another session: %s. Demo cannot start.", exc)
            print(f"\n[ERROR] Camera currently in use: {exc}\n")
            return
        except CameraAuthError as exc:
            logger.error("Camera control authentication failed: %s. Check PRESENTO_CONTROL_TOKEN.", exc)
            print(f"\n[ERROR] Camera control authentication failed: {exc}\n")
            return
        except Exception as exc:
            logger.error("Could not activate camera session: %s", exc)
            print(f"\n[ERROR] Could not activate camera session: {exc}")
            print("Running camera discovery diagnostics to check network connectivity...\n")
            try:
                from .discovery import main as discovery_main
                discovery_main(argv=["--host", target_host, "--port", str(target_stream_port), "--timeout", str(timeout or 1.5)])
            except Exception as disc_err:
                logger.debug("Discovery diagnostic execution failed: %s", disc_err)
            return

        logger.info("Connecting to Raspberry Pi camera stream at %s:%d...", target_host, target_stream_port)
        camera = FrameReceiver(host=target_host, port=target_stream_port)
        camera.start()

    elif is_webcam:
        # Laptop webcam / Local camera via OpenCV (DO NOT contact Pi, port 5001/5002, or CameraSession)
        source_label = "LAPTOP WEBCAM"
        cam_session = None
        cv_src: int | str = settings.camera_index if src_lower in ("webcam", "laptop", "local") else int(src_str)
        logger.info("Initializing laptop webcam with OpenCV (index=%s)...", cv_src)

        if os.name == "nt":
            camera = cv2.VideoCapture(cv_src, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera = cv2.VideoCapture(cv_src)
        else:
            camera = cv2.VideoCapture(cv_src)

        if not camera.isOpened():
            msg = (
                "\n" + "=" * 55 + "\n"
                "[ERROR] Could not open laptop webcam.\n\n"
                f"Failed to access camera index: {cv_src}\n\n"
                "Troubleshooting Steps:\n"
                "1. Another camera index may be selected. Try setting:\n"
                "   CAMERA_INDEX=1   (or 2, 3)\n"
                "   in your .env file or environment variables.\n"
                "2. Check Windows Camera Privacy Settings (Settings -> Privacy & Security -> Camera)\n"
                "   and verify 'Let desktop apps access your camera' is ON.\n"
                "3. Make sure another application (Zoom, Teams, Skype, Browser) is not using the camera.\n"
                + "=" * 55 + "\n"
            )
            print(msg)
            logger.error("Could not open laptop webcam at index %s", cv_src)
            return

        # Preserve native 30 FPS mode unless custom resolution is explicitly requested.
        # Unconditionally forcing 1280x720 over DirectShow causes hardware-level 10 FPS
        # throttling on many integrated laptop webcams.
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

        actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Laptop webcam opened successfully: %dx%d (index %s)", actual_w, actual_h, cv_src)
        camera = LatestFrameCamera(camera)

    else:
        # Fallback for custom video sources or mock
        source_label = src_str.upper()
        if src_str == "mock":
            cam_session = CameraSession(is_local_camera=True, session_type="demo")
            cam_session.start()
        cv_src = src_str
        camera = cv2.VideoCapture(cv_src)
        if not camera.isOpened():
            if cam_session is not None:
                cam_session.stop()
            logger.error("Local camera source '%s' could not be opened.", cv_src)
            print(f"\n[ERROR] Local camera source '{cv_src}' could not be opened.\n")
            return
        logger.info("Local camera source opened: %s", cv_src)
        camera = LatestFrameCamera(camera)

    detector = None
    as_worker = None
    attendance_worker = None

    try:
        # 3. Modern SCRFD Face Detector & ArcFace Embedder
        detector = SCRFDWorker(
            model_path=settings.scrfd_model_path,
            device=active_device,
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

        # 4. Face Enrollment Store & Matcher
        enrollment_store = FaceEnrollmentStore(settings.face_enrollment_path)
        matcher = FaceMatcher(enrollment_store)
        enrolled_count = len(enrollment_store.get_all_students())
        logger.info("FaceMatcher initialized with %d enrolled students from %s", enrolled_count, settings.face_enrollment_path)

        # 5. Asynchronous Dual Anti-Spoof Worker (MN3 + MiniFASNet V2 SE)
        as_worker = AntiSpoofWorker(
            mn3_model_path=settings.anti_spoof_model_path,
            minifas_model_path=settings.minifas_model_path,
            device=active_device,
            window_size=15,
            real_ratio_threshold=0.80,
            spoof_ratio_threshold=0.80,
        )
        as_worker.start()

        # 6. Asynchronous Attendance I/O Worker
        attendance_worker = AttendanceIOWorker(flush_interval=10.0)
        attendance_worker.start()

        # Metrics & State
        metrics_tracker = DemoMetricsTracker()
        cooldown: Dict[str, float] = {}
        active_session_id: Optional[int] = None
        last_session_query_time = 0.0

        current_mode = initial_mode
        pipeline_panel_open = show_pipeline
        is_paused = False
        paused_frame = None

        frame_count = 0
        frame_timestamps: deque[float] = deque(maxlen=30)
        display_fps = 0.0

        tolerance = settings.tolerance

        # In-GUI Non-blocking Enrollment State Machine
        gui_state = GUI_STATE_NORMAL
        enroll_input_id = ""
        enroll_input_name = ""
        enroll_captured_embeddings: List[np.ndarray] = []
        enroll_feedback_msg = "Look at camera to begin."
        enroll_feedback_color = COLOR_EMERALD
        enroll_feedback_time = 0.0
        enroll_saved_path = ""
        enroll_complete_start = 0.0
        enroll_target_samples = settings.enrollment_samples_target
        enroll_min_samples = settings.enrollment_samples_min
        enroll_duplicate_threshold = settings.enrollment_duplicate_threshold

        logger.info(
            "Presento AI Vision Demo running (Device: %s, Source: %s, Headless: %s)",
            settings.openvino_device,
            cam_src_setting,
            headless,
        )

        demo_start_time = time.monotonic()
        last_perf_log = demo_start_time

        while True:
            # 1. Frame Acquisition
            if not is_paused:
                try:
                    ok, frame = camera.read(timeout=0.035)
                except TypeError:
                    ok, frame = camera.read()

                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                # Submit original unmirrored frame to detector & embedder
                detector.submit_frame(frame)
                paused_frame = frame.copy()
            else:
                frame = paused_frame.copy() if paused_frame is not None else None
                if frame is None:
                    time.sleep(0.02)
                    continue

            # 2. Get latest AI inferences
            faces: List[SCRFDFace] = detector.get_latest_detections()

            # FPS calculation
            frame_count += 1
            now = time.monotonic()
            frame_timestamps.append(now)
            if len(frame_timestamps) >= 2:
                dt = frame_timestamps[-1] - frame_timestamps[0]
                if dt > 0:
                    display_fps = round((len(frame_timestamps) - 1) / dt, 1)

            det_stats = detector.get_stats()
            det_fps = det_stats.get("detection_fps", 0.0)
            arc_fps = det_stats.get("arcface_fps", 0.0)
            arc_lat = det_stats.get("arcface_latency_ms", 0.0)

            if now - last_perf_log >= 5.0:
                last_perf_log = now
                logger.info(
                    "[DEMO PERF] Display: %.1f FPS | SCRFD: %.1f FPS | ArcFace: %.1f FPS (%.1fms) | Faces: %d",
                    display_fps,
                    det_fps,
                    arc_fps,
                    arc_lat,
                    len(faces),
                )

            # Match live ArcFace embeddings
            face_matches: Dict[int, Optional[MatchResult]] = {}
            for face in faces:
                if face.embedding is not None:
                    m = matcher.match(face.embedding, face_id=face.face_id, frame_id=frame_count)
                    face_matches[face.face_id] = m

            # Submit face crops for asynchronous Anti-Spoof inference
            if not is_paused:
                active_fids = {f.face_id for f in faces}
                for face in faces:
                    as_worker.submit_face(face.face_id, frame_count, frame, face.bbox)
                as_worker.prune_stale(active_fids, max_idle_seconds=3.0)

            # Retrieve latest anti-spoof results
            face_anti_spoof: Dict[int, Optional[AntiSpoofResult]] = {}
            for face in faces:
                as_res = as_worker.get_result(face.face_id)
                face_anti_spoof[face.face_id] = as_res

            as_stats = as_worker.get_stats()
            as_lat = as_stats.get("latest_latency_ms", 0.0)

            # 3. Attendance Gating & Cooldown Management
            # Criteria: Matched Enrolled Student + Identity Stable + Anti-Spoof Temporal REAL + Fresh + Cooldown
            # Strict Safety Gate: Evaluated ONLY during normal live recognition (never during enrollment)
            if gui_state == GUI_STATE_NORMAL:
                for face in faces:
                    fid = face.face_id
                    m = face_matches.get(fid)
                    as_res = face_anti_spoof.get(fid)
                    if m is not None and m.best_match is not None and as_res is not None:
                        cand = m.best_match
                        fus = as_res.fusion_result
                        is_fresh = (now - as_res.updated_at) <= 2.0

                        # CLOSED-SET ENROLLED-ONLY GATE:
                        # Must strictly pass recognition acceptance criteria (similarity >= tolerance)
                        if cand.similarity < tolerance or not getattr(m, "is_recognized", True):
                            continue  # UNKNOWN face - BLOCKED from attendance

                        if m.is_stable and fus.temporal_label == "REAL" and is_fresh:
                            rem_cd = cooldown.get(cand.student_id, 0.0) - now
                            if rem_cd <= 0.0:
                                # Dynamically query active session from FastAPI backend
                                if active_session_id is None or (now - last_session_query_time) > 30.0:
                                    last_session_query_time = now
                                    sess_info = get_active_session()
                                    if sess_info:
                                        active_session_id = sess_info.get("id")

                                # Explicit temporal liveness state from AI engine
                                l_state = "REAL_LOCKED"
                                if getattr(as_res, "liveness_decision", None) is not None:
                                    l_state = as_res.liveness_decision.state.value

                                # Attendance event triggered!
                                cooldown[cand.student_id] = now + settings.cooldown_seconds
                                event = {
                                    "event_id": str(uuid.uuid4()),
                                    "student_id": cand.student_id,
                                    "session_id": active_session_id or 1,
                                    "confidence": round(float(cand.similarity), 3),
                                    "liveness_state": l_state,
                                    "liveness_score": round(float(fus.avg_p_real), 3),
                                    "source": "AI_CAMERA",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "device_id": settings.device_id,
                                    "observed_at": datetime.now(timezone.utc).isoformat(),
                                    "face_quality_score": round(float(face.confidence), 3),
                                    "room": settings.room,
                                }
                                attendance_worker.submit_event(event)
                                metrics_tracker.log_attendance(
                                    name=cand.name,
                                    student_id=cand.student_id,
                                    similarity=cand.similarity,
                                    anti_spoof_label=f"REAL ({fus.both_real_ratio * 100:.0f}%)",
                                )
                                beep()
                                logger.info(
                                    ">>> DEMO ATTENDANCE RECORDED: %s (%s) [Sim: %.3f, Liveness: %s (%.0f%% REAL), Session: %s] <<<",
                                    cand.name,
                                    cand.student_id,
                                    cand.similarity,
                                    l_state,
                                    fus.both_real_ratio * 100,
                                    active_session_id or "default",
                                )

            # Register encounters into metrics tracker
            face_states = {
                f.face_id: classify_face_state(
                    f,
                    face_matches.get(f.face_id),
                    face_anti_spoof.get(f.face_id),
                    cooldown.get(
                        face_matches[f.face_id].best_match.student_id if (face_matches.get(f.face_id) and face_matches[f.face_id].best_match) else "",
                        0.0,
                    ) - now,
                    tolerance,
                )[0]
                for f in faces
            }
            metrics_tracker.register_faces(faces, face_states, face_matches, face_anti_spoof)

            # 4. Horizontal Mirroring and Visual Rendering
            if not headless:
                display_frame = cv2.flip(frame, 1)
                disp_h, disp_w = display_frame.shape[:2]
                cursor_visible = int(now * 2) % 2 == 0

                if gui_state == GUI_STATE_NORMAL:
                    # Draw each detected face
                    for face in faces:
                        m_bbox = transform_mirrored_bbox(face.bbox, disp_w)
                        m_landmarks = transform_mirrored_landmarks(face.landmarks, disp_w)

                        m = face_matches.get(face.face_id)
                        as_res = face_anti_spoof.get(face.face_id)
                        cd_rem = 0.0
                        if m is not None and m.best_match is not None:
                            cd_rem = max(0.0, cooldown.get(m.best_match.student_id, 0.0) - now)

                        render_face_callout_card(
                            img=display_frame,
                            face=face,
                            m_bbox=m_bbox,
                            m_landmarks=m_landmarks,
                            match_res=m,
                            as_res=as_res,
                            cooldown_rem=cd_rem,
                            current_mode=current_mode,
                            tolerance=tolerance,
                        )

                    # Render Pipeline Architecture HUD (if toggled)
                    if pipeline_panel_open:
                        render_pipeline_debug_panel(
                            img=display_frame,
                            det_fps=det_fps,
                            arc_fps=arc_fps,
                            arc_lat=arc_lat,
                            as_lat=as_lat,
                            enrolled_count=enrolled_count,
                            active_faces=len(faces),
                            camera_source=str(cam_src_setting),
                            source_label=source_label,
                            device_label=device_label,
                            device_short=device_short,
                        )

                    # Render Mode 5 Summary Dashboard (if Mode 5 is active)
                    if current_mode == 5:
                        render_mode5_summary_dashboard(
                            img=display_frame,
                            tracker=metrics_tracker,
                            display_fps=display_fps,
                            det_fps=det_fps,
                            arc_fps=arc_fps,
                            enrolled_count=enrolled_count,
                            source_label=source_label,
                            device_label=device_label,
                            device_short=device_short,
                        )

                    # Render Top Header HUD and Bottom Mode Banner
                    render_top_header_hud(
                        img=display_frame,
                        current_mode=current_mode,
                        display_fps=display_fps,
                        det_fps=det_fps,
                        arc_fps=arc_fps,
                        arc_lat=arc_lat,
                        enrolled_count=enrolled_count,
                        camera_active=True,
                        is_paused=is_paused,
                        source_label=source_label,
                        device_label=device_label,
                        device_short=device_short,
                        cpu_label=cpu_hud_str,
                        gpu_label=gpu_hud_str,
                        ai_label=ai_hud_str,
                    )

                    # Unknown Person Detected Notification Prompt (above bottom mode banner)
                    has_unknown_face = any(s == STATE_UNKNOWN for s in face_states.values())
                    if has_unknown_face:
                        prompt_h = 24
                        prompt_y = disp_h - 36 - prompt_h - 4
                        draw_glass_rect(display_frame, 0, prompt_y, disp_w, prompt_h, bg_color=(20, 20, 35), alpha=0.92, border_color=COLOR_CYAN)
                        cv2.putText(
                            display_frame,
                            "UNKNOWN PERSON DETECTED  |  [E] Enroll this person (Demo)  |  [R] Retry Recognition  |  [Q] Quit",
                            (16, prompt_y + 17),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            COLOR_CYAN,
                            1,
                            cv2.LINE_AA,
                        )

                    render_bottom_mode_banner(img=display_frame, current_mode=current_mode)

                elif gui_state == GUI_STATE_ENROLL_ID:
                    render_enrollment_input_dialog(
                        img=display_frame,
                        step_title="DEMO ENROLLMENT - ENTER STUDENT ID (Step 1 of 3)",
                        field_label="STUDENT ID (e.g. 23UAM123, GUEST01):",
                        value=enroll_input_id,
                        hint_text="Type student ID, then press [ENTER] to proceed  |  [ESC] Cancel",
                        extra_info="Enroll new student into Presento local face database",
                        cursor_visible=cursor_visible,
                    )

                elif gui_state == GUI_STATE_ENROLL_NAME:
                    render_enrollment_input_dialog(
                        img=display_frame,
                        step_title="DEMO ENROLLMENT - ENTER STUDENT NAME (Step 2 of 3)",
                        field_label="FULL NAME (e.g. Alex Morgan):",
                        value=enroll_input_name,
                        hint_text="Type student name, then press [ENTER] to confirm  |  [ESC] Cancel",
                        extra_info=f"Target Student ID: {enroll_input_id}",
                        cursor_visible=cursor_visible,
                    )

                elif gui_state == GUI_STATE_ENROLL_CONFIRM:
                    render_enrollment_confirm_dialog(
                        img=display_frame,
                        student_id=enroll_input_id,
                        student_name=enroll_input_name,
                        target_samples=enroll_target_samples,
                    )

                elif gui_state == GUI_STATE_ENROLL_CAPTURE:
                    is_quality_ok, quality_msg, valid_face = assess_enrollment_quality(faces, frame)
                    for face in faces:
                        m_bbox = transform_mirrored_bbox(face.bbox, disp_w)
                        m_kps = transform_mirrored_landmarks(face.landmarks, disp_w)
                        box_color = COLOR_EMERALD if is_quality_ok else COLOR_RED
                        cv2.rectangle(display_frame, (m_bbox[0], m_bbox[1]), (m_bbox[2], m_bbox[3]), box_color, 2)
                        for pt in m_kps:
                            cv2.circle(display_frame, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 255), -1)

                    curr_cnt = len(enroll_captured_embeddings)
                    curr_pose = SMART_POSES[min(curr_cnt, len(SMART_POSES) - 1)]

                    if now - enroll_feedback_time > 2.5:
                        enroll_feedback_msg = quality_msg
                        enroll_feedback_color = COLOR_EMERALD if is_quality_ok else COLOR_AMBER

                    render_enrollment_capture_overlay(
                        img=display_frame,
                        student_name=enroll_input_name,
                        student_id=enroll_input_id,
                        current_count=curr_cnt,
                        target_samples=enroll_target_samples,
                        min_samples=enroll_min_samples,
                        current_pose_instruction=curr_pose,
                        feedback_msg=enroll_feedback_msg,
                        feedback_color=enroll_feedback_color,
                    )

                elif gui_state == GUI_STATE_ENROLL_COMPLETE:
                    render_enrollment_complete_dialog(
                        img=display_frame,
                        student_name=enroll_input_name,
                        student_id=enroll_input_id,
                        samples_count=len(enroll_captured_embeddings),
                        store_path=enroll_saved_path or str(settings.embeddings_path),
                    )

                # Display the rendering window
                cv2.imshow("Presento AI Vision Demo", display_frame)

                # Keyboard Interactivity - cv2.waitKey(1) runs on EVERY single frame!
                key = cv2.waitKey(1) & 0xFF
                if key in (255, 0):
                    if not _stdin_key_queue.empty():
                        key = _stdin_key_queue.get_nowait()
                    else:
                        try:
                            import msvcrt
                            if msvcrt.kbhit():
                                key = ord(msvcrt.getch())
                        except Exception:
                            pass

                if gui_state == GUI_STATE_NORMAL:
                    # Quit on 'q' or ESC
                    if key in (ord("q"), ord("Q"), 27):
                        logger.info("User requested demo shutdown (key '%s').", chr(key) if key != 27 else "ESC")
                        break

                    # Direct mode keys 1 to 5
                    elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                        current_mode = int(chr(key))
                        logger.info("Switched to Demo Mode %d", current_mode)

                    # Mode cycling on 'm' or TAB (key 9)
                    elif key in (ord("m"), ord("M"), 9):
                        current_mode = (current_mode % 5) + 1
                        logger.info("Cycled to Demo Mode %d", current_mode)

                    # Toggle pipeline panel on 'p'
                    elif key in (ord("p"), ord("P")):
                        pipeline_panel_open = not pipeline_panel_open
                        logger.info("Pipeline Architecture Panel: %s", "VISIBLE" if pipeline_panel_open else "HIDDEN")

                    # Freeze / Pause on Spacebar
                    elif key == 32:
                        is_paused = not is_paused
                        logger.info("Demo playback: %s", "FROZEN (Pause)" if is_paused else "RESUMED (Live)")

                    # Enroll Unknown Student on 'e' or 'E'
                    elif key in (ord("e"), ord("E")):
                        logger.info("User requested demo unknown enrollment flow via in-GUI state machine.")
                        gui_state = GUI_STATE_ENROLL_ID
                        enroll_input_id = ""
                        enroll_input_name = ""
                        enroll_captured_embeddings.clear()
                        enroll_feedback_msg = "Look at camera to begin."
                        enroll_feedback_color = COLOR_EMERALD
                        enroll_feedback_time = now

                    # Retry recognition / clear tracker cache on 'r' or 'R'
                    elif key in (ord("r"), ord("R")):
                        logger.info("User requested recognition retry: clearing stability and anti-spoof cache.")
                        matcher.reset_history()
                        as_worker._state.clear()

                    # Diagnostic Dump on 'd' or 'D'
                    elif key in (ord("d"), ord("D")):
                        print(device_decision.format_diagnostic_tree(display_fps, det_fps, arc_fps, as_lat))
                        print("  ACTIVE FACES IN VIEW:")
                        if not faces:
                            print("  [No faces detected in current frame]")
                        else:
                            for f in faces:
                                fid = f.face_id
                                m = face_matches.get(fid)
                                as_r = face_anti_spoof.get(fid)
                                cand_str = f"{m.best_match.name} ({m.best_match.student_id}) [Sim: {m.best_match.similarity:.3f}]" if (m and m.best_match) else "Unknown"
                                as_str = as_r.fusion_result.temporal_label if as_r else "CHECKING"
                                print(f"  [Face #{fid}] State: {face_states.get(fid)} | Candidate: {cand_str} | Anti-Spoof: {as_str}")
                        print("============================================================\n")

                elif gui_state == GUI_STATE_ENROLL_ID:
                    if key == 27:  # ESC -> Cancel
                        gui_state = GUI_STATE_NORMAL
                        logger.info("Demo enrollment cancelled by user (ESC).")
                    elif key in (13, 10):  # ENTER
                        cleaned_id = enroll_input_id.strip().upper()
                        if cleaned_id:
                            enroll_input_id = cleaned_id
                            gui_state = GUI_STATE_ENROLL_NAME
                        else:
                            enroll_input_id = "23UAM999"
                            gui_state = GUI_STATE_ENROLL_NAME
                    elif key in (8, 127):  # Backspace
                        enroll_input_id = enroll_input_id[:-1]
                    elif 32 <= key <= 126:
                        if len(enroll_input_id) < 20:
                            enroll_input_id += chr(key).upper()

                elif gui_state == GUI_STATE_ENROLL_NAME:
                    if key == 27:  # ESC -> Cancel
                        gui_state = GUI_STATE_NORMAL
                        logger.info("Demo enrollment cancelled by user (ESC).")
                    elif key in (13, 10):  # ENTER
                        cleaned_name = enroll_input_name.strip()
                        if cleaned_name:
                            enroll_input_name = cleaned_name
                            gui_state = GUI_STATE_ENROLL_CONFIRM
                        else:
                            enroll_input_name = "Demo Guest"
                            gui_state = GUI_STATE_ENROLL_CONFIRM
                    elif key in (8, 127):  # Backspace
                        enroll_input_name = enroll_input_name[:-1]
                    elif 32 <= key <= 126:
                        if len(enroll_input_name) < 32:
                            enroll_input_name += chr(key)

                elif gui_state == GUI_STATE_ENROLL_CONFIRM:
                    if key in (ord("y"), ord("Y"), 13, 10):  # Confirm
                        gui_state = GUI_STATE_ENROLL_CAPTURE
                        enroll_captured_embeddings.clear()
                        enroll_feedback_msg = "Enrollment started - look at camera"
                        enroll_feedback_color = COLOR_EMERALD
                        enroll_feedback_time = now
                        logger.info("Enrollment confirmed for %s (%s). Starting capture.", enroll_input_name, enroll_input_id)
                    elif key in (ord("n"), ord("N"), 27):  # Cancel
                        gui_state = GUI_STATE_NORMAL
                        logger.info("Enrollment cancelled by user at confirmation.")

                elif gui_state == GUI_STATE_ENROLL_CAPTURE:
                    is_quality_ok, quality_msg, valid_face = assess_enrollment_quality(faces, frame)
                    if key in (ord("q"), ord("Q"), 27):  # Cancel
                        gui_state = GUI_STATE_NORMAL
                        logger.info("Enrollment capture aborted by user.")
                    elif key in (ord("s"), ord("S")):  # Save early
                        if len(enroll_captured_embeddings) >= enroll_min_samples:
                            enrollment_store.add_student(
                                student_id=enroll_input_id,
                                name=enroll_input_name,
                                embeddings=enroll_captured_embeddings,
                            )
                            enroll_saved_path = enrollment_store.save()
                            enrolled_count = len(enrollment_store.get_all_students())
                            matcher = FaceMatcher(enrollment_store, tolerance=tolerance)
                            matcher.reset_history()
                            as_worker._state.clear()
                            gui_state = GUI_STATE_ENROLL_COMPLETE
                            enroll_complete_start = now
                            logger.info(
                                "Demo enrollment saved early (%d samples) for %s (%s). FaceMatcher reloaded.",
                                len(enroll_captured_embeddings),
                                enroll_input_name,
                                enroll_input_id,
                            )
                        else:
                            enroll_feedback_msg = f"Need at least {enroll_min_samples} samples! (Have {len(enroll_captured_embeddings)})"
                            enroll_feedback_color = COLOR_RED
                            enroll_feedback_time = now
                    elif key in (ord("c"), ord("C"), 32):  # Capture sample
                        if not is_quality_ok or valid_face is None:
                            enroll_feedback_msg = f"Rejected: {quality_msg}"
                            enroll_feedback_color = COLOR_RED
                            enroll_feedback_time = now
                        else:
                            aligned = align_face(frame, valid_face.landmarks)
                            emb = None
                            try:
                                underlying_emb = (
                                    detector.arcface_worker.embedder
                                    if (hasattr(detector, "arcface_worker") and detector.arcface_worker)
                                    else ArcFaceEmbedder(settings.arcface_model_path, device=active_device)
                                )
                                emb = underlying_emb.get_embedding(aligned)
                            except Exception as exc:
                                enroll_feedback_msg = f"Extraction error: {exc}"
                                enroll_feedback_color = COLOR_RED
                                enroll_feedback_time = now

                            if emb is not None:
                                if enroll_captured_embeddings:
                                    sims = [compute_cosine_similarity(emb, prev) for prev in enroll_captured_embeddings]
                                    max_sim = max(sims)
                                    if max_sim > enroll_duplicate_threshold:
                                        enroll_feedback_msg = f"Pose too similar ({max_sim:.3f} > {enroll_duplicate_threshold:.2f}) - adjust angle"
                                        enroll_feedback_color = COLOR_AMBER
                                        enroll_feedback_time = now
                                        emb = None

                            if emb is not None:
                                enroll_captured_embeddings.append(emb)
                                sample_num = len(enroll_captured_embeddings)
                                enroll_feedback_msg = f"Sample {sample_num}/{enroll_target_samples} Captured!"
                                enroll_feedback_color = COLOR_EMERALD
                                enroll_feedback_time = now
                                beep()

                                if len(enroll_captured_embeddings) >= enroll_target_samples:
                                    enrollment_store.add_student(
                                        student_id=enroll_input_id,
                                        name=enroll_input_name,
                                        embeddings=enroll_captured_embeddings,
                                    )
                                    enroll_saved_path = enrollment_store.save()
                                    enrolled_count = len(enrollment_store.get_all_students())
                                    matcher = FaceMatcher(enrollment_store, tolerance=tolerance)
                                    matcher.reset_history()
                                    as_worker._state.clear()
                                    gui_state = GUI_STATE_ENROLL_COMPLETE
                                    enroll_complete_start = now
                                    logger.info(
                                        "Demo enrollment complete (%d samples) for %s (%s). FaceMatcher hot-reloaded.",
                                        len(enroll_captured_embeddings),
                                        enroll_input_name,
                                        enroll_input_id,
                                    )

                elif gui_state == GUI_STATE_ENROLL_COMPLETE:
                    # Press any key or timeout after 3.0 seconds
                    if (key != 255 and key != 0) or (now - enroll_complete_start > 3.0):
                        gui_state = GUI_STATE_NORMAL
                        logger.info("Returning to live recognition. Enrolled: %s (%s)", enroll_input_name, enroll_input_id)

            if headless:
                # Key check in headless mode
                key = 255
                if not _stdin_key_queue.empty():
                    key = _stdin_key_queue.get_nowait()
                else:
                    try:
                        import msvcrt
                        if msvcrt.kbhit():
                            key = ord(msvcrt.getch())
                    except Exception:
                        pass
                if key in (ord("q"), ord("Q"), 27):
                    logger.info("User requested demo shutdown (key '%s').", chr(key) if key != 27 else "ESC")
                    break

            # Max frames check (for testing)
            if max_frames > 0 and frame_count >= max_frames:
                logger.info("Reached maximum frames limit (%d) - exiting cleanly.", max_frames)
                break

            # Duration check (for testing)
            if duration > 0 and (now - demo_start_time) >= duration:
                logger.info("Reached duration limit (%.1f seconds) - exiting cleanly.", duration)
                break

    finally:
        # ── GUARANTEED CLEAN SHUTDOWN & HARDWARE RELEASE ────────────────────────
        logger.info("Shutting down Presento AI Vision Demo...")

        if "detector" in locals() and detector is not None:
            stats = detector.get_stats()
            logger.info(
                "Final Performance Metrics -- Display: %.1f FPS, SCRFD: %.1f FPS, ArcFace: %.1f FPS",
                display_fps,
                stats.get("detection_fps", 0.0),
                stats.get("arcface_fps", 0.0),
            )

        if "attendance_worker" in locals() and attendance_worker is not None:
            attendance_worker.stop()

        if "as_worker" in locals() and as_worker is not None:
            as_worker.stop()

        if "detector" in locals() and detector is not None:
            detector.stop()

        if camera is not None:
            camera.release()

        if not headless:
            cv2.destroyAllWindows()

        if cam_session is not None:
            logger.info("Releasing camera hardware session and turning off Pi service...")
            cam_session.stop(force=True)

        logger.info("Presento AI Vision Demo stopped cleanly. Camera hardware released.")


def main():
    parser = argparse.ArgumentParser(description="Presento AI Vision Demo Mode")
    parser.add_argument("--source", type=str, default=None, help="Camera source (e.g. 'webcam', 'stream', '0', or 'IP:PORT')")
    parser.add_argument("--host", type=str, default=None, help=f"Raspberry Pi host/IP override (default: {settings.pi_ip})")
    parser.add_argument("--port", type=int, default=None, help=f"Stream port override (default: {settings.stream_port})")
    parser.add_argument("--timeout", type=float, default=None, help="Camera control timeout in seconds")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Initial demo mode (1-5)")
    parser.add_argument("--debug-pipeline", action="store_true", help="Start with pipeline architecture panel visible")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after N frames (for automated testing)")
    parser.add_argument("--duration", type=float, default=0.0, help="Exit after N seconds (for automated testing)")

    args = parser.parse_args()

    run_demo(
        source=args.source,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        initial_mode=args.mode,
        show_pipeline=args.debug_pipeline,
        headless=args.headless,
        max_frames=args.max_frames,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
