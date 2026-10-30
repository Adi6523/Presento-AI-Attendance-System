r"""
Presento Phase 2B: Recognition Validation Benchmark Tool.
Measures enrolled-person similarity versus unknown-person similarity using the live Pi C270 stream
or a local camera source without modifying production recognition behavior.

Usage:
  # Run live physical benchmark with Raspberry Pi C270 stream (20s per scenario):
  .venv\Scripts\python.exe -m edge.recognition_benchmark --source 192.168.1.100:5001 --duration 20

  # Run with local webcam for testing:
  .venv\Scripts\python.exe -m edge.recognition_benchmark --source 0 --duration 10

  # Headless execution (useful for automated testing):
  .venv\Scripts\python.exe -m edge.recognition_benchmark --source 192.168.1.100:5001 --duration 20 --headless
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import settings
from .face_enrollment import FaceEnrollmentStore
from .scrfd_detector import SCRFDDetector, SCRFDFace
from .arcface_embedder import (
    ArcFaceEmbedder,
    align_face,
    transform_mirrored_bbox,
)
from .face_matcher import FaceMatcher, MatchResult
from .frame_receiver import FrameReceiver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("presento.edge.recognition_benchmark")

DEFAULT_CAMERA_STREAM = "192.168.1.100:5001"
DEFAULT_DURATION = 20.0
DEFAULT_OUTPUT_PATH = "data/recognition_benchmark_results.json"

# Exact guided benchmark scenarios required for Phase 2B recognition validation
BENCHMARK_SCENARIOS = [
    {
        "id": "scenario_1",
        "name": "KNOWN - ADITYA",
        "search_name": "Aditya",
        "default_id": "23UAM090",
        "default_name": "Aditya Patil",
        "is_known": True,
    },
    {
        "id": "scenario_2",
        "name": "KNOWN - PIYUSH",
        "search_name": "Piyush",
        "default_id": "23UAM092",
        "default_name": "Piyush",
        "is_known": True,
    },
    {
        "id": "scenario_3",
        "name": "KNOWN - PARYUSHA",
        "search_name": "Paryusha",
        "default_id": "23UAM084",
        "default_name": "Paryusha",
        "is_known": True,
    },
    {
        "id": "scenario_4",
        "name": "KNOWN - ABHISHEK J",
        "search_name": "Abhishek",
        "default_id": "23UCS041",
        "default_name": "Abhishek J",
        "is_known": True,
    },
    {
        "id": "scenario_5",
        "name": "KNOWN - MOM",
        "search_name": "Mom",
        "default_id": "TEMP01",
        "default_name": "Mom",
        "is_known": True,
    },
    {
        "id": "scenario_6",
        "name": "UNKNOWN",
        "search_name": None,
        "default_id": None,
        "default_name": "Unenrolled Person",
        "is_known": False,
    },
]


@dataclass
class FaceObservation:
    """Record of a single face recognition observation in a frame."""
    timestamp: float
    frame_id: int
    face_id: int
    best_student_id: str
    best_student_name: str
    best_similarity: float
    second_student_id: Optional[str]
    second_student_name: Optional[str]
    second_similarity: float
    margin: float
    is_correct: Optional[bool]


def compute_margin(best_similarity: float, second_similarity: Optional[float]) -> float:
    """Calculate margin between best match and second-best candidate."""
    if second_similarity is not None:
        return float(best_similarity - second_similarity)
    return float(best_similarity)


def evaluate_observation_correctness(
    best_student_id: str,
    expected_student_id: Optional[str],
) -> Optional[bool]:
    """
    Determine if the top candidate matches the expected student ID.
    Returns True/False for known scenarios, or None for UNKNOWN scenarios.
    """
    if expected_student_id is None:
        return None
    return best_student_id.strip().lower() == expected_student_id.strip().lower()


def calculate_distribution_stats(values: List[float] | np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics: min, max, mean, median, p5, p95, and std.
    """
    if len(values) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "std": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }


def compute_histogram(values: List[float] | np.ndarray, num_bins: int = 10) -> List[Dict[str, Any]]:
    """Compute normalized histogram bins from 0.0 to 1.0."""
    bins: List[Dict[str, Any]] = []
    step = 1.0 / num_bins
    vals = [float(v) for v in values]
    total = len(vals)

    for i in range(num_bins):
        low = round(i * step, 2)
        high = round((i + 1) * step, 2)
        if i == num_bins - 1:
            cnt = sum(1 for v in vals if low <= v <= high)
        else:
            cnt = sum(1 for v in vals if low <= v < high)
        pct = (cnt / total * 100.0) if total > 0 else 0.0
        bins.append({
            "range": f"[{low:.1f}, {high:.1f})" if i < num_bins - 1 else f"[{low:.1f}, {high:.1f}]",
            "low": low,
            "high": high,
            "count": cnt,
            "percent": float(pct),
        })
    return bins


def calculate_scenario_statistics(
    scenario_name: str,
    expected_id: Optional[str],
    expected_name: str,
    is_known: bool,
    observations: List[FaceObservation],
    duration_seconds: float,
    enrolled_students: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a completed scenario:
    - Top-1 correct percentage (for known persons)
    - Similarity statistics (min, max, mean, median, p5, p95)
    - Margin statistics
    - Candidate distribution (especially how UNKNOWN is classified)
    """
    sims = [obs.best_similarity for obs in observations]
    margins = [obs.margin for obs in observations]

    sim_stats = calculate_distribution_stats(sims)
    margin_stats = calculate_distribution_stats(margins)

    total_obs = len(observations)

    # Top-1 Correct calculation
    if is_known and expected_id:
        correct_count = sum(1 for obs in observations if obs.is_correct is True)
        top1_correct_pct = float(correct_count / total_obs * 100.0) if total_obs > 0 else 0.0
    else:
        correct_count = None
        top1_correct_pct = None

    # Candidate assignment distribution
    candidate_counts: Dict[str, Dict[str, Any]] = {}
    if enrolled_students:
        for st in enrolled_students:
            sid = st["student_id"]
            candidate_counts[sid] = {
                "student_id": sid,
                "name": st["name"],
                "count": 0,
                "percent": 0.0,
            }

    for obs in observations:
        sid = obs.best_student_id
        if sid not in candidate_counts:
            candidate_counts[sid] = {
                "student_id": sid,
                "name": obs.best_student_name,
                "count": 0,
                "percent": 0.0,
            }
        candidate_counts[sid]["count"] += 1

    if total_obs > 0:
        for sid in candidate_counts:
            candidate_counts[sid]["percent"] = float(candidate_counts[sid]["count"] / total_obs * 100.0)

    return {
        "scenario_name": scenario_name,
        "expected_id": expected_id,
        "expected_name": expected_name,
        "is_known": is_known,
        "duration_seconds": duration_seconds,
        "total_observations": total_obs,
        "top1_correct_count": correct_count,
        "top1_correct_percent": top1_correct_pct,
        "similarity_stats": sim_stats,
        "margin_stats": margin_stats,
        "candidate_distribution": candidate_counts,
        "observations": [asdict(obs) for obs in observations],
    }


def calculate_overall_distributions(
    scenario_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate KNOWN and UNKNOWN similarity distributions and calculate separation metrics.
    """
    known_sims: List[float] = []
    unknown_sims: List[float] = []
    known_margins: List[float] = []
    unknown_margins: List[float] = []

    for sc in scenario_results:
        is_known = sc["is_known"]
        for obs in sc["observations"]:
            sim = float(obs["best_similarity"])
            margin = float(obs["margin"])
            if is_known:
                known_sims.append(sim)
                known_margins.append(margin)
            else:
                unknown_sims.append(sim)
                unknown_margins.append(margin)

    known_sim_stats = calculate_distribution_stats(known_sims)
    unknown_sim_stats = calculate_distribution_stats(unknown_sims)
    known_margin_stats = calculate_distribution_stats(known_margins)
    unknown_margin_stats = calculate_distribution_stats(unknown_margins)

    known_hist = compute_histogram(known_sims, num_bins=10)
    unknown_hist = compute_histogram(unknown_sims, num_bins=10)

    has_data = (len(known_sims) > 0 and len(unknown_sims) > 0)
    separation_gap = float(known_sim_stats["p5"] - unknown_sim_stats["p95"]) if has_data else 0.0
    min_max_gap = float(known_sim_stats["min"] - unknown_sim_stats["max"]) if has_data else 0.0
    has_overlap = bool(unknown_sim_stats["max"] >= known_sim_stats["min"]) if has_data else False

    return {
        "known": {
            "stats": known_sim_stats,
            "margin_stats": known_margin_stats,
            "histogram": known_hist,
        },
        "unknown": {
            "stats": unknown_sim_stats,
            "margin_stats": unknown_margin_stats,
            "histogram": unknown_hist,
        },
        "separation": {
            "known_p5": known_sim_stats["p5"],
            "unknown_p95": unknown_sim_stats["p95"],
            "separation_gap_p5_to_p95": separation_gap,
            "known_min": known_sim_stats["min"],
            "unknown_max": unknown_sim_stats["max"],
            "separation_gap_min_to_max": min_max_gap,
            "has_overlap": has_overlap,
        },
    }


def generate_summary_table(scenario_results: List[Dict[str, Any]]) -> str:
    """
    Produce the exact summary table required:
    Scenario | Expected | Top-1 Correct % | Best Sim Mean | Best Sim Min | Best Sim Max | Margin Mean
    """
    header = f"{'Scenario':<20} | {'Expected':<14} | {'Top-1 Correct %':<17} | {'Best Sim Mean':<13} | {'Best Sim Min':<12} | {'Best Sim Max':<12} | {'Margin Mean':<11}"
    divider = "-" * len(header)
    lines = [header, divider]

    for sc in scenario_results:
        sc_name = sc["scenario_name"]
        exp_id = sc["expected_id"] if sc["expected_id"] else "N/A (Unknown)"
        top1 = f"{sc['top1_correct_percent']:.1f}%" if sc["top1_correct_percent"] is not None else "N/A"
        sim_stats = sc["similarity_stats"]
        margin_stats = sc["margin_stats"]

        row = (
            f"{sc_name:<20} | "
            f"{exp_id:<14} | "
            f"{top1:<17} | "
            f"{sim_stats['mean']:<13.4f} | "
            f"{sim_stats['min']:<12.4f} | "
            f"{sim_stats['max']:<12.4f} | "
            f"{margin_stats['mean']:<11.4f}"
        )
        lines.append(row)

    return "\n".join(lines)


def format_distribution_text(title: str, dist_data: Dict[str, Any]) -> str:
    """Format ASCII distribution chart for console display."""
    stats = dist_data["stats"]
    lines = [
        f"\n[{title}] (N = {stats['count']})",
        f"  Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f} | Min: {stats['min']:.4f} | Max: {stats['max']:.4f} | P5: {stats['p5']:.4f} | P95: {stats['p95']:.4f} | Std: {stats['std']:.4f}",
    ]
    for b in dist_data["histogram"]:
        bar_len = int(round(b["percent"] / 2.5))  # max 40 chars
        bar = "#" * bar_len
        lines.append(f"  {b['range']:<12} : {bar:<40} ({b['percent']:5.1f}%, n={b['count']})")
    return "\n".join(lines)


def format_unknown_breakdown_text(scenario_results: List[Dict[str, Any]]) -> str:
    """Format table showing how UNKNOWN was classified against enrolled identities."""
    unknown_sc = next((sc for sc in scenario_results if not sc["is_known"]), None)
    if not unknown_sc:
        return ""

    lines = [
        "\n[UNKNOWN Top-1 Identity Breakdown]",
        f"{'Enrolled Identity':<35} | {'Count':<8} | {'Percentage':<10}",
        "-" * 60,
    ]
    cand_dist = unknown_sc["candidate_distribution"]
    sorted_cands = sorted(cand_dist.values(), key=lambda c: c["count"], reverse=True)
    for c in sorted_cands:
        label = f"{c['student_id']} - {c['name']}"
        lines.append(f"{label:<35} | {c['count']:<8} | {c['percent']:>8.1f}%")
    return "\n".join(lines)


def _to_serializable(val: Any) -> Any:
    """Recursively convert numpy types into standard Python types for JSON."""
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    return val


def save_benchmark_results(results_data: Dict[str, Any], output_path: str = DEFAULT_OUTPUT_PATH) -> str:
    """Serialize benchmark findings to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    clean_data = _to_serializable(results_data)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2)
    logger.info("Saved recognition benchmark results to %s", output_path)
    return output_path


def open_camera(source: str | int = DEFAULT_CAMERA_STREAM, connect_timeout: float = 3.0) -> Any:
    """
    Open camera stream via FrameReceiver for HOST:PORT or OpenCV VideoCapture for local devices.
    """
    src_str = str(source).strip()

    # Numeric local camera index
    if src_str.isdigit():
        idx = int(src_str)
        logger.info("Opening local camera index %d...", idx)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"[ERROR] Camera source '{idx}' could not be opened.", file=sys.stderr)
            return None
        return cap

    # TCP stream HOST:PORT
    if ":" in src_str:
        parts = src_str.split(":")
        if len(parts) == 2:
            host, port_str = parts[0].strip(), parts[1].strip()
            try:
                port = int(port_str)
            except ValueError:
                print(f"[ERROR] Invalid port in stream address '{src_str}'.", file=sys.stderr)
                return None

            logger.info("Connecting to Presento camera stream at %s:%d...", host, port)
            receiver = FrameReceiver(host=host, port=port)
            receiver.start()

            ok, frame = receiver.read(timeout=connect_timeout)
            if not ok or frame is None:
                receiver.stop()
                print(f"[ERROR] Could not connect to Presento camera stream at {host}:{port}", file=sys.stderr)
                return None

            logger.info("Connected to camera stream at %s:%d (frame size: %dx%d)", host, port, frame.shape[1], frame.shape[0])
            return receiver

    # Fallback to OpenCV
    cap = cv2.VideoCapture(src_str)
    if not cap.isOpened():
        print(f"[ERROR] Camera source '{src_str}' could not be opened.", file=sys.stderr)
        return None
    return cap


def close_camera(camera: Any) -> None:
    """Safely release camera resources."""
    if camera is None:
        return
    if hasattr(camera, "release"):
        camera.release()
    elif hasattr(camera, "stop"):
        camera.stop()


def draw_hud(
    frame: np.ndarray,
    scenario_idx: int,
    total_scenarios: int,
    scenario_name: str,
    expected_text: str,
    state: str,
    time_remaining: float,
    duration: float,
    observations_count: int,
    latest_obs: Optional[FaceObservation],
    top1_correct_rate: Optional[float],
) -> np.ndarray:
    """Render translucent HUD card with live benchmark metrics onto display frame."""
    disp_h, disp_w = frame.shape[:2]

    # Translucent card background
    card_w = min(680, disp_w - 20)
    card_h = 190
    card_x = 10
    card_y = 10

    sub = frame[card_y : card_y + card_h, card_x : card_x + card_w]
    black = np.zeros(sub.shape, dtype=np.uint8)
    frame[card_y : card_y + card_h, card_x : card_x + card_w] = cv2.addWeighted(sub, 0.25, black, 0.75, 1.0)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # Line 1: Title
    cv2.putText(
        frame,
        f"Presento Phase 2B Recognition Benchmark [{scenario_idx}/{total_scenarios}]",
        (card_x + 12, card_y + 25),
        font,
        0.58,
        (255, 255, 255),
        2,
    )

    # Line 2: Scenario and Expected Person
    sc_color = (0, 255, 255) if "UNKNOWN" not in scenario_name else (255, 180, 0)
    cv2.putText(
        frame,
        f"Scenario: {scenario_name}",
        (card_x + 12, card_y + 52),
        font,
        0.65,
        sc_color,
        2,
    )
    cv2.putText(
        frame,
        f"Target: {expected_text}",
        (card_x + 12, card_y + 78),
        font,
        0.55,
        (200, 200, 200),
        1,
    )

    # Line 3: State and Timer
    if state == "WAITING":
        cv2.putText(
            frame,
            "STATUS: READY -- Press SPACE / ENTER to start 20s recording",
            (card_x + 12, card_y + 110),
            font,
            0.55,
            (0, 255, 0),
            2,
        )
    else:
        elapsed = duration - time_remaining
        pct = min(1.0, max(0.0, elapsed / duration))
        bar_w = int(250 * pct)
        cv2.rectangle(frame, (card_x + 12, card_y + 98), (card_x + 12 + 250, card_y + 110), (50, 50, 50), -1)
        cv2.rectangle(frame, (card_x + 12, card_y + 98), (card_x + 12 + bar_w, card_y + 110), (0, 255, 0), -1)
        cv2.rectangle(frame, (card_x + 12, card_y + 98), (card_x + 12 + 250, card_y + 110), (120, 120, 120), 1)

        cv2.putText(
            frame,
            f"RECORDING: {elapsed:4.1f}s / {duration:.1f}s  (Frames: {observations_count})",
            (card_x + 275, card_y + 109),
            font,
            0.52,
            (0, 255, 0),
            1,
        )

    # Line 4: Live recognition metrics
    if latest_obs is not None:
        best_str = f"Top-1: {latest_obs.best_student_name} ({latest_obs.best_similarity:.3f}) | 2nd: {latest_obs.second_student_name or 'None'} ({latest_obs.second_similarity:.3f})"
        margin_str = f"Margin: {latest_obs.margin:.3f}"
        if top1_correct_rate is not None:
            margin_str += f" | Top-1 Correct: {top1_correct_rate:5.1f}%"

        cv2.putText(frame, best_str, (card_x + 12, card_y + 140), font, 0.50, (255, 255, 255), 1)
        cv2.putText(frame, margin_str, (card_x + 12, card_y + 165), font, 0.50, (0, 255, 255), 1)
    else:
        cv2.putText(frame, "Waiting for face in camera view...", (card_x + 12, card_y + 150), font, 0.50, (150, 150, 150), 1)

    # Bottom helper hint
    hint_text = "SPACE: Start/Finish Early | 'S': Skip Scenario | 'Q'/ESC: Quit"
    cv2.putText(frame, hint_text, (card_x + 12, disp_h - 15), font, 0.50, (180, 180, 180), 1)
    return frame


def run_recognition_benchmark(
    source: str = DEFAULT_CAMERA_STREAM,
    duration: float = DEFAULT_DURATION,
    output_path: str = DEFAULT_OUTPUT_PATH,
    headless: bool = False,
    auto: bool = False,
) -> Dict[str, Any]:
    """
    Execute the Phase 2B face recognition validation benchmark sequentially across all scenarios.
    """
    print("=" * 80)
    print("       Presento Phase 2B: Recognition Validation Benchmark (Physical/Live)")
    print("=" * 80)
    print(f"  Camera Source : {source}")
    print(f"  Duration/Scen : {duration:.1f} seconds")
    print(f"  Results File  : {output_path}")
    print(f"  Headless Mode : {headless}")
    print(f"  Auto Progress : {auto}")
    print("=" * 80)

    # 1. Initialize Face Enrollment Store
    store = FaceEnrollmentStore(settings.face_enrollment_path)
    enrolled_students = store.get_all_students()
    print(f"\n[Enrollment Store] Loaded {len(enrolled_students)} enrolled students from {settings.face_enrollment_path}:")
    for s in enrolled_students:
        print(f"  - {s['student_id']}: {s['name']} ({len(s['embeddings'])} reference vectors)")

    # Resolve scenario targets against actual enrolled identities
    scenarios_to_run: List[Dict[str, Any]] = []
    for sc in BENCHMARK_SCENARIOS:
        sc_copy = dict(sc)
        if sc_copy["is_known"] and sc_copy["search_name"]:
            match = next(
                (s for s in enrolled_students if sc_copy["search_name"].lower() in s["name"].lower()),
                None,
            )
            if match:
                sc_copy["expected_id"] = match["student_id"]
                sc_copy["expected_name"] = match["name"]
            else:
                sc_copy["expected_id"] = sc["default_id"]
                sc_copy["expected_name"] = sc["default_name"]
        else:
            sc_copy["expected_id"] = None
            sc_copy["expected_name"] = "Unenrolled Person"
        scenarios_to_run.append(sc_copy)

    # 2. Initialize Camera Source
    camera = open_camera(source)
    if camera is None:
        raise SystemExit(1)

    # 3. Initialize SCRFD detector, ArcFace embedder, and FaceMatcher
    print("\n[Hardware Model Initialization]")
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
    matcher = FaceMatcher(enrollment_store=store)

    completed_scenarios: List[Dict[str, Any]] = []
    total_scenarios = len(scenarios_to_run)

    try:
        for idx, sc in enumerate(scenarios_to_run, start=1):
            sc_name = sc["name"]
            exp_id = sc["expected_id"]
            exp_name = sc["expected_name"]
            is_known = sc["is_known"]

            print("\n" + "=" * 70)
            print(f"  SCENARIO {idx}/{total_scenarios}: {sc_name}")
            print(f"  Target Person: {exp_name} (ID: {exp_id or 'N/A'})")
            print(f"  Duration     : {duration:.1f} seconds")
            if is_known:
                print("  Instruction  : Ensure the designated enrolled person is seated in front of the camera.")
            else:
                print("  Instruction  : Ensure an UNENROLLED person is seated in front of the camera.")
            print("=" * 70)

            # State machine: WAITING -> RECORDING -> DONE
            state = "RECORDING" if auto else "WAITING"
            scenario_observations: List[FaceObservation] = []
            rec_start_time: Optional[float] = None
            frame_counter = 0

            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                frame_counter += 1
                now = time.monotonic()

                # Process face detection & recognition
                faces = detector.detect(frame)
                latest_obs: Optional[FaceObservation] = None

                if faces:
                    # Sort faces by bounding box area to focus on the primary user in front of camera
                    primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    aligned = align_face(frame, primary_face.landmarks)
                    emb = embedder.get_embedding(aligned)
                    match_res = matcher.match(emb, face_id=0, frame_id=frame_counter)

                    best_c = match_res.best_match
                    second_c = match_res.second_best_match
                    best_sid = best_c.student_id if best_c else "None"
                    best_sname = best_c.name if best_c else "None"
                    best_sim = float(best_c.similarity) if best_c else 0.0

                    second_sid = second_c.student_id if second_c else None
                    second_sname = second_c.name if second_c else None
                    second_sim = float(second_c.similarity) if second_c else 0.0
                    margin = compute_margin(best_sim, second_sim if second_c else None)

                    is_correct = evaluate_observation_correctness(best_sid, exp_id)

                    obs = FaceObservation(
                        timestamp=now,
                        frame_id=frame_counter,
                        face_id=0,
                        best_student_id=best_sid,
                        best_student_name=best_sname,
                        best_similarity=best_sim,
                        second_student_id=second_sid,
                        second_student_name=second_sname,
                        second_similarity=second_sim,
                        margin=margin,
                        is_correct=is_correct,
                    )
                    latest_obs = obs

                    if state == "RECORDING":
                        scenario_observations.append(obs)

                # State handling
                time_remaining = 0.0
                if state == "RECORDING":
                    if rec_start_time is None:
                        rec_start_time = now
                    elapsed = now - rec_start_time
                    time_remaining = max(0.0, duration - elapsed)
                    if elapsed >= duration:
                        print(f"\n[COMPLETE] Scenario {idx} completed: {len(scenario_observations)} observations captured.")
                        break

                # Live top-1 correct metric for display
                current_rate: Optional[float] = None
                if is_known and scenario_observations:
                    c_cnt = sum(1 for o in scenario_observations if o.is_correct is True)
                    current_rate = (c_cnt / len(scenario_observations) * 100.0)

                # GUI rendering (if not headless)
                if not headless:
                    display_frame = cv2.flip(frame, 1)
                    disp_w = display_frame.shape[1]

                    # Annotate face
                    if faces:
                        m_bbox = transform_mirrored_bbox(primary_face.bbox, disp_w)
                        box_color = (0, 255, 0) if (latest_obs and latest_obs.is_correct) else (0, 140, 255)
                        if not is_known:
                            box_color = (255, 200, 0)
                        cv2.rectangle(display_frame, (m_bbox[0], m_bbox[1]), (m_bbox[2], m_bbox[3]), box_color, 2)

                    exp_disp = f"{exp_name} ({exp_id})" if exp_id else "Unenrolled Person (N/A)"
                    draw_hud(
                        display_frame,
                        scenario_idx=idx,
                        total_scenarios=total_scenarios,
                        scenario_name=sc_name,
                        expected_text=exp_disp,
                        state=state,
                        time_remaining=time_remaining,
                        duration=duration,
                        observations_count=len(scenario_observations),
                        latest_obs=latest_obs,
                        top1_correct_rate=current_rate,
                    )

                    cv2.imshow("Presento Phase 2B - Recognition Benchmark", display_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):  # 'q' or ESC -> quit early
                        print("\n[USER QUIT] Quitting benchmark early...")
                        break
                    elif key == ord("s"):  # 's' -> skip scenario
                        print(f"\n[SKIPPED] Skipped scenario {idx}: {sc_name}")
                        scenario_observations.clear()
                        break
                    elif key in (ord(" "), 10, 13):  # SPACE or ENTER
                        if state == "WAITING":
                            state = "RECORDING"
                            rec_start_time = time.monotonic()
                            print(f"[START] Recording begun for {sc_name}...")
                        elif state == "RECORDING":
                            print(f"[EARLY STOP] Finished scenario {idx} early with {len(scenario_observations)} observations.")
                            break
                else:
                    # Headless auto progression
                    if state == "WAITING":
                        state = "RECORDING"
                        rec_start_time = now

            # Compute and store scenario statistics
            sc_stats = calculate_scenario_statistics(
                scenario_name=sc_name,
                expected_id=exp_id,
                expected_name=exp_name,
                is_known=is_known,
                observations=scenario_observations,
                duration_seconds=duration,
                enrolled_students=enrolled_students,
            )
            completed_scenarios.append(sc_stats)

            # Print brief scenario log
            sim_stats = sc_stats["similarity_stats"]
            if is_known:
                print(f"  Summary: Top-1 Correct: {sc_stats['top1_correct_percent']:.1f}% | Best Sim Mean: {sim_stats['mean']:.4f} (Min: {sim_stats['min']:.4f}, Max: {sim_stats['max']:.4f})")
            else:
                print(f"  Summary: Unknown Impostor Sim Mean: {sim_stats['mean']:.4f} (Min: {sim_stats['min']:.4f}, Max: {sim_stats['max']:.4f})")

    finally:
        close_camera(camera)
        if not headless:
            cv2.destroyAllWindows()

    # 4. Generate Overall Analysis and Output
    overall_dists = calculate_overall_distributions(completed_scenarios)
    summary_table_text = generate_summary_table(completed_scenarios)
    known_dist_text = format_distribution_text("KNOWN Similarity Distribution", overall_dists["known"])
    unknown_dist_text = format_distribution_text("UNKNOWN Similarity Distribution", overall_dists["unknown"])
    unknown_breakdown_text = format_unknown_breakdown_text(completed_scenarios)

    # Print Final Benchmark Report to Console
    print("\n" + "=" * 90)
    print("           PRESENTO PHASE 2B: RECOGNITION VALIDATION BENCHMARK RESULTS")
    print("=" * 90)
    print(summary_table_text)
    print(unknown_breakdown_text)
    print(known_dist_text)
    print(unknown_dist_text)

    sep = overall_dists["separation"]
    print("\n[Distribution Separation Analysis]")
    print(f"  Known P5 (Conservative Enrolled)   : {sep['known_p5']:.4f}")
    print(f"  Unknown P95 (Upper Impostor Bound) : {sep['unknown_p95']:.4f}")
    print(f"  P5-to-P95 Separation Gap           : {sep['separation_gap_p5_to_p95']:+.4f}")
    print(f"  Known Min vs Unknown Max Gap       : {sep['separation_gap_min_to_max']:+.4f} (Overlap: {sep['has_overlap']})")
    print("=" * 90)

    # 5. Save JSON Artifact
    final_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_source": str(source),
        "duration_per_scenario_seconds": duration,
        "scenarios": completed_scenarios,
        "distributions": overall_dists,
        "summary_table": summary_table_text,
    }
    saved_path = save_benchmark_results(final_output, output_path)
    print(f"\n[OUTPUT] Complete benchmark data saved to: {saved_path}")

    return final_output


def main():
    parser = argparse.ArgumentParser(description="Presento Phase 2B Face Recognition Validation Benchmark")
    parser.add_argument("--source", type=str, default=DEFAULT_CAMERA_STREAM, help=f"Camera source (e.g. {DEFAULT_CAMERA_STREAM} or 0)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help=f"Duration per scenario in seconds (default: {DEFAULT_DURATION})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help=f"JSON results output path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI preview window")
    parser.add_argument("--auto", action="store_true", help="Auto advance through scenarios without waiting for user SPACE prompt")
    args = parser.parse_args()

    run_recognition_benchmark(
        source=args.source,
        duration=args.duration,
        output_path=args.output,
        headless=args.headless,
        auto=args.auto,
    )


if __name__ == "__main__":
    main()
