r"""
Presento Face Anti-Spoofing Benchmark.
Evaluates anti-spoofing models on OpenVINO Core (preferring Intel Iris Xe GPU).

Supported Models:
  1. 'mn3'     : OpenVINO Open Model Zoo anti-spoof-mn3 (MobileNetV3 on CelebA-Spoof)
  2. 'minifas' : MiniFASNet V2 SE (Squeeze-and-Excitation on CelebA-Spoof)
  3. 'hybrid'  : Dual-model fusion combining MN3 and MiniFASNet with AND, OR, AVERAGE, and TEMPORAL strategies.

Usage:
  # Synthetic GPU benchmark:
  .venv\Scripts\python.exe -m edge.anti_spoof_benchmark --mode synthetic --model mn3
  .venv\Scripts\python.exe -m edge.anti_spoof_benchmark --mode synthetic --model minifas
  .venv\Scripts\python.exe -m edge.anti_spoof_benchmark --mode synthetic --model hybrid

  # Live camera benchmark (manual A/B/C/D/E/F interactive mode):
  .venv\Scripts\python.exe -m edge.anti_spoof_benchmark --mode live --model hybrid --source 0

  # Automatic long-duration physical benchmark (sequential 6 scenarios with SPACE prompt):
  .venv\Scripts\python.exe -m edge.anti_spoof_benchmark --mode auto --model hybrid --duration 20 --source 0
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import openvino as ov

from .config import settings
from .anti_spoof import AntiSpoofClassifier, crop_face
from .minifas_antispoof import MiniFASNetV2SEClassifier, crop_face_minifas
from .anti_spoof_fusion import AntiSpoofFusionEngine, FusionResult
from .scrfd_detector import SCRFDDetector, SCRFDFace
from .arcface_embedder import transform_mirrored_bbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("presento.benchmark.antispoof")

# The exact ordered scenarios for automatic sequential benchmark
AUTO_SCENARIOS: List[str] = [
    "REAL FACE",
    "PRINTED PHOTO",
    "PHONE PHOTO",
    "PHONE VIDEO",
    "LIGHTING VARIATION",
    "FACE DISTANCE",
]


def resolve_model_and_classifier(
    model_arg: str,
    device: str = "GPU",
) -> Tuple[Any, Callable, str, str]:
    """
    Resolve single-model selection to classifier instance, face cropping function, display name, and file path.
    Supported options:
      - 'mn3', 'anti-spoof-mn3', or path to anti-spoof-mn3.onnx
      - 'minifas', 'minifasnet', 'minifasnet_v2se', or path to minifasnet_v2se.onnx
    """
    arg_str = str(model_arg).strip().lower()

    if "minifas" in arg_str:
        model_path = model_arg if os.path.exists(model_arg) else "data/models/minifasnet_v2se.onnx"
        classifier = MiniFASNetV2SEClassifier(model_path=model_path, device=device)
        crop_fn = crop_face_minifas
        model_name = "MiniFASNet V2 SE"
        return classifier, crop_fn, model_name, model_path
    else:
        model_path = model_arg if os.path.exists(model_arg) else "data/models/anti-spoof-mn3.onnx"
        classifier = AntiSpoofClassifier(model_path=model_path, device=device)
        crop_fn = crop_face
        model_name = "anti-spoof-mn3 (MobileNetV3)"
        return classifier, crop_fn, model_name, model_path


def calculate_longest_run(sequence: List[str], target: str) -> int:
    """Calculate the longest consecutive run of a target prediction in a sequence."""
    max_run = 0
    curr_run = 0
    for item in sequence:
        if item == target:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 0
    return max_run


def compute_temporal_metrics(
    real_probs: List[float],
    spoof_probs: List[float],
    predictions: List[str],
) -> Dict[str, Any]:
    """
    Calculate temporal consistency statistics for single models:
    - Percentage of frames with REAL probability >= 0.5
    - Percentage of frames with SPOOF probability >= 0.5
    - Longest consecutive REAL run
    - Longest consecutive SPOOF run
    """
    n = len(predictions)
    if n == 0:
        return {
            "real_ge_05_pct": 0.0,
            "spoof_ge_05_pct": 0.0,
            "longest_real_run": 0,
            "longest_spoof_run": 0,
        }

    real_ge_05 = sum(1 for p in real_probs if p >= 0.5)
    spoof_ge_05 = sum(1 for p in spoof_probs if p >= 0.5)

    longest_real = calculate_longest_run(predictions, "REAL")
    longest_spoof = calculate_longest_run(predictions, "SPOOF")

    return {
        "real_ge_05_pct": round((real_ge_05 / n) * 100.0, 1),
        "spoof_ge_05_pct": round((spoof_ge_05 / n) * 100.0, 1),
        "longest_real_run": longest_real,
        "longest_spoof_run": longest_spoof,
    }


def compute_scenario_metrics(
    name: str,
    real_probs: List[float],
    spoof_probs: List[float],
    predictions: List[str],
    latencies: List[float],
    duration_sec: float,
) -> Dict[str, Any]:
    """
    Compute full statistical and temporal metrics for a single-model benchmark scenario.
    Does not assume any fixed frame count.
    """
    n = len(predictions)
    if n == 0:
        return {
            "name": name,
            "frames": 0,
            "duration_sec": round(duration_sec, 2),
            "real_mean": 0.0,
            "real_median": 0.0,
            "real_min": 0.0,
            "real_max": 0.0,
            "real_std": 0.0,
            "spoof_mean": 0.0,
            "spoof_median": 0.0,
            "spoof_min": 0.0,
            "spoof_max": 0.0,
            "spoof_std": 0.0,
            "real_count": 0,
            "spoof_count": 0,
            "real_pct": 0.0,
            "spoof_pct": 0.0,
            "latency_mean": 0.0,
            "latency_median": 0.0,
            "latency_p95": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "fps": 0.0,
            "temporal": compute_temporal_metrics([], [], []),
        }

    r_arr = np.array(real_probs, dtype=np.float64)
    s_arr = np.array(spoof_probs, dtype=np.float64)
    l_arr = np.array(latencies, dtype=np.float64)

    real_cnt = sum(1 for p in predictions if p == "REAL")
    spoof_cnt = sum(1 for p in predictions if p == "SPOOF")
    fps = float(n / duration_sec) if duration_sec > 0 else 0.0

    temporal = compute_temporal_metrics(real_probs, spoof_probs, predictions)

    return {
        "name": name,
        "frames": n,
        "duration_sec": round(duration_sec, 2),
        "real_mean": float(np.mean(r_arr)),
        "real_median": float(np.median(r_arr)),
        "real_min": float(np.min(r_arr)),
        "real_max": float(np.max(r_arr)),
        "real_std": float(np.std(r_arr)),
        "spoof_mean": float(np.mean(s_arr)),
        "spoof_median": float(np.median(s_arr)),
        "spoof_min": float(np.min(s_arr)),
        "spoof_max": float(np.max(s_arr)),
        "spoof_std": float(np.std(s_arr)),
        "real_count": real_cnt,
        "spoof_count": spoof_cnt,
        "real_pct": round((real_cnt / n) * 100.0, 1),
        "spoof_pct": round((spoof_cnt / n) * 100.0, 1),
        "latency_mean": float(np.mean(l_arr)),
        "latency_median": float(np.median(l_arr)),
        "latency_p95": float(np.percentile(l_arr, 95)),
        "latency_min": float(np.min(l_arr)),
        "latency_max": float(np.max(l_arr)),
        "fps": round(fps, 1),
        "temporal": temporal,
    }


def compute_hybrid_scenario_metrics(
    name: str,
    duration_sec: float,
    mn3_records: Dict[str, Any],
    mf_records: Dict[str, Any],
    fusion_records: Dict[str, Any],
    hybrid_latencies: List[float],
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for hybrid multi-model fusion benchmark.
    Reports individual model stats and all four fusion strategies (AND, OR, AVERAGE, TEMPORAL).
    """
    n = len(hybrid_latencies)
    if n == 0:
        return {
            "name": name,
            "frames": 0,
            "duration_sec": round(duration_sec, 2),
            "effective_fps": 0.0,
            "mn3": compute_scenario_metrics("MN3", [], [], [], [], duration_sec),
            "minifas": compute_scenario_metrics("MiniFAS", [], [], [], [], duration_sec),
            "fusion": {},
            "latency": {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0},
        }

    # Individual models
    mn3_m = compute_scenario_metrics(
        "MN3",
        mn3_records["real_probs"],
        mn3_records["spoof_probs"],
        mn3_records["predictions"],
        mn3_records["latencies"],
        duration_sec,
    )
    mf_m = compute_scenario_metrics(
        "MiniFAS",
        mf_records["real_probs"],
        mf_records["spoof_probs"],
        mf_records["predictions"],
        mf_records["latencies"],
        duration_sec,
    )

    and_preds = fusion_records["and_preds"]
    or_preds = fusion_records["or_preds"]
    avg_probs = fusion_records["avg_probs"]
    avg_preds = fusion_records["avg_preds"]
    temporal_preds = fusion_records["temporal_preds"]

    # 1. AND Strategy Metrics
    and_r = sum(1 for p in and_preds if p == "REAL")
    and_s = sum(1 for p in and_preds if p == "SPOOF")
    and_metrics = {
        "real_count": and_r,
        "spoof_count": and_s,
        "real_pct": round((and_r / n) * 100.0, 1),
        "spoof_pct": round((and_s / n) * 100.0, 1),
        "longest_real_run": calculate_longest_run(and_preds, "REAL"),
        "longest_spoof_run": calculate_longest_run(and_preds, "SPOOF"),
    }

    # 2. OR Strategy Metrics
    or_r = sum(1 for p in or_preds if p == "REAL")
    or_s = sum(1 for p in or_preds if p == "SPOOF")
    or_metrics = {
        "real_count": or_r,
        "spoof_count": or_s,
        "real_pct": round((or_r / n) * 100.0, 1),
        "spoof_pct": round((or_s / n) * 100.0, 1),
        "longest_real_run": calculate_longest_run(or_preds, "REAL"),
        "longest_spoof_run": calculate_longest_run(or_preds, "SPOOF"),
    }

    # 3. AVERAGE Strategy Metrics
    avg_r = sum(1 for p in avg_preds if p == "REAL")
    avg_s = sum(1 for p in avg_preds if p == "SPOOF")
    avg_arr = np.array(avg_probs, dtype=np.float64)
    avg_ge_05 = sum(1 for p in avg_probs if p >= 0.5)
    avg_metrics = {
        "real_count": avg_r,
        "spoof_count": avg_s,
        "real_pct": round((avg_r / n) * 100.0, 1),
        "spoof_pct": round((avg_s / n) * 100.0, 1),
        "mean_p_real": float(np.mean(avg_arr)),
        "median_p_real": float(np.median(avg_arr)),
        "min_p_real": float(np.min(avg_arr)),
        "max_p_real": float(np.max(avg_arr)),
        "std_p_real": float(np.std(avg_arr)),
        "longest_real_run": calculate_longest_run(avg_preds, "REAL"),
        "longest_spoof_run": calculate_longest_run(avg_preds, "SPOOF"),
        "real_ge_05_pct": round((avg_ge_05 / n) * 100.0, 1),
    }

    # 4. TEMPORAL Strategy Metrics
    temp_r = sum(1 for p in temporal_preds if p == "REAL")
    temp_s = sum(1 for p in temporal_preds if p == "SPOOF")
    temp_u = sum(1 for p in temporal_preds if p == "UNCERTAIN")
    temporal_metrics = {
        "real_count": temp_r,
        "spoof_count": temp_s,
        "uncertain_count": temp_u,
        "real_pct": round((temp_r / n) * 100.0, 1),
        "spoof_pct": round((temp_s / n) * 100.0, 1),
        "uncertain_pct": round((temp_u / n) * 100.0, 1),
        "longest_real_run": calculate_longest_run(temporal_preds, "REAL"),
        "longest_spoof_run": calculate_longest_run(temporal_preds, "SPOOF"),
        "longest_uncertain_run": calculate_longest_run(temporal_preds, "UNCERTAIN"),
    }

    # Total Hybrid Latencies
    l_arr = np.array(hybrid_latencies, dtype=np.float64)
    fps = float(n / duration_sec) if duration_sec > 0 else 0.0

    return {
        "name": name,
        "frames": n,
        "duration_sec": round(duration_sec, 2),
        "effective_fps": round(fps, 1),
        "mn3": mn3_m,
        "minifas": mf_m,
        "fusion": {
            "and": and_metrics,
            "or": or_metrics,
            "average": avg_metrics,
            "temporal": temporal_metrics,
        },
        "latency": {
            "mean": float(np.mean(l_arr)),
            "median": float(np.median(l_arr)),
            "p95": float(np.percentile(l_arr, 95)),
            "min": float(np.min(l_arr)),
            "max": float(np.max(l_arr)),
        },
    }


def save_benchmark_results(
    results_data: Dict[str, Any],
    filepath: str = "data/anti_spoof_benchmark_results.json",
) -> str:
    """
    Save complete benchmark results to JSON without overwriting prior runs.
    Maintains a history list in filepath and writes a timestamped copy as backup.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    history = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    history = loaded
                elif isinstance(loaded, dict):
                    if "runs" in loaded and isinstance(loaded["runs"], list):
                        history = loaded["runs"]
                    else:
                        history = [loaded]
        except Exception as e:
            logger.warning("Could not read existing results from %s: %s", filepath, e)
            history = []

    history.append(results_data)

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(history),
        "runs": history,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Also save an isolated timestamped copy to prevent any data loss
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_filename = os.path.join(
        os.path.dirname(os.path.abspath(filepath)),
        f"anti_spoof_benchmark_{ts_str}.json",
    )
    try:
        with open(ts_filename, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
    except Exception as e:
        logger.warning("Could not write timestamped results copy: %s", e)

    return filepath


def init_camera(source_override: Optional[str] = None) -> Any:
    """Initialize camera stream: Raspberry Pi FrameReceiver or local webcam."""
    cam_source = source_override or settings.camera_source
    if str(cam_source).lower() in ("stream", "pi", "network"):
        from .frame_receiver import FrameReceiver
        logger.info("Connecting to Raspberry Pi camera stream at %s:%d...", settings.pi_ip, settings.stream_port)
        camera = FrameReceiver(host=settings.pi_ip, port=settings.stream_port)
        camera.start()
        return camera
    else:
        try:
            cam_src: int | str = int(cam_source)
        except ValueError:
            cam_src = cam_source
        camera = cv2.VideoCapture(cam_src)
        if not camera.isOpened():
            print(f"[ERROR] Camera source '{cam_src}' could not be opened.", file=sys.stderr)
            return None
        logger.info("Camera opened: source=%s", cam_src)
        return camera


def close_camera(camera: Any):
    """Safely release or stop camera resource."""
    if camera is None:
        return
    if hasattr(camera, "stop"):
        camera.stop()
    if hasattr(camera, "release"):
        camera.release()


def draw_hud_card(
    frame: np.ndarray,
    lines: List[Tuple[str, Tuple[int, int, int], float, int]],
    x: int = 15,
    y: int = 15,
    padding: int = 12,
):
    """Render a clean translucent dark background card with colored text lines."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_w = 0
    total_h = 0
    line_metrics = []

    for text, color, scale, thickness in lines:
        if text == "":
            total_h += 12
            line_metrics.append((0, 12))
            continue
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        max_w = max(max_w, w)
        line_spacing = h + baseline + 6
        total_h += line_spacing
        line_metrics.append((w, line_spacing))

    card_w = max_w + (padding * 2)
    card_h = total_h + (padding * 2)

    h_img, w_img = frame.shape[:2]
    x2 = min(w_img - 5, x + card_w)
    y2 = min(h_img - 5, y + card_h)

    sub_img = frame[y:y2, x:x2]
    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.35, black_rect, 0.65, 1.0)
    frame[y:y2, x:x2] = res
    cv2.rectangle(frame, (x, y), (x2, y2), (70, 70, 70), 1)

    curr_y = y + padding
    for i, (text, color, scale, thickness) in enumerate(lines):
        if text == "":
            curr_y += 12
            continue
        _, line_sp = line_metrics[i]
        curr_y += int(line_sp * 0.75)
        cv2.putText(frame, text, (x + padding, curr_y), font, scale, color, thickness, cv2.LINE_AA)
        curr_y += int(line_sp * 0.25)


# ---------------------------------------------------------------------------
# HYBRID BENCHMARK IMPLEMENTATION (SYNTHETIC, LIVE, AUTO)
# ---------------------------------------------------------------------------

def run_hybrid_synthetic_benchmark(device: str = "GPU", iterations: int = 100) -> dict:
    """Run standalone synthetic GPU benchmark for the hybrid fusion pipeline."""
    print("=" * 70)
    print("      Presento Hybrid Anti-Spoof Fusion Synthetic Benchmark")
    print("=" * 70)

    clf_mn3 = AntiSpoofClassifier(model_path="data/models/anti-spoof-mn3.onnx", device=device)
    clf_mf = MiniFASNetV2SEClassifier(model_path="data/models/minifasnet_v2se.onnx", device=device)
    fusion_engine = AntiSpoofFusionEngine(window_size=15)

    print(f"Hardware Device : {clf_mn3.active_device} -> {clf_mn3.full_device_name}")
    print(f"Models Combined : anti-spoof-mn3 (MobileNetV3) + MiniFASNet V2 SE")
    print(f"Strategies      : AND, OR, AVERAGE, TEMPORAL")
    print(f"Iterations      : {iterations}")
    print("-" * 70)

    lat_mn3_list = []
    lat_mf_list = []
    lat_tot_list = []

    for _ in range(iterations):
        dummy_crop = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        r_mn3, s_mn3, _, lat_mn3 = clf_mn3.predict(dummy_crop)
        r_mf, s_mf, _, lat_mf = clf_mf.predict(dummy_crop)
        _ = fusion_engine.update(r_mn3, s_mn3, r_mf, s_mf)
        tot_lat = (time.perf_counter() - t0) * 1000.0

        lat_mn3_list.append(lat_mn3)
        lat_mf_list.append(lat_mf)
        lat_tot_list.append(tot_lat)

    avg_mn3 = float(np.mean(lat_mn3_list))
    avg_mf = float(np.mean(lat_mf_list))
    avg_tot = float(np.mean(lat_tot_list))
    fps_tot = 1000.0 / avg_tot if avg_tot > 0 else 0.0

    print("[Synthetic Hybrid Latency Results]")
    print(f"  MN3 Mean Latency        : {avg_mn3:.2f} ms")
    print(f"  MiniFAS Mean Latency    : {avg_mf:.2f} ms")
    print(f"  Hybrid Total Latency    : {avg_tot:.2f} ms (p95: {np.percentile(lat_tot_list, 95):.2f} ms)")
    print(f"  Hybrid Throughput       : {fps_tot:.1f} FPS")
    print("=" * 70)

    return {
        "model": "hybrid",
        "mn3_avg_ms": avg_mn3,
        "minifas_avg_ms": avg_mf,
        "hybrid_avg_ms": avg_tot,
        "hybrid_fps": fps_tot,
    }


def run_hybrid_auto_benchmark(
    device: str = "GPU",
    duration: float = 20.0,
    source_override: Optional[str] = None,
    output_path: str = "data/anti_spoof_benchmark_results.json",
):
    """
    Run automated long-duration physical benchmark using hybrid fusion across all 6 scenarios.
    Measures and compares AND, OR, AVERAGE, and TEMPORAL strategies.
    """
    print("\n" + "=" * 70)
    print("      PRESENTO: HYBRID ANTI-SPOOF FUSION BENCHMARK (AUTO MODE)")
    print("=" * 70)
    print(f"Models Used         : anti-spoof-mn3 + MiniFASNet V2 SE")
    print(f"Fusion Strategies   : AND, OR, AVERAGE, TEMPORAL (Window=15, Thresh=80%)")
    print(f"Configured Duration : {duration:.1f} seconds per scenario")
    print(f"Scenarios ({len(AUTO_SCENARIOS)})        : {', '.join(AUTO_SCENARIOS)}")
    print(f"Results File        : {output_path}")
    print("=" * 70 + "\n")

    camera = init_camera(source_override)
    if camera is None:
        return

    detector = SCRFDDetector(
        model_path=settings.scrfd_model_path,
        device=settings.openvino_device,
        input_size=settings.scrfd_input_size,
        conf_thresh=settings.scrfd_conf_thresh,
        nms_thresh=settings.scrfd_nms_thresh,
    )
    clf_mn3 = AntiSpoofClassifier(model_path="data/models/anti-spoof-mn3.onnx", device=device)
    clf_mf = MiniFASNetV2SEClassifier(model_path="data/models/minifasnet_v2se.onnx", device=device)
    fusion_engine = AntiSpoofFusionEngine(window_size=15, real_ratio_threshold=0.80, spoof_ratio_threshold=0.80)

    all_scenario_metrics: List[Dict[str, Any]] = []
    all_per_frame_records: Dict[str, List[Dict[str, Any]]] = {}

    cv2.namedWindow("Presento - Hybrid Anti-Spoof Benchmark (Auto Mode)", cv2.WINDOW_AUTOSIZE)

    try:
        for sc_idx, scenario_name in enumerate(AUTO_SCENARIOS, start=1):
            print(f"\n[{sc_idx}/{len(AUTO_SCENARIOS)}] PREPARING SCENARIO: {scenario_name}")
            print(f"  -> Setup physical presentation: {scenario_name}")
            print("  -> Press [SPACE] in the preview window when ready...")

            fusion_engine.reset()

            # ---------------------------------------------------------------
            # 1. PREPARATION PHASE (SPACE prompt)
            # ---------------------------------------------------------------
            ready_to_start = False
            while not ready_to_start:
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)

                prep_lines = [
                    ("Presento Hybrid Anti-Spoof Benchmark", (0, 255, 255), 0.65, 2),
                    ("Models: MN3 + MiniFASNet V2 SE (Dual GPU)", (180, 220, 255), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"NEXT TEST [{sc_idx}/{len(AUTO_SCENARIOS)}]: {scenario_name}", (0, 215, 255), 0.70, 2),
                    ("Prepare the test object in front of camera.", (220, 220, 220), 0.55, 1),
                    (f"Collection Duration: {duration:.1f} seconds", (180, 180, 180), 0.50, 1),
                    ("Strategies: AND | OR | AVERAGE | TEMPORAL", (200, 200, 200), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (">>> PRESS [SPACE] WHEN READY <<<", (0, 255, 0), 0.65, 2),
                    ("Press 'Q' to quit benchmark.", (150, 150, 150), 0.45, 1),
                ]
                draw_hud_card(display_frame, prep_lines, x=20, y=20)
                cv2.imshow("Presento - Hybrid Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return
                elif key == 32:  # SPACE
                    ready_to_start = True

            # ---------------------------------------------------------------
            # 2. COUNTDOWN PHASE (3s)
            # ---------------------------------------------------------------
            countdown_start = time.monotonic()
            while True:
                elapsed_cd = time.monotonic() - countdown_start
                if elapsed_cd >= 3.0:
                    break
                secs_left = int(3.0 - elapsed_cd) + 1

                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)

                cd_lines = [
                    ("Presento Hybrid Anti-Spoof Benchmark", (0, 255, 255), 0.60, 2),
                    (f"Test: {scenario_name}", (0, 215, 255), 0.65, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"STARTING IN: {secs_left}...", (0, 140, 255), 0.90, 2),
                    ("Hold position steadily.", (200, 200, 200), 0.55, 1),
                ]
                draw_hud_card(display_frame, cd_lines, x=20, y=20)
                cv2.imshow("Presento - Hybrid Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return

            # ---------------------------------------------------------------
            # 3. COLLECTION PHASE
            # ---------------------------------------------------------------
            print(f"  -> Collecting frames for {duration:.1f} seconds...")
            collect_start = time.monotonic()

            mn3_recs = {"real_probs": [], "spoof_probs": [], "predictions": [], "latencies": []}
            mf_recs = {"real_probs": [], "spoof_probs": [], "predictions": [], "latencies": []}
            fus_recs = {"and_preds": [], "or_preds": [], "avg_probs": [], "avg_preds": [], "temporal_preds": []}
            tot_latencies = []
            sc_per_frame = []

            # Defaults for HUD
            mn3_r, mf_r = 0.5, 0.5
            and_res, or_res, avg_res, temp_res = "INIT", "INIT", "INIT", "INIT"
            lat_mn3, lat_mf, tot_lat = 0.0, 0.0, 0.0

            while True:
                now = time.monotonic()
                elapsed_col = now - collect_start
                if elapsed_col >= duration:
                    break

                time_remaining = max(0.0, duration - elapsed_col)

                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue

                faces = detector.detect(frame)
                largest_face: Optional[SCRFDFace] = None

                if faces:
                    largest_face = max(faces, key=lambda f: (f.right - f.left) * (f.bottom - f.top))

                    # 1. Crop for MN3 (1.2x scale)
                    crop_mn3 = crop_face(frame, largest_face.bbox, scale=1.2)
                    # 2. Crop for MiniFAS (1.5x scale with reflection padding)
                    crop_mf = crop_face_minifas(frame, largest_face.bbox, scale=1.5)

                    # Inferences
                    t0 = time.perf_counter()
                    mn3_r, mn3_s, mn3_lbl, lat_mn3 = clf_mn3.predict(crop_mn3)
                    mf_r, mf_s, mf_lbl, lat_mf = clf_mf.predict(crop_mf)
                    fus = fusion_engine.update(mn3_r, mn3_s, mf_r, mf_s)
                    tot_lat = (time.perf_counter() - t0) * 1000.0

                    # Store records
                    mn3_recs["real_probs"].append(mn3_r)
                    mn3_recs["spoof_probs"].append(mn3_s)
                    mn3_recs["predictions"].append(mn3_lbl)
                    mn3_recs["latencies"].append(lat_mn3)

                    mf_recs["real_probs"].append(mf_r)
                    mf_recs["spoof_probs"].append(mf_s)
                    mf_recs["predictions"].append(mf_lbl)
                    mf_recs["latencies"].append(lat_mf)

                    fus_recs["and_preds"].append(fus.and_label)
                    fus_recs["or_preds"].append(fus.or_label)
                    fus_recs["avg_probs"].append(fus.avg_p_real)
                    fus_recs["avg_preds"].append(fus.avg_label)
                    fus_recs["temporal_preds"].append(fus.temporal_label)
                    tot_latencies.append(tot_lat)

                    and_res = fus.and_label
                    or_res = fus.or_label
                    avg_res = f"{fus.avg_p_real:.3f} ({fus.avg_label})"
                    temp_res = fus.temporal_label

                    sc_per_frame.append({
                        "frame_idx": len(tot_latencies),
                        "time_sec": round(elapsed_col, 3),
                        "mn3": {"p_real": round(mn3_r, 4), "p_spoof": round(mn3_s, 4), "label": mn3_lbl, "lat_ms": round(lat_mn3, 2)},
                        "minifas": {"p_real": round(mf_r, 4), "p_spoof": round(mf_s, 4), "label": mf_lbl, "lat_ms": round(lat_mf, 2)},
                        "fusion": {
                            "and": fus.and_label,
                            "or": fus.or_label,
                            "average": {"p_real": round(fus.avg_p_real, 4), "label": fus.avg_label},
                            "temporal": fus.temporal_label,
                        },
                        "total_lat_ms": round(tot_lat, 2),
                    })

                # Mirrored display frame
                display_frame = cv2.flip(frame, 1)
                disp_h, disp_w = display_frame.shape[:2]

                if largest_face is not None:
                    m_left, m_top, m_right, m_bottom = transform_mirrored_bbox(largest_face.bbox, disp_w)
                    # Color by temporal decision: Green (REAL), Red (SPOOF), Yellow/Orange (UNCERTAIN)
                    if temp_res == "REAL":
                        box_color = (0, 255, 0)
                    elif temp_res == "SPOOF":
                        box_color = (0, 0, 255)
                    else:
                        box_color = (0, 200, 255)

                    cv2.rectangle(display_frame, (m_left, m_top), (m_right, m_bottom), box_color, 2)
                    cv2.putText(
                        display_frame,
                        f"Temporal: {temp_res}",
                        (m_left, max(25, m_top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )

                # Exactly requested HUD elements
                col_lines = [
                    ("Presento Anti-Spoof Benchmark (HYBRID)", (0, 255, 255), 0.58, 2),
                    (f"Test: {scenario_name}", (0, 215, 255), 0.62, 2),
                    (f"Time remaining: {time_remaining:.1f}s", (0, 255, 0) if time_remaining > 3.0 else (0, 140, 255), 0.65, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"MN3 Real : {mn3_r:.3f}", (0, 255, 0) if mn3_r >= 0.5 else (0, 0, 255), 0.55, 1),
                    (f"MiniFAS  : {mf_r:.3f}", (0, 255, 0) if mf_r >= 0.5 else (0, 0, 255), 0.55, 1),
                    ("", (0, 0, 0), 0, 0),
                    ("FUSION STRATEGIES:", (220, 220, 220), 0.50, 1),
                    (f"- AND     : {and_res}", (0, 255, 0) if and_res == "REAL" else (0, 0, 255), 0.55, 2 if and_res == "REAL" else 1),
                    (f"- OR      : {or_res}", (0, 255, 0) if or_res == "REAL" else (0, 0, 255), 0.55, 2 if or_res == "REAL" else 1),
                    (f"- AVERAGE : {avg_res}", (0, 255, 0) if "REAL" in avg_res else (0, 0, 255), 0.55, 1),
                    (f"- TEMPORAL: {temp_res}", (0, 255, 0) if temp_res == "REAL" else ((0, 0, 255) if temp_res == "SPOOF" else (0, 200, 255)), 0.60, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"Collected: {len(tot_latencies)} frames", (255, 255, 255), 0.50, 1),
                    (f"Lat: MN3 {lat_mn3:.1f}ms | MF {lat_mf:.1f}ms | Total {tot_lat:.1f}ms", (200, 200, 200), 0.48, 1),
                    (f"Device: OpenVINO GPU ({clf_mn3.active_device})", (0, 200, 255), 0.48, 1),
                ]

                draw_hud_card(display_frame, col_lines, x=20, y=20)
                cv2.imshow("Presento - Hybrid Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return

            actual_duration = time.monotonic() - collect_start

            sc_metrics = compute_hybrid_scenario_metrics(
                name=scenario_name,
                duration_sec=actual_duration,
                mn3_records=mn3_recs,
                mf_records=mf_recs,
                fusion_records=fus_recs,
                hybrid_latencies=tot_latencies,
            )
            all_scenario_metrics.append(sc_metrics)
            all_per_frame_records[scenario_name] = sc_per_frame

            print_hybrid_scenario_result(sc_metrics)

            # ---------------------------------------------------------------
            # 4. POST-TEST DISPLAY PHASE (3s)
            # ---------------------------------------------------------------
            post_start = time.monotonic()
            while time.monotonic() - post_start < 3.0:
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)
                t_left = max(0.0, 3.0 - (time.monotonic() - post_start))

                fus_info = sc_metrics["fusion"]
                res_lines = [
                    (f"TEST COMPLETED: {scenario_name}", (0, 255, 0), 0.65, 2),
                    (f"Frames: {sc_metrics['frames']} | Duration: {sc_metrics['duration_sec']:.1f}s ({sc_metrics['effective_fps']:.1f} FPS)", (255, 255, 255), 0.52, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"AND Strategy : REAL {fus_info['and']['real_pct']}% | SPOOF {fus_info['and']['spoof_pct']}%", (200, 200, 200), 0.52, 1),
                    (f"OR Strategy  : REAL {fus_info['or']['real_pct']}% | SPOOF {fus_info['or']['spoof_pct']}%", (200, 200, 200), 0.52, 1),
                    (f"AVERAGE      : REAL {fus_info['average']['real_pct']}% | Mean Prob: {fus_info['average']['mean_p_real']:.3f}", (200, 200, 200), 0.52, 1),
                    (f"TEMPORAL     : REAL {fus_info['temporal']['real_pct']}% | SPOOF {fus_info['temporal']['spoof_pct']}% | UNC {fus_info['temporal']['uncertain_pct']}%", (0, 215, 255), 0.55, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"Proceeding in {t_left:.1f}s (or press SPACE)...", (0, 140, 255), 0.55, 1),
                ]
                draw_hud_card(display_frame, res_lines, x=20, y=20)
                cv2.imshow("Presento - Hybrid Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return
                elif key == 32:
                    break

        # -------------------------------------------------------------------
        # 5. FINAL REPORT & JSON EXPORT
        # -------------------------------------------------------------------
        print_hybrid_full_summary(all_scenario_metrics, clf_mn3, clf_mf, duration)

        json_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "hybrid",
            "models_used": ["anti-spoof-mn3", "minifasnet_v2se"],
            "fusion_strategies": ["and", "or", "average", "temporal"],
            "hardware": clf_mn3.full_device_name,
            "device": clf_mn3.active_device,
            "duration": duration,
            "configured_duration_sec": duration,
            "scenario_metrics": all_scenario_metrics,
            "per_frame_data": all_per_frame_records,
        }

        saved_path = save_benchmark_results(json_record, output_path)
        print(f"\n[Results Saved] Detailed hybrid records saved to: {saved_path}")
        print("Benchmark complete.\n")

    finally:
        close_camera(camera)
        cv2.destroyAllWindows()


def print_hybrid_scenario_result(metrics: Dict[str, Any]):
    """Print immediate statistics after a hybrid scenario completes."""
    print("\n" + "=" * 75)
    print(f"  HYBRID SCENARIO RESULT: {metrics['name']} ({metrics['duration_sec']:.1f}s, {metrics['frames']} frames)")
    print("=" * 75)
    mn3_m = metrics["mn3"]
    mf_m = metrics["minifas"]
    fus = metrics["fusion"]
    lat = metrics["latency"]

    print(f"  [Individual Models]")
    print(f"    MN3      : REAL={mn3_m['real_pct']}% (mean={mn3_m['real_mean']:.3f}) | Latency: {mn3_m['latency_mean']:.2f}ms")
    print(f"    MiniFAS  : REAL={mf_m['real_pct']}% (mean={mf_m['real_mean']:.3f}) | Latency: {mf_m['latency_mean']:.2f}ms")
    print(f"  [Fusion Strategies]")
    print(f"    AND      : REAL={fus['and']['real_count']} ({fus['and']['real_pct']}%), SPOOF={fus['and']['spoof_count']} ({fus['and']['spoof_pct']}%) | Runs: R={fus['and']['longest_real_run']}, S={fus['and']['longest_spoof_run']}")
    print(f"    OR       : REAL={fus['or']['real_count']} ({fus['or']['real_pct']}%), SPOOF={fus['or']['spoof_count']} ({fus['or']['spoof_pct']}%) | Runs: R={fus['or']['longest_real_run']}, S={fus['or']['longest_spoof_run']}")
    print(f"    AVERAGE  : REAL={fus['average']['real_count']} ({fus['average']['real_pct']}%), SPOOF={fus['average']['spoof_count']} ({fus['average']['spoof_pct']}%) | Mean Prob={fus['average']['mean_p_real']:.3f}")
    print(f"    TEMPORAL : REAL={fus['temporal']['real_count']} ({fus['temporal']['real_pct']}%), SPOOF={fus['temporal']['spoof_count']} ({fus['temporal']['spoof_pct']}%), UNC={fus['temporal']['uncertain_count']} ({fus['temporal']['uncertain_pct']}%) | Runs: R={fus['temporal']['longest_real_run']}, S={fus['temporal']['longest_spoof_run']}, U={fus['temporal']['longest_uncertain_run']}")
    print(f"  [Latency & Throughput]")
    print(f"    Total Hybrid Latency: mean={lat['mean']:.2f}ms, median={lat['median']:.2f}ms, p95={lat['p95']:.2f}ms | Effective FPS: {metrics['effective_fps']:.1f}")
    print("=" * 75)


def print_hybrid_full_summary(
    stats_list: List[Dict[str, Any]],
    clf_mn3: Any,
    clf_mf: Any,
    duration: float,
):
    """Print complete formatted comparison tables for the hybrid benchmark."""
    print("\n" + "=" * 130)
    print("                               PRESENTO HYBRID ANTI-SPOOF FUSION SUMMARY")
    print("=" * 130)
    print(f"Models   : anti-spoof-mn3 + MiniFASNet V2 SE")
    print(f"Hardware : {clf_mn3.full_device_name} ({clf_mn3.active_device})")
    print(f"Duration : {duration:.1f}s per scenario")
    print("=" * 130)

    # Table 1: Strategy Comparison (REAL % vs SPOOF %)
    header_1 = (
        f"{'Scenario':<18} | {'Frames':<6} | {'MN3 R%':<7} | {'MF R%':<7} | "
        f"{'AND R%':<7} | {'OR R%':<7} | {'AVG R%':<7} | {'TEMP R%':<8} | {'TEMP S%':<8} | {'TEMP U%':<8}"
    )
    print(header_1)
    print("-" * 130)

    for st in stats_list:
        f = st["fusion"]
        row = (
            f"{st['name']:<18} | {st['frames']:<6} | {st['mn3']['real_pct']:<6.1f}% | {st['minifas']['real_pct']:<6.1f}% | "
            f"{f['and']['real_pct']:<6.1f}% | {f['or']['real_pct']:<6.1f}% | {f['average']['real_pct']:<6.1f}% | "
            f"{f['temporal']['real_pct']:<7.1f}% | {f['temporal']['spoof_pct']:<7.1f}% | {f['temporal']['uncertain_pct']:<7.1f}%"
        )
        print(row)
    print("=" * 130)

    # Table 2: Temporal Consistency & Longest Runs
    print("\n" + "-" * 95)
    print("                      TEMPORAL CONSISTENCY & LONGEST CONSECUTIVE RUNS")
    print("-" * 95)
    header_2 = f"{'Scenario':<20} | {'AND (R/S Run)':<16} | {'OR (R/S Run)':<16} | {'AVG (R/S Run)':<16} | {'TEMPORAL (R/S/U)':<18}"
    print(header_2)
    print("-" * 95)

    for st in stats_list:
        f = st["fusion"]
        and_str = f"R:{f['and']['longest_real_run']} / S:{f['and']['longest_spoof_run']}"
        or_str = f"R:{f['or']['longest_real_run']} / S:{f['or']['longest_spoof_run']}"
        avg_str = f"R:{f['average']['longest_real_run']} / S:{f['average']['longest_spoof_run']}"
        temp_str = f"R:{f['temporal']['longest_real_run']} / S:{f['temporal']['longest_spoof_run']} / U:{f['temporal']['longest_uncertain_run']}"
        row_2 = f"{st['name']:<20} | {and_str:<16} | {or_str:<16} | {avg_str:<16} | {temp_str:<18}"
        print(row_2)
    print("=" * 95)

    # Table 3: Latency & Hardware Throughput
    print("\n" + "-" * 85)
    print("                        LATENCY & THROUGHPUT PERFORMANCE")
    print("-" * 85)
    header_3 = f"{'Pipeline / Model':<28} | {'Mean Latency':<14} | {'Median':<10} | {'p95':<10} | {'FPS':<10}"
    print(header_3)
    print("-" * 85)

    all_mn3_lats = [st["mn3"]["latency_mean"] for st in stats_list if st["frames"] > 0]
    all_mf_lats = [st["minifas"]["latency_mean"] for st in stats_list if st["frames"] > 0]
    all_tot_lats = [st["latency"]["mean"] for st in stats_list if st["frames"] > 0]

    mn3_mean = float(np.mean(all_mn3_lats)) if all_mn3_lats else 0.0
    mf_mean = float(np.mean(all_mf_lats)) if all_mf_lats else 0.0
    tot_mean = float(np.mean(all_tot_lats)) if all_tot_lats else 0.0

    print(f"{'anti-spoof-mn3':<28} | {f'{mn3_mean:.2f} ms':<14} | {'-':<10} | {'-':<10} | {f'{1000.0/mn3_mean:.1f}':<10}")
    print(f"{'MiniFASNet V2 SE':<28} | {f'{mf_mean:.2f} ms':<14} | {'-':<10} | {'-':<10} | {f'{1000.0/mf_mean:.1f}':<10}")
    print(f"{'Hybrid Pipeline (Dual+Fusion)':<28} | {f'{tot_mean:.2f} ms':<14} | {'-':<10} | {'-':<10} | {f'{1000.0/tot_mean:.1f}':<10}")
    print("=" * 85)


# ---------------------------------------------------------------------------
# SINGLE MODEL BENCHMARK MODES (EXISTING)
# ---------------------------------------------------------------------------

def run_synthetic_benchmark(
    model: str = "mn3",
    device: str = "GPU",
    iterations: int = 100,
) -> dict:
    """Run standalone synthetic GPU benchmark for single anti-spoofing model or hybrid."""
    if model == "hybrid":
        return run_hybrid_synthetic_benchmark(device=device, iterations=iterations)

    print("=" * 70)
    print("      Presento Standalone Face Anti-Spoofing Benchmark")
    print("=" * 70)

    core = ov.Core()
    available_devices = [d.upper() for d in core.available_devices]
    print(f"OpenVINO Available Devices: {available_devices}")

    t_load_start = time.perf_counter()
    classifier, _, model_name, model_path = resolve_model_and_classifier(model, device=device)
    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0

    print(f"Model Name                : {model_name}")
    print(f"Model File                : {model_path}")
    print(f"Active Device             : {classifier.active_device} -> {classifier.full_device_name}")
    print(f"Model Load Time           : {classifier.load_time_ms:.1f} ms")
    print(f"Warm-up Latency           : {classifier.warmup_time_ms:.2f} ms")
    print(f"Input Shape               : [1, 3, 128, 128]")
    print(f"Output Shape              : [1, 2] (Class 0: REAL, Class 1: SPOOF)")
    print("-" * 70)

    np.random.seed(42)
    latencies = []
    print(f"Running {iterations} benchmark iterations on {classifier.full_device_name}...")

    for _ in range(iterations):
        dummy_crop = np.random.randint(40, 220, (128, 128, 3), dtype=np.uint8)
        p_real, p_spoof, label, lat = classifier.predict(dummy_crop)
        latencies.append(lat)

    latencies = np.array(latencies)
    avg_latency = float(np.mean(latencies))
    median_latency = float(np.median(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n[Benchmark Latency & Throughput Results]")
    print(f"  Model                   : {model_name}")
    print(f"  Average Latency         : {avg_latency:.2f} ms")
    print(f"  Median Latency          : {median_latency:.2f} ms")
    print(f"  Min Latency             : {min_latency:.2f} ms")
    print(f"  Max Latency             : {max_latency:.2f} ms")
    print(f"  95th Percentile Latency : {p95_latency:.2f} ms")
    print(f"  Throughput (FPS)        : {fps:.1f} FPS")
    print("=" * 70)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "device": classifier.active_device,
        "full_device_name": classifier.full_device_name,
        "load_time_ms": load_time_ms,
        "warmup_time_ms": classifier.warmup_time_ms,
        "avg_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "fps": fps,
    }


class BatchCollector:
    """Collects multi-frame predictions for interactive physical testing scenarios."""

    def __init__(self, name: str, target_frames: int = 30):
        self.name = name
        self.target_frames = target_frames
        self.real_probs: List[float] = []
        self.spoof_probs: List[float] = []
        self.predictions: List[str] = []
        self.latencies: List[float] = []
        self.is_active = False

    def start(self):
        self.real_probs.clear()
        self.spoof_probs.clear()
        self.predictions.clear()
        self.latencies.clear()
        self.is_active = True

    def add(self, p_real: float, p_spoof: float, label: str, latency: float):
        if not self.is_active:
            return
        self.real_probs.append(p_real)
        self.spoof_probs.append(p_spoof)
        self.predictions.append(label)
        self.latencies.append(latency)
        if len(self.real_probs) >= self.target_frames:
            self.is_active = False

    @property
    def is_complete(self) -> bool:
        return len(self.real_probs) >= self.target_frames

    def get_summary(self) -> dict:
        if not self.real_probs:
            return {}
        n = len(self.real_probs)
        r_avg = float(np.mean(self.real_probs))
        r_min = float(np.min(self.real_probs))
        r_max = float(np.max(self.real_probs))
        s_avg = float(np.mean(self.spoof_probs))
        real_count = sum(1 for p in self.predictions if p == "REAL")
        spoof_count = sum(1 for p in self.predictions if p == "SPOOF")
        avg_lat = float(np.mean(self.latencies))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        return {
            "name": self.name,
            "frames": n,
            "real_avg": r_avg,
            "real_min": r_min,
            "real_max": r_max,
            "spoof_avg": s_avg,
            "real_count": f"{real_count}/{n}",
            "spoof_count": f"{spoof_count}/{n}",
            "consistency": f"{(max(real_count, spoof_count) / n) * 100:.1f}%",
            "avg_latency_ms": avg_lat,
            "fps": fps,
        }

    def print_report(self):
        sm = self.get_summary()
        if not sm:
            print(f"[{self.name}] No frames collected.")
            return
        print("\n" + "=" * 55)
        print(f"  {sm['name'].upper()}")
        print("=" * 55)
        print(f"  Frames Collected : {sm['frames']}")
        print(f"  REAL avg prob    : {sm['real_avg']:.3f} (min: {sm['real_min']:.3f}, max: {sm['real_max']:.3f})")
        print(f"  SPOOF avg prob   : {sm['spoof_avg']:.3f}")
        print(f"  REAL prediction  : {sm['real_count']}")
        print(f"  SPOOF prediction : {sm['spoof_count']}")
        print(f"  Consistency      : {sm['consistency']}")
        print(f"  Average Latency  : {sm['avg_latency_ms']:.2f} ms ({sm['fps']:.1f} FPS)")
        print("=" * 55)


def run_live_benchmark(
    model: str = "mn3",
    device: str = "GPU",
    target_frames_per_test: int = 30,
    source_override: Optional[str] = None,
):
    """
    Run live camera face anti-spoofing benchmark using SCRFD detection.
    Supports manual key-triggered tests (A..F).
    """
    print("=" * 70)
    print("      Presento Live Camera Anti-Spoof Benchmark")
    print("=" * 70)

    camera = init_camera(source_override)
    if camera is None:
        return

    detector = SCRFDDetector(
        model_path=settings.scrfd_model_path,
        device=settings.openvino_device,
        input_size=settings.scrfd_input_size,
        conf_thresh=settings.scrfd_conf_thresh,
        nms_thresh=settings.scrfd_nms_thresh,
    )

    is_hybrid = (model == "hybrid")
    if is_hybrid:
        clf_mn3 = AntiSpoofClassifier(model_path="data/models/anti-spoof-mn3.onnx", device=device)
        clf_mf = MiniFASNetV2SEClassifier(model_path="data/models/minifasnet_v2se.onnx", device=device)
        fusion_engine = AntiSpoofFusionEngine(window_size=15)
        model_name = "HYBRID FUSION"
    else:
        classifier, crop_fn, model_name, model_path = resolve_model_and_classifier(model, device=device)

    test_collectors: Dict[str, BatchCollector] = {
        "A": BatchCollector("Test A: Real Face", target_frames_per_test),
        "B": BatchCollector("Test B: Printed Photo", target_frames_per_test),
        "C": BatchCollector("Test C: Phone Photo", target_frames_per_test),
        "D": BatchCollector("Test D: Phone Video", target_frames_per_test),
        "E": BatchCollector("Test E: Lighting Variation", target_frames_per_test),
        "F": BatchCollector("Test F: Face Distance", target_frames_per_test),
    }
    active_test_key: Optional[str] = None

    print(f"\n[Model: {model_name}]")
    print("[Testing Controls]")
    print("  'A' : Run Test A (Real face, 30 frames)")
    print("  'B' : Run Test B (Printed photo, 30 frames)")
    print("  'C' : Run Test C (Phone photo, 30 frames)")
    print("  'D' : Run Test D (Phone video, 30 frames)")
    print("  'E' : Run Test E (Lighting variation, 30 frames)")
    print("  'F' : Run Test F (Face distance, 30 frames)")
    print("  'S' : Print full summary report of all completed tests")
    print("  'Q' : Quit")
    print("-" * 70)

    frame_count = 0
    start_time = time.monotonic()
    display_fps = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= 1.0:
                display_fps = round(frame_count / elapsed, 1)
                frame_count = 0
                start_time = now

            faces = detector.detect(frame)
            p_real, p_spoof, label, lat = 0.5, 0.5, "UNKNOWN", 0.0
            largest_face = None

            if faces:
                largest_face = max(faces, key=lambda f: (f.right - f.left) * (f.bottom - f.top))

                if is_hybrid:
                    crop_m = crop_face(frame, largest_face.bbox, scale=1.2)
                    crop_f = crop_face_minifas(frame, largest_face.bbox, scale=1.5)
                    t0 = time.perf_counter()
                    mn3_r, mn3_s, _, _ = clf_mn3.predict(crop_m)
                    mf_r, mf_s, _, _ = clf_mf.predict(crop_f)
                    fus = fusion_engine.update(mn3_r, mn3_s, mf_r, mf_s)
                    lat = (time.perf_counter() - t0) * 1000.0
                    p_real, p_spoof, label = fus.avg_p_real, fus.avg_p_spoof, fus.temporal_label
                else:
                    crop = crop_fn(frame, largest_face.bbox)
                    p_real, p_spoof, label, lat = classifier.predict(crop)

                if active_test_key is not None:
                    col = test_collectors[active_test_key]
                    if col.is_active:
                        col.add(p_real, p_spoof, label, lat)
                        if col.is_complete:
                            col.print_report()
                            active_test_key = None

            display_frame = cv2.flip(frame, 1)
            disp_h, disp_w = display_frame.shape[:2]

            if largest_face is not None:
                m_left, m_top, m_right, m_bottom = transform_mirrored_bbox(largest_face.bbox, disp_w)
                box_color = (0, 255, 0) if label == "REAL" else ((0, 200, 255) if label == "UNCERTAIN" else (0, 0, 255))
                cv2.rectangle(display_frame, (m_left, m_top), (m_right, m_bottom), box_color, 2)

                label_text = f"Pred: {label} | Real: {p_real:.3f} | Spoof: {p_spoof:.3f}"
                cv2.putText(
                    display_frame,
                    label_text,
                    (m_left, max(25, m_top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    box_color,
                    2,
                )

            active_info = "None"
            if active_test_key is not None:
                col = test_collectors[active_test_key]
                active_info = f"{col.name} ({len(col.real_probs)}/{col.target_frames})"

            hud_1 = f"Model: {model_name} | Latency: {lat:.1f} ms | Device: GPU"
            hud_2 = f"Display: {display_fps:.1f} FPS | Active: {active_info} | Keys: A..F, S: Summary, Q: Quit"

            cv2.putText(display_frame, hud_1, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display_frame, hud_2, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2)

            cv2.imshow("Presento - Anti-Spoof Benchmark", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("a"), ord("A")):
                active_test_key = "A"
                test_collectors["A"].start()
                print("\n[Started] Test A: Real Face (collecting 30 frames)...")
            elif key in (ord("b"), ord("B")):
                active_test_key = "B"
                test_collectors["B"].start()
                print("\n[Started] Test B: Printed Photo (collecting 30 frames)...")
            elif key in (ord("c"), ord("C")):
                active_test_key = "C"
                test_collectors["C"].start()
                print("\n[Started] Test C: Phone Photo (collecting 30 frames)...")
            elif key in (ord("d"), ord("D")):
                active_test_key = "D"
                test_collectors["D"].start()
                print("\n[Started] Test D: Phone Video (collecting 30 frames)...")
            elif key in (ord("e"), ord("E")):
                active_test_key = "E"
                test_collectors["E"].start()
                print("\n[Started] Test E: Lighting Variation (collecting 30 frames)...")
            elif key in (ord("f"), ord("F")):
                active_test_key = "F"
                test_collectors["F"].start()
                print("\n[Started] Test F: Face Distance (collecting 30 frames)...")
            elif key in (ord("s"), ord("S")):
                print_full_summary(test_collectors, model_name)

    finally:
        close_camera(camera)
        cv2.destroyAllWindows()
        print_full_summary(test_collectors, model_name)


def print_full_summary(collectors: Dict[str, BatchCollector], model_name: str):
    """Print structured summary table of all physical tests."""
    print("\n" + "=" * 70)
    print("          PRESENTO ANTI-SPOOF BENCHMARK SUMMARY REPORT")
    print("=" * 70)
    print(f"Model       : {model_name}")
    print("=" * 70)
    print(f"{'Test Scenario':<26} | {'Frames':<7} | {'REAL avg':<9} | {'SPOOF avg':<9} | {'Pred Count':<11} | {'Avg Lat':<8}")
    print("-" * 70)

    for key in ["A", "B", "C", "D", "E", "F"]:
        col = collectors[key]
        sm = col.get_summary()
        if sm:
            count_str = f"R:{sm['real_count'].split('/')[0]} S:{sm['spoof_count'].split('/')[0]}"
            print(
                f"{sm['name']:<26} | {sm['frames']:<7} | {sm['real_avg']:<9.3f} | {sm['spoof_avg']:<9.3f} | {count_str:<11} | {sm['avg_latency_ms']:<6.2f}ms"
            )
        else:
            print(f"{col.name:<26} | {'(not run)':<7} | {'-':<9} | {'-':<9} | {'-':<11} | {'-':<8}")
    print("=" * 70 + "\n")


def run_auto_benchmark(
    model: str = "mn3",
    device: str = "GPU",
    duration: float = 10.0,
    source_override: Optional[str] = None,
    output_path: str = "data/anti_spoof_benchmark_results.json",
):
    """
    Run automated, long-duration physical anti-spoof benchmark sequentially across 6 scenarios.
    Dispatches to hybrid auto benchmark if model == 'hybrid', otherwise runs single model.
    """
    if model == "hybrid":
        return run_hybrid_auto_benchmark(
            device=device,
            duration=duration,
            source_override=source_override,
            output_path=output_path,
        )

    classifier, crop_fn, model_name, model_path = resolve_model_and_classifier(model, device=device)

    print("\n" + "=" * 70)
    print("      PRESENTO: AUTOMATIC ANTI-SPOOF BENCHMARK")
    print("=" * 70)
    print(f"Model Name          : {model_name}")
    print(f"Model Path          : {model_path}")
    print(f"Configured Duration : {duration:.1f} seconds per scenario")
    print(f"Scenarios ({len(AUTO_SCENARIOS)})        : {', '.join(AUTO_SCENARIOS)}")
    print(f"Target Device       : {classifier.active_device} -> {classifier.full_device_name}")
    print(f"Results File        : {output_path}")
    print("=" * 70 + "\n")

    camera = init_camera(source_override)
    if camera is None:
        return

    detector = SCRFDDetector(
        model_path=settings.scrfd_model_path,
        device=settings.openvino_device,
        input_size=settings.scrfd_input_size,
        conf_thresh=settings.scrfd_conf_thresh,
        nms_thresh=settings.scrfd_nms_thresh,
    )

    all_scenario_stats: List[Dict[str, Any]] = []
    all_per_frame_records: Dict[str, List[Dict[str, Any]]] = {}

    cv2.namedWindow("Presento - Anti-Spoof Benchmark (Auto Mode)", cv2.WINDOW_AUTOSIZE)

    try:
        for sc_idx, scenario_name in enumerate(AUTO_SCENARIOS, start=1):
            print(f"\n[{sc_idx}/{len(AUTO_SCENARIOS)}] PREPARING SCENARIO: {scenario_name}")
            print(f"  -> Prepare the physical test setup for: {scenario_name}")
            print("  -> Press [SPACE] in the camera preview window when ready...")

            # Preparation
            ready_to_start = False
            while not ready_to_start:
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)

                prep_lines = [
                    (f"Presento Anti-Spoof Benchmark [{model_name}]", (0, 255, 255), 0.60, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"NEXT TEST [{sc_idx}/{len(AUTO_SCENARIOS)}]: {scenario_name}", (0, 215, 255), 0.70, 2),
                    ("Prepare the test object in front of camera.", (220, 220, 220), 0.55, 1),
                    (f"Collection Duration: {duration:.1f} seconds", (180, 180, 180), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (">>> PRESS [SPACE] WHEN READY <<<", (0, 255, 0), 0.65, 2),
                    ("Press 'Q' to quit benchmark.", (150, 150, 150), 0.45, 1),
                ]
                draw_hud_card(display_frame, prep_lines, x=20, y=20)
                cv2.imshow("Presento - Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return
                elif key == 32:
                    ready_to_start = True

            # Countdown
            countdown_start = time.monotonic()
            while True:
                elapsed_cd = time.monotonic() - countdown_start
                if elapsed_cd >= 3.0:
                    break
                secs_left = int(3.0 - elapsed_cd) + 1

                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)

                cd_lines = [
                    (f"Presento Anti-Spoof Benchmark ({model_name})", (0, 255, 255), 0.60, 2),
                    (f"Test: {scenario_name}", (0, 215, 255), 0.65, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"STARTING IN: {secs_left}...", (0, 140, 255), 0.90, 2),
                    ("Hold position steadily.", (200, 200, 200), 0.55, 1),
                ]
                draw_hud_card(display_frame, cd_lines, x=20, y=20)
                cv2.imshow("Presento - Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return

            # Collection
            print(f"  -> Collecting frames for {duration:.1f} seconds...")
            collect_start = time.monotonic()
            sc_real_probs: List[float] = []
            sc_spoof_probs: List[float] = []
            sc_predictions: List[str] = []
            sc_latencies: List[float] = []
            sc_per_frame: List[Dict[str, Any]] = []

            p_real, p_spoof, label, lat = 0.5, 0.5, "AWAITING FACE", 0.0

            while True:
                now = time.monotonic()
                elapsed_col = now - collect_start
                if elapsed_col >= duration:
                    break

                time_remaining = max(0.0, duration - elapsed_col)

                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.002)
                    continue

                faces = detector.detect(frame)
                largest_face: Optional[SCRFDFace] = None

                if faces:
                    largest_face = max(faces, key=lambda f: (f.right - f.left) * (f.bottom - f.top))
                    crop = crop_fn(frame, largest_face.bbox)
                    p_real, p_spoof, label, lat = classifier.predict(crop)

                    sc_real_probs.append(p_real)
                    sc_spoof_probs.append(p_spoof)
                    sc_predictions.append(label)
                    sc_latencies.append(lat)

                    sc_per_frame.append({
                        "frame_idx": len(sc_real_probs),
                        "time_sec": round(elapsed_col, 3),
                        "p_real": round(p_real, 4),
                        "p_spoof": round(p_spoof, 4),
                        "pred": label,
                        "lat_ms": round(lat, 2),
                    })

                display_frame = cv2.flip(frame, 1)
                disp_h, disp_w = display_frame.shape[:2]

                if largest_face is not None:
                    m_left, m_top, m_right, m_bottom = transform_mirrored_bbox(largest_face.bbox, disp_w)
                    box_color = (0, 255, 0) if label == "REAL" else (0, 0, 255)
                    cv2.rectangle(display_frame, (m_left, m_top), (m_right, m_bottom), box_color, 2)
                    cv2.putText(
                        display_frame,
                        f"Pred: {label} ({p_real:.3f})",
                        (m_left, max(25, m_top - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2,
                    )

                real_cnt = sum(1 for p in sc_predictions if p == "REAL")
                spoof_cnt = sum(1 for p in sc_predictions if p == "SPOOF")

                col_lines = [
                    ("Presento Anti-Spoof Benchmark", (0, 255, 255), 0.58, 2),
                    (f"Model: {model_name}", (180, 220, 255), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"Test: {scenario_name}", (0, 215, 255), 0.62, 2),
                    (f"Time remaining: {time_remaining:.1f}s", (0, 255, 0) if time_remaining > 3.0 else (0, 140, 255), 0.65, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"REAL: {p_real:.3f}", (0, 255, 0) if p_real >= 0.5 else (180, 180, 180), 0.55, 1),
                    (f"SPOOF: {p_spoof:.3f}", (0, 0, 255) if p_spoof >= 0.5 else (180, 180, 180), 0.55, 1),
                    (f"Prediction: {label}", (0, 255, 0) if label == "REAL" else (0, 0, 255), 0.60, 2),
                    ("", (0, 0, 0), 0, 0),
                    (f"Collected: {len(sc_real_probs)}", (255, 255, 255), 0.55, 1),
                    (f"Real: {real_cnt}", (0, 255, 0), 0.55, 1),
                    (f"Spoof: {spoof_cnt}", (0, 0, 255), 0.55, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"Latency: {lat:.1f} ms", (200, 200, 200), 0.50, 1),
                    (f"Device: OpenVINO {classifier.active_device}", (0, 200, 255), 0.50, 1),
                ]

                draw_hud_card(display_frame, col_lines, x=20, y=20)
                cv2.imshow("Presento - Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return

            actual_duration = time.monotonic() - collect_start

            metrics = compute_scenario_metrics(
                name=scenario_name,
                real_probs=sc_real_probs,
                spoof_probs=sc_spoof_probs,
                predictions=sc_predictions,
                latencies=sc_latencies,
                duration_sec=actual_duration,
            )
            all_scenario_stats.append(metrics)
            all_per_frame_records[scenario_name] = sc_per_frame

            print_scenario_result(metrics)

            # Post-test
            post_start = time.monotonic()
            while time.monotonic() - post_start < 3.0:
                ok, frame = camera.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                display_frame = cv2.flip(frame, 1)
                t_left = max(0.0, 3.0 - (time.monotonic() - post_start))

                res_lines = [
                    (f"TEST COMPLETED: {scenario_name}", (0, 255, 0), 0.65, 2),
                    (f"Model: {model_name}", (180, 220, 255), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"Frames: {metrics['frames']} | Duration: {metrics['duration_sec']:.1f}s ({metrics['fps']:.1f} FPS)", (255, 255, 255), 0.55, 1),
                    (f"REAL Avg: {metrics['real_mean']:.3f} | SPOOF Avg: {metrics['spoof_mean']:.3f}", (200, 200, 200), 0.55, 1),
                    (f"Predictions: REAL={metrics['real_pct']}% | SPOOF={metrics['spoof_pct']}%", (0, 215, 255), 0.60, 2),
                    (f"Longest Runs: REAL={metrics['temporal']['longest_real_run']} | SPOOF={metrics['temporal']['longest_spoof_run']}", (180, 180, 180), 0.50, 1),
                    ("", (0, 0, 0), 0, 0),
                    (f"Proceeding in {t_left:.1f}s (or press SPACE)...", (0, 140, 255), 0.55, 1),
                ]
                draw_hud_card(display_frame, res_lines, x=20, y=20)
                cv2.imshow("Presento - Anti-Spoof Benchmark (Auto Mode)", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[User Aborted] Benchmark cancelled by user.")
                    return
                elif key == 32:
                    break

        print_auto_benchmark_summary(all_scenario_stats, classifier, duration, model_name, model_path)

        json_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "model": model_path,
            "model_path": model_path,
            "device": classifier.active_device,
            "hardware": classifier.full_device_name,
            "configured_duration_sec": duration,
            "duration": duration,
            "scenarios": {s["name"]: s for s in all_scenario_stats},
            "per_frame_data": all_per_frame_records,
        }

        saved_path = save_benchmark_results(json_record, output_path)
        print(f"\n[Results Saved] Detailed benchmark records saved to: {saved_path}")
        print("Benchmark complete.\n")

    finally:
        close_camera(camera)
        cv2.destroyAllWindows()


def print_scenario_result(metrics: Dict[str, Any]):
    """Print immediate statistics after a scenario completes."""
    print("\n" + "=" * 65)
    print(f"  SCENARIO RESULT: {metrics['name']} (Duration: {metrics['duration_sec']:.1f}s)")
    print("=" * 65)
    print(f"  Frames Collected : {metrics['frames']}")
    print(f"  Throughput (FPS) : {metrics['fps']:.1f} FPS")
    print(
        f"  REAL Probability : mean={metrics['real_mean']:.3f}, median={metrics['real_median']:.3f}, "
        f"min={metrics['real_min']:.3f}, max={metrics['real_max']:.3f}, std={metrics['real_std']:.3f}"
    )
    print(
        f"  SPOOF Probability: mean={metrics['spoof_mean']:.3f}, median={metrics['spoof_median']:.3f}, "
        f"min={metrics['spoof_min']:.3f}, max={metrics['spoof_max']:.3f}, std={metrics['spoof_std']:.3f}"
    )
    print(
        f"  Predictions      : REAL={metrics['real_count']} ({metrics['real_pct']}%), "
        f"SPOOF={metrics['spoof_count']} ({metrics['spoof_pct']}%)"
    )
    print(
        f"  Latency          : mean={metrics['latency_mean']:.2f} ms, median={metrics['latency_median']:.2f} ms, "
        f"p95={metrics['latency_p95']:.2f} ms, min={metrics['latency_min']:.2f} ms, max={metrics['latency_max']:.2f} ms"
    )
    temp = metrics.get("temporal", {})
    print(
        f"  Temporal Consist : REAL >= 0.5: {temp.get('real_ge_05_pct', 0.0)}% | SPOOF >= 0.5: {temp.get('spoof_ge_05_pct', 0.0)}% | "
        f"Longest REAL: {temp.get('longest_real_run', 0)} | Longest SPOOF: {temp.get('longest_spoof_run', 0)}"
    )
    print("=" * 65)


def print_auto_benchmark_summary(
    stats_list: List[Dict[str, Any]],
    classifier: Any,
    duration: float,
    model_name: str,
    model_path: str,
):
    """Print the complete formatted summary table and temporal consistency table."""
    print("\n" + "=" * 125)
    print("                               PRESENTO ANTI-SPOOF BENCHMARK SUMMARY (AUTO MODE)")
    print("=" * 125)
    print(f"Model      : {model_name} ({model_path})")
    print(f"Device     : {classifier.active_device} -> {classifier.full_device_name}")
    print(f"Duration   : {duration:.1f} seconds per scenario")
    print("=" * 125)
    header_1 = (
        f"{'Scenario':<20} | {'Frames':<6} | {'REAL avg':<8} | {'REAL med':<8} | "
        f"{'REAL min':<8} | {'REAL max':<8} | {'SPOOF avg':<9} | {'SPOOF med':<9} | "
        f"{'REAL %':<7} | {'SPOOF %':<7} | {'Latency':<8} | {'FPS':<6}"
    )
    print(header_1)
    print("-" * 125)

    for st in stats_list:
        lat_str = f"{st['latency_mean']:.2f}ms"
        fps_str = f"{st['fps']:.1f}"
        real_pct_str = f"{st['real_pct']:.1f}%"
        spoof_pct_str = f"{st['spoof_pct']:.1f}%"
        row = (
            f"{st['name']:<20} | {st['frames']:<6} | {st['real_mean']:<8.3f} | {st['real_median']:<8.3f} | "
            f"{st['real_min']:<8.3f} | {st['real_max']:<8.3f} | {st['spoof_mean']:<9.3f} | {st['spoof_median']:<9.3f} | "
            f"{real_pct_str:<7} | {spoof_pct_str:<7} | {lat_str:<8} | {fps_str:<6}"
        )
        print(row)

    print("=" * 125)

    # Section 2: Temporal Consistency
    print("\n" + "-" * 80)
    print("                             TEMPORAL CONSISTENCY")
    print("-" * 80)
    header_2 = f"{'Scenario':<22} | {'REAL %':<8} | {'SPOOF %':<8} | {'Longest REAL run':<18} | {'Longest SPOOF run':<18}"
    print(header_2)
    print("-" * 80)

    for st in stats_list:
        temp = st.get("temporal", {})
        r_run = f"{temp.get('longest_real_run', 0)} frames"
        s_run = f"{temp.get('longest_spoof_run', 0)} frames"
        real_pct_str = f"{st['real_pct']:.1f}%"
        spoof_pct_str = f"{st['spoof_pct']:.1f}%"
        row_2 = (
            f"{st['name']:<22} | {real_pct_str:<8} | {spoof_pct_str:<8} | "
            f"{r_run:<18} | {s_run:<18}"
        )
        print(row_2)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Presento Face Anti-Spoofing Benchmark")
    parser.add_argument(
        "--mode",
        choices=["synthetic", "live", "auto"],
        default="synthetic",
        help="Benchmark mode: synthetic (offline GPU latency), live (manual camera test), or auto (sequential long-duration)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mn3",
        help="Model selection: 'mn3' (anti-spoof-mn3), 'minifas' (MiniFASNet V2 SE), 'hybrid' (Dual Fusion), or model path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Collection duration in seconds per scenario in auto mode (e.g. --duration 10, --duration 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="OpenVINO device (GPU or CPU)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Frames per test batch (in manual live mode)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Camera source override (e.g. 0, 1, or 'stream')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/anti_spoof_benchmark_results.json",
        help="Path to JSON file to save benchmark results",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_benchmark(
            model=args.model,
            device=args.device,
            iterations=100,
        )
    elif args.mode == "live":
        run_live_benchmark(
            model=args.model,
            device=args.device,
            target_frames_per_test=args.frames,
            source_override=args.source,
        )
    elif args.mode == "auto":
        run_auto_benchmark(
            model=args.model,
            device=args.device,
            duration=args.duration,
            source_override=args.source,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
