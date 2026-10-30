"""
Modern Face AI Benchmark: SCRFD (Detection + 5-point Landmarks) & ArcFace (512-D Normalized Embeddings)
Compares legacy HOG + dlib against modern neural models across ONNX Runtime CPU, OpenVINO CPU, and OpenVINO GPU (Intel Iris Xe).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import psutil

# Ensure models exist
DETECTOR_MODEL_PATH = "data/models/scrfd_500m_bnkps.onnx"
RECOGNITION_MODEL_PATH = "data/models/w600k_mbf.onnx"

# Standard 112x112 ArcFace reference landmarks
ARCFACE_REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)


def get_gpu_utilization() -> float | None:
    """Query current Windows GPU utilization percentage across GPU engines."""
    try:
        cmd = "powershell -Command \"(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum\""
        out = subprocess.check_output(cmd, shell=True, timeout=2.0).decode().strip()
        if out:
            return round(float(out), 1)
    except Exception:
        pass
    return None


class SCRFDDecoder:
    """Vectorized anchor generation, box decoding, landmark decoding, and NMS for SCRFD."""

    def __init__(self, strides: tuple[int, ...] = (8, 16, 32), num_anchors: int = 2):
        self.strides = strides
        self.num_anchors = num_anchors
        self._anchor_cache: dict[tuple[int, int], dict[int, np.ndarray]] = {}

    def get_anchors(self, height: int, width: int) -> dict[int, np.ndarray]:
        key = (height, width)
        if key in self._anchor_cache:
            return self._anchor_cache[key]

        anchors_dict = {}
        for stride in self.strides:
            feat_h = height // stride
            feat_w = width // stride
            grid_y, grid_x = np.meshgrid(np.arange(feat_h), np.arange(feat_w), indexing="ij")
            anchor_centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) * stride
            # Duplicate for num_anchors=2
            anchor_centers = np.repeat(anchor_centers, self.num_anchors, axis=0).astype(np.float32)
            anchors_dict[stride] = anchor_centers

        self._anchor_cache[key] = anchors_dict
        return anchors_dict

    def decode(
        self,
        scores_list: list[np.ndarray],
        bboxes_list: list[np.ndarray],
        kpss_list: list[np.ndarray],
        input_shape: tuple[int, int],
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decodes network outputs into bounding boxes, confidence scores, and 5 landmarks.
        Returns:
            boxes: (N, 4) in [x1, y1, x2, y2]
            scores: (N,)
            kpss: (N, 5, 2) in [x, y]
        """
        h, w = input_shape
        anchors = self.get_anchors(h, w)

        all_boxes, all_scores, all_kpss = [], [], []

        for idx, stride in enumerate(self.strides):
            scores = scores_list[idx].reshape(-1)
            bboxes = bboxes_list[idx].reshape(-1, 4)
            kpss = kpss_list[idx].reshape(-1, 10)
            anchor_pts = anchors[stride]

            # Filter by confidence threshold
            valid_mask = scores >= conf_thresh
            if not np.any(valid_mask):
                continue

            valid_scores = scores[valid_mask]
            valid_bboxes = bboxes[valid_mask] * stride
            valid_kpss = kpss[valid_mask] * stride
            valid_anchors = anchor_pts[valid_mask]

            # Decode bboxes: [cx - l, cy - t, cx + r, cy + b]
            x1 = valid_anchors[:, 0] - valid_bboxes[:, 0]
            y1 = valid_anchors[:, 1] - valid_bboxes[:, 1]
            x2 = valid_anchors[:, 0] + valid_bboxes[:, 2]
            y2 = valid_anchors[:, 1] + valid_bboxes[:, 3]
            decoded_boxes = np.column_stack([x1, y1, x2, y2])

            # Decode keypoints: [cx + dx, cy + dy]
            decoded_kps = np.zeros_like(valid_kpss)
            for k in range(5):
                decoded_kps[:, 2 * k] = valid_anchors[:, 0] + valid_kpss[:, 2 * k]
                decoded_kps[:, 2 * k + 1] = valid_anchors[:, 1] + valid_kpss[:, 2 * k + 1]

            all_boxes.append(decoded_boxes)
            all_scores.append(valid_scores)
            all_kpss.append(decoded_kps)

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        kpss = np.vstack(all_kpss).reshape(-1, 5, 2)

        # Apply NMS via cv2.dnn.NMSBoxes
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in boxes],
            scores=[float(s) for s in scores],
            score_threshold=conf_thresh,
            nms_threshold=nms_thresh,
        )

        if len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        keep = indices.flatten()
        return boxes[keep], scores[keep], kpss[keep]


class BaseInferenceEngine:
    def infer(self, input_blob: np.ndarray) -> Any:
        raise NotImplementedError

    def get_device_info(self) -> str:
        raise NotImplementedError


class ORTInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider", provider_options: dict | None = None):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # Error/Fatal only, suppress dynamic shape warnings

        providers = [provider]
        options = [provider_options] if provider_options else None

        t0 = time.monotonic()
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers, provider_options=options)
        self.load_time_ms = (time.monotonic() - t0) * 1000.0

        self.input_name = self.session.get_inputs()[0].name
        self.active_provider = self.session.get_providers()[0]

    def infer(self, input_blob: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: input_blob})

    def get_device_info(self) -> str:
        return f"ORT ({self.active_provider})"


class OpenVINOInferenceEngine(BaseInferenceEngine):
    def __init__(self, model_path: str, device: str = "CPU"):
        import openvino as ov

        core = ov.Core()
        self.device = device

        t0 = time.monotonic()
        model = core.read_model(model_path)
        self.compiled_model = core.compile_model(model, device)
        self.load_time_ms = (time.monotonic() - t0) * 1000.0

        self.infer_request = self.compiled_model.create_infer_request()
        self.input_name = self.compiled_model.inputs[0].any_name
        self.full_device_name = core.get_property(device, "FULL_DEVICE_NAME")

    def infer(self, input_blob: np.ndarray) -> list[np.ndarray]:
        res = self.infer_request.infer({self.input_name: input_blob})
        return [res[out] for out in self.compiled_model.outputs]

    def get_device_info(self) -> str:
        return f"OpenVINO {self.device} ({self.full_device_name})"


def preprocess_scrfd(image: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, float, float]:
    """
    Preprocess image for SCRFD with letterbox / direct resize.
    Returns: blob (1, 3, H, W), scale_x, scale_y
    """
    target_h, target_w = target_shape
    orig_h, orig_w = image.shape[:2]

    # Resize image
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    scale_x = orig_w / float(target_w)
    scale_y = orig_h / float(target_h)

    # Normalize: (img - 127.5) / 128.0, transpose to NCHW
    blob = (resized.astype(np.float32) - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob, scale_x, scale_y


def align_face_112(image: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Align face to standard 112x112 using partial affine similarity transform."""
    M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_REFERENCE_POINTS, method=cv2.LMEDS)
    if M is None:
        # Fallback to simple crop and resize if matrix estimation fails
        x1, y1 = int(np.min(kps[:, 0])), int(np.min(kps[:, 1]))
        x2, y2 = int(np.max(kps[:, 0])), int(np.max(kps[:, 1]))
        crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112))
    return cv2.warpAffine(image, M, (112, 112), borderValue=0.0)


def preprocess_arcface(face_112: np.ndarray) -> np.ndarray:
    """Preprocess 112x112 aligned face crop for ArcFace model."""
    blob = (face_112[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def benchmark_function(
    func: Any,
    args: tuple,
    warmup_iters: int = 10,
    measure_iters: int = 40,
    record_gpu: bool = False,
) -> dict:
    """Execute warmup and timing iterations, recording detailed latency and utilization stats."""
    # Warmup
    for _ in range(warmup_iters):
        func(*args)

    latencies = []
    cpu_measurements = []
    gpu_measurements = []

    process = psutil.Process()
    process.cpu_percent(interval=None)

    for _ in range(measure_iters):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        cpu_measurements.append(process.cpu_percent(interval=None))
        if record_gpu:
            gpu_val = get_gpu_utilization()
            if gpu_val is not None:
                gpu_measurements.append(gpu_val)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    valid_cpu = [c for c in cpu_measurements if c > 0]
    avg_cpu = float(np.mean(valid_cpu)) if valid_cpu else 0.0
    avg_gpu = float(np.mean(gpu_measurements)) if gpu_measurements else 0.0

    return {
        "mean_ms": mean_lat,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "fps": fps,
        "cpu_pct": avg_cpu,
        "gpu_pct": avg_gpu,
        "iterations": measure_iters,
    }


def run_synthetic_benchmark() -> dict:
    """Run full synthetic benchmark comparing HOG, SCRFD, and ArcFace across backends."""
    print("=" * 80)
    print("PRESENTO MODERN FACE AI BENCHMARK: SCRFD & ARCFACE EVALUATION")
    print("=" * 80)

    # Synthetic 1280x720 test frame with synthetic face feature
    test_frame_1280 = np.full((720, 1280, 3), 128, dtype=np.uint8)
    cv2.circle(test_frame_1280, (640, 360), 80, (220, 220, 220), -1)
    cv2.circle(test_frame_1280, (610, 330), 10, (50, 50, 50), -1)
    cv2.circle(test_frame_1280, (670, 330), 10, (50, 50, 50), -1)
    cv2.ellipse(test_frame_1280, (640, 390), (30, 15), 0, 0, 180, (50, 50, 50), 3)

    results: dict[str, dict] = {}
    decoder = SCRFDDecoder()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Benchmark Legacy HOG (dlib)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/5] Benchmarking Legacy HOG face detection...")
    import face_recognition

    # HOG on full 1280x720
    rgb_1280 = np.ascontiguousarray(test_frame_1280[:, :, ::-1])
    res_hog_1280 = benchmark_function(face_recognition.face_locations, (rgb_1280,), warmup_iters=2, measure_iters=10)
    print(f"  HOG @ 1280x720: {res_hog_1280['mean_ms']:.1f} ms ({res_hog_1280['fps']:.1f} FPS, P95: {res_hog_1280['p95_ms']:.1f} ms, CPU: {res_hog_1280['cpu_pct']:.1f}%)")

    # HOG on downscaled 640x360
    frame_640_360 = cv2.resize(test_frame_1280, (640, 360))
    rgb_640_360 = np.ascontiguousarray(frame_640_360[:, :, ::-1])
    res_hog_640 = benchmark_function(face_recognition.face_locations, (rgb_640_360,), warmup_iters=2, measure_iters=15)
    print(f"  HOG @ 640x360:  {res_hog_640['mean_ms']:.1f} ms ({res_hog_640['fps']:.1f} FPS, P95: {res_hog_640['p95_ms']:.1f} ms, CPU: {res_hog_640['cpu_pct']:.1f}%)")

    # Legacy dlib 128-D face embedding
    dummy_loc = [(280, 720, 440, 560)]
    res_dlib_emb = benchmark_function(face_recognition.face_encodings, (rgb_1280, dummy_loc), warmup_iters=2, measure_iters=10)
    print(f"  dlib 128-D Embedding: {res_dlib_emb['mean_ms']:.1f} ms (P95: {res_dlib_emb['p95_ms']:.1f} ms)")

    results["HOG_1280"] = res_hog_1280
    results["HOG_640"] = res_hog_640
    results["dlib_emb"] = res_dlib_emb

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Benchmark SCRFD on Backends (ORT CPU, OpenVINO CPU, OpenVINO GPU)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/5] Initializing SCRFD Detection Backends...")

    engines_scrfd: dict[str, BaseInferenceEngine] = {}

    # ORT CPU
    try:
        engines_scrfd["ORT_CPU"] = ORTInferenceEngine(DETECTOR_MODEL_PATH, provider="CPUExecutionProvider")
        print(f"  Loaded ORT CPU (startup: {engines_scrfd['ORT_CPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  ORT CPU failed: {e}")

    # OpenVINO CPU
    try:
        engines_scrfd["OV_CPU"] = OpenVINOInferenceEngine(DETECTOR_MODEL_PATH, device="CPU")
        print(f"  Loaded OpenVINO CPU (startup: {engines_scrfd['OV_CPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  OpenVINO CPU failed: {e}")

    # OpenVINO GPU (Intel Iris Xe)
    try:
        engines_scrfd["OV_GPU"] = OpenVINOInferenceEngine(DETECTOR_MODEL_PATH, device="GPU")
        print(f"  Loaded OpenVINO GPU - Intel Iris Xe (startup: {engines_scrfd['OV_GPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  OpenVINO GPU failed: {e}")

    # Resolutions to test: 640x640 (standard SCRFD), 640x360 (16:9 downscaled), 1280x720 (full resolution)
    test_resolutions = [
        (640, 640, "640x640 (Standard SCRFD)"),
        (360, 640, "640x360 (Widescreen Downscaled)"),
        (720, 1280, "1280x720 (Full Resolution)"),
    ]

    for target_h, target_w, label in test_resolutions:
        print(f"\n[3/5] Benchmarking SCRFD Detection at {label}...")
        blob, _, _ = preprocess_scrfd(test_frame_1280, (target_h, target_w))

        for eng_name, engine in engines_scrfd.items():
            is_gpu = "GPU" in eng_name
            res = benchmark_function(engine.infer, (blob,), warmup_iters=10, measure_iters=35, record_gpu=is_gpu)
            key = f"SCRFD_{eng_name}_{target_w}x{target_h}"
            results[key] = res
            gpu_str = f", GPU: {res['gpu_pct']:.1f}%" if is_gpu and res["gpu_pct"] > 0 else ""
            print(f"  {engine.get_device_info():<45} : {res['mean_ms']:.2f} ms ({res['fps']:.1f} FPS, P95: {res['p95_ms']:.2f} ms, CPU: {res['cpu_pct']:.1f}%{gpu_str})")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Benchmark ArcFace Embedding on Backends
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[4/5] Initializing ArcFace MobileFaceNet Backends...")

    engines_arcface: dict[str, BaseInferenceEngine] = {}

    try:
        engines_arcface["ORT_CPU"] = ORTInferenceEngine(RECOGNITION_MODEL_PATH, provider="CPUExecutionProvider")
        print(f"  Loaded ORT CPU (startup: {engines_arcface['ORT_CPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  ORT CPU failed: {e}")

    try:
        engines_arcface["OV_CPU"] = OpenVINOInferenceEngine(RECOGNITION_MODEL_PATH, device="CPU")
        print(f"  Loaded OpenVINO CPU (startup: {engines_arcface['OV_CPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  OpenVINO CPU failed: {e}")

    try:
        engines_arcface["OV_GPU"] = OpenVINOInferenceEngine(RECOGNITION_MODEL_PATH, device="GPU")
        print(f"  Loaded OpenVINO GPU - Intel Iris Xe (startup: {engines_arcface['OV_GPU'].load_time_ms:.1f} ms)")
    except Exception as e:
        print(f"  OpenVINO GPU failed: {e}")

    # Prepare aligned 112x112 crop
    dummy_kps = np.array([[610, 330], [670, 330], [640, 360], [620, 390], [660, 390]], dtype=np.float32)
    aligned_112 = align_face_112(test_frame_1280, dummy_kps)
    arcface_blob = preprocess_arcface(aligned_112)

    def arcface_pipeline(engine: BaseInferenceEngine, blob: np.ndarray) -> np.ndarray:
        out = engine.infer(blob)[0]
        # L2-normalize embedding vector
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / (norm + 1e-10)

    print("\n[5/5] Benchmarking ArcFace 512-D Embedding Extraction...")
    for eng_name, engine in engines_arcface.items():
        is_gpu = "GPU" in eng_name
        res = benchmark_function(arcface_pipeline, (engine, arcface_blob), warmup_iters=10, measure_iters=40, record_gpu=is_gpu)
        key = f"ArcFace_{eng_name}"
        results[key] = res
        gpu_str = f", GPU: {res['gpu_pct']:.1f}%" if is_gpu and res["gpu_pct"] > 0 else ""
        print(f"  {engine.get_device_info():<45} : {res['mean_ms']:.2f} ms (P95: {res['p95_ms']:.2f} ms, CPU: {res['cpu_pct']:.1f}%{gpu_str})")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Tables
    # ─────────────────────────────────────────────────────────────────────────
    print_summary_tables(results, engines_scrfd, engines_arcface)
    return results


def print_summary_tables(results: dict, engines_scrfd: dict, engines_arcface: dict) -> None:
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)

    print("\nTABLE 1: FACE DETECTION COMPARISON (Recommended Input Resolution)")
    print("-" * 80)
    print(f"{'Metric':<20} | {'Current HOG (640x360)':<22} | {'SCRFD CPU (640x640)':<22} | {'SCRFD GPU (640x640)':<22}")
    print("-" * 80)

    hog = results.get("HOG_640", {})
    sc_cpu = results.get("SCRFD_OV_CPU_640x640", results.get("SCRFD_ORT_CPU_640x640", {}))
    sc_gpu = results.get("SCRFD_OV_GPU_640x640", {})

    print(f"{'Detection FPS':<20} | {hog.get('fps', 0.0):<22.1f} | {sc_cpu.get('fps', 0.0):<22.1f} | {sc_gpu.get('fps', 0.0):<22.1f}")

    h_mean = f"{hog.get('mean_ms', 0.0):.2f} ms"
    sc_cpu_mean = f"{sc_cpu.get('mean_ms', 0.0):.2f} ms"
    sc_gpu_mean = f"{sc_gpu.get('mean_ms', 0.0):.2f} ms"
    print(f"{'Mean Latency':<20} | {h_mean:<22} | {sc_cpu_mean:<22} | {sc_gpu_mean:<22}")

    h_med = f"{hog.get('median_ms', 0.0):.2f} ms"
    sc_cpu_med = f"{sc_cpu.get('median_ms', 0.0):.2f} ms"
    sc_gpu_med = f"{sc_gpu.get('median_ms', 0.0):.2f} ms"
    print(f"{'Median Latency':<20} | {h_med:<22} | {sc_cpu_med:<22} | {sc_gpu_med:<22}")

    h_p95 = f"{hog.get('p95_ms', 0.0):.2f} ms"
    sc_cpu_p95 = f"{sc_cpu.get('p95_ms', 0.0):.2f} ms"
    sc_gpu_p95 = f"{sc_gpu.get('p95_ms', 0.0):.2f} ms"
    print(f"{'P95 Latency':<20} | {h_p95:<22} | {sc_cpu_p95:<22} | {sc_gpu_p95:<22}")

    h_range = f"{hog.get('min_ms', 0.0):.1f}/{hog.get('max_ms', 0.0):.1f} ms"
    sc_cpu_range = f"{sc_cpu.get('min_ms', 0.0):.1f}/{sc_cpu.get('max_ms', 0.0):.1f} ms"
    sc_gpu_range = f"{sc_gpu.get('min_ms', 0.0):.1f}/{sc_gpu.get('max_ms', 0.0):.1f} ms"
    print(f"{'Min / Max Latency':<20} | {h_range:<22} | {sc_cpu_range:<22} | {sc_gpu_range:<22}")

    h_cpu = f"{hog.get('cpu_pct', 0.0):.1f}%"
    sc_cpu_cpu = f"{sc_cpu.get('cpu_pct', 0.0):.1f}%"
    sc_gpu_cpu = f"{sc_gpu.get('cpu_pct', 0.0):.1f}%"
    print(f"{'CPU Usage':<20} | {h_cpu:<22} | {sc_cpu_cpu:<22} | {sc_gpu_cpu:<22}")

    sc_gpu_gpu = f"{sc_gpu.get('gpu_pct', 0.0):.1f}%"
    print(f"{'GPU Usage':<20} | {'0.0% (N/A)':<22} | {'0.0% (N/A)':<22} | {sc_gpu_gpu:<22}")
    print(f"{'Landmarks':<20} | {'None (BBox only)':<22} | {'5 Keypoints built-in':<22} | {'5 Keypoints built-in':<22}")
    print("-" * 80)

    print("\nTABLE 2: SCRFD RESOLUTION SCALING ON INTEL HARDWARE")
    print("-" * 80)
    print(f"{'Resolution':<25} | {'OpenVINO CPU FPS (Lat)':<25} | {'OpenVINO GPU FPS (Lat)':<25}")
    print("-" * 80)
    for res_name, res_tag in [("640x640 (Square)", "640x640"), ("640x360 (Widescreen)", "640x360"), ("1280x720 (Full)", "1280x720")]:
        cpu_r = results.get(f"SCRFD_OV_CPU_{res_tag}", results.get(f"SCRFD_ORT_CPU_{res_tag}", {}))
        gpu_r = results.get(f"SCRFD_OV_GPU_{res_tag}", {})
        cpu_str = f"{cpu_r.get('fps', 0.0):.1f} FPS ({cpu_r.get('mean_ms', 0.0):.1f} ms)" if cpu_r else "N/A"
        gpu_str = f"{gpu_r.get('fps', 0.0):.1f} FPS ({gpu_r.get('mean_ms', 0.0):.1f} ms)" if gpu_r else "N/A"
        print(f"{res_name:<25} | {cpu_str:<25} | {gpu_str:<25}")
    print("-" * 80)

    print("\nTABLE 3: FACE EMBEDDING COMPARISON (112x112 Normalized 512-D vs Legacy 128-D)")
    print("-" * 80)
    print(f"{'Metric':<22} | {'dlib 128-D CPU':<18} | {'ArcFace CPU (512-D)':<20} | {'ArcFace GPU (512-D)':<20}")
    print("-" * 80)
    dlib = results.get("dlib_emb", {})
    af_cpu = results.get("ArcFace_OV_CPU", results.get("ArcFace_ORT_CPU", {}))
    af_gpu = results.get("ArcFace_OV_GPU", {})

    dlib_mean = f"{dlib.get('mean_ms', 0.0):.2f} ms"
    af_cpu_mean = f"{af_cpu.get('mean_ms', 0.0):.2f} ms"
    af_gpu_mean = f"{af_gpu.get('mean_ms', 0.0):.2f} ms"
    print(f"{'Embedding Latency':<22} | {dlib_mean:<18} | {af_cpu_mean:<20} | {af_gpu_mean:<20}")

    dlib_med = f"{dlib.get('median_ms', 0.0):.2f} ms"
    af_cpu_med = f"{af_cpu.get('median_ms', 0.0):.2f} ms"
    af_gpu_med = f"{af_gpu.get('median_ms', 0.0):.2f} ms"
    print(f"{'Median Latency':<22} | {dlib_med:<18} | {af_cpu_med:<20} | {af_gpu_med:<20}")

    dlib_p95 = f"{dlib.get('p95_ms', 0.0):.2f} ms"
    af_cpu_p95 = f"{af_cpu.get('p95_ms', 0.0):.2f} ms"
    af_gpu_p95 = f"{af_gpu.get('p95_ms', 0.0):.2f} ms"
    print(f"{'P95 Latency':<22} | {dlib_p95:<18} | {af_cpu_p95:<20} | {af_gpu_p95:<20}")

    dlib_range = f"{dlib.get('min_ms', 0.0):.1f}/{dlib.get('max_ms', 0.0):.1f} ms"
    af_cpu_range = f"{af_cpu.get('min_ms', 0.0):.1f}/{af_cpu.get('max_ms', 0.0):.1f} ms"
    af_gpu_range = f"{af_gpu.get('min_ms', 0.0):.1f}/{af_gpu.get('max_ms', 0.0):.1f} ms"
    print(f"{'Min / Max Latency':<22} | {dlib_range:<18} | {af_cpu_range:<20} | {af_gpu_range:<20}")
    print(f"{'Provider / Hardware':<22} | {'dlib C++ CPU':<18} | {'OpenVINO CPU (i5)':<20} | {'Intel Iris Xe GPU':<20}")
    print("-" * 80)


def run_live_benchmark(seconds: float = 15.0, backend: str = "GPU", headless: bool = False) -> None:
    """Run live stream benchmark with SCRFD and ArcFace on incoming frames."""
    from edge.frame_receiver import FrameReceiver
    from edge.config import settings

    print("\n" + "=" * 80)
    print(f"STARTING LIVE STREAM BENCHMARK (Backend: {backend}, Duration: {seconds}s)")
    print("=" * 80)

    try:
        detector_engine = OpenVINOInferenceEngine(DETECTOR_MODEL_PATH, device=backend)
        recognizer_engine = OpenVINOInferenceEngine(RECOGNITION_MODEL_PATH, device=backend)
    except Exception as e:
        print(f"Failed to load OpenVINO {backend} engine: {e}. Falling back to OpenVINO CPU...")
        detector_engine = OpenVINOInferenceEngine(DETECTOR_MODEL_PATH, device="CPU")
        recognizer_engine = OpenVINOInferenceEngine(RECOGNITION_MODEL_PATH, device="CPU")

    print(f"Active Detector Backend: {detector_engine.get_device_info()}")
    print(f"Active Recognizer Backend: {recognizer_engine.get_device_info()}")

    if str(settings.camera_source).lower() in ("stream", "pi", "network"):
        print(f"Connecting to Pi stream at {settings.pi_ip}:{settings.stream_port}...")
        camera = FrameReceiver(host=settings.pi_ip, port=settings.stream_port)
        camera.start()
    else:
        try:
            cam_src = int(settings.camera_source)
        except ValueError:
            cam_src = settings.camera_source
        camera = cv2.VideoCapture(cam_src)

    decoder = SCRFDDecoder()
    start_time = time.monotonic()
    frame_count = 0
    faces_detected_total = 0
    detection_latencies = []
    embedding_latencies = []
    display_fps_readings = []

    fps_frame_count = 0
    fps_start = time.monotonic()
    current_display_fps = 0.0

    print("Live benchmark running. Press 'q' to stop early...")

    no_frame_count = 0
    fallback_camera = None

    try:
        while time.monotonic() - start_time < seconds:
            ok, frame = camera.read(timeout=0.1) if hasattr(camera, 'read') else camera.read()
            if not ok or frame is None:
                no_frame_count += 1
                if no_frame_count > 15 and fallback_camera is None:
                    print("Camera stream offline or no frames received; switching to local test pattern...")
                    fallback_camera = "synthetic"
                if fallback_camera == "synthetic":
                    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
                    cv2.circle(frame, (640, 360), 80, (220, 220, 220), -1)
                    time.sleep(0.033)  # simulate ~30 FPS source
                else:
                    time.sleep(0.01)
                    continue
            else:
                no_frame_count = 0

            frame_count += 1
            fps_frame_count += 1
            now = time.monotonic()

            if now - fps_start >= 1.0:
                current_display_fps = round(fps_frame_count / (now - fps_start), 1)
                display_fps_readings.append(current_display_fps)
                fps_frame_count = 0
                fps_start = now

            orig_h, orig_w = frame.shape[:2]
            det_blob, scale_x, scale_y = preprocess_scrfd(frame, (640, 640))
            t_det0 = time.perf_counter()
            outputs = detector_engine.infer(det_blob)
            t_det1 = time.perf_counter()
            detection_latencies.append((t_det1 - t_det0) * 1000.0)

            scores_list = [outputs[0], outputs[1], outputs[2]]
            bboxes_list = [outputs[3], outputs[4], outputs[5]]
            kpss_list = [outputs[6], outputs[7], outputs[8]]
            boxes, scores, kpss = decoder.decode(scores_list, bboxes_list, kpss_list, (640, 640), conf_thresh=0.5)

            faces_detected_total += len(boxes)

            for box, kps in zip(boxes, kpss):
                x1 = int(round(box[0] * scale_x))
                y1 = int(round(box[1] * scale_y))
                x2 = int(round(box[2] * scale_x))
                y2 = int(round(box[3] * scale_y))

                scaled_kps = kps.copy()
                scaled_kps[:, 0] *= scale_x
                scaled_kps[:, 1] *= scale_y

                t_emb0 = time.perf_counter()
                aligned_112 = align_face_112(frame, scaled_kps)
                arc_blob = preprocess_arcface(aligned_112)
                raw_emb = recognizer_engine.infer(arc_blob)[0]
                norm_emb = raw_emb / (np.linalg.norm(raw_emb, axis=1, keepdims=True) + 1e-10)
                t_emb1 = time.perf_counter()
                embedding_latencies.append((t_emb1 - t_emb0) * 1000.0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for pt in scaled_kps:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

            last_det = np.mean(detection_latencies[-10:]) if detection_latencies else 0.0
            cv2.putText(
                frame,
                f"Display: {current_display_fps:.1f} FPS | Det: {last_det:.1f}ms | Faces: {len(boxes)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if not headless:
                cv2.imshow("Presento Modern Face AI Benchmark", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        if not headless:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    avg_disp_fps = frame_count / elapsed if elapsed > 0 else 0.0
    mean_det = float(np.mean(detection_latencies)) if detection_latencies else 0.0
    p95_det = float(np.percentile(detection_latencies, 95)) if detection_latencies else 0.0
    mean_emb = float(np.mean(embedding_latencies)) if embedding_latencies else 0.0
    p95_emb = float(np.percentile(embedding_latencies, 95)) if embedding_latencies else 0.0

    print("\n" + "=" * 80)
    print("LIVE STREAM BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Runtime:           {elapsed:.1f} seconds")
    print(f"Total Frames Processed:  {frame_count}")
    print(f"Average Display FPS:     {avg_disp_fps:.1f} FPS")
    print(f"Total Faces Detected:    {faces_detected_total}")
    det_fps_val = 1000.0 / mean_det if mean_det > 0 else 0.0
    print(f"SCRFD Detection Mean:    {mean_det:.2f} ms ({det_fps_val:.1f} FPS)")
    print(f"SCRFD Detection P95:     {p95_det:.2f} ms")
    print(f"ArcFace Embedding Mean:  {mean_emb:.2f} ms")
    print(f"ArcFace Embedding P95:   {p95_emb:.2f} ms")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Presento Modern Face AI Benchmark")
    parser.add_argument("--mode", choices=["synthetic", "live"], default="synthetic", help="Benchmark mode")
    parser.add_argument("--backend", choices=["CPU", "GPU"], default="GPU", help="Inference device for live mode")
    parser.add_argument("--seconds", type=float, default=15.0, help="Duration for live mode in seconds")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=35, help="Measurement iterations")
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_synthetic_benchmark()
    else:
        run_live_benchmark(seconds=args.seconds, backend=args.backend, headless=args.headless)


if __name__ == "__main__":
    main()
