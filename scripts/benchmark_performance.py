"""
Presento — Phase F Physical Performance Benchmarking & Metric Collection Script
Measures and records:
- Source stream FPS (frames received / duration)
- Processed FPS & Display throughput
- Dropped frames & drop rate (single-slot buffer & backlog drain)
- OpenVINO inference latencies (Mean and P95 for SCRFD, ArcFace, MobileNetV3, MiniFASNet V2 SE)
- CPU and memory usage
- Attendance API latency (POST /api/attendance/events)
- Camera lifecycle stop and reconnect duration
"""
from __future__ import annotations

import os
import sys
import time
import json
import uuid
import psutil
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edge.config import settings as edge_settings
from backend.config import settings as backend_settings
from edge.camera_control import get_camera_status, stop_camera_service, CameraSession
from edge.frame_receiver import FrameReceiver
from edge.scrfd_detector import SCRFDWorker, SCRFDDetector
from edge.arcface_embedder import ArcFaceEmbedder
from edge.anti_spoof import AntiSpoofClassifier, crop_face
from edge.minifas_antispoof import MiniFASNetV2SEClassifier, crop_face_minifas
from edge.anti_spoof_worker import AntiSpoofWorker


def run_performance_benchmark(duration_sec: float = 20.0):
    print("=" * 75)
    print("   PRESENTO PHASE F: PHYSICAL CAMERA & AI PERFORMANCE BENCHMARK")
    print("=" * 75)
    print(f"Target Pi: {edge_settings.pi_ip}:{edge_settings.stream_port}")
    print(f"Benchmark duration: {duration_sec:.1f} seconds")

    process = psutil.Process(os.getpid())

    # 1. Start Camera Session on Raspberry Pi
    print("\n[1] Starting Pi camera session...")
    # First make sure any stale camera session on Pi is stopped
    stop_camera_service(host=edge_settings.pi_ip, port=edge_settings.control_port, token=edge_settings.control_token, force=True)
    time.sleep(2.5)

    cam_session = CameraSession(
        host=edge_settings.pi_ip,
        control_port=edge_settings.control_port,
        stream_port=edge_settings.stream_port,
        token=edge_settings.control_token,
        session_type="benchmark_phase_f",
    )
    t_cam_start = time.time()
    cam_session.start()
    time.sleep(2.5)
    cam_start_latency = (time.time() - t_cam_start) * 1000.0

    stat = get_camera_status(host=edge_settings.pi_ip, port=edge_settings.control_port, token=edge_settings.control_token)
    assert stat.get("service_active") == True, "Failed to start Pi camera service!"
    print(f"  Pi camera active. Start latency: {cam_start_latency:.1f} ms")

    # 2. Connect FrameReceiver Client
    print("\n[2] Connecting FrameReceiver client...")
    camera = FrameReceiver(host=edge_settings.pi_ip, port=edge_settings.stream_port)
    camera.start()

    # Wait for first frame
    t_wait_first = time.time()
    first_frame = None
    while time.time() - t_wait_first < 8.0:
        ret, frame = camera.read(timeout=1.0)
        if ret and frame is not None:
            first_frame = frame
            break

    assert first_frame is not None, "Did not receive first frame from camera stream!"
    h, w, c = first_frame.shape
    time_to_first_frame = (time.time() - t_wait_first) * 1000.0
    print(f"  First frame received in {time_to_first_frame:.1f} ms (Resolution: {w}x{h}, Channels: {c})")

    # 3. Model Benchmark (Direct Inference Latency Profiling)
    print("\n[3] Profiling direct model latencies on OpenVINO...")
    t_model_init = time.time()
    raw_detector = SCRFDDetector(
        model_path=edge_settings.scrfd_model_path,
        device="CPU",
        input_size=edge_settings.scrfd_input_size,
    )
    embedder = ArcFaceEmbedder(
        model_path=edge_settings.arcface_model_path,
        device="CPU",
    )
    mn3_classifier = AntiSpoofClassifier(
        model_path=edge_settings.anti_spoof_model_path,
        device="CPU",
    )
    minifas_classifier = MiniFASNetV2SEClassifier(
        model_path=edge_settings.minifas_model_path,
        device="CPU",
    )
    model_init_time = (time.time() - t_model_init) * 1000.0
    print(f"  Models compiled in {model_init_time:.1f} ms on {raw_detector.full_device_name}")

    # Measure exact per-model latencies over 20 iterations
    scrfd_latencies = []
    arcface_latencies = []
    mn3_latencies = []
    minifas_latencies = []
    total_ai_latencies = []

    test_crop = first_frame[100:260, 100:260]
    test_bbox = (100, 100, 260, 260)
    crop_mn3 = crop_face(first_frame, test_bbox)
    crop_mf = crop_face_minifas(first_frame, test_bbox)

    for _ in range(20):
        t0 = time.perf_counter()
        _ = raw_detector.detect(first_frame)
        t_scrfd = (time.perf_counter() - t0) * 1000.0
        scrfd_latencies.append(t_scrfd)

        t_emb = time.perf_counter()
        _ = embedder.get_embedding(test_crop)
        t_arc = (time.perf_counter() - t_emb) * 1000.0
        arcface_latencies.append(t_arc)

        t_mn3 = time.perf_counter()
        _, _, _, m_lat = mn3_classifier.predict(crop_mn3)
        t_mn3_total = (time.perf_counter() - t_mn3) * 1000.0
        mn3_latencies.append(m_lat)

        t_mf = time.perf_counter()
        _, _, _, mf_lat = minifas_classifier.predict(crop_mf)
        t_mf_total = (time.perf_counter() - t_mf) * 1000.0
        minifas_latencies.append(mf_lat)

        total_ai_latencies.append(t_scrfd + t_arc + m_lat + mf_lat)

    # 4. Sustained Production Loop (Decoupled Asynchronous Workers)
    print(f"\n[4] Running sustained decoupled pipeline for {duration_sec:.1f}s...")
    detector_worker = SCRFDWorker(
        model_path=edge_settings.scrfd_model_path,
        device="CPU",
        input_size=edge_settings.scrfd_input_size,
        conf_thresh=edge_settings.scrfd_conf_thresh,
        nms_thresh=edge_settings.scrfd_nms_thresh,
        target_fps=edge_settings.scrfd_target_fps,
        arcface_enabled=True,
    )
    detector_worker.start()

    anti_spoof_worker = AntiSpoofWorker(
        mn3_model_path=edge_settings.anti_spoof_model_path,
        minifas_model_path=edge_settings.minifas_model_path,
        window_size=15,
        real_ratio_threshold=0.80,
        spoof_ratio_threshold=0.80,
        device="CPU",
    )
    anti_spoof_worker.start()

    frame_count = 0
    cpu_measurements = []
    mem_measurements = []
    start_bench = time.monotonic()
    t_end = start_bench + duration_sec

    while time.monotonic() < t_end:
        ret, frame = camera.read(timeout=0.035)
        if not ret or frame is None:
            continue

        frame_count += 1
        detector_worker.submit_frame(frame)
        faces = detector_worker.get_latest_detections()

        active_fids = {f.face_id for f in faces}
        for face in faces:
            anti_spoof_worker.submit_face(face.face_id, frame_count, frame, face.bbox)
        anti_spoof_worker.prune_stale(active_fids, max_idle_seconds=3.0)

        for face in faces:
            _ = anti_spoof_worker.get_result(face.face_id)

        if frame_count % 15 == 0:
            cpu_measurements.append(psutil.cpu_percent(interval=None))
            mem_measurements.append(process.memory_info().rss / (1024 * 1024))

    actual_duration = time.monotonic() - start_bench
    rcv_stats = camera.get_stats()
    det_worker_stats = detector_worker.get_stats()
    as_worker_stats = anti_spoof_worker.get_stats()

    detector_worker.stop()
    anti_spoof_worker.stop()

    recv_frames = rcv_stats["received_frames"]
    decoded_frames = rcv_stats["decoded_frames"]
    dropped_frames = rcv_stats["dropped_frames"]

    source_fps = recv_frames / actual_duration
    decoded_fps = decoded_frames / actual_duration
    processed_fps = frame_count / actual_duration

    # 5. Measure Attendance API Latency
    print("\n[5] Measuring Attendance API round-trip latency...")
    api_latencies = []
    ai_headers = {"Authorization": f"Bearer {backend_settings.presento_ai_api_key}"}
    for _ in range(10):
        test_event = {
            "event_id": str(uuid.uuid4()),
            "session_id": 999999,  # Non-existent session probes validation & return
            "student_id": "23UAM090",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.95,
            "liveness_state": "REAL_LOCKED",
            "liveness_score": 0.98,
            "device_id": edge_settings.device_id,
            "source": "BENCHMARK_PROBE"
        }
        t_req = time.perf_counter()
        r = requests.post(f"{edge_settings.api_url}/api/attendance/events", json=test_event, headers=ai_headers, timeout=5)
        api_latencies.append((time.perf_counter() - t_req) * 1000.0)
    api_latency_mean = float(np.mean(api_latencies))
    api_latency_p95 = float(np.percentile(api_latencies, 95))

    # 6. Measure Camera Reconnect Cycle Time
    print("\n[6] Measuring Camera Reconnect Cycle Time...")
    t_rec_stop = time.time()
    camera.stop()
    cam_session.stop()
    time.sleep(2.5)
    cam_stop_duration = (time.time() - t_rec_stop) * 1000.0

    t_rec_start = time.time()
    reconnect_session = CameraSession(
        host=edge_settings.pi_ip,
        control_port=edge_settings.control_port,
        stream_port=edge_settings.stream_port,
        token=edge_settings.control_token,
        session_type="benchmark_reconnect",
    )
    reconnect_session.start()
    time.sleep(2.5)
    reconnect_camera = FrameReceiver(host=edge_settings.pi_ip, port=edge_settings.stream_port)
    reconnect_camera.start()

    rec_frame = None
    t_wait_rec = time.time()
    while time.time() - t_wait_rec < 6.0:
        ret, frame = reconnect_camera.read(timeout=0.5)
        if ret and frame is not None:
            rec_frame = frame
            break
    total_reconnect_time = (time.time() - t_rec_start) * 1000.0

    reconnect_camera.stop()
    reconnect_session.stop()
    stop_camera_service(host=edge_settings.pi_ip, port=edge_settings.control_port, token=edge_settings.control_token, force=True)
    time.sleep(0.5)

    # 7. Compile Report Metrics
    metrics = {
        "hardware": {
            "camera": "Logitech C270 (V4L2 /dev/video0)",
            "camera_host": f"Raspberry Pi 4 Model B ({edge_settings.pi_ip})",
            "resolution": f"{w}x{h}",
            "channels": c,
            "edge_device": f"Windows 11 ({raw_detector.full_device_name})",
        },
        "stream_performance": {
            "duration_seconds": round(actual_duration, 2),
            "frames_received": recv_frames,
            "source_stream_fps": round(source_fps, 2),
            "frames_decoded": decoded_frames,
            "decoded_fps": round(decoded_fps, 2),
            "frames_processed": frame_count,
            "processed_fps": round(processed_fps, 2),
            "frames_dropped": dropped_frames,
            "drop_rate_pct": round((dropped_frames / max(recv_frames, 1)) * 100.0, 2),
        },
        "latencies_ms": {
            "model_initialization_ms": round(model_init_time, 2),
            "time_to_first_frame_ms": round(time_to_first_frame, 2),
            "scrfd_mean_ms": round(float(np.mean(scrfd_latencies)), 2),
            "scrfd_p95_ms": round(float(np.percentile(scrfd_latencies, 95)), 2),
            "arcface_mean_ms": round(float(np.mean(arcface_latencies)), 2),
            "arcface_p95_ms": round(float(np.percentile(arcface_latencies, 95)), 2),
            "mn3_mean_ms": round(float(np.mean(mn3_latencies)), 2),
            "mn3_p95_ms": round(float(np.percentile(mn3_latencies, 95)), 2),
            "minifas_mean_ms": round(float(np.mean(minifas_latencies)), 2),
            "minifas_p95_ms": round(float(np.percentile(minifas_latencies, 95)), 2),
            "anti_spoof_total_mean_ms": round(float(np.mean(mn3_latencies) + np.mean(minifas_latencies)), 2),
            "anti_spoof_total_p95_ms": round(float(np.percentile(mn3_latencies, 95) + np.percentile(minifas_latencies, 95)), 2),
            "total_ai_per_frame_mean_ms": round(float(np.mean(total_ai_latencies)), 2),
            "total_ai_per_frame_p95_ms": round(float(np.percentile(total_ai_latencies, 95)), 2),
            "attendance_api_mean_ms": round(api_latency_mean, 2),
            "attendance_api_p95_ms": round(api_latency_p95, 2),
            "camera_stop_ms": round(cam_stop_duration, 2),
            "camera_reconnect_total_ms": round(total_reconnect_time, 2),
        },
        "system_resources": {
            "process_ram_mb": round(float(np.mean(mem_measurements)) if mem_measurements else 0.0, 2),
            "cpu_percent_avg": round(float(np.mean(cpu_measurements)) if cpu_measurements else 0.0, 2),
            "cpu_percent_max": round(float(np.max(cpu_measurements)) if cpu_measurements else 0.0, 2),
        }
    }

    print("\n" + "=" * 75)
    print("                    PERFORMANCE BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Source Stream Rate       : {metrics['stream_performance']['source_stream_fps']} FPS")
    print(f"Frames Received          : {metrics['stream_performance']['frames_received']}")
    print(f"Frames Decoded           : {metrics['stream_performance']['frames_decoded']} ({metrics['stream_performance']['decoded_fps']} FPS)")
    print(f"Frames AI Processed      : {metrics['stream_performance']['frames_processed']} ({metrics['stream_performance']['processed_fps']} FPS)")
    print(f"Frames Dropped           : {metrics['stream_performance']['frames_dropped']} ({metrics['stream_performance']['drop_rate_pct']}%)")
    print("-" * 75)
    print(f"SCRFD Latency (Mean/P95) : {metrics['latencies_ms']['scrfd_mean_ms']} ms / {metrics['latencies_ms']['scrfd_p95_ms']} ms")
    print(f"ArcFace Latency (Mean/P95: {metrics['latencies_ms']['arcface_mean_ms']} ms / {metrics['latencies_ms']['arcface_p95_ms']} ms")
    print(f"MN3 Latency (Mean/P95)   : {metrics['latencies_ms']['mn3_mean_ms']} ms / {metrics['latencies_ms']['mn3_p95_ms']} ms")
    print(f"MiniFASNet (Mean/P95)    : {metrics['latencies_ms']['minifas_mean_ms']} ms / {metrics['latencies_ms']['minifas_p95_ms']} ms")
    print(f"Total AI Frame Latency   : {metrics['latencies_ms']['total_ai_per_frame_mean_ms']} ms / {metrics['latencies_ms']['total_ai_per_frame_p95_ms']} ms")
    print("-" * 75)
    print(f"Attendance API Latency   : {metrics['latencies_ms']['attendance_api_mean_ms']} ms (P95: {metrics['latencies_ms']['attendance_api_p95_ms']} ms)")
    print(f"Camera Reconnect Time    : {metrics['latencies_ms']['camera_reconnect_total_ms']} ms")
    print(f"Process RAM Usage        : {metrics['system_resources']['process_ram_mb']} MB")
    print(f"System CPU Usage         : {metrics['system_resources']['cpu_percent_avg']}% (Peak: {metrics['system_resources']['cpu_percent_max']}%)")
    print("=" * 75)

    out_path = Path("data/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_path.resolve()}")

    return metrics


if __name__ == "__main__":
    run_performance_benchmark(duration_sec=20.0)
