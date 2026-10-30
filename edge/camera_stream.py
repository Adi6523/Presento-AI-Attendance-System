from __future__ import annotations

import argparse
import os
import signal
import socket
import struct
import subprocess
import sys
import time
import cv2

DEFAULT_CAMERA = os.getenv("CAMERA_DEVICE", "/dev/video0")
DEFAULT_HOST = os.getenv("PRESENTO_STREAM_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PRESENTO_STREAM_PORT", "5001"))
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30
DEFAULT_JPEG_QUALITY = 85

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    print(f"\nShutdown signal received (signal {signum}) — stopping stream...", flush=True)
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


def apply_v4l2_settings(device: str) -> None:
    """Ensure exposure_dynamic_framerate=0 and format on Linux V4L2 device."""
    try:
        subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--set-ctrl=exposure_dynamic_framerate=0"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                str(device),
                f"--set-fmt-video=width={DEFAULT_WIDTH},height={DEFAULT_HEIGHT},pixelformat=MJPG",
                f"--set-parm={DEFAULT_FPS}",
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        # Non-critical fallback if v4l2-ctl is unavailable
        pass


def open_camera(device: str | int, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, fps: int = DEFAULT_FPS) -> cv2.VideoCapture | None:
    """Open camera with MJPEG 1280x720 at 30 FPS."""
    dev = int(device) if str(device).isdigit() else device
    backend = cv2.CAP_V4L2 if isinstance(dev, str) and dev.startswith("/dev") else cv2.CAP_ANY

    cap = cv2.VideoCapture(dev, backend)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS on {device}", flush=True)
    return cap


def is_valid_frame(frame) -> bool:
    """Validate frame and reject known C270 startup dark/black frames."""
    if frame is None or frame.size == 0:
        return False

    h, w = frame.shape[:2]
    if w < 320 or h < 240:
        return False

    mean_val = float(frame.mean())
    min_val = int(frame.min())
    max_val = int(frame.max())

    # Detect known C270 startup dark/black condition without artificial sleep delays
    if mean_val <= 2.5 and min_val >= 1 and max_val <= 3:
        return False

    return True


def run_server(
    device: str | int = DEFAULT_CAMERA,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> None:
    """Main camera streaming loop serving live frames to the Windows receiver."""
    global _shutdown_requested

    if isinstance(device, str) and device.startswith("/dev"):
        apply_v4l2_settings(device)

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)

    print(f"Presento camera stream server listening on {host}:{port}", flush=True)

    cap: cv2.VideoCapture | None = None

    try:
        while not _shutdown_requested:
            if cap is None or not cap.isOpened():
                print(f"Opening camera {device}...", flush=True)
                cap = open_camera(device)
                if cap is None:
                    print("Camera unavailable; retrying in 2s...", flush=True)
                    time.sleep(2.0)
                    continue

            # Wait for incoming Windows connection
            try:
                conn, client_addr = server_sock.accept()
            except socket.timeout:
                continue

            print(f"Windows receiver connected from {client_addr[0]}:{client_addr[1]}", flush=True)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            try:
                while not _shutdown_requested:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("Camera read failed; reopening camera...", flush=True)
                        cap.release()
                        cap = None
                        break

                    if not is_valid_frame(frame):
                        continue

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                    )
                    if not ok:
                        continue

                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))

                    try:
                        conn.sendall(header + payload)
                    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                        print(f"Receiver disconnected: {exc}", flush=True)
                        break

            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print("Connection closed; waiting for new receiver connection...", flush=True)

    finally:
        print("Shutting down camera stream server...", flush=True)
        if cap is not None:
            cap.release()
        try:
            server_sock.close()
        except Exception:
            pass
        print("Camera stream server stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Presento Raspberry Pi Camera Streaming Service")
    parser.add_argument("--device", default=DEFAULT_CAMERA, help=f"Camera device path or index (default: {DEFAULT_CAMERA})")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"TCP port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY, help=f"JPEG quality 1-100 (default: {DEFAULT_JPEG_QUALITY})")

    args = parser.parse_args()
    run_server(device=args.device, host=args.host, port=args.port, jpeg_quality=args.quality)


if __name__ == "__main__":
    main()
