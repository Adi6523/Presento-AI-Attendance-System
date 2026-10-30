from __future__ import annotations

import argparse
import os
import select
import socket
import struct
import threading
import time
import cv2
import numpy as np

DEFAULT_PI_IP = os.getenv("PI_IP", "presento-pi.local")
DEFAULT_PORT = int(os.getenv("PRESENTO_STREAM_PORT", "5001"))
MAX_FRAME_SIZE = 5_000_000


def recv_exact(sock: socket.socket, size: int) -> bytes | None:
    """Read exact number of bytes from socket."""
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        except (socket.timeout, OSError):
            return None
    return bytes(data)


class FrameReceiver:
    """
    Thread-safe client that connects to the Raspberry Pi camera stream
    and retains ONLY the latest received frame (single-slot buffer, zero unbounded queue).
    Supports automatic hostname/mDNS resolution, candidate host rotation, and reconnection backoff.
    """

    def __init__(
        self,
        host: str = DEFAULT_PI_IP,
        port: int = DEFAULT_PORT,
        reconnect_delay: float = 1.5,
        max_reconnect_delay: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        from .discovery import HostCandidateRotator
        self._rotator = HostCandidateRotator(
            primary_host=self.host,
            port=self.port,
            min_retry_delay=self.reconnect_delay,
            max_retry_delay=self.max_reconnect_delay,
        )

        self._latest_frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.current_host: str = self.host
        self.resolved_ip: str | None = None
        self.received_frames: int = 0
        self.dropped_frames: int = 0
        self.decoded_frames: int = 0

    def start(self) -> "FrameReceiver":
        """Start the background receiver thread."""
        if self._running:
            return self

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, name="PresentoFrameReceiver", daemon=True)
        self._thread.start()
        return self

    def _recv_loop(self) -> None:
        """Background thread loop: handles connect, receive, decode, and reconnect with auto-discovery."""
        from .discovery import resolve_hostname

        while self._running:
            target_host = self._rotator.get_current_candidate()
            self.current_host = target_host
            try:
                # Dynamic re-resolution (re-queries DNS/mDNS so IP changes are automatically caught)
                resolved = resolve_hostname(target_host, timeout=3.0)
                connect_target = resolved or target_host
                self.resolved_ip = resolved

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                except Exception:
                    pass
                sock.settimeout(5.0)

                disp_target = f"{target_host} ({resolved})" if (resolved and resolved != target_host) else target_host
                print(f"[FrameReceiver] Connecting to Raspberry Pi at {disp_target}:{self.port}...", flush=True)
                sock.connect((connect_target, self.port))
                sock.settimeout(3.0)
                self._sock = sock
                self._connected = True
                self._rotator.record_connection_success(target_host, resolved)
                print(f"[FrameReceiver] Connected to camera stream on {disp_target}:{self.port}", flush=True)

                while self._running:
                    header = recv_exact(sock, 4)
                    if header is None:
                        print("[FrameReceiver] Pi disconnected or header read failed.", flush=True)
                        break

                    frame_size = struct.unpack("!I", header)[0]
                    if frame_size <= 0 or frame_size > MAX_FRAME_SIZE:
                        print(f"[FrameReceiver] Invalid frame size received: {frame_size}", flush=True)
                        break

                    jpeg_bytes = recv_exact(sock, frame_size)
                    if jpeg_bytes is None:
                        print("[FrameReceiver] Frame payload read failed.", flush=True)
                        break

                    self.received_frames += 1

                    # Backlog drain: If the consumer has not consumed the previous frame
                    # and more complete frames are waiting in the socket buffer,
                    # discard stale payloads without decoding and keep only the newest complete payload.
                    socket_closed_after_frame = False
                    while self._running and self._new_frame_event.is_set():
                        rlist, _, _ = select.select([sock], [], [], 0.0)
                        if not rlist:
                            break
                        next_header = recv_exact(sock, 4)
                        if next_header is None:
                            socket_closed_after_frame = True
                            break
                        next_size = struct.unpack("!I", next_header)[0]
                        if next_size <= 0 or next_size > MAX_FRAME_SIZE:
                            socket_closed_after_frame = True
                            break
                        next_payload = recv_exact(sock, next_size)
                        if next_payload is None:
                            socket_closed_after_frame = True
                            break

                        # We have successfully read another complete frame
                        jpeg_bytes = next_payload
                        self.received_frames += 1
                        self.dropped_frames += 1

                    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.decoded_frames += 1

                        # Single-slot latest frame: overwrite immediately.
                        with self._lock:
                            self._latest_frame = frame
                            self._new_frame_event.set()

                    if socket_closed_after_frame:
                        print("[FrameReceiver] Stream closed after frame.", flush=True)
                        break

            except (ConnectionRefusedError, TimeoutError, OSError) as exc:
                if self._running:
                    wait_delay = self._rotator.record_connection_failure()
                    next_cand = self._rotator.get_current_candidate()
                    print(
                        f"[FrameReceiver] Connection to {target_host}:{self.port} failed ({exc}); "
                        f"next attempt {next_cand} in {wait_delay:.1f}s...",
                        flush=True,
                    )
                    time.sleep(wait_delay)
            finally:
                self._connected = False
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None

    def read(self, timeout: float = 0.5) -> tuple[bool, np.ndarray | None]:
        """
        Fetch the newest frame. Compatible with cv2.VideoCapture.read().
        Waits up to timeout seconds for a genuinely new frame to arrive.
        Consumes the new-frame notification so the same frame is not returned repeatedly.
        """
        if not self._running and not self._new_frame_event.is_set():
            return False, None

        if timeout is not None and timeout > 0:
            if not self._new_frame_event.wait(timeout):
                return False, None
        else:
            if not self._new_frame_event.is_set():
                return False, None

        with self._lock:
            self._new_frame_event.clear()
            if self._latest_frame is None:
                return False, None
            # Return a copy to ensure safe concurrent manipulation
            return True, self._latest_frame.copy()

    def isOpened(self) -> bool:
        """Returns True if the receiver is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Returns True if the socket is actively connected."""
        return self._connected

    @property
    def active_host(self) -> str:
        """Returns the currently active or last tried host."""
        return self.current_host

    def get_stats(self) -> dict[str, int]:
        """Return frame reception, dropping, and decoding metrics."""
        return {
            "received_frames": self.received_frames,
            "dropped_frames": self.dropped_frames,
            "decoded_frames": self.decoded_frames,
        }

    def release(self) -> None:
        """Stop receiver and release all network resources."""
        self.stop()

    def stop(self) -> None:
        """Stop receiver thread cleanly."""
        self._running = False
        self._new_frame_event.set()  # Unblock any waiting read() calls
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        self._connected = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Presento Live Frame Receiver Preview")
    parser.add_argument("--host", default=DEFAULT_PI_IP, help=f"Raspberry Pi IP (default: {DEFAULT_PI_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    receiver = FrameReceiver(host=args.host, port=args.port)
    receiver.start()

    print("Presento live frame receiver preview started.")
    print("Press 'q' in the preview window to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = receiver.read(timeout=0.1)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Display FPS overlay on preview
            cv2.putText(
                frame,
                f"Pi Stream: {fps:.1f} FPS",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Presento - Live C270 Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        receiver.stop()
        cv2.destroyAllWindows()
        print("Receiver preview closed.")


if __name__ == "__main__":
    main()
