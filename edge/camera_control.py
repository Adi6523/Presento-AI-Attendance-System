"""
Presento Windows Camera On-Demand Control Client.
Provides client APIs and a thread-safe `CameraSession` context manager
for commanding the Raspberry Pi camera service on demand.

Features:
- Token-authenticated start, stop, status, and heartbeat requests.
- Mutual exclusion awareness: raises `CameraInUseError` (HTTP 409) if camera is locked by another session.
- Authentication error handling: raises `CameraAuthError` (HTTP 401).
- Automatic background heartbeat keepalive thread while session is active.
- Stream readiness probing before client connection.
- Safe no-op behavior for local OpenCV camera sources.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from .config import settings

logger = logging.getLogger("presento.edge.camera_control")


class CameraControlError(Exception):
    """Base exception for camera control errors."""
    pass


class CameraAuthError(CameraControlError):
    """Raised when authentication with Pi control server fails (HTTP 401)."""
    pass


class CameraInUseError(CameraControlError):
    """Raised when camera is locked by a competing session (HTTP 409)."""
    def __init__(self, message: str, active_session_type: Optional[str] = None, owner: Optional[str] = None):
        super().__init__(message)
        self.active_session_type = active_session_type
        self.owner = owner


class CameraStreamUnavailableError(CameraControlError):
    """Raised when the TCP frame stream does not become available within timeout."""
    pass


def _make_http_request(
    url: str,
    method: str = "GET",
    token: Optional[str] = None,
    data: Optional[dict] = None,
    timeout: float = 5.0,
) -> dict:
    """Send an HTTP request to the Pi control server with authentication."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "Presento-Windows-Client/1.0",
    }
    auth_token = token or settings.control_token
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    body_bytes = None
    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read().decode("utf-8", errors="replace")
        err_json = {}
        try:
            err_json = json.loads(raw_err)
        except Exception:
            pass

        if exc.code == 401:
            raise CameraAuthError("Authentication failed: invalid or missing PRESENTO_CONTROL_TOKEN.") from exc
        elif exc.code == 409:
            active_info = err_json.get("active_session", {})
            active_type = active_info.get("session_type") or err_json.get("session_type") or "another operation"
            owner = active_info.get("client_id") or err_json.get("owner")
            msg = err_json.get("message") or f"Camera currently in use by '{active_type}'."
            raise CameraInUseError(msg, active_session_type=active_type, owner=owner) from exc
        else:
            msg = err_json.get("error") or err_json.get("message") or f"HTTP {exc.code}: {exc.reason}"
            raise CameraControlError(f"Pi control server error ({exc.code}): {msg}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CameraControlError(f"Could not reach Pi control server at {url}: {exc}") from exc


def get_camera_status(
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    timeout: float = 3.0,
) -> dict:
    """Query current camera service status and active session info from Pi."""
    target_host = host or settings.pi_ip
    target_port = port or settings.control_port
    url = f"http://{target_host}:{target_port}/camera/status"
    return _make_http_request(url, method="GET", token=token, timeout=timeout)


def start_camera_service(
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    session_type: str = "attendance",
    client_id: Optional[str] = None,
    timeout: float = 6.0,
) -> dict:
    """
    Request the Pi control server to start the camera stream service.
    Raises CameraInUseError if another session already owns the camera.
    """
    target_host = host or settings.pi_ip
    target_port = port or settings.control_port
    url = f"http://{target_host}:{target_port}/camera/start"
    payload = {
        "session_type": session_type,
        "client_id": client_id or socket.gethostname(),
    }
    logger.info("Requesting camera activation on %s:%d for session '%s'...", target_host, target_port, session_type)
    res = _make_http_request(url, method="POST", token=token, data=payload, timeout=timeout)
    logger.info("Pi camera service activation confirmed: %s", res.get("status", "ok"))
    return res


def stop_camera_service(
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    session_type: str = "attendance",
    client_id: Optional[str] = None,
    force: bool = False,
    timeout: float = 6.0,
) -> dict:
    """Request the Pi control server to stop the camera stream service and release the C270."""
    target_host = host or settings.pi_ip
    target_port = port or settings.control_port
    url = f"http://{target_host}:{target_port}/camera/stop"
    payload = {
        "session_type": session_type,
        "client_id": client_id or socket.gethostname(),
        "force": force,
    }
    logger.info("Requesting camera deactivation on %s:%d for session '%s' (force=%s)...", target_host, target_port, session_type, force)
    res = _make_http_request(url, method="POST", token=token, data=payload, timeout=timeout)
    logger.info("Pi camera service deactivation confirmed: %s", res.get("status", "stopped"))
    return res


def send_heartbeat(
    host: Optional[str] = None,
    port: Optional[int] = None,
    token: Optional[str] = None,
    session_type: str = "attendance",
    client_id: Optional[str] = None,
    timeout: float = 3.0,
) -> dict:
    """Send heartbeat to refresh the camera session lease on the Pi."""
    target_host = host or settings.pi_ip
    target_port = port or settings.control_port
    url = f"http://{target_host}:{target_port}/camera/heartbeat"
    payload = {
        "session_type": session_type,
        "client_id": client_id or socket.gethostname(),
    }
    return _make_http_request(url, method="POST", token=token, data=payload, timeout=timeout)


def wait_for_stream_availability(
    host: str,
    stream_port: int,
    timeout: float = 8.0,
    poll_interval: float = 0.25,
) -> bool:
    """
    Wait until the camera TCP stream server on host:stream_port is accepting connections.
    Returns True if reachable within timeout, False otherwise.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, stream_port), timeout=1.0) as sock:
                sock.close()
                return True
        except (socket.error, OSError):
            time.sleep(poll_interval)
    return False


class CameraSession:
    """
    Context manager for strictly scoped, on-demand camera usage.
    
    Guarantees:
    - Starts the camera service upon entering the context.
    - Runs a background heartbeat thread to maintain session lease.
    - Probes stream availability before yielding control.
    - Automatically stops the camera service upon context exit (normal or exception).
    - Releases C270 hardware cleanly.
    - Completely no-ops for local OpenCV cameras.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        control_port: Optional[int] = None,
        stream_port: Optional[int] = None,
        token: Optional[str] = None,
        session_type: str = "attendance",
        client_id: Optional[str] = None,
        heartbeat_interval: Optional[float] = None,
        wait_stream_timeout: float = 8.0,
        is_local_camera: bool = False,
    ):
        self.host = host or settings.pi_ip
        self.control_port = control_port or settings.control_port
        self.stream_port = stream_port or settings.stream_port
        self.token = token or settings.control_token
        self.session_type = session_type
        self.client_id = client_id or socket.gethostname()
        self.heartbeat_interval = heartbeat_interval or settings.camera_heartbeat_interval
        self.wait_stream_timeout = wait_stream_timeout
        self.is_local_camera = is_local_camera

        self._active = False
        self._lock = threading.RLock()
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the camera session explicitly."""
        if self.is_local_camera:
            logger.debug("CameraSession: Local camera mode; skipping remote Pi activation.")
            self._active = True
            return

        with self._lock:
            if self._active:
                logger.debug("CameraSession for '%s' already active.", self.session_type)
                return

            # 1. Request Pi control server to start camera service
            start_camera_service(
                host=self.host,
                port=self.control_port,
                token=self.token,
                session_type=self.session_type,
                client_id=self.client_id,
            )
            self._active = True

            # 2. IMMEDIATELY start background heartbeat keepalive thread
            # Runs continuously in background while waiting for stream and during AI model loading
            self._start_heartbeat()

        # 3. Wait for TCP stream port to become available
        logger.info("Waiting for camera stream port %s:%d to accept connections...", self.host, self.stream_port)
        ready = wait_for_stream_availability(
            host=self.host,
            stream_port=self.stream_port,
            timeout=self.wait_stream_timeout,
        )
        if not ready:
            # Clean up immediately on failure to prevent orphaned Pi service or abandoned session
            logger.error(
                "Camera stream port %s:%d failed to accept connections within %.1fs timeout",
                self.host,
                self.stream_port,
                self.wait_stream_timeout,
            )
            self.stop(force=True)
            raise CameraStreamUnavailableError(
                f"Camera stream port {self.host}:{self.stream_port} did not become ready after {self.wait_stream_timeout}s."
            )

        logger.info("Camera stream is ready at %s:%d", self.host, self.stream_port)

    def _start_heartbeat(self) -> None:
        """Starts background heartbeat thread if not already running."""
        with self._lock:
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                logger.debug("Heartbeat thread already running for session '%s'", self.session_type)
                return

            self._heartbeat_stop_event.clear()

            def _heartbeat_worker():
                interval = max(0.05, float(self.heartbeat_interval))
                while not self._heartbeat_stop_event.wait(interval):
                    if not self._active or self._heartbeat_stop_event.is_set():
                        break
                    try:
                        send_heartbeat(
                            host=self.host,
                            port=self.control_port,
                            token=self.token,
                            session_type=self.session_type,
                            client_id=self.client_id,
                        )
                    except CameraControlError as exc:
                        err_msg = str(exc)
                        if "404" in err_msg or "No active camera session" in err_msg:
                            logger.warning(
                                "Camera session '%s' abandoned or expired on server (HTTP 404). Stopping heartbeat thread.",
                                self.session_type,
                            )
                            self._active = False
                            break
                        logger.warning("Heartbeat transmission to %s:%d failed: %s", self.host, self.control_port, exc)
                    except Exception as exc:
                        logger.warning("Heartbeat transmission to %s:%d failed: %s", self.host, self.control_port, exc)

            self._heartbeat_thread = threading.Thread(
                target=_heartbeat_worker,
                name=f"CameraHeartbeat-{self.session_type}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """Stop background heartbeat thread cleanly."""
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

    def stop(self, force: bool = False) -> None:
        """Stop the camera session and release C270."""
        self._stop_heartbeat()

        if self.is_local_camera:
            self._active = False
            return

        with self._lock:
            was_active = self._active
            self._active = False

        if was_active or force:
            try:
                stop_camera_service(
                    host=self.host,
                    port=self.control_port,
                    token=self.token,
                    session_type=self.session_type,
                    client_id=self.client_id,
                    force=force,
                )
            except Exception as exc:
                logger.warning("Error stopping Pi camera service: %s", exc)

    def __enter__(self) -> CameraSession:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
