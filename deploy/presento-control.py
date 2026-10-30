#!/usr/bin/env python3
"""
Presento Raspberry Pi Camera On-Demand Control Server.
Runs as a lightweight HTTP micro-service on the Raspberry Pi (default port: 5002).

See edge/pi_control_server.py for detailed architecture.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PiControl] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("presento.pi_control")

DEFAULT_HOST = os.getenv("PRESENTO_CONTROL_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PRESENTO_CONTROL_PORT", "5002"))
DEFAULT_STREAM_PORT = int(os.getenv("PRESENTO_STREAM_PORT", "5001"))
DEFAULT_CAMERA_SERVICE = os.getenv("PRESENTO_CAMERA_SERVICE", "presento-camera.service")
DEFAULT_TIMEOUT_SEC = max(60.0, float(os.getenv("PRESENTO_CAMERA_SESSION_TIMEOUT", "60.0")))


def load_token_from_env_or_file() -> str:
    token = os.getenv("PRESENTO_CONTROL_TOKEN", "").strip()
    if token:
        return token

    config_file = "/etc/presento/control.env"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("PRESENTO_CONTROL_TOKEN="):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        if val:
                            return val
        except Exception as exc:
            logger.warning("Could not read %s: %s", config_file, exc)

    return ""


class CameraSessionState:
    def __init__(self, default_timeout: float = DEFAULT_TIMEOUT_SEC):
        self._lock = threading.Lock()
        self.default_timeout = default_timeout
        self.active_session_type: Optional[str] = None
        self.active_client_id: Optional[str] = None
        self.started_at: float = 0.0
        self.last_heartbeat: float = 0.0

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self.active_session_type is not None

    def get_snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            if self.active_session_type is None:
                return {
                    "in_use": False,
                    "session_type": None,
                    "client_id": None,
                    "uptime_seconds": 0.0,
                    "lease_remaining_seconds": 0.0,
                }
            lease_remaining = max(0.0, self.default_timeout - (now - self.last_heartbeat))
            return {
                "in_use": True,
                "session_type": self.active_session_type,
                "client_id": self.active_client_id,
                "uptime_seconds": round(now - self.started_at, 2),
                "lease_remaining_seconds": round(lease_remaining, 2),
            }

    def acquire(self, session_type: str, client_id: str) -> tuple[bool, str, int]:
        with self._lock:
            now = time.time()
            if self.active_session_type is not None:
                if now - self.last_heartbeat > self.default_timeout:
                    logger.warning(
                        "Expiring abandoned session '%s' (idle %.1fs > %.1fs)",
                        self.active_session_type,
                        now - self.last_heartbeat,
                        self.default_timeout,
                    )
                elif self.active_session_type == session_type:
                    self.last_heartbeat = now
                    if client_id:
                        self.active_client_id = client_id
                    return True, f"Session '{session_type}' refreshed", 200
                else:
                    elapsed = now - self.started_at
                    msg = (
                        f"Camera currently in use by '{self.active_session_type}' "
                        f"(client: {self.active_client_id or 'unknown'}, active: {elapsed:.1f}s)"
                    )
                    return False, msg, 409

            self.active_session_type = session_type
            self.active_client_id = client_id or "unknown"
            self.started_at = now
            self.last_heartbeat = now
            return True, f"Acquired camera lock for '{session_type}'", 200

    def renew_heartbeat(self, session_type: str, client_id: Optional[str] = None) -> tuple[bool, str, int]:
        with self._lock:
            now = time.time()
            if self.active_session_type is None:
                return False, "No active camera session to heartbeat", 404

            if session_type and self.active_session_type != session_type:
                return False, f"Heartbeat mismatch: camera is locked by '{self.active_session_type}'", 409

            self.last_heartbeat = now
            remaining = max(0.0, self.default_timeout)
            return True, f"Heartbeat received (lease reset to {remaining:.0f}s)", 200

    def release(self, session_type: Optional[str] = None, force: bool = False) -> tuple[bool, str, int]:
        with self._lock:
            if self.active_session_type is None:
                return True, "No active camera session", 200

            if not force and session_type and self.active_session_type != session_type:
                return False, f"Cannot stop camera: owned by '{self.active_session_type}'", 409

            released_type = self.active_session_type
            self.active_session_type = None
            self.active_client_id = None
            self.started_at = 0.0
            self.last_heartbeat = 0.0
            return True, f"Released camera lock for '{released_type}'", 200

    def check_watchdog_expiry(self) -> tuple[bool, Optional[str]]:
        with self._lock:
            if self.active_session_type is None:
                return False, None
            now = time.time()
            if now - self.last_heartbeat > self.default_timeout:
                expired_type = self.active_session_type
                self.active_session_type = None
                self.active_client_id = None
                self.started_at = 0.0
                self.last_heartbeat = 0.0
                return True, expired_type
            return False, None


class CameraServiceManager:
    def __init__(
        self,
        service_name: str = DEFAULT_CAMERA_SERVICE,
        systemctl_runner: Optional[Callable[[list[str]], tuple[int, str]]] = None,
    ):
        self.service_name = service_name
        self._runner = systemctl_runner or self._default_systemctl_runner

    @staticmethod
    def _default_systemctl_runner(cmd: list[str]) -> tuple[int, str]:
        try:
            res = subprocess.run(
                ["sudo", "systemctl"] + cmd,
                capture_output=True,
                text=True,
                timeout=8.0,
                check=False,
            )
            output = (res.stdout or "") + (res.stderr or "")
            return res.returncode, output.strip()
        except Exception as exc:
            return 1, str(exc)

    def start_service(self) -> tuple[bool, str]:
        code, out = self._runner(["start", self.service_name])
        if code == 0:
            return True, f"Started {self.service_name}"
        return False, f"Failed to start {self.service_name} (code {code}): {out}"

    def stop_service(self) -> tuple[bool, str]:
        code, out = self._runner(["stop", self.service_name])
        if code == 0:
            return True, f"Stopped {self.service_name}"
        return False, f"Failed to stop {self.service_name} (code {code}): {out}"

    def is_active(self) -> bool:
        code, out = self._runner(["is-active", self.service_name])
        return code == 0 and out.strip() == "active"


class ControlServerApp:
    def __init__(
        self,
        token: str,
        service_name: str = DEFAULT_CAMERA_SERVICE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SEC,
        stream_port: int = DEFAULT_STREAM_PORT,
        systemctl_runner: Optional[Callable[[list[str]], tuple[int, str]]] = None,
    ):
        self.token = token
        self.stream_port = stream_port
        self.session_state = CameraSessionState(default_timeout=timeout_seconds)
        self.service_manager = CameraServiceManager(
            service_name=service_name,
            systemctl_runner=systemctl_runner,
        )
        self._watchdog_stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    def start_watchdog(self, interval: Optional[float] = None) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        self._watchdog_stop_event.clear()
        poll_interval = interval if interval is not None else min(5.0, max(0.2, self.session_state.default_timeout / 4.0))

        def _watchdog_loop():
            while not self._watchdog_stop_event.wait(poll_interval):
                expired, expired_type = self.session_state.check_watchdog_expiry()
                if expired:
                    logger.warning(
                        "[WATCHDOG] Camera session '%s' abandoned (no heartbeat). Auto-stopping camera service...",
                        expired_type,
                    )
                    self.service_manager.stop_service()

        self._watchdog_thread = threading.Thread(target=_watchdog_loop, name="PiControlWatchdog", daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop_event.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

    def verify_token(self, auth_header: Optional[str], custom_header: Optional[str]) -> bool:
        if not self.token:
            return False

        provided = None
        if auth_header and auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
        elif custom_header:
            provided = custom_header.strip()

        if not provided:
            return False

        return hmac.compare_digest(provided.encode("utf-8"), self.token.encode("utf-8"))


def make_request_handler(app: ControlServerApp):
    class ControlRequestHandler(BaseHTTPRequestHandler):
        server_version = "PresentoPiControl/1.0"

        def log_message(self, fmt, *args):
            logger.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), fmt % args)

        def _send_json(self, status_code: int, data: dict):
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _check_auth(self) -> bool:
            auth_header = self.headers.get("Authorization")
            custom_header = self.headers.get("X-Control-Token")
            if not app.verify_token(auth_header, custom_header):
                self._send_json(401, {"error": "Unauthorized", "message": "Missing or invalid PRESENTO_CONTROL_TOKEN"})
                return False
            return True

        def _read_body_json(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return {}
                raw = self.rfile.read(length).decode("utf-8")
                return json.loads(raw) if raw else {}
            except Exception:
                return {}

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")

            if path == "/health" or path == "":
                self._send_json(200, {"status": "ok", "service": "presento-control"})
                return

            if path == "/camera/status":
                if not self._check_auth():
                    return

                snap = app.session_state.get_snapshot()
                is_active = app.service_manager.is_active()

                self._send_json(
                    200,
                    {
                        "service": app.service_manager.service_name,
                        "service_active": is_active,
                        "stream_port": app.stream_port,
                        "session": snap,
                    },
                )
                return

            self._send_json(404, {"error": "Not Found", "path": path})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            body = self._read_body_json()

            if not self._check_auth():
                return

            session_type = str(body.get("session_type", "attendance")).strip()
            client_id = str(body.get("client_id", self.client_address[0])).strip()

            if path == "/camera/start":
                ok, msg, status_code = app.session_state.acquire(session_type, client_id)
                if not ok:
                    snap = app.session_state.get_snapshot()
                    self._send_json(
                        status_code,
                        {
                            "error": "Camera currently in use",
                            "message": msg,
                            "active_session": snap,
                        },
                    )
                    return

                svc_ok, svc_msg = app.service_manager.start_service()
                if not svc_ok:
                    app.session_state.release(session_type, force=True)
                    self._send_json(500, {"error": "Failed to start camera service", "detail": svc_msg})
                    return

                logger.info("Camera started on-demand for session '%s' (client: %s)", session_type, client_id)
                self._send_json(
                    200,
                    {
                        "status": "started",
                        "active": True,
                        "session_type": session_type,
                        "client_id": client_id,
                        "stream_port": app.stream_port,
                        "message": "Camera service started successfully",
                    },
                )
                return

            elif path == "/camera/heartbeat":
                ok, msg, status_code = app.session_state.renew_heartbeat(session_type, client_id)
                if not ok:
                    self._send_json(status_code, {"error": msg})
                    return

                snap = app.session_state.get_snapshot()
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "session_type": session_type,
                        "lease_remaining_seconds": snap["lease_remaining_seconds"],
                    },
                )
                return

            elif path == "/camera/stop":
                force = bool(body.get("force", False))
                ok, msg, status_code = app.session_state.release(session_type, force=force)
                if not ok:
                    self._send_json(status_code, {"error": msg})
                    return

                svc_ok, svc_msg = app.service_manager.stop_service()
                logger.info("Camera stopped on-demand for session '%s' (force=%s)", session_type, force)
                self._send_json(
                    200,
                    {
                        "status": "stopped",
                        "active": False,
                        "service_message": svc_msg,
                    },
                )
                return

            self._send_json(404, {"error": "Not Found", "path": path})

    return ControlRequestHandler


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: Optional[str] = None,
    service_name: str = DEFAULT_CAMERA_SERVICE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SEC,
    stream_port: int = DEFAULT_STREAM_PORT,
    systemctl_runner: Optional[Callable[[list[str]], tuple[int, str]]] = None,
) -> tuple[HTTPServer, ControlServerApp]:
    secret = token or load_token_from_env_or_file()
    if not secret:
        logger.warning(
            "[SECURITY WARNING] No PRESENTO_CONTROL_TOKEN configured! "
            "Requests will be rejected until a token is provided in environment or /etc/presento/control.env."
        )

    app = ControlServerApp(
        token=secret,
        service_name=service_name,
        timeout_seconds=timeout_seconds,
        stream_port=stream_port,
        systemctl_runner=systemctl_runner,
    )
    app.start_watchdog()

    handler_cls = make_request_handler(app)
    httpd = HTTPServer((host, port), handler_cls)
    logger.info("Presento Pi Camera Control Server listening on %s:%d (Stream: %d)", host, port, stream_port)
    return httpd, app


def main():
    parser = argparse.ArgumentParser(description="Presento Raspberry Pi Camera Control Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind (default: {DEFAULT_PORT})")
    parser.add_argument("--token", default=None, help="Shared control token (defaults to PRESENTO_CONTROL_TOKEN)")
    parser.add_argument("--service", default=DEFAULT_CAMERA_SERVICE, help="Systemd camera service unit name")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="Watchdog grace period in seconds")
    args = parser.parse_args()

    httpd, app = run_server(
        host=args.host,
        port=args.port,
        token=args.token,
        service_name=args.service,
        timeout_seconds=args.timeout,
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Pi control server...")
    finally:
        app.stop_watchdog()
        httpd.server_close()
        logger.info("Pi control server stopped.")


if __name__ == "__main__":
    main()
