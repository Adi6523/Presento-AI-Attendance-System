"""
Comprehensive test suite for Presento On-Demand Camera Operation.
Tests:
1. Systemd unit definitions and installation configuration (camera service not enabled on boot).
2. Token authentication (401 Unauthorized for missing/invalid tokens, 200 for valid Bearer / X-Control-Token).
3. Mutual exclusion single-session locking (409 Conflict when competing session attempts to start).
4. Same-session idempotent refresh.
5. Heartbeat lease extension and watchdog auto-stop upon abandonment.
6. CameraSession context manager lifecycle (start, heartbeat, stop, stream probing).
7. Enrollment lifecycle (starts on enrollment, cleans up on exit/cancellation/error).
8. Attendance lifecycle (starts on attendance, stops on close/exit).
9. Idle guarantee (launcher scripts and idle backend do not turn on camera).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from edge.pi_control_server import (
    ControlServerApp,
    run_server,
    CameraSessionState,
    CameraServiceManager,
)
from edge.camera_control import (
    CameraSession,
    CameraInUseError,
    CameraAuthError,
    CameraControlError,
    CameraStreamUnavailableError,
    start_camera_service,
    stop_camera_service,
    send_heartbeat,
    get_camera_status,
    wait_for_stream_availability,
)


class TestOnDemandCameraArchitecture(unittest.TestCase):
    """Test 1: Verify service definitions, deployment script, and configuration."""

    def test_camera_service_unit_file_and_installer_disable_on_boot(self):
        """Verify that presento-camera.service has Restart=always and install script disables it on boot."""
        service_path = os.path.join("deploy", "presento-camera.service")
        self.assertTrue(os.path.exists(service_path), "deploy/presento-camera.service must exist")

        with open(service_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Service must restart if crashes while active
        self.assertIn("Restart=always", content)
        self.assertIn("camera_stream.py", content)

        # Installer script must explicitly disable camera service on boot (on-demand mode)
        installer_path = os.path.join("deploy", "install_pi_service.sh")
        self.assertTrue(os.path.exists(installer_path), "deploy/install_pi_service.sh must exist")

        with open(installer_path, "r", encoding="utf-8") as f:
            install_content = f.read()

        self.assertIn("systemctl disable presento-camera.service", install_content)
        self.assertIn("presento-control.service", install_content)
        self.assertIn("systemctl enable presento-control.service", install_content)

    def test_control_service_unit_file(self):
        """Verify that deploy/presento-control.service exists and is configured for port 5002."""
        service_path = os.path.join("deploy", "presento-control.service")
        self.assertTrue(os.path.exists(service_path), "deploy/presento-control.service must exist")

        with open(service_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("5002", content)
        self.assertIn("WantedBy=multi-user.target", content)

    def test_launchers_do_not_auto_start_camera(self):
        """Verify that Start-Presento.bat does NOT start edge.runner at boot."""
        bat_path = "Start-Presento.bat"
        with open(bat_path, "r", encoding="utf-8") as f:
            bat_content = f.read()

        self.assertNotIn("start \"Presento Camera Runner\"", bat_content)
        self.assertIn("On-Demand", bat_content)


class TestPiControlServerLive(unittest.TestCase):
    """Test 2-5: Live HTTP control server tests with token auth, session locking, and watchdog."""

    TEST_TOKEN = "secret-test-token-12345"

    @classmethod
    def setUpClass(cls):
        cls.mock_service_state = {"active": False, "starts": 0, "stops": 0}

        def mock_runner(cmd: list[str]) -> tuple[int, str]:
            action = cmd[0]
            if action == "start":
                cls.mock_service_state["active"] = True
                cls.mock_service_state["starts"] += 1
                return 0, "Started"
            elif action == "stop":
                cls.mock_service_state["active"] = False
                cls.mock_service_state["stops"] += 1
                return 0, "Stopped"
            elif action == "is-active":
                return (0, "active") if cls.mock_service_state["active"] else (3, "inactive")
            return 0, ""

        # Bind to ephemeral port
        cls.httpd, cls.app = run_server(
            host="127.0.0.1",
            port=0,
            token=cls.TEST_TOKEN,
            timeout_seconds=2.0,  # short 2.0s watchdog for testing
            stream_port=5999,
            systemctl_runner=mock_runner,
        )
        cls.server_port = cls.httpd.server_address[1]

        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.stop_watchdog()
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # Reset server session state before each test
        self.app.session_state.release(force=True)
        self.mock_service_state["active"] = False

    def test_unauthenticated_requests_rejected(self):
        """Verify HTTP 401 Unauthorized when token is missing or incorrect."""
        # 1. Missing token
        with self.assertRaises(CameraAuthError):
            start_camera_service(
                host="127.0.0.1",
                port=self.server_port,
                token=None,
                session_type="attendance",
            )

        # 2. Invalid token
        with self.assertRaises(CameraAuthError):
            start_camera_service(
                host="127.0.0.1",
                port=self.server_port,
                token="wrong-token",
                session_type="attendance",
            )

    def test_authenticated_start_and_status(self):
        """Verify camera starts on-demand with valid token and reflects active status."""
        res = start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
            client_id="test-client-1",
        )
        self.assertEqual(res["status"], "started")
        self.assertTrue(res["active"])
        self.assertEqual(res["session_type"], "attendance")
        self.assertTrue(self.mock_service_state["active"])

        # Query status
        status = get_camera_status(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
        )
        self.assertTrue(status["service_active"])
        self.assertTrue(status["session"]["in_use"])
        self.assertEqual(status["session"]["session_type"], "attendance")

    def test_mutual_exclusion_competing_sessions_rejected(self):
        """Verify that an active attendance session rejects an enrollment session with 409 Conflict."""
        # 1. Start attendance session
        start_res = start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
            client_id="attendance-client",
        )
        self.assertEqual(start_res["status"], "started")

        # 2. Attempt to start enrollment while attendance is active
        with self.assertRaises(CameraInUseError) as ctx:
            start_camera_service(
                host="127.0.0.1",
                port=self.server_port,
                token=self.TEST_TOKEN,
                session_type="enrollment",
                client_id="enrollment-client",
            )

        self.assertIn("attendance", str(ctx.exception).lower())

        # 3. Same attendance session refreshes idempotently (200 OK)
        refresh_res = start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
            client_id="attendance-client",
        )
        self.assertIn(refresh_res["status"], ("started", "ok"))

        # 4. Stop attendance session
        stop_res = stop_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
        )
        self.assertEqual(stop_res["status"], "stopped")
        self.assertFalse(self.mock_service_state["active"])

        # 5. Now enrollment session can acquire the lock cleanly
        enroll_res = start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="enrollment",
            client_id="enrollment-client",
        )
        self.assertEqual(enroll_res["status"], "started")
        self.assertEqual(enroll_res["session_type"], "enrollment")

        # Clean up
        stop_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="enrollment",
        )

    def test_heartbeat_lease_renewal(self):
        """Verify that heartbeat resets remaining lease duration."""
        start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
        )

        hb_res = send_heartbeat(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
        )
        self.assertEqual(hb_res["status"], "ok")
        self.assertGreater(hb_res["lease_remaining_seconds"], 0.0)

    def test_watchdog_auto_stop_abandoned_session(self):
        """Verify that watchdog automatically stops the camera if heartbeat is missed beyond timeout."""
        start_camera_service(
            host="127.0.0.1",
            port=self.server_port,
            token=self.TEST_TOKEN,
            session_type="attendance",
        )
        self.assertTrue(self.mock_service_state["active"])

        # Wait for timeout (configured to 2.0s in setUpClass) + grace
        time.sleep(2.6)

        # Watchdog should have auto-stopped the camera service
        self.assertFalse(self.mock_service_state["active"])
        self.assertFalse(self.app.session_state.is_active)


class TestCameraSessionContextManager(unittest.TestCase):
    """Test 6: CameraSession context manager lifecycle and stream waiting."""

    def test_stream_wait_probing(self):
        """Verify wait_for_stream_availability detects when a TCP socket becomes open."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        test_port = server_sock.getsockname()[1]
        server_sock.listen(1)

        try:
            # Stream port should be immediately detected as available
            ready = wait_for_stream_availability("127.0.0.1", test_port, timeout=1.0)
            self.assertTrue(ready)
        finally:
            server_sock.close()

        # Once closed, should report false when timeout expires
        ready_closed = wait_for_stream_availability("127.0.0.1", test_port, timeout=0.3, poll_interval=0.05)
        self.assertFalse(ready_closed)

    def test_camera_session_lifecycle(self):
        """Verify CameraSession enters, runs heartbeat, and cleanly exits stopping the camera."""
        mock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        mock_server.bind(("127.0.0.1", 0))
        mock_stream_port = mock_server.getsockname()[1]
        mock_server.listen(5)

        try:
            with patch("edge.camera_control.start_camera_service") as mock_start, \
                 patch("edge.camera_control.stop_camera_service") as mock_stop, \
                 patch("edge.camera_control.send_heartbeat") as mock_hb:

                mock_start.return_value = {"status": "started", "active": True}
                mock_stop.return_value = {"status": "stopped", "active": False}

                session = CameraSession(
                    host="127.0.0.1",
                    control_port=5002,
                    stream_port=mock_stream_port,
                    token="dummy-token",
                    session_type="attendance",
                    heartbeat_interval=0.1,  # fast heartbeat for test
                    wait_stream_timeout=1.0,
                )

                with session:
                    self.assertTrue(session._active)
                    mock_start.assert_called_once()
                    time.sleep(0.35)  # allow a few heartbeats
                    self.assertGreaterEqual(mock_hb.call_count, 1)

                # Exiting context should stop camera
                mock_stop.assert_called_once()
                self.assertFalse(session._active)
        finally:
            mock_server.close()

    def test_local_camera_mode_is_passthrough(self):
        """Verify CameraSession with is_local_camera=True performs no remote network calls."""
        with patch("edge.camera_control.start_camera_service") as mock_start, \
             patch("edge.camera_control.stop_camera_service") as mock_stop:

            with CameraSession(is_local_camera=True, session_type="enrollment") as sess:
                self.assertTrue(sess._active)
                mock_start.assert_not_called()

            mock_stop.assert_not_called()
            self.assertFalse(sess._active)

    def test_heartbeat_starts_immediately_before_stream_ready(self):
        """Verify heartbeat thread is running while waiting for stream availability."""
        heartbeat_started_during_stream_wait = False

        def mock_wait_for_stream(*args, **kwargs):
            nonlocal heartbeat_started_during_stream_wait
            # Check that heartbeat thread is active right now while wait is executing
            if session._heartbeat_thread and session._heartbeat_thread.is_alive():
                heartbeat_started_during_stream_wait = True
            return True

        with patch("edge.camera_control.start_camera_service") as mock_start, \
             patch("edge.camera_control.stop_camera_service"), \
             patch("edge.camera_control.wait_for_stream_availability", side_effect=mock_wait_for_stream):

            mock_start.return_value = {"status": "started", "active": True}
            session = CameraSession(
                host="127.0.0.1",
                session_type="demo",
                heartbeat_interval=0.1,
            )
            session.start()
            self.assertTrue(heartbeat_started_during_stream_wait)
            session.stop()

    def test_heartbeat_no_duplicate_threads(self):
        """Verify repeated calls to _start_heartbeat do not spawn duplicate threads."""
        session = CameraSession(host="127.0.0.1", session_type="demo", heartbeat_interval=1.0)
        session._active = True
        session._start_heartbeat()
        t1 = session._heartbeat_thread
        self.assertIsNotNone(t1)
        self.assertTrue(t1.is_alive())

        # Second call
        session._start_heartbeat()
        t2 = session._heartbeat_thread
        self.assertIs(t1, t2)
        session.stop()

    def test_heartbeat_stops_on_404(self):
        """Verify heartbeat thread shuts down cleanly when server returns HTTP 404."""
        with patch("edge.camera_control.send_heartbeat") as mock_hb:
            mock_hb.side_effect = CameraControlError("HTTP 404: No active camera session to heartbeat")
            session = CameraSession(
                host="127.0.0.1",
                session_type="demo",
                heartbeat_interval=0.05,
            )
            session._active = True
            session._start_heartbeat()
            time.sleep(0.2)
            # Thread should have detected 404 and terminated
            self.assertFalse(session._active)
            if session._heartbeat_thread:
                self.assertFalse(session._heartbeat_thread.is_alive())
            session.stop()

    def test_stream_failure_stops_heartbeat_and_camera(self):
        """Verify stream availability failure triggers stop(force=True) and cancels heartbeat."""
        with patch("edge.camera_control.start_camera_service") as mock_start, \
             patch("edge.camera_control.stop_camera_service") as mock_stop, \
             patch("edge.camera_control.wait_for_stream_availability", return_value=False):

            mock_start.return_value = {"status": "started", "active": True}
            session = CameraSession(
                host="127.0.0.1",
                session_type="demo",
                wait_stream_timeout=0.1,
            )
            with self.assertRaises(CameraStreamUnavailableError):
                session.start()

            mock_stop.assert_called_with(
                host="127.0.0.1",
                port=session.control_port,
                token=session.token,
                session_type="demo",
                client_id=socket.gethostname(),
                force=True,
            )
            self.assertFalse(session._active)
            self.assertIsNone(session._heartbeat_thread)


class TestAttendanceAndEnrollmentLifecycle(unittest.TestCase):
    """Test 7-8: Enrollment and Attendance camera lifecycle integration."""

    def test_enrollment_conflict_handling(self):
        """Verify enroll_face handles CameraInUseError with clear error reporting."""
        from edge.enroll_face import run_enrollment

        with patch("edge.enroll_face.parse_camera_source", return_value=("tcp", "127.0.0.1", 5001)), \
             patch("edge.camera_control.CameraSession.start") as mock_start:

            # Simulate another session currently holding the camera lock
            mock_start.side_effect = CameraInUseError("Camera currently in use by 'attendance'")

            res = run_enrollment(
                student_id="TEST001",
                name="Test Student",
                source_override="127.0.0.1:5001",
            )
            # Should gracefully return False without crashing
            self.assertFalse(res)

    def test_backend_camera_endpoints(self):
        """Verify backend /v1/camera/session start, status, and stop endpoints."""
        from fastapi.testclient import TestClient
        from legacy.backend.main import app
        from legacy.backend.auth import get_current_user

        client = TestClient(app)
        app.dependency_overrides[get_current_user] = lambda: {
            "id": 1,
            "username": "admin1",
            "role": "admin",
            "active": 1,
        }

        try:
            with patch("edge.camera_control.CameraSession.start") as mock_start, \
                 patch("edge.camera_control.CameraSession.stop") as mock_stop, \
                 patch("edge.camera_control.get_camera_status") as mock_get_status:

                mock_get_status.return_value = {"service_active": False, "in_use": False}

                # 1. Query status initially
                status_resp = client.get("/v1/camera/session/status")
                self.assertEqual(status_resp.status_code, 200)
                self.assertFalse(status_resp.json()["backend_session_active"])

                # 2. Start camera session
                start_resp = client.post("/v1/camera/session/start")
                self.assertEqual(start_resp.status_code, 200)
                self.assertEqual(start_resp.json()["status"], "started")
                mock_start.assert_called_once()

                # 3. Stop camera session
                stop_resp = client.post("/v1/camera/session/stop")
                self.assertEqual(stop_resp.status_code, 200)
                self.assertEqual(stop_resp.json()["status"], "stopped")
                mock_stop.assert_called_once()
        finally:
            app.dependency_overrides.pop(get_current_user, None)


if __name__ == "__main__":
    unittest.main()
