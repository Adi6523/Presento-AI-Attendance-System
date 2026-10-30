from __future__ import annotations

import io
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path when running this test file individually
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

from edge.config import settings
from edge.enroll_face import open_camera_source, parse_camera_source
from edge.frame_receiver import FrameReceiver


class TestEnrollFaceSource(unittest.TestCase):
    """
    Unit test suite for Phase 2B face enrollment camera source handling:
    - numeric source -> OpenCV camera path
    - HOST:PORT source -> FrameReceiver path
    - source parsing
    - clean shutdown
    - invalid HOST:PORT handling
    """

    # -------------------------------------------------------------------------
    # 1. Source Parsing Tests
    # -------------------------------------------------------------------------
    def test_parse_numeric_sources(self):
        """Verify numeric source formats map to OpenCV camera indices."""
        self.assertEqual(parse_camera_source(0), ("opencv", 0))
        self.assertEqual(parse_camera_source(1), ("opencv", 1))
        self.assertEqual(parse_camera_source("0"), ("opencv", 0))
        self.assertEqual(parse_camera_source("1"), ("opencv", 1))
        self.assertEqual(parse_camera_source("  2  "), ("opencv", 2))

    def test_parse_named_stream_keywords(self):
        """Verify named stream keywords map to TCP stream using configured default host/port."""
        expected = ("tcp", settings.pi_ip, settings.stream_port)
        self.assertEqual(parse_camera_source("stream"), expected)
        self.assertEqual(parse_camera_source("STREAM"), expected)
        self.assertEqual(parse_camera_source("pi"), expected)
        self.assertEqual(parse_camera_source("network"), expected)

    def test_parse_none_source(self):
        """Verify None source falls back to settings.camera_source."""
        parsed = parse_camera_source(None)
        if settings.camera_source.lower() in ("stream", "pi", "network"):
            self.assertEqual(parsed, ("tcp", settings.pi_ip, settings.stream_port))
        elif settings.camera_source.isdigit():
            self.assertEqual(parsed, ("opencv", int(settings.camera_source)))
        else:
            self.assertEqual(parsed[0], "tcp" if ":" in settings.camera_source else "opencv")

    def test_parse_host_port_sources(self):
        """Verify HOST:PORT formats are correctly extracted."""
        self.assertEqual(
            parse_camera_source("10.0.0.50:5001"),
            ("tcp", "10.0.0.50", 5001),
        )
        self.assertEqual(
            parse_camera_source("localhost:8080"),
            ("tcp", "localhost", 8080),
        )
        self.assertEqual(
            parse_camera_source("10.0.0.51:5001"),
            ("tcp", "10.0.0.51", 5001),
        )
        self.assertEqual(
            parse_camera_source("  10.0.0.5 : 5002  "),
            ("tcp", "10.0.0.5", 5002),
        )

    def test_parse_file_path_sources(self):
        """Verify file paths with and without Windows drive letters are treated as OpenCV sources."""
        self.assertEqual(parse_camera_source("sample.mp4"), ("opencv", "sample.mp4"))
        self.assertEqual(parse_camera_source(r"C:\videos\enrollment.mp4"), ("opencv", r"C:\videos\enrollment.mp4"))
        self.assertEqual(parse_camera_source("D:/videos/enrollment.mp4"), ("opencv", "D:/videos/enrollment.mp4"))

    # -------------------------------------------------------------------------
    # 2. Invalid HOST:PORT Handling Tests
    # -------------------------------------------------------------------------
    def test_invalid_host_port_syntax_raises_value_error(self):
        """Verify invalid HOST:PORT syntaxes raise ValueError with informative message."""
        # Non-integer port
        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:abc")
        self.assertIn("port must be an integer", str(ctx.exception))

        # Port out of valid range (1 - 65535)
        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:99999")
        self.assertIn("out of range", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:0")
        self.assertIn("out of range", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:-1")
        self.assertIn("out of range", str(ctx.exception))

        # Empty host
        with self.assertRaises(ValueError) as ctx:
            parse_camera_source(":5001")
        self.assertIn("host cannot be empty", str(ctx.exception))

        # Empty port
        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:")
        self.assertIn("port cannot be empty", str(ctx.exception))

        # Multiple colons
        with self.assertRaises(ValueError) as ctx:
            parse_camera_source("10.0.0.50:5001:extra")
        self.assertIn("expected HOST:PORT", str(ctx.exception))

    def test_open_camera_source_with_invalid_syntax_returns_none(self):
        """Verify open_camera_source handles invalid syntax gracefully, returning None and logging error."""
        stderr_trap = io.StringIO()
        with patch("sys.stderr", stderr_trap):
            cam = open_camera_source("10.0.0.50:invalid_port")
        self.assertIsNone(cam)
        self.assertIn("[ERROR]", stderr_trap.getvalue())
        self.assertIn("port must be an integer", stderr_trap.getvalue())

    def test_open_camera_source_connection_failure(self):
        """Verify open_camera_source reports clear connection error when TCP target is unreachable."""
        # Use a high unassigned port on localhost that refuses connections
        stderr_trap = io.StringIO()
        with patch("sys.stderr", stderr_trap):
            cam = open_camera_source("127.0.0.1:59998", connect_timeout=0.3)
        self.assertIsNone(cam)
        err_output = stderr_trap.getvalue()
        self.assertIn("[ERROR] Could not connect to Presento camera stream at 127.0.0.1:59998", err_output)

    # -------------------------------------------------------------------------
    # 3. Numeric Source -> OpenCV Camera Path Tests
    # -------------------------------------------------------------------------
    @patch("cv2.VideoCapture")
    def test_numeric_source_opencv_success(self, mock_videocapture_cls):
        """Verify numeric source instantiates OpenCV VideoCapture and returns it when opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_videocapture_cls.return_value = mock_cap

        cam = open_camera_source("0")
        self.assertIs(cam, mock_cap)
        mock_videocapture_cls.assert_called_once_with(0)

    @patch("cv2.VideoCapture")
    def test_numeric_source_opencv_failure(self, mock_videocapture_cls):
        """Verify when OpenCV camera cannot be opened, returns None and outputs clear error."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_videocapture_cls.return_value = mock_cap

        stderr_trap = io.StringIO()
        with patch("sys.stderr", stderr_trap):
            cam = open_camera_source(1)
        self.assertIsNone(cam)
        self.assertIn("[ERROR] Camera source '1' could not be opened.", stderr_trap.getvalue())

    # -------------------------------------------------------------------------
    # 4. HOST:PORT Source -> FrameReceiver Path Tests
    # -------------------------------------------------------------------------
    def test_host_port_frame_receiver_success(self):
        """Verify HOST:PORT connects to TCP stream and delivers valid BGR frames."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        stop_server = threading.Event()
        dummy_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

        def _mock_server():
            conn, _ = server_sock.accept()
            try:
                ok, encoded = cv2.imencode(".jpg", dummy_frame)
                if ok:
                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))
                    while not stop_server.is_set():
                        try:
                            conn.sendall(header + payload)
                            time.sleep(0.03)
                        except OSError:
                            break
            finally:
                conn.close()

        server_thread = threading.Thread(target=_mock_server, daemon=True)
        server_thread.start()

        try:
            cam = open_camera_source(f"127.0.0.1:{test_port}", connect_timeout=2.0)
            self.assertIsNotNone(cam)
            self.assertIsInstance(cam, FrameReceiver)
            self.assertTrue(cam.isOpened())

            # Verify frame read
            ok, frame = cam.read(timeout=1.0)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertEqual(frame.dtype, np.uint8)

        finally:
            stop_server.set()
            if cam is not None:
                cam.release()
            server_sock.close()
            server_thread.join(timeout=1.0)

    # -------------------------------------------------------------------------
    # 5. Clean Shutdown Tests
    # -------------------------------------------------------------------------
    def test_clean_shutdown(self):
        """Verify calling camera.release() on FrameReceiver stops thread and closes socket cleanly."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        stop_server = threading.Event()
        dummy_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

        def _mock_server():
            conn, _ = server_sock.accept()
            try:
                ok, encoded = cv2.imencode(".jpg", dummy_frame)
                if ok:
                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))
                    while not stop_server.is_set():
                        try:
                            conn.sendall(header + payload)
                            time.sleep(0.03)
                        except OSError:
                            break
            finally:
                conn.close()

        server_thread = threading.Thread(target=_mock_server, daemon=True)
        server_thread.start()

        try:
            cam = open_camera_source(f"127.0.0.1:{test_port}", connect_timeout=2.0)
            self.assertIsNotNone(cam)
            self.assertTrue(cam.isOpened())

            # Execute release (clean shutdown)
            cam.release()

            # Verify state
            self.assertFalse(cam.isOpened())
            self.assertFalse(cam._running)
            self.assertFalse(cam._connected)
            self.assertIsNone(cam._sock)
            if cam._thread is not None:
                self.assertFalse(cam._thread.is_alive())

        finally:
            stop_server.set()
            server_sock.close()
            server_thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
