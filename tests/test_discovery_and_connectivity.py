from __future__ import annotations

from pathlib import Path
import socket
import struct
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

# Ensure project root is in sys.path when running tests individually
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from edge.discovery import (
    DEFAULT_MDNS_HOSTNAME,
    DEFAULT_NETBIOS_HOSTNAME,
    HostCandidateRotator,
    build_candidate_hosts,
    discover_pi_stream,
    is_ip_address,
    probe_stream,
    resolve_hostname,
)
from edge.frame_receiver import FrameReceiver


class TestDiscoveryAndConnectivity(unittest.TestCase):
    """
    Unit test suite verifying Presento Pi <-> Windows connectivity and discovery:
    - configured IP available
    - hostname available
    - Pi temporarily unavailable
    - reconnect after disconnect
    - invalid/unreachable address
    - no Pi found
    - connection recovery
    """

    # -------------------------------------------------------------------------
    # 1. Candidate Generation & Resolution Tests
    # -------------------------------------------------------------------------
    def test_configured_ip_available_as_primary(self):
        """Verify that an explicitly configured IP address is prioritized first in the candidate list."""
        cands = build_candidate_hosts("10.0.0.50")
        self.assertEqual(cands[0], "10.0.0.50")
        self.assertIn("presento-pi.local", cands)
        self.assertIn("presento-pi", cands)

    def test_hostname_available_as_primary_when_no_ip_configured(self):
        """Verify that when no explicit IP is provided, mDNS hostname is prioritized."""
        cands = build_candidate_hosts(None)
        self.assertEqual(cands[0], DEFAULT_MDNS_HOSTNAME)
        self.assertEqual(cands[1], DEFAULT_NETBIOS_HOSTNAME)

    def test_is_ip_address(self):
        """Verify IP address string detection."""
        self.assertTrue(is_ip_address("10.0.0.51"))
        self.assertTrue(is_ip_address("10.0.0.50"))
        self.assertTrue(is_ip_address("127.0.0.1"))
        self.assertFalse(is_ip_address("presento-pi.local"))
        self.assertFalse(is_ip_address("presento-pi"))

    def test_resolve_hostname_with_ip_literal(self):
        """Verify IP literals resolve immediately without DNS overhead."""
        self.assertEqual(resolve_hostname("127.0.0.1"), "127.0.0.1")
        self.assertEqual(resolve_hostname("192.168.1.50"), "192.168.1.50")

    @patch("socket.getaddrinfo")
    def test_resolve_hostname_via_mdns(self, mock_getaddrinfo):
        """Verify hostname resolution correctly extracts IPv4 from socket getaddrinfo."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.100", 5001))
        ]
        res = resolve_hostname("presento-pi.local")
        self.assertEqual(res, "192.168.1.100")
        mock_getaddrinfo.assert_called_once()

    @patch("socket.getaddrinfo")
    def test_invalid_unreachable_address_resolution_failure(self, mock_getaddrinfo):
        """Verify unresolvable hostname returns None safely without raising an unhandled exception."""
        mock_getaddrinfo.side_effect = socket.gaierror(11001, "getaddrinfo failed")
        res = resolve_hostname("nonexistent-bogus-pi.local")
        self.assertIsNone(res)

    # -------------------------------------------------------------------------
    # 2. Stream Probing & Discovery Tests
    # -------------------------------------------------------------------------
    def test_probe_stream_success(self):
        """Verify probe_stream reports True when the stream port is open."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        try:
            reachable, resolved_ip, err = probe_stream("127.0.0.1", port=test_port, timeout=1.0)
            self.assertTrue(reachable)
            self.assertEqual(resolved_ip, "127.0.0.1")
            self.assertIsNone(err)
        finally:
            server_sock.close()

    def test_probe_stream_unreachable_port(self):
        """Verify probe_stream reports False with error when port is closed."""
        reachable, resolved_ip, err = probe_stream("127.0.0.1", port=59996, timeout=0.2)
        self.assertFalse(reachable)
        self.assertEqual(resolved_ip, "127.0.0.1")
        self.assertIsNotNone(err)

    def test_discover_pi_stream_finds_active_candidate(self):
        """Verify discover_pi_stream identifies the active stream amongst candidate options."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        try:
            active_host, active_ip = discover_pi_stream(configured_host="127.0.0.1", port=test_port, timeout=0.5)
            self.assertEqual(active_host, "127.0.0.1")
            self.assertEqual(active_ip, "127.0.0.1")
        finally:
            server_sock.close()

    def test_no_pi_found_returns_none(self):
        """Verify discover_pi_stream safely returns (None, None) when no candidate is listening."""
        active_host, active_ip = discover_pi_stream(configured_host="127.0.0.1", port=59995, timeout=0.1)
        self.assertIsNone(active_host)
        self.assertIsNone(active_ip)

    # -------------------------------------------------------------------------
    # 3. Host Candidate Rotator & Backoff Tests
    # -------------------------------------------------------------------------
    def test_host_candidate_rotator_backoff_and_reset(self):
        """Verify rotator steps through backoff delays and resets upon success."""
        rotator = HostCandidateRotator(
            primary_host="127.0.0.1",
            port=5001,
            min_retry_delay=1.0,
            max_retry_delay=4.0,
            backoff_multiplier=2.0,
        )

        cand1 = rotator.get_current_candidate()
        self.assertEqual(cand1, "127.0.0.1")

        # First failure: delay is 1.0s, next becomes 2.0s
        d1 = rotator.record_connection_failure()
        self.assertEqual(d1, 1.0)
        self.assertEqual(rotator.current_delay, 2.0)

        # Second failure: delay is 2.0s, next becomes 4.0s (capped)
        d2 = rotator.record_connection_failure()
        self.assertEqual(d2, 2.0)
        self.assertEqual(rotator.current_delay, 4.0)

        # Third failure: delay capped at max 4.0s
        d3 = rotator.record_connection_failure()
        self.assertEqual(d3, 4.0)
        self.assertEqual(rotator.current_delay, 4.0)

        # Success resets backoff
        rotator.record_connection_success("127.0.0.1", "127.0.0.1")
        self.assertEqual(rotator.current_delay, 1.0)

    # -------------------------------------------------------------------------
    # 4. FrameReceiver Reconnection & Recovery Tests
    # -------------------------------------------------------------------------
    def test_frame_receiver_reconnect_after_disconnect(self):
        """Verify FrameReceiver detects server disconnection and automatically reconnects upon stream recovery."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(2)
        test_port = server_sock.getsockname()[1]

        stop_session = threading.Event()
        dummy_frame = np.full((480, 640, 3), 150, dtype=np.uint8)

        def _mock_server():
            # Connection 1
            try:
                conn1, _ = server_sock.accept()
                ok, encoded = cv2.imencode(".jpg", dummy_frame)
                payload = encoded.tobytes()
                header = struct.pack("!I", len(payload))
                conn1.sendall(header + payload)
                time.sleep(0.05)
                # Abruptly close connection 1 to simulate Pi reboot/restart
                conn1.close()
            except Exception:
                pass

            # Connection 2 (after recovery)
            try:
                conn2, _ = server_sock.accept()
                ok, encoded = cv2.imencode(".jpg", dummy_frame)
                payload = encoded.tobytes()
                header = struct.pack("!I", len(payload))
                while not stop_session.is_set():
                    try:
                        conn2.sendall(header + payload)
                        time.sleep(0.03)
                    except OSError:
                        break
                conn2.close()
            except Exception:
                pass

        server_thread = threading.Thread(target=_mock_server, daemon=True)
        server_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port, reconnect_delay=0.1)
        receiver.start()

        try:
            # 1. Read first frame from connection 1
            ok1, frame1 = receiver.read(timeout=2.0)
            self.assertTrue(ok1)
            self.assertIsNotNone(frame1)

            # 2. Wait for disconnection to be detected
            time.sleep(0.2)

            # 3. Verify automatic reconnection on connection 2
            ok2, frame2 = receiver.read(timeout=2.5)
            self.assertTrue(ok2)
            self.assertIsNotNone(frame2)
            self.assertTrue(receiver.is_connected)

        finally:
            stop_session.set()
            receiver.stop()
            server_sock.close()
            server_thread.join(timeout=1.0)

    def test_frame_receiver_pi_temporarily_unavailable_recovery(self):
        """Verify FrameReceiver retries without crashing when Pi is initially offline and connects once available."""
        # Find an unused local port
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_sock.bind(("127.0.0.1", 0))
        test_port = temp_sock.getsockname()[1]
        temp_sock.close()  # Port is now closed

        # Start receiver pointing to closed port
        receiver = FrameReceiver(host="127.0.0.1", port=test_port, reconnect_delay=0.1)
        receiver.start()

        self.assertFalse(receiver.is_connected)
        # Verify read times out safely without crashing
        ok, frame = receiver.read(timeout=0.2)
        self.assertFalse(ok)
        self.assertIsNone(frame)

        # Now start server on test_port (simulating Pi booting up)
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", test_port))
        server_sock.listen(1)

        stop_session = threading.Event()
        dummy_frame = np.full((480, 640, 3), 200, dtype=np.uint8)

        def _delayed_server():
            conn, _ = server_sock.accept()
            try:
                ok, encoded = cv2.imencode(".jpg", dummy_frame)
                payload = encoded.tobytes()
                header = struct.pack("!I", len(payload))
                while not stop_session.is_set():
                    try:
                        conn.sendall(header + payload)
                        time.sleep(0.03)
                    except OSError:
                        break
            finally:
                conn.close()

        server_thread = threading.Thread(target=_delayed_server, daemon=True)
        server_thread.start()

        try:
            # Wait for FrameReceiver to automatically recover and connect
            ok_rec, frame_rec = receiver.read(timeout=3.0)
            self.assertTrue(ok_rec)
            self.assertIsNotNone(frame_rec)
            self.assertTrue(receiver.is_connected)
        finally:
            stop_session.set()
            receiver.stop()
            server_sock.close()
            server_thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
