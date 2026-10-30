from __future__ import annotations

import socket
import struct
import threading
import time
import unittest
import cv2
import numpy as np

from edge.frame_receiver import FrameReceiver
from edge.camera_stream import is_valid_frame


class TestCameraStreamAndReceiver(unittest.TestCase):
    def test_startup_dark_frame_validation(self):
        """Verify startup dark/black frames are rejected and valid frames accepted."""
        # Empty/None checks
        self.assertFalse(is_valid_frame(None))
        self.assertFalse(is_valid_frame(np.empty((0, 0, 3), dtype=np.uint8)))

        # Dark startup frame typical of Logitech C270 (mean <= 2.5, min >= 1, max <= 3)
        dark_frame = np.full((720, 1280, 3), 2, dtype=np.uint8)
        self.assertFalse(is_valid_frame(dark_frame))

        # Normal classroom/face frame
        normal_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        self.assertTrue(is_valid_frame(normal_frame))

    def test_receiver_single_slot_latest_frame(self):
        """Verify TCP frame reception, decoding, and single-slot latest frame behavior."""
        # Setup a local test server socket
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        stop_server = threading.Event()

        def _mock_sender():
            conn, _ = server_sock.accept()
            try:
                # Send 5 test frames with increasing brightness markers
                for val in [50, 75, 100, 150, 200]:
                    if stop_server.is_set():
                        break
                    frame = np.full((720, 1280, 3), val, dtype=np.uint8)
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        continue
                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))
                    conn.sendall(header + payload)
                    time.sleep(0.02)
            finally:
                conn.close()

        server_thread = threading.Thread(target=_mock_sender, daemon=True)
        server_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        try:
            # Read first frame
            ok, frame = receiver.read(timeout=2.0)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (720, 1280, 3))

            # Wait for all frames to arrive at server
            time.sleep(0.2)

            # Read latest frame: should have updated to the newest frame (brightness around 200)
            ok, latest = receiver.read(timeout=1.0)
            self.assertTrue(ok)
            self.assertIsNotNone(latest)
            # JPEG lossy compression might slightly adjust pixel values, so check range
            self.assertGreater(float(latest.mean()), 180.0)

        finally:
            stop_server.set()
            receiver.stop()
            server_sock.close()
            server_thread.join(timeout=1.0)

    def test_new_frame_event_and_no_duplicate_rendering(self):
        """Verify read() waits for a genuinely new frame and does not return duplicates."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        conn_holder = []

        def _accept():
            c, _ = server_sock.accept()
            conn_holder.append(c)

        acc_thread = threading.Thread(target=_accept, daemon=True)
        acc_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        def _send_frame(conn, brightness):
            f = np.full((120, 160, 3), brightness, dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", f)
            assert ok
            p = enc.tobytes()
            conn.sendall(struct.pack("!I", len(p)) + p)

        try:
            acc_thread.join(timeout=2.0)
            self.assertEqual(len(conn_holder), 1)
            conn = conn_holder[0]

            # 1. Send first frame (brightness 70)
            _send_frame(conn, 70)
            ok1, f1 = receiver.read(timeout=1.5)
            self.assertTrue(ok1)
            self.assertIsNotNone(f1)
            self.assertAlmostEqual(float(f1.mean()), 70.0, delta=5.0)

            # 2. Immediate second read without new frame arriving MUST return False (no duplicate)
            ok_dup, f_dup = receiver.read(timeout=0.05)
            self.assertFalse(ok_dup)
            self.assertIsNone(f_dup)

            # 3. Send second frame (brightness 170)
            _send_frame(conn, 170)
            ok2, f2 = receiver.read(timeout=1.5)
            self.assertTrue(ok2)
            self.assertIsNotNone(f2)
            self.assertAlmostEqual(float(f2.mean()), 170.0, delta=5.0)

            # 4. Immediate third read without new frame again returns False
            ok_dup2, f_dup2 = receiver.read(timeout=0.05)
            self.assertFalse(ok_dup2)
            self.assertIsNone(f_dup2)

        finally:
            if conn_holder:
                conn_holder[0].close()
            receiver.stop()
            server_sock.close()

    def test_backlog_stale_frame_dropping(self):
        """Verify receiver drops stale queued frames from backlog and only decodes newest."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        conn_holder = []

        def _accept():
            c, _ = server_sock.accept()
            conn_holder.append(c)

        acc_thread = threading.Thread(target=_accept, daemon=True)
        acc_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        def _send_frame(conn, brightness):
            f = np.full((120, 160, 3), brightness, dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", f)
            assert ok
            p = enc.tobytes()
            conn.sendall(struct.pack("!I", len(p)) + p)

        try:
            acc_thread.join(timeout=2.0)
            self.assertEqual(len(conn_holder), 1)
            conn = conn_holder[0]

            # Send 5 frames simultaneously in single atomic write without client reading
            payloads = []
            for val in [30, 60, 90, 120, 210]:
                f = np.full((120, 160, 3), val, dtype=np.uint8)
                ok, enc = cv2.imencode(".jpg", f)
                assert ok
                p = enc.tobytes()
                payloads.append(struct.pack("!I", len(p)) + p)
            conn.sendall(b"".join(payloads))

            # Wait briefly for all frames to be buffered in OS socket
            time.sleep(0.08)

            # Client reads: should get the latest frame (brightness ~210)
            ok, frame = receiver.read(timeout=1.5)
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            self.assertGreater(float(frame.mean()), 190.0)

            stats = receiver.get_stats()
            # Stale frames should have been dropped and total decoded frames is less than received
            self.assertGreaterEqual(stats["dropped_frames"], 1)
            self.assertLess(stats["decoded_frames"], stats["received_frames"])

        finally:
            if conn_holder:
                conn_holder[0].close()
            receiver.stop()
            server_sock.close()

    def test_partial_tcp_payload_safety(self):
        """Verify receiver safely handles fragmented/chunked TCP delivery without frame corruption."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        conn_holder = []

        def _accept():
            c, _ = server_sock.accept()
            conn_holder.append(c)

        acc_thread = threading.Thread(target=_accept, daemon=True)
        acc_thread.start()

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        try:
            acc_thread.join(timeout=2.0)
            self.assertEqual(len(conn_holder), 1)
            conn = conn_holder[0]

            f = np.full((120, 160, 3), 140, dtype=np.uint8)
            ok, enc = cv2.imencode(".jpg", f)
            self.assertTrue(ok)
            p = enc.tobytes()
            header = struct.pack("!I", len(p))

            # Send header in two separate 2-byte chunks
            conn.sendall(header[:2])
            time.sleep(0.01)
            conn.sendall(header[2:])
            time.sleep(0.01)

            # Send payload in small 128-byte chunks
            chunk_size = 128
            for i in range(0, len(p), chunk_size):
                conn.sendall(p[i : i + chunk_size])
                time.sleep(0.002)

            ok_read, decoded_frame = receiver.read(timeout=2.0)
            self.assertTrue(ok_read)
            self.assertIsNotNone(decoded_frame)
            self.assertEqual(decoded_frame.shape, (120, 160, 3))
            self.assertAlmostEqual(float(decoded_frame.mean()), 140.0, delta=5.0)

        finally:
            if conn_holder:
                conn_holder[0].close()
            receiver.stop()
            server_sock.close()

    def test_clean_shutdown_unblocks_waiting_read(self):
        """Verify calling stop() cleanly unblocks a thread waiting in read()."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        test_port = server_sock.getsockname()[1]

        receiver = FrameReceiver(host="127.0.0.1", port=test_port)
        receiver.start()

        read_result = []

        def _reader():
            ok, f = receiver.read(timeout=5.0)
            read_result.append((ok, f))

        r_thread = threading.Thread(target=_reader, daemon=True)
        r_thread.start()

        # Give receiver a moment to enter wait
        time.sleep(0.1)
        t0 = time.monotonic()
        receiver.stop()
        r_thread.join(timeout=1.0)
        t_elapsed = time.monotonic() - t0

        server_sock.close()

        self.assertFalse(r_thread.is_alive())
        self.assertLess(t_elapsed, 1.0)
        self.assertEqual(len(read_result), 1)
        self.assertFalse(read_result[0][0])
        self.assertIsNone(read_result[0][1])


if __name__ == "__main__":
    unittest.main()
