import time
from pathlib import Path
import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edge.frame_receiver import FrameReceiver


HOST = "192.168.1.15"
PORT = 5001
DURATION = 30


def main():
    receiver = FrameReceiver(
        host=HOST,
        port=PORT
    )

    receiver.start()

    print("Connecting to Raspberry Pi...")

    start_wait = time.time()

    while True:
        ok, frame = receiver.read(timeout=1.0)

        if ok and frame is not None:
            break

        if time.time() - start_wait > 10:
            print("ERROR: No frames received for 10 seconds.")
            receiver.stop()
            return

    height, width = frame.shape[:2]

    output = Path("data/test/stream_30s.avi")

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (width, height)
    )

    print("Writer opened:", writer.isOpened())
    print("Recording stream for 30 seconds...")

    start = time.time()
    frames = 0

    while time.time() - start < DURATION:
        ok, frame = receiver.read(timeout=1.0)

        if not ok or frame is None:
            continue

        writer.write(frame)
        frames += 1

    writer.release()
    receiver.stop()

    print()
    print("===== STREAM TEST =====")
    print("Frames recorded:", frames)
    print("Video:", output)
    print("Video size:", output.stat().st_size)
    print("=======================")


if __name__ == "__main__":
    main()