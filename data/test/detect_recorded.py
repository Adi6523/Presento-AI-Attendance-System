import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import face_recognition

from edge.vision import assess_face_quality


INPUT = Path("data/test/stream_30s.avi")
OUTPUT = Path("data/test/stream_30s_faces.avi")

SAMPLE_EVERY = 5


def main():
    cap = cv2.VideoCapture(str(INPUT))

    if not cap.isOpened():
        print("ERROR: Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not open output video.")
        cap.release()
        return

    print("Input:", INPUT)
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    print(f"Total frames: {total}")
    print(f"Running HOG detection every {SAMPLE_EVERY} frames...")

    frame_number = 0
    processed = 0
    frames_with_faces = 0
    total_faces = 0

    last_locations = []

    start = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if frame_number % SAMPLE_EVERY == 0:
            processed += 1

            rgb = frame[:, :, ::-1]

            last_locations = face_recognition.face_locations(
                rgb,
                model="hog"
            )

            if last_locations:
                frames_with_faces += 1
                total_faces += len(last_locations)

        for location in last_locations:

            top, right, bottom, left = location

            quality = assess_face_quality(
                frame,
                location
            )

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Quality: {quality.score:.2f}",
                (left, max(25, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        frame_number += 1

    cap.release()
    writer.release()

    elapsed = time.time() - start

    detection_percentage = (
        frames_with_faces / processed * 100
        if processed
        else 0
    )

    average_faces = (
        total_faces / frames_with_faces
        if frames_with_faces
        else 0
    )

    print()
    print("===== FACE DETECTION RESULT =====")
    print(f"Video frames:               {frame_number}")
    print(f"Frames processed:           {processed}")
    print(f"Frames with faces:          {frames_with_faces}")
    print(f"Total face detections:      {total_faces}")
    print(f"Detection percentage:       {detection_percentage:.2f}%")
    print(f"Average faces/detected:     {average_faces:.2f}")
    print(f"Processing time:            {elapsed:.2f}s")
    print(f"Output video:               {OUTPUT}")
    print(f"Output size:                {OUTPUT.stat().st_size} bytes")
    print("=================================")


if __name__ == "__main__":
    main()