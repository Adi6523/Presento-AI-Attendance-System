import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import face_recognition

from edge.frame_receiver import FrameReceiver
from edge.vision import assess_face_quality


HOST = "192.168.1.15"
PORT = 5001
DURATION = 30

OUTPUT = Path("data/test/phase2_face_detection_30s.avi")


def main():
    receiver = FrameReceiver(
        host=HOST,
        port=PORT
    )

    print("Connecting to Raspberry Pi...")
    receiver.start()

    first_frame = None

    while first_frame is None:
        ok, frame = receiver.read(timeout=1.0)

        if ok and frame is not None:
            first_frame = frame
        else:
            time.sleep(0.01)

    height, width = first_frame.shape[:2]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: VideoWriter could not be opened.")
        receiver.stop()
        return

    print("Connected.")
    print(f"Recording {DURATION} seconds...")
    print("Go to the room with the Raspberry Pi.")
    print("Stand in front of the camera for 30 seconds.")

    start = time.time()

    total_frames = 0
    frames_with_faces = 0
    total_faces = 0

    while time.time() - start < DURATION:

        ok, frame = receiver.read(timeout=0.5)

        if not ok or frame is None:
            continue

        total_frames += 1

        rgb = frame[:, :, ::-1]

        locations = face_recognition.face_locations(
            rgb,
            model="hog"
        )

        if locations:
            frames_with_faces += 1
            total_faces += len(locations)

        for location in locations:

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

        elapsed = time.time() - start
        remaining = max(
            0,
            DURATION - elapsed
        )

        cv2.putText(
            frame,
            f"Time: {remaining:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)

    writer.release()
    receiver.stop()

    detection_percentage = (
        frames_with_faces / total_frames * 100
        if total_frames
        else 0
    )

    average_faces = (
        total_faces / frames_with_faces
        if frames_with_faces
        else 0
    )

    print()
    print("===== PHASE 2 RESULT =====")
    print(f"Total frames:               {total_frames}")
    print(f"Frames with faces:          {frames_with_faces}")
    print(f"Total face detections:      {total_faces}")
    print(f"Detection percentage:       {detection_percentage:.2f}%")
    print(f"Average faces/detected:     {average_faces:.2f}")
    print(f"Saved video:                {OUTPUT}")
    print(f"Video size:                 {OUTPUT.stat().st_size} bytes")
    print("===========================")


if __name__ == "__main__":
    main()