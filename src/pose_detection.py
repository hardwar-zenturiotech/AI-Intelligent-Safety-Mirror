"""Stage 1: webcam-based full-body pose detection for the Safety Mirror."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from src.scba_tracking import calculate_scba_anchor
except ImportError:
    from scba_tracking import calculate_scba_anchor

try:
    from src.rtmpose_tracker import RTMPoseTracker
except ImportError:
    from rtmpose_tracker import RTMPoseTracker


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (27, 31), (28, 32),
]


def visible(landmark, threshold: float) -> bool:
    return getattr(landmark, "visibility", 1.0) >= threshold


def draw_pose(
    frame,
    landmarks,
    visibility_threshold: float = 0.5,
) -> None:
    height, width = frame.shape[:2]
    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in landmarks
    ]

    for start, end in CONNECTIONS:
        if visible(landmarks[start], visibility_threshold) and visible(
            landmarks[end], visibility_threshold
        ):
            cv2.line(frame, points[start], points[end], (0, 220, 100), 2)

    for index, landmark in enumerate(landmarks):
        if visible(landmark, visibility_threshold):
            cv2.circle(frame, points[index], 4, (40, 80, 255), -1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--engine",
        type=str,
        choices=["rtmpose", "mediapipe"],
        default="rtmpose",
        help="Pose tracking engine: OpenMMLab RTMPose (SOTA) or MediaPipe (backup)",
    )
    parser.add_argument(
        "--rtmpose-model",
        type=Path,
        default=Path("models/rtmpose-m.onnx"),
        help="Path to RTMPose ONNX model",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_lite.task"),
        help="Path to MediaPipe Pose Landmarker .task model",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("recordings"),
        help="Directory to save MP4 recordings in",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally mirror the camera image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tracker = RTMPoseTracker(
        model_path=args.rtmpose_model,
        backup_model_path=args.model,
        preferred_engine=args.engine,
    )
    print(f"Safety Mirror active tracking engine: {tracker.active_engine}")

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        print(f"Could not open camera device {args.camera}.")
        return 1

    args.record_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = args.record_dir / f"safety_mirror_{timestamp}.mp4"
    writer = None

    previous_frame_time = time.perf_counter()
    ctrl_pressed = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            if writer is None:
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    print(f"Could not create recording file: {output_path}")
                    writer = None

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            result = tracker.detect(frame)

            scba_status = ""
            if result.pose_landmarks:
                for pose in result.pose_landmarks:
                    draw_pose(frame, pose)
                status = "PERSON DETECTED"
                status_colour = (0, 220, 100)
                if result.pose_world_landmarks:
                    anchor = calculate_scba_anchor(result.pose_world_landmarks[0])
                    if anchor is not None:
                        scba_status = "SCBA CYLINDER ANCHOR: ATTACHED"
            else:
                status = "WAITING FOR PERSON"
                status_colour = (0, 180, 255)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_frame_time, 1e-6)
            previous_frame_time = now

            cv2.putText(
                frame, status, (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_colour, 2,
            )
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            if scba_status:
                cv2.putText(
                    frame, scba_status, (20, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2,
                )
            cv2.putText(
                frame, "Press Q or Ctrl+Q to quit", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow("AI Intelligent Safety Mirror | Stage 1", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 17:
                ctrl_pressed = True
                continue

            if ctrl_pressed and key in (ord("q"), ord("Q")):
                break

            ctrl_pressed = False
            if key in (ord("q"), ord("Q")):
                break
    finally:
        tracker.close()
        if writer is not None:
            writer.release()
            print(f"Recording saved to: {output_path}")
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
