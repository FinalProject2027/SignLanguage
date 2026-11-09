import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

if __package__:
    from .extract_landmarks import (
        FACE_INDICES,
        POSE_INDICES,
        get_points,
        normalize_landmarks,
    )
else:
    from extract_landmarks import (
        FACE_INDICES,
        POSE_INDICES,
        get_points,
        normalize_landmarks,
    )

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "holistic_landmarker.task"

DEFAULT_BUFFER_SIZE = 60
MAX_BUFFER_SIZE = 90
WINDOW_NAME = "Live Sign Language Camera"


@dataclass(frozen=True)
class LandmarkFrame:
    left_hand: np.ndarray
    right_hand: np.ndarray
    pose: np.ndarray
    face: np.ndarray


class LandmarkBuffer:
    def __init__(self, capacity=DEFAULT_BUFFER_SIZE):
        if not 1 <= capacity <= MAX_BUFFER_SIZE:
            raise ValueError(
                f"Buffer size must be between 1 and {MAX_BUFFER_SIZE}."
            )
        self.frames = deque(maxlen=capacity)

    def append(self, frame):
        self.frames.append(frame)

    def __len__(self):
        return len(self.frames)

    def arrays(self):
        if not self.frames:
            raise RuntimeError("The landmark buffer is empty.")
        return {
            "left_hand": np.stack([frame.left_hand for frame in self.frames]),
            "right_hand": np.stack([frame.right_hand for frame in self.frames]),
            "pose": np.stack([frame.pose for frame in self.frames]),
            "face": np.stack([frame.face for frame in self.frames]),
        }


def extract_frame_landmarks(frame, detector, rgb_frame=None):
    if rgb_frame is None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )
    result = detector.detect(mp_image)

    if result.pose_landmarks:
        pose_raw = get_points(result.pose_landmarks, POSE_INDICES)
        left_shoulder = pose_raw[11]
        right_shoulder = pose_raw[12]
        center = (left_shoulder + right_shoulder) / 2.0
        scale = max(float(np.linalg.norm(left_shoulder - right_shoulder)), 1e-6)
        pose_points = normalize_landmarks(pose_raw, center, scale)

        if result.left_hand_landmarks:
            left_points = normalize_landmarks(
                get_points(result.left_hand_landmarks),
                center,
                scale,
            )
        else:
            left_points = np.zeros((21, 3), dtype=np.float32)

        if result.right_hand_landmarks:
            right_points = normalize_landmarks(
                get_points(result.right_hand_landmarks),
                center,
                scale,
            )
        else:
            right_points = np.zeros((21, 3), dtype=np.float32)

        if result.face_landmarks:
            face_points = normalize_landmarks(
                get_points(result.face_landmarks, FACE_INDICES),
                center,
                scale,
            )
        else:
            face_points = np.zeros((40, 3), dtype=np.float32)
    else:
        pose_points = np.zeros((25, 3), dtype=np.float32)
        left_points = np.zeros((21, 3), dtype=np.float32)
        right_points = np.zeros((21, 3), dtype=np.float32)
        face_points = np.zeros((40, 3), dtype=np.float32)

    return LandmarkFrame(
        left_hand=left_points,
        right_hand=right_points,
        pose=pose_points,
        face=face_points,
    )


def run_camera(camera_index=0, buffer_size=DEFAULT_BUFFER_SIZE):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

    landmark_buffer = LandmarkBuffer(buffer_size)
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )
    capture = None

    try:
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            capture = cv2.VideoCapture(camera_index)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open webcam at camera index {camera_index}."
                )

            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                landmark_buffer.append(extract_frame_landmarks(frame, detector))

                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Buffered frames: {len(landmark_buffer)}/{buffer_size}  |  Q/Esc to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()

    return landmark_buffer


def main():
    parser = argparse.ArgumentParser(
        description="Process webcam frames into a sliding landmark window."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE,
        help=f"Sliding-window capacity from 1 to {MAX_BUFFER_SIZE} frames.",
    )
    args = parser.parse_args()

    run_camera(args.camera_index, args.buffer_size)


if __name__ == "__main__":
    main()
