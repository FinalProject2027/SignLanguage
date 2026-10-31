
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

project_root = Path(__file__).resolve().parent.parent
video_path = (
    project_root
    / "data"
    / "raw"
    / "train"
    / "raw_videos"
    / "--7E2sU6zP4_10-5-rgb_front.mp4"
)
model_path = project_root / "models" / "holistic_landmarker.task"

if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

try:
    success, frame = cap.read()
    if not success:
        raise RuntimeError("Could not read the first video frame.")

    # OpenCV reads BGR images; MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    with vision.HolisticLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    print("Video opened successfully.")
    print(f"Frame dimensions: {frame.shape[1]} x {frame.shape[0]}")
    print(f"Face landmarks: {len(result.face_landmarks)}")
    print(f"Pose landmarks: {len(result.pose_landmarks)}")
    print(f"Left-hand landmarks: {len(result.left_hand_landmarks)}")
    print(f"Right-hand landmarks: {len(result.right_hand_landmarks)}")

finally:
    cap.release()
