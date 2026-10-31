
from pathlib import Path
import re
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw" / "train" / "raw_videos"
CSV_PATH = ROOT / "data" / "raw" / "train" / "how2sign_train.csv"
MODEL_PATH = ROOT / "models" / "holistic_landmarker.task"
OUTPUT_DIR = ROOT / "data" / "processed" / "train"

SAMPLE_FPS = 5.0
MAX_VIDEOS = None 

# Proposed 40-point face subset: 20 lips, 8 eyebrows, 12 face contour.
FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    55, 65, 105, 107, 285, 295, 334, 336,
    10, 338, 297, 332, 284, 251, 454, 323, 361, 152, 172, 234,
]

# Blueprint specifies pose landmarks 0 through 24.
POSE_INDICES = list(range(25))


def normalize_landmarks(points, center, scale):
    """Center and scale an (N, 3) landmark array."""
    return ((points - center) / scale).astype(np.float32)


def get_points(landmarks, indices=None):
    """Convert MediaPipe landmarks to an (N, 3) float32 array."""
    if not landmarks:
        count = len(indices) if indices is not None else 21
        return np.zeros((count, 3), dtype=np.float32)

    if indices is None:
        selected = landmarks
    else:
        selected = [landmarks[i] for i in indices]

    return np.array(
        [[point.x, point.y, point.z] for point in selected],
        dtype=np.float32,
    )


def segment_id_from_clip(video_path):
    """
    Clip: --7E2sU6zP4_10-5-rgb_front.mp4
    Segment ID: --7E2sU6zP4_10
    """
    match = re.match(r"^(.*_\d+)-\d+-rgb_front$", video_path.stem)
    if not match:
        raise ValueError(f"Cannot derive segment ID from {video_path.name}")
    return match.group(1)


def process_video(video_path, detector, annotation_by_id):
    segment_id = segment_id_from_clip(video_path)

    if segment_id not in annotation_by_id:
        raise ValueError(f"No annotation for segment ID: {segment_id}")

    annotation = annotation_by_id[segment_id]
    sentence = str(annotation["SENTENCE"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0.0
    sample_every = max(1, round(fps / SAMPLE_FPS))

    left_frames = []
    right_frames = []
    pose_frames = []
    face_frames = []
    timestamps = []

    frame_index = 0
    processed = 0
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % sample_every == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )
                result = detector.detect(image)

                # Pose landmarks 11 and 12 are shoulders.
                if len(result.pose_landmarks) > 0:
                    pose_raw = get_points(
                        result.pose_landmarks, POSE_INDICES
                    )
                    left_shoulder = np.array(
                        [
                            result.pose_landmarks[11].x,
                            result.pose_landmarks[11].y,
                            result.pose_landmarks[11].z,
                        ],
                        dtype=np.float32,
                    )
                    right_shoulder = np.array(
                        [
                            result.pose_landmarks[12].x,
                            result.pose_landmarks[12].y,
                            result.pose_landmarks[12].z,
                        ],
                        dtype=np.float32,
                    )
                    center = (left_shoulder + right_shoulder) / 2.0
                    scale = float(np.linalg.norm(left_shoulder - right_shoulder))
                    scale = max(scale, 1e-6)

                   
                    pose_points = normalize_landmarks(pose_raw, center, scale)

# Normalize hand landmarks only when detected.
                    if result.left_hand_landmarks:
                            left_points = get_points(result.left_hand_landmarks)
                            left_points = normalize_landmarks(left_points, center, scale)
                    else:
                            left_points = np.zeros((21, 3), dtype=np.float32)

                    if result.right_hand_landmarks:
                            right_points = get_points(result.right_hand_landmarks)
                            right_points = normalize_landmarks(right_points, center, scale)
                    else:
                            right_points = np.zeros((21, 3), dtype=np.float32)

# Normalize selected face landmarks only when detected.
                    if result.face_landmarks:
                            face_points = get_points(result.face_landmarks, FACE_INDICES)
                            face_points = normalize_landmarks(face_points, center, scale)
                    else:
                            face_points = np.zeros((40, 3), dtype=np.float32)

                else:
                    pose_points = np.zeros((25, 3), dtype=np.float32)
                    left_points = np.zeros((21, 3), dtype=np.float32)
                    right_points = np.zeros((21, 3), dtype=np.float32)
                    face_points = np.zeros((40, 3), dtype=np.float32)

                left_frames.append(left_points)
                right_frames.append(right_points)
                pose_frames.append(pose_points)
                face_frames.append(face_points)
                timestamps.append(frame_index / fps if fps > 0 else 0.0)
                processed += 1

            frame_index += 1
    finally:
        cap.release()

    elapsed = time.perf_counter() - start_time

    output_path = OUTPUT_DIR / f"{video_path.stem}.npz"
    np.savez_compressed(
        output_path,
        left_hand=np.asarray(left_frames, dtype=np.float32),
        right_hand=np.asarray(right_frames, dtype=np.float32),
        pose=np.asarray(pose_frames, dtype=np.float32),
        face=np.asarray(face_frames, dtype=np.float32),
        text=sentence,
        sentence_id=segment_id,
        timestamps=np.asarray(timestamps, dtype=np.float32),
        frame_count=len(timestamps),
        original_frame_count=total_frames,
        original_fps=fps,
        clip_duration_seconds=duration,
    )

    return {
        "clip": video_path.name,
        "sentence_id": segment_id,
        "sampled_frames": processed,
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "output": str(output_path.relative_to(ROOT)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    annotations = pd.read_csv(CSV_PATH, sep="\t")
    annotation_by_id = {
        str(row["SENTENCE_ID"]).strip(): row
        for _, row in annotations.iterrows()
    }

    video_paths = sorted(VIDEO_DIR.glob("*.mp4"))
    if MAX_VIDEOS is not None:
        video_paths = video_paths[:MAX_VIDEOS]
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found in {VIDEO_DIR}")

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionTaskRunningMode.IMAGE,
    )

    results = []
    with vision.HolisticLandmarker.create_from_options(options) as detector:
        for index, video_path in enumerate(video_paths, start=1):
            try:
                result = process_video(video_path, detector, annotation_by_id)
                results.append(result)

                if index % 100 == 0 or index == len(video_paths):
                    print(f"Progress: {index}/{len(video_paths)} videos processed")

            except Exception as exc:
                print(f"ERROR processing {video_path.name}: {exc}")

    print("\nBatch summary:")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
