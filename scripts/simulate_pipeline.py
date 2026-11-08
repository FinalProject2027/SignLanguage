"""Simulate a webcam with real training clips (no camera, no trained model).

Each clip (about 4.8 samples per second) is stretched into a fake 25 FPS
camera stream, followed by 3 seconds of empty frames (hands lowered). The
stream goes through RealtimePipeline with a dummy model that just waits
50 ms. Time is simulated, so this runs instantly.

Run from the project root:
    python scripts/simulate_pipeline.py
"""
import glob
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend.realtime_pipeline import RealtimePipeline, SAMPLE_PERIOD_S  # noqa: E402

N_CLIPS = 15
CAMERA_FPS = 25.0
TAIL_S = 3.0


def dummy_model(arrays):
    time.sleep(0.05)               # pretend the model takes 50 ms
    return "dummy sentence"


def empty_frame():
    return SimpleNamespace(
        left_hand=np.zeros((21, 3), np.float32),
        right_hand=np.zeros((21, 3), np.float32),
        pose=np.zeros((25, 3), np.float32),
        face=np.zeros((40, 3), np.float32),
    )


files = sorted(glob.glob("data/processed/train/*.npz"))
if not files:
    sys.exit("No .npz files found. Run this from the project root.")

print("clip | true span | sent to model | wait (s) | model (ms) | events")
print("-----|-----------|---------------|----------|------------|-------")

shown = 0
exactly_one = 0
for path in files:
    if shown >= N_CLIPS:
        break
    d = np.load(path, allow_pickle=True)
    n = int(d["frame_count"])
    if n < 10:
        continue
    seen = (np.abs(d["left_hand"]).sum(axis=(1, 2)) > 0) | \
           (np.abs(d["right_hand"]).sum(axis=(1, 2)) > 0)
    if not seen.any():
        continue
    idx = np.where(seen)[0]
    true_span = int(idx.max() - idx.min() + 1)

    frames = [SimpleNamespace(left_hand=d["left_hand"][i], right_hand=d["right_hand"][i],
                              pose=d["pose"][i], face=d["face"][i]) for i in range(n)]
    pipe = RealtimePipeline(dummy_model)
    events = []
    steps = int((n * SAMPLE_PERIOD_S + TAIL_S) * CAMERA_FPS)
    for k in range(steps):
        t = k / CAMERA_FPS
        i = int(t / SAMPLE_PERIOD_S)
        ev = pipe.push(frames[i] if i < n else empty_frame(), t)
        if ev:
            events.append(ev)

    shown += 1
    if len(events) == 1:
        exactly_one += 1
    if events:
        e = events[0]
        print(f"{shown:>4} | {true_span:>9} | {e['n_samples']:>13} | "
              f"{e['wait_s']:>8.2f} | {e['L_inf_ms']:>10.0f} | {len(events):>6}")
    else:
        print(f"{shown:>4} | {true_span:>9} | {'-':>13} | {'-':>8} | {'-':>10} | {0:>6}")

print()
print(f"clips with exactly one event: {exactly_one}/{shown}")