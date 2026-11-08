"""Replay real training clips through the boundary detector.

Each clip is fed in one sample at a time, then followed by TAIL empty
samples (hands lowered). For every clip we check whether the detector
fires AFTER the signer's last hand movement (good) or BEFORE it
(the sentence was cut off early = a false trigger).

Run from the project root:
    python scripts/replay_boundary_test.py
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.backend.sign_boundary import SignBoundaryDetector  # noqa: E402

N_CLIPS = 500          # how many training clips to test
TAIL = 15              # empty samples appended after each clip
SAMPLE_PERIOD_S = 0.2085   # time between samples (from the dataset timestamps)
REST_VALUES = [3, 4, 5, 6, 8, 10]   # thresholds to compare

files = sorted(glob.glob("data/processed/train/*.npz"))[:N_CLIPS]
if not files:
    sys.exit("No .npz files found. Run this from the project root.")

clips = []
for f in files:
    d = np.load(f, allow_pickle=True)
    if int(d["frame_count"]) < 10:      # skip the tiny clips
        continue
    left, right = d["left_hand"], d["right_hand"]
    hands_seen = (np.abs(left).sum(axis=(1, 2)) > 0) | (np.abs(right).sum(axis=(1, 2)) > 0)
    if not hands_seen.any():
        continue                        # nobody signs in this clip
    last_active = int(np.where(hands_seen)[0].max())
    clips.append((left, right, last_active))

print(f"clips used: {len(clips)}")
print()
print("rest_samples | wait after signing | fired OK | cut early | never fired")
print("-------------|--------------------|----------|-----------|------------")

empty = np.zeros((21, 3), dtype=np.float32)
for rest in REST_VALUES:
    ok = early = never = 0
    for left, right, last_active in clips:
        det = SignBoundaryDetector(rest_samples=rest)
        fire_at = None
        for i in range(len(left) + TAIL):
            l = left[i] if i < len(left) else empty
            r = right[i] if i < len(right) else empty
            if det.update(l, r):
                fire_at = i
                break
        if fire_at is None:
            never += 1
        elif fire_at < last_active:
            early += 1
        else:
            ok += 1
    n = len(clips)
    print(f"{rest:>12} | {rest * SAMPLE_PERIOD_S:>16.2f} s | {ok / n:>7.1%} | "
          f"{early / n:>8.1%} | {never / n:>10.1%}")