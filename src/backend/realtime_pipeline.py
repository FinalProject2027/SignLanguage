"""Real-time pipeline glue (Member 4, Person B).

Sits between Person A's camera loop and Member 2's model.

    camera loop  --every frame-->  RealtimePipeline.push(frame)
                                        |
        1. counts frames          -> FPS
        2. resamples to ~4.8 Hz   -> same rate the model was trained on
        3. boundary detector      -> decides when a sentence has finished
        4. calls predict_fn       -> Member 2's model, timed
                                        |
                       returns None, or an "event" dict when a sentence ended

`frame` is any object with .left_hand, .right_hand, .pose and .face arrays,
for example Person A's LandmarkFrame from scripts/live_camera.py.

`predict_fn(arrays)` receives a dict with the same keys as the training
.npz files: left_hand (T,21,3), right_hand (T,21,3), pose (T,25,3),
face (T,40,3), and returns the predicted text.
"""
import time
from collections import deque

import numpy as np

from .sign_boundary import SignBoundaryDetector, hand_present

# Training data keeps every 5th frame of ~23.976 fps video = one sample
# every 0.2085 s (about 4.8 samples per second).
SAMPLE_PERIOD_S = 0.2085
DEFAULT_BUFFER_SIZE = 90
KEYS = ("left_hand", "right_hand", "pose", "face")


class FpsCounter:
    """Frames per second over a short sliding window."""

    def __init__(self, window_s=2.0):
        self.window_s = window_s
        self.times = deque()

    def tick(self, t):
        self.times.append(t)
        while self.times and t - self.times[0] > self.window_s:
            self.times.popleft()

    def fps(self):
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class RealtimePipeline:
    def __init__(self, predict_fn, detector=None,
                 sample_period_s=SAMPLE_PERIOD_S,
                 buffer_size=DEFAULT_BUFFER_SIZE, clock=time.perf_counter):
        self.predict_fn = predict_fn
        self.detector = detector or SignBoundaryDetector()
        self.sample_period_s = sample_period_s
        self.samples = deque(maxlen=buffer_size)  # resampled frames only
        self.clock = clock
        self.fps_counter = FpsCounter()
        self._next_sample_t = None
        self._last_hand_t = None   # time of the newest sample that had a hand

    def fps(self):
        return self.fps_counter.fps()

    def push(self, frame, t=None):
        """Call for EVERY camera frame. `t` defaults to the current time.

        Returns None, or an event dict when a sentence just finished.
        """
        t = self.clock() if t is None else t
        self.fps_counter.tick(t)

        # Resample: keep one frame per sample period, drop the rest.
        if self._next_sample_t is None:
            self._next_sample_t = t
        if t < self._next_sample_t:
            return None
        self._next_sample_t += self.sample_period_s
        if self._next_sample_t <= t:   # running slower than the sample rate
            self._next_sample_t = t + self.sample_period_s

        if hand_present(frame.left_hand) or hand_present(frame.right_hand):
            self._last_hand_t = t
        self.samples.append(frame)

        if not self.detector.update(frame.left_hand, frame.right_hand):
            return None
        return self._finish_sentence(t)

    def _finish_sentence(self, t_trigger):
        frames = list(self.samples)
        end = len(frames) - self.detector.last_tail
        wanted = self.detector.last_length
        start = max(0, end - wanted)
        sentence = frames[start:end]
        arrays = {k: np.stack([getattr(f, k) for f in sentence]) for k in KEYS}

        t0 = time.perf_counter()
        text = self.predict_fn(arrays)
        infer_s = time.perf_counter() - t0

        return {
            "text": text,
            "n_samples": len(sentence),
            "truncated": len(sentence) < wanted,   # sentence longer than buffer
            # time between the signer's last visible hand and the trigger
            "wait_s": (t_trigger - self._last_hand_t
                       if self._last_hand_t is not None else None),
            "L_inf_ms": infer_s * 1000.0,
            "fps": self.fps(),
            # keys match TelemetryCollector.record_event. t_capture is the
            # moment the sentence was finalized (the trigger).
            "timestamps": {"t_capture": t_trigger,
                           "t_infer": t_trigger + infer_s},
        }