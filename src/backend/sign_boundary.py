"""Sign-boundary detector (Member 4, Person B).

Decides when a signed sentence has finished, so the app knows when to run
the model.

Input, once per sample (about 4.8 samples per second, the same rate as the
training data): the left-hand and right-hand landmark arrays, each shaped
(21, 3). A hand that was not detected is all zeros, exactly as in the
.npz training files.

Rule:
  IDLE    -> SIGNING  when at least one hand is seen for `start_samples`
                      samples in a row.
  SIGNING -> fires    when BOTH hands are missing for `rest_samples`
                      samples in a row AND the sentence had at least
                      `min_samples` samples of signing.
  If the hands vanish for `rest_samples` but the sentence was too short
  (a stray hand wave), it is thrown away without firing.
"""
import numpy as np


def hand_present(hand) -> bool:
    """True if the hand was detected (an undetected hand is all zeros)."""
    return bool(np.any(hand))


class SignBoundaryDetector:
    IDLE = "IDLE"
    SIGNING = "SIGNING"

    def __init__(self, start_samples=2, rest_samples=5, min_samples=5,
                 max_samples=None):
        self.start_samples = start_samples  # hand samples needed to start
        self.rest_samples = rest_samples    # empty samples needed to finish
        self.min_samples = min_samples      # shortest sentence we accept
        self.max_samples = max_samples      # optional forced cut-off
        self.last_length = 0                # signing samples in the last finished sentence
        self.last_tail = 0                  # empty samples that came right after it
        self.reset()

    def reset(self):
        self.state = self.IDLE
        self.active_run = 0   # consecutive samples with a hand (while IDLE)
        self.rest_run = 0     # consecutive samples with no hands (while SIGNING)
        self.length = 0       # samples since the sentence began

    def update(self, left_hand, right_hand) -> bool:
        """Call once per sample. Returns True when a sentence just finished."""
        present = hand_present(left_hand) or hand_present(right_hand)

        if self.state == self.IDLE:
            self.active_run = self.active_run + 1 if present else 0
            if self.active_run >= self.start_samples:
                self.state = self.SIGNING
                self.length = self.active_run
                self.rest_run = 0
            return False

        # state is SIGNING
        self.length += 1
        self.rest_run = 0 if present else self.rest_run + 1

        if self.rest_run >= self.rest_samples:
            signing_length = self.length - self.rest_run
            self.reset()
            if signing_length >= self.min_samples:
                self.last_length = signing_length
                self.last_tail = self.rest_samples
                return True
            return False  # too short: discard

        if self.max_samples is not None and self.length >= self.max_samples:
            self.last_length = self.length - self.rest_run
            self.last_tail = self.rest_run
            self.reset()
            return True

        return False