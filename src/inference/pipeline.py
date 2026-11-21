"""
Frame -> stable hand-sign command.

    pipeline = GesturePipeline()            # loads checkpoints/gesture_model.npz if present
    result = pipeline.process(frame_bgr)    # every camera frame
    result.command                          # e.g. "palm", or None = no command

Safety rules from the proposal (section 1, "Hand signs"):
  - only act on predictions with confidence >= conf_threshold
  - the same gesture must hold for hold_frames frames in a row
  - losing the hand, or "no_gesture", means NO command (car stops / holds)
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.inference.hand_landmarks import HandTracker, normalize

DEFAULT_MODEL_PATH = os.path.join("checkpoints", "gesture_model.npz")
CONF_THRESHOLD = 0.8
HOLD_FRAMES = 5


class FingerCountPlaceholder:
    """Stand-in until the trained model exists: counts extended fingers.
    Same interface as classifier.GestureModel."""

    classes = ["fist", "palm", "no_gesture"]
    _FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, middle joint)

    def predict(self, features):
        p = np.asarray(features).reshape(21, 2)  # wrist is already at (0, 0)
        extended = sum(np.linalg.norm(p[tip]) > 1.1 * np.linalg.norm(p[mid])
                       for tip, mid in self._FINGERS)
        if extended == 0:
            return "fist", 0.9
        if extended == 4:
            return "palm", 0.9
        return "no_gesture", 0.6


class CommandSmoother:
    """Turns noisy per-frame predictions into a stable command."""

    def __init__(self, threshold=CONF_THRESHOLD, hold_frames=HOLD_FRAMES):
        self.threshold, self.hold_frames = threshold, hold_frames
        self.candidate, self.count = None, 0

    def update(self, label, confidence):
        """label=None means no hand. Returns the command, or None for 'no command'."""
        if label is None or confidence < self.threshold or label == "no_gesture":
            self.candidate, self.count = None, 0
            return None
        if label == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = label, 1
        return label if self.count >= self.hold_frames else None


def load_model(path=DEFAULT_MODEL_PATH):
    """Trained model if it exists, otherwise the finger-count placeholder."""
    if os.path.exists(path):
        from src.inference.classifier import GestureModel
        return GestureModel(path), "trained model"
    return FingerCountPlaceholder(), "placeholder"


@dataclass
class GestureResult:
    landmarks: Optional[np.ndarray]  # (21, 2) pixels, or None if no hand
    label: Optional[str]             # this frame's raw prediction
    confidence: float
    command: Optional[str]           # smoothed command; None = do nothing


class GesturePipeline:
    def __init__(self, model_path=DEFAULT_MODEL_PATH,
                 conf_threshold=CONF_THRESHOLD, hold_frames=HOLD_FRAMES):
        self.model, self.model_name = load_model(model_path)
        self.tracker = HandTracker()
        self.smoother = CommandSmoother(conf_threshold, hold_frames)

    def process(self, frame_bgr):
        pts = self.tracker.landmarks(frame_bgr)
        features = normalize(pts) if pts is not None else None
        if features is None:
            label, conf = None, 0.0
        else:
            label, conf = self.model.predict(features)
        return GestureResult(pts, label, conf, self.smoother.update(label, conf))
