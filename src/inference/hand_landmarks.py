"""
Shared hand-landmark code for RoboPet hand signs.

Used by BOTH the training side (scripts/extract_hagrid.py, scripts/train*.py)
and the live side (scripts/live_demo.py, later the Jetson). The classifier
only works if training data and live data go through the EXACT same steps.
If you change normalize() or the CSV format, bump FEATURE_VERSION and tell
each other: old models will then refuse to load instead of silently failing.

Landmark order (MediaPipe): 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
13-16 ring, 17-20 pinky (each finger: base -> tip).
"""
import time

import cv2
import mediapipe as mp
import numpy as np

FEATURE_VERSION = "v1-xy-pixels-wrist-maxdist"
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 2  # 42

# ---- Dataset CSV format (shared by collect_data.py and extract_hagrid.py) ----
FEATURE_COLS = [f"f{i}" for i in range(NUM_FEATURES)]
RAW_COLS = [f"p{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y")]
HEADER = ["label", "timestamp", "source"] + FEATURE_COLS + RAW_COLS


class HandTracker:
    """Wraps MediaPipe Hands.

    static_image_mode=False: for video (faster, tracks between frames).
    static_image_mode=True:  for unrelated photos, e.g. the HaGRID dataset.
    """

    def __init__(self, static_image_mode=False, min_detection_confidence=0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def landmarks(self, frame_bgr):
        """Return a (21, 2) array of PIXEL coordinates, or None if no hand.

        Pixels (not MediaPipe's 0..1) so the hand's shape isn't stretched by
        the image's aspect ratio: HaGRID photos and the car camera have
        different shapes, and pixels make them comparable.
        """
        # OpenCV supplies BGR channels. MediaPipe expects the same pixels in RGB order.
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lm = res.multi_hand_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def close(self):
        self._hands.close()


def normalize(pts):
    """(21, 2) pixel landmarks -> 42 features for the classifier.

    The wrist moves to (0, 0) and the farthest landmark ends up at distance 1,
    so the features ignore where the hand is and how big it looks.
    Returns None for a degenerate hand (all points on top of each other).
    """
    pts = np.asarray(pts, dtype=np.float32)
    # Point 0 is the wrist. Subtract its x and y from every joint.
    pts = pts - pts[0]
    # axis=1 gives one wrist-to-joint distance per row. Use the longest as the hand size.
    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale < 1e-6:
        return None
    return (pts / scale).flatten()


def mirror_features(features):
    """Left hand <-> right hand. Used for training augmentation so the
    model works with either hand."""
    f = np.asarray(features, dtype=np.float32).reshape(NUM_LANDMARKS, 2).copy()
    # Flip x around the wrist while keeping y and the joint ordering intact.
    f[:, 0] *= -1
    return f.flatten()


def make_row(label, pts, source):
    """One CSV row in the shared dataset format, or None if unusable."""
    features = normalize(pts)
    if features is None:
        return None
    return ([label, f"{time.time():.3f}", source]
            + [f"{v:.5f}" for v in features]
            + [f"{v:.1f}" for v in np.asarray(pts).flatten()])


# ---- Display helpers (not needed for training) ----
CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


def mirror_for_display(frame):
    """Flip the image so the screen acts like a mirror.

    Only for what you SEE. Always detect on the un-flipped frame so the data
    matches the car camera, which is not mirrored.
    """
    return cv2.flip(frame, 1)


def draw(frame, pts, mirrored=True):
    """Draw pixel landmarks. mirrored=True if frame went through mirror_for_display()."""
    w = frame.shape[1]
    px = [(int(w - 1 - x) if mirrored else int(x), int(y)) for x, y in pts]
    for a, b in CONNECTIONS:
        cv2.line(frame, px[a], px[b], (255, 255, 255), 2)
    for i, p in enumerate(px):
        cv2.circle(frame, p, 5, (0, 200, 0), -1)
        cv2.putText(frame, str(i), (p[0] + 6, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
