"""
Hand landmark demo (H1). Webcam with the 21 MediaPipe hand landmarks drawn
on it. Press Q to quit.

    python scripts/hand_demo.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.inference.hand_landmarks import HandTracker, draw, mirror_for_display, normalize  # noqa: E402

CAMERA_INDEX = 0  # change to 1 if the wrong camera opens


def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1.")
        return

    last, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break

        pts = tracker.landmarks(frame)        # detect on the REAL image
        view = mirror_for_display(frame)      # mirror only for viewing
        if pts is not None:
            draw(view, pts)
            status = f"hand: yes  ({len(normalize(pts))} features)"
        else:
            status = "hand: no"

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(view, f"{status}   fps: {fps:.0f}   [Q] quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Hand landmarks", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
