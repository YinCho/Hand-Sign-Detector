"""
Live hand-sign recognition: camera -> landmarks -> classifier -> smoothed
command. Uses checkpoints/gesture_model.npz if it exists, otherwise a
finger-counting placeholder (fist / palm). Press Q to quit.

    python scripts/live_demo.py
    python scripts/live_demo.py --model checkpoints/gesture_model.npz --camera 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.inference.hand_landmarks import draw, mirror_for_display  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    CONF_THRESHOLD, DEFAULT_MODEL_PATH, HOLD_FRAMES, GesturePipeline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--hold", type=int, default=HOLD_FRAMES)
    args = ap.parse_args()

    # Keep one pipeline across frames so the tracker and gesture streak keep their state.
    pipeline = GesturePipeline(args.model, args.threshold, args.hold)
    print(f"Using {pipeline.model_name}: classes {pipeline.model.classes}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. Try --camera 1.")
        return

    last_command = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = pipeline.process(frame)
        # A held pose can return the same command on many frames. Print only changes, including release.
        if r.command != last_command:
            print(f"command: {r.command or '-- none --'}")
            last_command = r.command

        view = mirror_for_display(frame)
        if r.landmarks is not None:
            draw(view, r.landmarks)
        raw = f"{r.label} ({r.confidence:.2f})" if r.label else "no hand"
        cv2.putText(view, f"[{pipeline.model_name}] frame: {raw}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(view, f"COMMAND: {r.command or '-- none --'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 220, 0) if r.command else (0, 0, 255), 2)
        cv2.imshow("Live hand signs", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
