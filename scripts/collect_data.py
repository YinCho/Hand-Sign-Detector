"""
Gesture data collector: records hand landmarks + a label into a CSV in the
same format as the HaGRID data, so both can be used for training.

    python scripts/collect_data.py
    python scripts/collect_data.py --out data/oncar/gestures.csv --source jetson_cam

KEYS
  1-9    start recording that gesture (hold it, move your hand around a bit)
  SPACE  stop recording
  U      undo: delete the last recording burst
  Q      quit (running again ADDS to the same CSV)
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.inference.hand_landmarks import (  # noqa: E402
    HEADER, HandTracker, draw, make_row, mirror_for_display)

# Use HaGRID's class names EXACTLY so this data mixes with the HaGRID data.
# "no_gesture" = random hand poses that aren't commands, so the model learns
# to say "not a command". Placeholders: replace with the team's gestures.
GESTURES = ["no_gesture", "palm", "fist", "like", "peace"]
SAVE_EVERY_N_FRAMES = 2   # skip frames so samples aren't near-duplicates
CAMERA_INDEX = 0

assert len(GESTURES) <= 9, "Only keys 1-9 are available"


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]  # skip header


def write_all(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)


def append_rows(path, rows):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        w.writerows(rows)


def counts(rows):
    c = {g: 0 for g in GESTURES}
    for r in rows:
        if r and r[0] in c:
            c[r[0]] += 1
    return c


def put(frame, text, y, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "laptop", "gesture_data.csv"))
    ap.add_argument("--source", default="laptop_webcam", help="tag saved in each row")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            if next(csv.reader(f), None) != HEADER:
                print(f"{args.out} uses an old format. Rename or delete it, then run again.")
                return
    all_rows = load_rows(args.out)
    print(f"Loaded {len(all_rows)} existing samples from {args.out}")

    tracker = HandTracker()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera {CAMERA_INDEX}. Try CAMERA_INDEX = 1.")
        return

    recording = None   # gesture currently being recorded
    burst = []         # rows from the current/last burst (for undo)
    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames.")
            break
        frame_i += 1

        pts = tracker.landmarks(frame)
        view = mirror_for_display(frame)
        if pts is not None:
            draw(view, pts)

        if recording and pts is not None and frame_i % SAVE_EVERY_N_FRAMES == 0:
            row = make_row(recording, pts, args.source)
            if row:
                burst.append(row)
                all_rows.append(row)
                append_rows(args.out, [row])

        c = counts(all_rows)
        if recording:
            put(view, f"RECORDING: {recording}  (+{len(burst)})   SPACE = stop",
                25, (0, 0, 255), 0.65, 2)
            if pts is None:
                put(view, "no hand seen - not saving", 50, (0, 165, 255))
        else:
            put(view, "Hold a gesture, press its number to record. U=undo Q=quit",
                25, (0, 255, 0))
        for i, g in enumerate(GESTURES):
            put(view, f"[{i + 1}] {g}: {c[g]}", 80 + i * 22)

        cv2.imshow("Gesture data collector", view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            if recording:
                print(f"Stopped '{recording}': +{len(burst)} samples")
            recording = None
        elif ord("1") <= key <= ord(str(len(GESTURES))):
            recording = GESTURES[key - ord("1")]
            burst = []
            print(f"Recording '{recording}'...")
        elif key in (ord("u"), ord("U")) and burst:
            recording = None
            n = len(burst)
            del all_rows[-n:]
            write_all(args.out, all_rows)
            print(f"Undid last burst: removed {n} samples")
            burst = []

    cap.release()
    cv2.destroyAllWindows()

    print("\nSamples per gesture:")
    for g, n in counts(all_rows).items():
        print(f"  {g:>10}: {n}")
    print(f"Saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
