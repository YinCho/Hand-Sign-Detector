"""
Turn HaGRID V2 images into training rows (H2 prep).

Runs the SAME MediaPipe + normalize() as the car, and writes the SAME CSV
format as collect_data.py, so HaGRID rows and our own recordings can be
mixed in one training set.

Expected HaGRID layout (check against one downloaded file; adjust
find_image() if yours differs):
    <annotations>/<class>.json   {image_id: {"bboxes": [[x, y, w, h], ...],
                                             "labels": ["palm", ...], ...}}
    <images>/<class>/<image_id>.jpg
bboxes are normalized 0..1, top-left corner + width/height.

Example:
    python scripts/extract_hagrid.py --annotations data/hagrid/annotations/train
        --images data/hagrid/images --classes palm fist like peace no_gesture
        --max-per-class 3000
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from src.inference.hand_crop import square_crop  # noqa: E402
from src.inference.hand_landmarks import HEADER, HandTracker, make_row  # noqa: E402


def find_image(images_root, cls, image_id):
    for p in (os.path.join(images_root, cls, image_id + ".jpg"),
              os.path.join(images_root, image_id + ".jpg"),
              os.path.join(images_root, cls, image_id + ".jpeg")):
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True, help="folder of <class>.json files")
    ap.add_argument("--images", required=True, help="folder with <class>/<id>.jpg")
    ap.add_argument("--out", default=os.path.join("features", "landmarks", "hagrid_landmarks.csv"))
    ap.add_argument("--classes", nargs="*", help="only keep these labels (default: all)")
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no limit")
    ap.add_argument("--pad", type=float, default=0.5, help="crop padding around the hand box")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    keep = set(args.classes) if args.classes else None
    tracker = HandTracker(static_image_mode=True, min_detection_confidence=0.5)
    saved, missed, no_image = Counter(), Counter(), 0

    json_files = sorted(f for f in os.listdir(args.annotations) if f.endswith(".json"))
    with open(args.out, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(HEADER)
        for jf in json_files:
            cls = jf[:-5]
            with open(os.path.join(args.annotations, jf)) as f:
                data = json.load(f)
            print(f"{cls}: {len(data)} images")
            for image_id, ann in data.items():
                labels, bboxes = ann.get("labels"), ann.get("bboxes")
                if not labels or not bboxes:
                    continue
                # Pair each label with its own hand box; one photo can contain more than one labelled hand.
                wanted = [(l, b) for l, b in zip(labels, bboxes)
                          if (keep is None or l in keep)
                          and (not args.max_per_class or saved[l] < args.max_per_class)]
                if not wanted:
                    continue
                path = find_image(args.images, cls, image_id)
                img = cv2.imread(path) if path else None
                if img is None:
                    no_image += 1
                    continue
                for label, bbox in wanted:
                    crop = square_crop(img, bbox, args.pad)
                    pts = tracker.landmarks(crop) if crop is not None else None
                    row = make_row(label, pts, "hagrid") if pts is not None else None
                    if row is None:
                        missed[label] += 1
                        continue
                    writer.writerow(row)
                    saved[label] += 1

    print("\nlabel           saved   no hand found")
    for label in sorted(set(saved) | set(missed)):
        print(f"{label:<15} {saved[label]:>6}   {missed[label]:>6}")
    if no_image:
        print(f"\n{no_image} annotated images were not found on disk. Check --images.")
    print(f"\nWrote {sum(saved.values())} rows to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
