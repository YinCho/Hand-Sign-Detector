"""Cropping around a hand before running MediaPipe.

MediaPipe finds small hands much more reliably in a crop than in a full
photo. Used for HaGRID (which gives a box per hand), and later on the car
if hands are missed at a distance (proposal 6.2: crop around the owner).
"""


def square_crop(img, bbox, pad=0.5):
    """Square crop around a normalized [x, y, w, h] box (top-left corner,
    0..1 of image size), padded by `pad` so the whole hand is visible.
    Clipped at the image edges. Returns None if the crop is too small."""
    H, W = img.shape[:2]
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    side = max(w * W, h * H) * (1 + pad)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1]
