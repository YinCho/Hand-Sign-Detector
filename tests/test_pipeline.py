"""
Checks for the hand-sign pipeline using synthetic data (no camera, no
HaGRID download needed). Run from the repo root:

    python tests/test_pipeline.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import collect_data as cd  # noqa: E402
from src.inference import classifier as clf_mod  # noqa: E402
from src.inference import hand_landmarks as hl  # noqa: E402
from src.inference.hand_crop import square_crop  # noqa: E402
from src.inference.pipeline import CommandSmoother, FingerCountPlaceholder  # noqa: E402

rng = np.random.default_rng(0)
pts = rng.uniform(100, 400, (21, 2)).astype(np.float32)


def test_normalize():
    f = hl.normalize(pts)
    assert f.shape == (42,) and np.allclose(f[:2], 0), "wrist should be at the origin"
    assert np.isclose(np.max(np.linalg.norm(f.reshape(21, 2), axis=1)), 1.0)
    assert np.allclose(f, hl.normalize(pts * 0.3 + 57), atol=1e-5), "ignore position and size"
    assert hl.normalize(np.zeros((21, 2))) is None


def test_mirror():
    f = hl.normalize(pts)
    m = hl.mirror_features(f)
    assert np.allclose(hl.mirror_features(m), f)
    assert np.allclose(m[1::2], f[1::2]) and np.allclose(m[0::2], -f[0::2])


def test_pixels_avoid_aspect_distortion():
    hand01 = rng.uniform(0.3, 0.6, (21, 2))  # MediaPipe-style 0..1 coords in a 640x480 image
    in_pixels = hl.normalize(hand01 * [640, 480])
    assert np.allclose(in_pixels, hl.normalize(hand01 * [640, 480] * 2.25), atol=1e-5)
    assert not np.allclose(in_pixels, hl.normalize(hand01), atol=1e-3), \
        "0..1 coords of a non-square image stretch the hand"


def test_csv_rows_and_undo(tmp):
    row = hl.make_row("palm", pts, "test")
    assert len(row) == len(hl.HEADER) == 3 + 42 + 42
    p = os.path.join(tmp, "c.csv")
    for r in [hl.make_row("palm", pts, "t")] * 5 + [hl.make_row("fist", pts, "t")] * 3:
        cd.append_rows(p, [r])
    rows = cd.load_rows(p)
    assert cd.counts(rows)["palm"] == 5 and cd.counts(rows)["fist"] == 3
    del rows[-3:]
    cd.write_all(p, rows)
    assert cd.counts(cd.load_rows(p))["fist"] == 0
    with open(p) as fh:
        assert next(csv.reader(fh)) == hl.HEADER


def test_square_crop():
    img = np.zeros((1080, 1920, 3), np.uint8)
    c = square_crop(img, [0.5, 0.5, 0.05, 0.1], pad=0.5)
    assert abs(c.shape[0] - c.shape[1]) <= 1 and abs(c.shape[0] - 162) <= 1, c.shape
    assert square_crop(img, [0.98, 0.0, 0.05, 0.05]).size > 0, "clipped at edge, still valid"
    assert square_crop(img, [0.5, 0.5, 0.0001, 0.0001]) is None


def test_extract_hagrid_runs(tmp):
    ann_dir, img_dir = os.path.join(tmp, "ann"), os.path.join(tmp, "img")
    os.makedirs(ann_dir)
    os.makedirs(os.path.join(img_dir, "palm"))
    cv2.imwrite(os.path.join(img_dir, "palm", "id1.jpg"), np.full((480, 640, 3), 128, np.uint8))
    with open(os.path.join(ann_dir, "palm.json"), "w") as f:
        json.dump({"id1": {"bboxes": [[0.4, 0.4, 0.2, 0.2]], "labels": ["palm"]},
                   "id_missing": {"bboxes": [[0.4, 0.4, 0.2, 0.2]], "labels": ["palm"]},
                   "id_bad": {"something_else": 1}}, f)
    out = os.path.join(tmp, "h.csv")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "extract_hagrid.py"),
                        "--annotations", ann_dir, "--images", img_dir, "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "1 annotated images were not found" in r.stdout, r.stdout
    with open(out) as fh:
        assert next(csv.reader(fh)) == hl.HEADER


def test_numpy_runtime_matches_sklearn(tmp):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    warnings.filterwarnings("ignore")
    X = rng.normal(size=(600, 42)).astype(np.float32)
    y5 = np.array(["palm", "fist", "like", "peace", "no_gesture"])[np.argmax(X[:, :5], axis=1)]
    y2 = np.where(X[:, 0] > 0, "palm", "fist")
    cases = {
        "multiclass_relu": (MLPClassifier((64, 32), max_iter=300, random_state=0).fit(X, y5), X),
        "binary_tanh": (MLPClassifier((16,), activation="tanh", max_iter=300,
                                      random_state=0).fit(X, y2), X),
        "pipeline_scaler": (make_pipeline(StandardScaler(), MLPClassifier(
            (32,), max_iter=300, random_state=0)).fit(X * 5 + 3, y5), X * 5 + 3),
    }
    for name, (clf, Xs) in cases.items():
        path = os.path.join(tmp, name + ".npz")
        clf_mod.export_mlp(clf, path)
        gm = clf_mod.GestureModel(path)
        ours = np.array([gm.predict_proba(x) for x in Xs[:50]])
        assert np.allclose(ours, clf.predict_proba(Xs[:50]), atol=1e-6), name
        assert gm.predict(Xs[0])[0] == clf.predict(Xs[:1])[0], name

    # a model made with different features must refuse to load
    d = dict(np.load(os.path.join(tmp, "multiclass_relu.npz")))
    meta = json.loads(d["meta"].item())
    meta["feature_version"] = "old"
    d["meta"] = np.array(json.dumps(meta))
    np.savez(os.path.join(tmp, "old.npz"), **d)
    try:
        clf_mod.GestureModel(os.path.join(tmp, "old.npz"))
        raise AssertionError("should refuse a model trained on different features")
    except ValueError:
        pass


def test_smoother():
    s = CommandSmoother(threshold=0.8, hold_frames=3)
    assert [s.update("palm", 0.9) for _ in range(4)] == [None, None, "palm", "palm"]
    assert s.update(None, 0) is None and s.update("palm", 0.9) is None, "hand lost resets"
    assert s.update("palm", 0.5) is None and s.count == 0, "low confidence resets"
    s.update("palm", 0.9)
    s.update("palm", 0.9)
    assert s.update("fist", 0.95) is None, "switching gesture restarts the count"
    assert s.update("no_gesture", 0.99) is None, "no_gesture is never a command"


def test_placeholder():
    def hand(open_):
        p = np.zeros((21, 2))
        for k, base in enumerate([1, 5, 9, 13, 17]):
            for j in range(4):
                reach = 40 + j * 30 if open_ else [40, 60, 45, 35][j]
                p[base + j] = [(k - 2) * 20, -reach]
        return p + 300
    ph = FingerCountPlaceholder()
    assert ph.predict(hl.normalize(hand(True)))[0] == "palm"
    assert ph.predict(hl.normalize(hand(False)))[0] == "fist"


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    try:
        tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
        for name, fn in tests:
            fn(tmp) if fn.__code__.co_argcount else fn()
            print(f"PASS {name}")
        print(f"\nAll {len(tests)} tests passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
