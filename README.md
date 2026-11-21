# Hand-Sign-Detector

Hand-sign control for **RoboPet** (face-verified owner following with
hand-sign control). A camera frame goes in, a stable command for the car
comes out:

```
camera frame
  → MediaPipe Hands: 21 hand landmarks (pixels)     src/inference/hand_landmarks.py
  → normalize: 42 numbers, position/size removed    src/inference/hand_landmarks.py
  → MLP classifier: gesture + confidence            src/inference/classifier.py
  → smoothing + safety rules: command or nothing    src/inference/pipeline.py
  → car state machine (Nolé)                        src/robot/  (not written yet)
```

| Task | Who | Status |
|---|---|---|
| H1. Test the MediaPipe hand landmark tool on a laptop | David | Done: `scripts/hand_demo.py` |
| H2. Train the MLP classifier on HaGRID V2 | Jerry | Data extractor and working starter script ready |
| H3. Merge the landmark tool with the trained classifier | Jerry | `src/inference/pipeline.py` ready; just needs the trained model file |
| H4. Deploy the full hand-sign pipeline on the Jetson | David | Not started (Nov 28–30) |

---

## Quick start (laptop)

Python **3.10** (same as the Jetson).

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
```

Run everything from the repo root:

```powershell
python scripts/hand_demo.py          # see the 21 landmarks on your hand
python scripts/live_demo.py          # full pipeline (placeholder model until training is done)
python scripts/collect_data.py       # record your own gesture samples
python tests/test_pipeline.py        # checks, no camera needed
```

## For Jerry: training (H2) and handing the model back

1. **Download HaGRID V2** into `data/hagrid/` (`data/` is git-ignored).
   Check one annotation file matches the layout described at the top of
   `scripts/extract_hagrid.py`; if not, adjust `find_image()`.
2. **Turn images into landmark rows.** This has to use our extractor so
   training features are made exactly like the live ones:
   ```powershell
   python scripts/extract_hagrid.py --annotations data/hagrid/annotations/train --images data/hagrid/images --classes palm fist like peace no_gesture --max-per-class 3000
   ```
   Output: `features/landmarks/hagrid_landmarks.csv`. The printed table shows
   how many hands MediaPipe found per class.
3. **Train.** `scripts/train_mlp_example.py` works end to end; copy it into
   `scripts/train.py` and change what you like (layers, classes, more data):
   ```powershell
   python scripts/train_mlp_example.py features/landmarks/hagrid_landmarks.csv data/laptop/gesture_data.csv
   ```
4. **The one hard rule:** train on columns `f0`–`f41` and finish with
   `export_mlp(...)`. That writes `checkpoints/gesture_model.npz`, which
   `scripts/live_demo.py` and the Jetson pick up automatically.
5. **Pick the gestures with Nolé** (they map to car commands). Use HaGRID's
   exact class names, and update `GESTURES` in `scripts/collect_data.py` to match.

---

## What each file does

### `src/inference/` (the pipeline itself; runs on laptop and Jetson)

| File | What it does |
|---|---|
| `hand_landmarks.py` | **Shared core.** `HandTracker` (MediaPipe wrapper returning 21 pixel points), `normalize()` (points → 42 features), `mirror_features()` (left↔right hand), the CSV format (`HEADER`, `make_row`), and drawing helpers. Training and live code both import this, so the features are always made the same way. |
| `hand_crop.py` | `square_crop()`: square, padded crop around a hand box. Used for HaGRID, and available on the car if hands are missed at a distance (proposal 6.2). |
| `classifier.py` | `export_mlp()` saves a trained scikit-learn MLP as plain arrays; `GestureModel` runs it with **numpy only** and returns `(label, confidence)`. |
| `pipeline.py` | `GesturePipeline.process(frame)` → landmarks, raw prediction, and the **smoothed command**. Contains `CommandSmoother` (safety rules) and `FingerCountPlaceholder` (fist/palm stand-in used until the trained model exists). |

### `scripts/` (things you run)

| File | What it does |
|---|---|
| `hand_demo.py` | H1 demo: webcam with landmarks and FPS. |
| `live_demo.py` | Full pipeline on the webcam: frame prediction + smoothed command on screen; prints each command change. `--model`, `--camera`, `--threshold`, `--hold` options. |
| `collect_data.py` | Records labelled samples (number keys start, SPACE stops, U undoes the last burst). Writes the same CSV format as HaGRID extraction. `--out`, `--source` options (e.g. `--source jetson_cam` for on-car data). |
| `extract_hagrid.py` | HaGRID V2 images + annotations → landmark CSV. Crops each annotated hand, runs MediaPipe, reports per-class detection counts. |
| `train_mlp_example.py` | Working starter for H2: loads CSVs, mirror augmentation, StandardScaler + MLP, prints a per-class report, exports the model. |
| `train.py`, `evaluate.py`, `export_onnx.py` | Jerry's, still empty. |

### Other

| Path | What it is |
|---|---|
| `tests/test_pipeline.py` | 9 checks on synthetic data: normalization, aspect ratio, CSV format, undo, cropping, HaGRID extractor, numpy-vs-sklearn equality, smoothing, placeholder. |
| `requirements-train.txt` | Pinned laptop/training environment (installs cleanly, `pip check` passes). |
| `requirements-jetson.txt` | What the Jetson needs at runtime. **Documentation only: don't pip-install it on the Jetson.** |
| `checkpoints/` | Trained model goes here (`gesture_model.npz`). Not git-ignored, so the model can be shared through the repo. |
| `data/`, `features/`, `models/` | Datasets and generated CSVs. Git-ignored (too big for git). |
| `configs/training.yaml`, `src/robot/*` | Placeholders, still empty (Jerry / Nolé). |

### Dataset CSV format

One row per hand, made by `make_row()`:

| Columns | Meaning |
|---|---|
| `label` | Gesture name (HaGRID class names, e.g. `palm`, `no_gesture`) |
| `timestamp` | When the row was made |
| `source` | `hagrid`, `laptop_webcam`, `jetson_cam`, ... so sources can be compared or filtered |
| `f0`–`f41` | **Model input**: normalized x,y of the 21 landmarks |
| `px0, py0, … px20, py20` | Raw pixel landmarks, kept so we can re-normalize later without re-running MediaPipe |

---

## Design choices

**MediaPipe Hands + small classifier instead of an image model (YOLO etc.).**
MediaPipe gives a ready-made, pre-trained 21-point hand tracker, so we only
train a tiny classifier on 42 numbers instead of an image detector needing
thousands of labelled photos and GPU memory the face and body models also
need (proposal 5.3).

**MediaPipe's built-in `solutions.hands` API, version 0.10.14.**
The model ships inside the pip package, so there is no separate `.task` file
to download and keep in sync (the empty `assets/hand_landmarker.task`
placeholder was removed). 0.10.14 still has this API and works with Python 3.10.

**Same Python and library versions as the Jetson.** Python 3.10, numpy 1.x,
OpenCV 4.x on the laptop, so code that works here doesn't hit a missing
function on the car. (Newest OpenCV 5.x needs numpy 2, which the Jetson doesn't have.)

**Pixel coordinates, not MediaPipe's 0–1 values.** MediaPipe's 0–1 values are
fractions of image width and height, so the same hand looks stretched
differently in a 1920×1080 HaGRID photo than in a 640×480 car frame. Pixels
keep the real shape, so training and live features match. (Tested in
`test_pixels_avoid_aspect_distortion`.)

**Normalization: wrist at the origin, farthest point at distance 1.**
Removes where the hand is in the frame and how close it is, leaving only the
shape (proposal: "robust to camera distance"). Only x,y are used; MediaPipe's
z is a rough relative depth and noisier, so it would add noise for little gain.

**Mirror augmentation during training.** Every training sample is also added
flipped left↔right, so the model works with either hand without needing
twice the data.

**Detect on the real frame; mirror only the display.** A selfie-style mirrored
image turns a right hand into a left hand. The car camera isn't mirrored, so
recorded data mustn't be either. The screen still shows a mirror view so
it's easy to use.

**HaGRID: square, padded crops around each annotated hand.** MediaPipe finds
small hands in big photos poorly; cropping fixes that, and the square shape
avoids distortion. `static_image_mode=True` is used because HaGRID photos are
unrelated stills, not video.

**One shared module for training and live code.** `hand_landmarks.py` is
the only place features are made. `FEATURE_VERSION` is saved inside each
model; if normalization ever changes, old models refuse to load instead of
silently giving wrong answers.

**Model saved as numpy arrays instead of pickle or ONNX.** A pickled
scikit-learn model only loads in nearly the same scikit-learn version, and the
Jetson's apt version (0.23) is much older than the laptop's (1.7). ONNX would
need onnxruntime on the Jetson (a new dependency) or OpenCV DNN, which
may not support some extra operations scikit-learn's ONNX converter adds. An MLP is a few
matrix multiplications, so `classifier.py` does them in numpy; the tests
check the output matches scikit-learn to 6 decimal places. `export_onnx.py`
is still there if Jerry prefers ONNX.

**Safety smoothing (`CommandSmoother`).** From the proposal: act only on
confident predictions (≥ 0.8) that hold for several frames (5 in a row), and
give no command when the hand is lost. `no_gesture` is never a command. Both
numbers are flags in `live_demo.py` so they can be tuned on the car.

**`no_gesture` class.** Without it the model must always pick a gesture,
even for a hand that's just scratching a nose. HaGRID already has this class.

**Placeholder model.** `FingerCountPlaceholder` (fist vs. palm by counting
extended fingers) lets the full pipeline and car integration be tested
before training is finished; it's replaced automatically once
`checkpoints/gesture_model.npz` exists.

## Known gaps / next steps

- Not yet tested on real HaGRID images or on the Jetson (only synthetic data
  and a laptop webcam).
- `extract_hagrid.py` assumes HaGRID's annotation layout; confirm on a real download.
- Final gesture list and gesture → car command mapping: decide with Nolé (`src/robot/commands.py`).
- Proposal 6.2 contingencies (lower resolution, MediaPipe every 2nd–3rd
  frame, crop around the owner's hand) are not implemented yet; do them in
  H4 if the Jetson frame rate is too low.
- Record the exact Jetson package versions in `requirements-jetson.txt`.
