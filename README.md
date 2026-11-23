# RoboPet hand-sign detector

RoboPet is a CPEN 391 robot that uses face verification to identify its owner and follows them. This repository contains its hand-sign interface. A person can raise an open palm to request a stop or point down to call the car over. The detector reads a camera frame and returns a stable gesture label. RoboPet’s controller decides how to act on that request.

## From camera frame to command

MediaPipe Hands locates 21 joints in the camera frame. The detector converts their pixel coordinates into 42 features and passes them to a small multilayer perceptron (MLP). The classifier predicts a pose on every usable frame. A filter then checks whether the prediction is confident and consistent enough to send to the controller.

| Pose | Stable output | Intended use |
|---|---|---|
| Open palm facing camera | palm | Stop |
| Index finger pointing down | point_down | Come here |
| Thumb pointing sideways | thumb_side | Spin |
| Index finger pointing outward or sideways | point | Request owner switch |
| Neutral pose, missing hand or low confidence | None | No gesture command |

## Design choices

- **Classify the hand shape using landmarks.** RoboPet’s face and body tracking also need compute on the Jetson. Reusing a pretrained hand tracker lets us train just the gesture classifier on 42 numbers per hand. The trade-off is that a missed hand produces no command.
- **Preserve shape across camera formats.** MediaPipe expresses horizontal and vertical positions as fractions of image width and height. We convert them to pixels first so a wide frame does not distort the hand proportions. We then subtract the wrist position and divide by the longest wrist-to-joint distance. This removes position and scale while preserving orientation, which matters for distinguishing a downward point from a sideways one.
- **Share the feature calculation.** Data collection and live inference call the same normalization function. Each CSV row keeps the label, timestamp, source, 42 model inputs and original pixel coordinates. Keeping the pixels lets us revisit normalization without processing the video again.
- **Mirror samples during training.** Flipping the horizontal coordinates adds an opposite-hand example while preserving the direction of a downward point. Live detection uses the original camera frame. Only the webcam preview is mirrored for the person giving the gesture.
- **Export the classifier as NumPy arrays.** The checkpoint stores the feature scaler, weights, biases and labels together. The Jetson can run the classifier with NumPy, avoiding a runtime dependency on the laptop’s scikit-learn version. The pipeline tests compare the exported model’s predictions with scikit-learn’s.
- **Give the controller a neutral state.** An ordinary hand pose can be confidently recognized as `no_gesture`. The frame filter also clears the command when tracking drops or confidence falls. Requiring a short hold reduces accidental requests at the cost of a few frames of response time.

## Test the camera on Windows

Open PowerShell in the repository root. With Python 3.10 installed, run the following once:

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
~~~

With the webcam connected (or the phone already available as a Windows webcam), run:

~~~powershell
.\.venv\Scripts\python.exe scripts/live_demo.py --camera 0
~~~

Look for **Using trained model** in the terminal. If it says **placeholder**, check that checkpoints/gesture_model.npz exists and that you started the demo in the project root. If you get the wrong camera or no feed, close other camera demos and try --camera 1. Hold your palm facing the camera until COMMAND shows palm, then lower your hand and check that the command clears. Press Q in the preview to quit.

## Controller handoff

Create one GesturePipeline per camera stream and reuse it so the tracker and smoothing state persist.

~~~python
from src.inference.pipeline import GesturePipeline

pipeline = GesturePipeline(model_path="checkpoints/gesture_model.npz")
# In the camera callback, supply the OpenCV BGR frame.
result = pipeline.process(frame_bgr)
command = result.command
~~~

A command requires five consecutive matching predictions with confidence at least 0.8. A held gesture continues returning its label. The receiving controller handles repeat suppression for actions such as spin and owner switch. The robot controller and live Jetson camera still need integration testing.

## Check the saved model without a camera

~~~powershell
.\.venv\Scripts\python.exe tests/test_pipeline.py
.\.venv\Scripts\python.exe scripts/evaluate.py data/laptop/jerry_session6.csv
~~~

The earlier checkpoint scored **1,226 / 1,334 frames (91.9%)** on session 5 before that session was added to training. The final checkpoint includes sessions 1 through 5. To measure its accuracy on new data, record a fresh session and pass that CSV to the evaluation script. The matrix shows actual gestures in rows and predictions in columns.

## Record or retrain

The CSVs contain labelled landmarks. The data folder is git-ignored; request the session CSVs from Jerry to reproduce training or evaluation. To record, run scripts/collect_data.py with --out and a new CSV path. Keys 1 through 5 select no_gesture, palm, point_down, thumb_side and point, respectively. Space stops the burst, U undoes the last burst and Q quits. Reusing a filename appends rows.

To retrain from the earlier sessions, run the following from the project root. This replaces the checkpoint. The evaluation example above assumes you saved a fresh recording as jerry_session6.csv.

~~~powershell
.\.venv\Scripts\python.exe scripts/train.py data/laptop/jerry_session1.csv data/laptop/jerry_session2.csv data/laptop/jerry_session3.csv data/laptop/jerry_session4.csv data/laptop/no_gesture_extra.csv data/laptop/jerry_session5.csv
~~~

## What each file does

### Live gesture detection

| File | Role |
|---|---|
| `src/inference/hand_landmarks.py` | Tracks 21 hand landmarks and converts them into the same 42 features used during recording and live prediction. Also defines the CSV columns and draws the landmarks for the preview. |
| `src/inference/classifier.py` | Exports the trained MLP and scaler, then loads those arrays to predict a gesture and confidence using NumPy. |
| `src/inference/pipeline.py` | Processes one camera frame at a time. Returns the raw prediction for debugging and the filtered command for the controller. Includes a finger-count fallback for testing without a checkpoint. |
| `src/inference/hand_crop.py` | Makes a square, padded crop around an annotated hand for HaGRID extraction. |
| `src/__init__.py`, `src/inference/__init__.py` | Mark the source and inference directories as Python packages. |

### Scripts

| File | When to use it |
|---|---|
| `scripts/hand_demo.py` | Check whether MediaPipe can see the hand. The preview shows landmarks and frame rate. |
| `scripts/live_demo.py` | Try the full detector on a webcam and compare the current prediction with the command that passes the hold filter. |
| `scripts/collect_data.py` | Record labelled bursts of webcam landmarks to a CSV. Undo the last burst if a pose was labelled incorrectly. |
| `scripts/extract_hagrid.py` | Prepare optional HaGRID training data by cropping annotated hands and writing detected landmarks in the webcam recorder’s CSV format. |
| `scripts/train.py` | Train on one or more session CSVs, augment the samples by mirroring, fit the scaler and MLP, and export the checkpoint. |
| `scripts/evaluate.py` | Score a checkpoint on a labelled session. Prints frame accuracy and a confusion matrix showing which gestures were mixed up. |

### Model, tests and environment

| File | Role |
|---|---|
| `checkpoints/gesture_model.npz` | Saved model used by the live demo. Bundles the weights, scaler and gesture labels. |
| `tests/test_pipeline.py` | Runs camera-free checks for feature geometry, CSV recording and undo, cropping, HaGRID extraction, model export and command filtering. |
| `requirements-train.txt` | Pins the laptop dependencies for data collection, training and the webcam demo. |
| `requirements-jetson.txt` | Documents the Jetson runtime environment. Use it as a reference when checking the board’s installed packages. |
| `.gitignore` | Keeps local files such as recorded session data out of version control. |

## Known gaps and next steps

- **Evaluate the final checkpoint on unseen recordings.** Session 5 became part of training after the earlier evaluation. Record a fresh session and run `scripts/evaluate.py` on it. Include another person and vary distance and lighting to see where recognition breaks down.
- **Measure response time on the Jetson.** Run the live pipeline with the car camera and record the package versions, frame rate and time from forming a gesture to receiving a stable command. The five-frame hold takes longer as frame rate drops.
- **Test the handoff to RoboPet’s controller.** Hold a spin gesture and confirm the controller starts a single spin. Lower the hand, verify that the command clears, then repeat the gesture to check that a second request is accepted.
