"""Score a saved checkpoint against a labelled session."""
import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.classifier import GestureModel
from src.inference.hand_landmarks import FEATURE_COLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--model", default="checkpoints/gesture_model.npz")
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    model = GestureModel(args.model)
    if data.empty or not data["label"].isin(model.classes).all():
        parser.error("The CSV must contain samples labelled with this model's classes.")
    # predict returns (label, confidence). Compare the label with the key recorded for each frame.
    predictions = [model.predict(row)[0] for row in data[FEATURE_COLS].to_numpy()]
    correct = data["label"] == predictions
    print(f"Correct: {correct.sum()} / {len(data)} frames ({correct.mean():.1%})")
    # Rows are recorded gestures and columns are predictions. Off-diagonal counts show which poses were confused.
    print(pd.crosstab(data["label"], predictions, rownames=["Actual"], colnames=["Predicted"]))
