"""
Working starter for H2. scripts/train.py is Jerry's to write; copy from this
or replace it. The only hard requirements for the live side are:
  1. train on the f0..f41 columns (made by hand_landmarks.normalize)
  2. finish with export_mlp(...) so the Jetson can run it without sklearn

    python scripts/train_mlp_example.py features/landmarks/hagrid_landmarks.csv [more.csv ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.inference.classifier import export_mlp  # noqa: E402
from src.inference.hand_landmarks import FEATURE_COLS, mirror_features  # noqa: E402
from src.inference.pipeline import DEFAULT_MODEL_PATH  # noqa: E402

df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
print(df["label"].value_counts(), "\n")

X = df[FEATURE_COLS].to_numpy(np.float32)
y = df["label"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0)

# Mirror augmentation: every training sample also as the other hand.
X_train = np.vstack([X_train, np.array([mirror_features(x) for x in X_train])])
y_train = np.concatenate([y_train, y_train])

clf = make_pipeline(
    StandardScaler(),
    # early_stopping=True crashes with text labels in scikit-learn 1.7.
    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0),
)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))

os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
export_mlp(clf, DEFAULT_MODEL_PATH)
