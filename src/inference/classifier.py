"""
Run the trained MLP using ONLY numpy, so the Jetson doesn't need a
matching scikit-learn.

Why: a scikit-learn model saved with pickle/joblib only loads in (almost)
the same scikit-learn version it was trained with. The Jetson's apt
scikit-learn is 0.23 (2020); a training laptop has 1.x. Instead we save
the raw weights and do the math ourselves: an MLP is a few matrix
multiplications.

Training side (has scikit-learn):
    from src.inference.classifier import export_mlp
    export_mlp(clf, "checkpoints/gesture_model.npz")                 # MLPClassifier
    export_mlp(clf, "checkpoints/gesture_model.npz", scaler=scaler)  # + StandardScaler
    export_mlp(pipeline, "checkpoints/gesture_model.npz")            # Pipeline(scaler, mlp)

Live side (numpy only):
    model = GestureModel("checkpoints/gesture_model.npz")
    label, confidence = model.predict(features)    # features from normalize()
"""
import json

import numpy as np

from src.inference.hand_landmarks import FEATURE_VERSION, NUM_FEATURES

_ACTIVATIONS = {
    "relu": lambda x: np.maximum(x, 0),
    "tanh": np.tanh,
    "logistic": lambda x: 1 / (1 + np.exp(-x)),
    "identity": lambda x: x,
}


def export_mlp(clf, path, scaler=None):
    """Save a fitted sklearn MLPClassifier (optionally with the StandardScaler
    used before it) as plain arrays. Also accepts Pipeline(StandardScaler, MLP)."""
    if hasattr(clf, "steps"):  # sklearn Pipeline
        steps = [s for _, s in clf.steps]
        clf = steps[-1]
        scalers = [s for s in steps[:-1] if hasattr(s, "mean_")]
        if len(steps) > 2 or (len(steps) == 2 and not scalers):
            raise ValueError("Only Pipeline(StandardScaler, MLPClassifier) is supported")
        scaler = scalers[0] if scalers else None
    if clf.coefs_[0].shape[0] != NUM_FEATURES:
        raise ValueError(f"Model expects {clf.coefs_[0].shape[0]} inputs, features have {NUM_FEATURES}")
    arrays = {}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    if scaler is not None:
        arrays["scaler_mean"] = scaler.mean_
        arrays["scaler_scale"] = scaler.scale_
    meta = {
        "activation": clf.activation,
        "out_activation": clf.out_activation_,
        "classes": [str(c) for c in clf.classes_],
        "n_layers": len(clf.coefs_),
        "feature_version": FEATURE_VERSION,
    }
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    print(f"Exported {meta['n_layers']}-layer MLP, classes {meta['classes']} -> {path}")


class GestureModel:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(d["meta"].item())
        if meta["feature_version"] != FEATURE_VERSION:
            raise ValueError(
                f"Model was trained on features '{meta['feature_version']}', but this code "
                f"makes '{FEATURE_VERSION}'. Retrain or use the matching hand_landmarks.py.")
        self.classes = meta["classes"]
        self._act = _ACTIVATIONS[meta["activation"]]
        self._out = meta["out_activation"]
        self._layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(meta["n_layers"])]
        self._scaler = (d["scaler_mean"], d["scaler_scale"]) if "scaler_mean" in d else None

    def predict_proba(self, features):
        """Probability for each class, in the order of self.classes."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if self._scaler is not None:
            x = (x - self._scaler[0]) / self._scaler[1]
        for W, b in self._layers[:-1]:
            x = self._act(x @ W + b)
        W, b = self._layers[-1]
        z = (x @ W + b)[0]
        if self._out == "softmax":
            e = np.exp(z - z.max())
            return e / e.sum()
        p = 1 / (1 + np.exp(-z[0]))  # 2 classes: sklearn uses a single logistic output
        return np.array([1 - p, p])

    def predict(self, features):
        """(label, confidence) of the most likely class."""
        p = self.predict_proba(features)
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])
