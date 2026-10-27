"""Section model: features, context windows, and the small trained network.

Shared by training (tools/train_sections.py) and runtime (Analyzer), so both
see exactly the same numbers. One vector per second of audio; the model reads
a causal window of the last 16 s (it cannot see the future in real time).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = [
    "energy", "bass", "kick", "tempo_confidence", "onset_strength", "onset_jump",
    "tension", "openness", "balance", "harmony_confidence", "chord_change",
    "vocal", "singing", "speech", "level",
]
# The spectral summary (music/spectral.py) is appended to each second's row.
SPECTRAL = [f"band{i}" for i in range(8)] + ["flux", "flatness", "centroid"]
ROW_WIDTH = len(FEATURES) + len(SPECTRAL)
WINDOW_SECONDS = 16
STATS = 5  # last value, mean, change over the window, standard deviation, max
WINDOW_SIZE = STATS * ROW_WIDTH
SECTIONS = ["intro", "groove", "breakdown", "build", "climax"]


def vector(features: Any) -> np.ndarray:
    """One tick's features (a Features dataclass or its asdict()) as numbers."""

    f = asdict(features) if is_dataclass(features) else features
    beat, harmony, timbre = f["beat"], f["harmony"], f["timbre"]
    return np.array([
        f["energy"],
        f["bass"],
        1.0 if beat.get("kick") else 0.0,
        beat.get("confidence") or 0.0,
        min(3.0, beat.get("onset_strength") or 0.0) / 3,
        np.log1p(min(20.0, beat.get("onset_jump") or 0.0)) / np.log1p(20.0),
        timbre["tension"],
        timbre["openness"],
        harmony.get("balance") or 0.0,
        harmony.get("confidence") or 0.0,
        1.0 if harmony.get("change") else 0.0,
        timbre["vocal"],
        timbre.get("singing") or 0.0,
        timbre.get("speech") or 0.0,
        max(-1.0, min(0.0, (timbre.get("level_db") or -120.0) / 60)),
    ], dtype=np.float32)


class Seconds:
    """Average tick vectors into one per second (events take their maximum)."""

    EVENTS = [FEATURES.index("chord_change"), FEATURES.index("onset_jump")]

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []
        self._elapsed = 0.0

    def add(self, v: np.ndarray, dt: float) -> np.ndarray | None:
        self._rows.append(v)
        self._elapsed += dt
        if self._elapsed < 1.0:
            return None
        rows = np.array(self._rows)
        out = rows.mean(axis=0)
        out[self.EVENTS] = rows[:, self.EVENTS].max(axis=0)
        self._rows, self._elapsed = [], self._elapsed - 1.0
        return out


def window(seconds: np.ndarray, i: int) -> np.ndarray:
    """Context for second i from the last WINDOW_SECONDS rows (padded at the start)."""

    start = i - WINDOW_SECONDS + 1
    rows = seconds[max(0, start) : i + 1]
    if start < 0:
        rows = np.concatenate([np.repeat(rows[:1], -start, axis=0), rows])
    return np.concatenate([rows[-1], rows.mean(axis=0), rows[-1] - rows[0], rows.std(axis=0), rows.max(axis=0)])


class Model:
    """The trained section model, run in numpy (trained by tools/train_sections.py).

    Either gradient-boosted trees (weights["kind"] == "trees") or a small
    multilayer perceptron.
    """

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.classes = [str(c) for c in weights["classes"]]
        self.kind = str(weights.get("kind", "mlp"))
        if self.kind == "trees":
            self.baseline = weights["baseline"]
            self.feature, self.threshold = weights["feature"], weights["threshold"]
            self.left, self.right = weights["left"], weights["right"]
            self.leaf, self.value = weights["leaf"], weights["value"]
            self.roots, self.tree_class = weights["roots"], weights["tree_class"]
            self.depth = int(weights["depth"])
            return
        self.means, self.scales = weights["means"], weights["scales"]
        self.layers = []
        k = 0
        while f"W{k}" in weights:
            self.layers.append((weights[f"W{k}"], weights[f"b{k}"]))
            k += 1

    def proba_batch(self, X: np.ndarray) -> np.ndarray:
        """Probabilities, one row per input, columns in self.classes order."""

        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.kind == "trees":
            raw = np.tile(self.baseline, (len(X), 1))
            rows = np.arange(len(X))
            for root, cls in zip(self.roots, self.tree_class):
                node = np.full(len(X), root)
                for _ in range(self.depth):
                    leaf = self.leaf[node]
                    if leaf.all():
                        break
                    go_left = X[rows, self.feature[node]] <= self.threshold[node]
                    node = np.where(leaf, node, np.where(go_left, self.left[node], self.right[node]))
                raw[:, cls] += self.value[node]
            z = raw
        else:
            h = (X - self.means) / self.scales
            for W, b in self.layers[:-1]:
                h = np.maximum(0.0, h @ W + b)  # ReLU, as scikit-learn's MLPClassifier
            W, b = self.layers[-1]
            z = h @ W + b
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    @classmethod
    def load(cls, path: Path) -> "Model | None":
        try:
            with np.load(path, allow_pickle=False) as data:
                return cls({k: data[k] for k in data.files})
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return None

    def proba(self, x: np.ndarray) -> dict[str, float]:
        return {c: float(v) for c, v in zip(self.classes, self.proba_batch(x)[0])}


# Which section may follow which: the model proposes, the music decides.
ALLOWED = {
    "intro": {"groove", "breakdown", "build"},
    "groove": {"breakdown", "build", "climax"},  # a drop need not follow a marked build
    "breakdown": {"build", "climax", "groove"},
    "build": {"climax", "breakdown"},
    "climax": {"groove", "breakdown", "build"},
}
CONFIDENT = 0.5
HOLD_SECONDS = 2  # the model's top section must hold this long before a switch
MIN_STAY = 8  # seconds a section lasts at least once entered (sections run 20 to 60 s)


class SectionDecoder:
    """Turns per-second model probabilities into sections, the same way in
    training (for scoring) and at runtime."""

    def __init__(self, start: str = "groove") -> None:
        self.section = start
        self.released = False  # the last step entered a climax from a breakdown or build
        self._candidate: str | None = None
        self._held = 0
        self._stayed = MIN_STAY

    def step(self, proba: dict[str, float]) -> str:
        self.released = False
        self._stayed += 1
        if self._stayed < MIN_STAY:
            return self.section
        top = max(proba, key=proba.get)
        if proba[top] < CONFIDENT or top == self.section or top not in ALLOWED.get(self.section, ()):
            self._candidate, self._held = None, 0
            return self.section
        self._held = self._held + 1 if top == self._candidate else 1
        self._candidate = top
        if self._held >= HOLD_SECONDS:
            self.released = top == "climax" and self.section in ("breakdown", "build")
            self.section, self._candidate, self._held, self._stayed = top, None, 0, 0
        return self.section

    def follow(self, section: str) -> None:
        """Keep in step when something else (silence, a fast release) moved the section."""

        if section != self.section:
            self.section, self._candidate, self._held = section, None, 0
