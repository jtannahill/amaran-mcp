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
WINDOW_SECONDS = 16
STATS = 5  # last value, mean, change over the window, standard deviation, max
WINDOW_SIZE = STATS * len(FEATURES)
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
    """A small multilayer perceptron, run in numpy (trained by tools/train_sections.py)."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.means, self.scales = weights["means"], weights["scales"]
        self.layers = []
        k = 0
        while f"W{k}" in weights:
            self.layers.append((weights[f"W{k}"], weights[f"b{k}"]))
            k += 1
        self.classes = [str(c) for c in weights["classes"]]

    @classmethod
    def load(cls, path: Path) -> "Model | None":
        try:
            with np.load(path, allow_pickle=False) as data:
                return cls({k: data[k] for k in data.files})
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return None

    def proba(self, x: np.ndarray) -> dict[str, float]:
        h = (np.asarray(x, dtype=np.float64) - self.means) / self.scales
        for W, b in self.layers[:-1]:
            h = np.maximum(0.0, h @ W + b)  # ReLU, as scikit-learn's MLPClassifier
        W, b = self.layers[-1]
        z = h @ W + b
        z = np.exp(z - z.max())
        p = z / z.sum()
        return {c: float(v) for c, v in zip(self.classes, p)}
