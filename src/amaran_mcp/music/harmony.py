"""Which chord is sounding, whether the harmony leans major or minor, and lifts.

12-bin chroma from a long FFT window (about 0.35 s, so low notes resolve),
smoothed over about a second, matched against the 24 major and minor triads.
A lift is the harmony opening upward: a change onto a major chord, or the
major/minor balance rising quickly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
PITCH_RANGE = (55.0, 2000.0)
WINDOW_SECONDS = 0.35
UPDATE_SECONDS = 0.1
CHROMA_SMOOTHING = 1.0  # seconds
HOLD_SECONDS = 0.5  # a new chord must hold this long to count as a change
CONFIDENT = 0.5
BALANCE_SECONDS = 25.0  # about 12 bars at trance tempo
LIFT_RISE = 0.3  # balance rise within LIFT_WINDOW that counts as a lift
LIFT_WINDOW = 2.0
LIFT_DEBOUNCE = 4.0  # one lift per harmonic move


@dataclass
class Harmony:
    chord: str | None  # e.g. "Am"; None when confidence is low
    root: int | None  # pitch class 0..11 (C = 0)
    major: bool | None
    confidence: float  # 0..1
    balance: float  # -1 all minor .. +1 all major, over the last ~25 s
    change: bool  # the chord changed in this block
    lift: bool  # the harmony opened upward in this block


def _templates() -> tuple[np.ndarray, list[tuple[int, bool]]]:
    rows, labels = [], []
    for root in range(12):
        for major in (True, False):
            row = np.zeros(12)
            row[[root, (root + (4 if major else 3)) % 12, (root + 7) % 12]] = 1
            rows.append(row / np.linalg.norm(row))
            labels.append((root, major))
    return np.array(rows), labels


TEMPLATES, LABELS = _templates()


class HarmonyTracker:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.size = 1 << int(math.ceil(math.log2(rate * WINDOW_SECONDS)))
        freqs = np.fft.rfftfreq(self.size, 1 / rate)
        usable = (freqs >= PITCH_RANGE[0]) & (freqs <= PITCH_RANGE[1])
        self._bins = np.nonzero(usable)[0]
        midi = 69 + 12 * np.log2(freqs[self._bins] / 440.0)
        self._pitch_class = np.round(midi).astype(int) % 12
        self._window = np.hanning(self.size)
        self._buffer = np.zeros(self.size, np.float32)
        self._chroma = np.zeros(12)
        self._now = 0.0
        self._next_update = 0.0
        self._current: tuple[int, bool] | None = None
        self._candidate: tuple[int, bool] | None = None
        self._candidate_since = 0.0
        self._confidence = 0.0
        self._history: deque[tuple[float, float]] = deque()  # (time, +1 major / -1 minor)
        self._balances: deque[tuple[float, float]] = deque()
        self._last_lift = -math.inf

    def feed(self, samples: np.ndarray) -> Harmony:
        n = len(samples)
        dt = n / self.rate
        self._now += dt
        if n >= self.size:
            self._buffer = samples[-self.size :].astype(np.float32)
        else:
            self._buffer = np.concatenate([self._buffer[n:], samples.astype(np.float32)])
        change = lift = False
        if self._now >= self._next_update:
            self._next_update = self._now + UPDATE_SECONDS
            change, lift = self._update(UPDATE_SECONDS)
        balance = self._balance()
        if self._current is None or self._confidence < CONFIDENT:
            return Harmony(None, None, None, self._confidence, balance, change, lift)
        root, major = self._current
        name = NAMES[root] + ("" if major else "m")
        return Harmony(name, root, major, self._confidence, balance, change, lift)

    def _update(self, dt: float) -> tuple[bool, bool]:
        spectrum = np.abs(np.fft.rfft(self._buffer * self._window))[self._bins]
        frame = np.bincount(self._pitch_class, weights=np.log1p(spectrum * 100), minlength=12)
        self._chroma += (frame - self._chroma) * (1 - math.exp(-dt / CHROMA_SMOOTHING))
        norm = np.linalg.norm(self._chroma)
        if norm <= 1e-9:
            self._confidence = 0.0
            return False, False
        scores = TEMPLATES @ (self._chroma / norm)
        best = int(np.argmax(scores))
        # A flat chroma (noise) scores about 0.5 against any triad; a clean
        # triad scores well above 0.85.
        self._confidence = float(max(0.0, min(1.0, (scores[best] - 0.55) / 0.3)))
        if self._confidence < CONFIDENT:
            self._candidate = None
            return False, False
        label = LABELS[best]
        if label == self._current:
            self._candidate = None
            self._history.append((self._now, 1.0 if label[1] else -1.0))
            return False, self._balance_lift()
        if label != self._candidate:
            self._candidate, self._candidate_since = label, self._now
            return False, False
        if self._now - self._candidate_since < HOLD_SECONDS:
            return False, False
        previous, self._current, self._candidate = self._current, label, None
        self._history.append((self._now, 1.0 if label[1] else -1.0))
        onto_major = previous is not None and label[1]
        lift = (onto_major or self._balance_lift()) and self._now - self._last_lift >= LIFT_DEBOUNCE
        if lift:
            self._last_lift = self._now
        return True, lift

    def _balance(self) -> float:
        while self._history and self._history[0][0] < self._now - BALANCE_SECONDS:
            self._history.popleft()
        if not self._history:
            return 0.0
        return float(np.mean([v for _, v in self._history]))

    def _balance_lift(self) -> bool:
        balance = self._balance()
        self._balances.append((self._now, balance))
        while self._balances and self._balances[0][0] < self._now - LIFT_WINDOW:
            self._balances.popleft()
        rise = balance - min(b for _, b in self._balances)
        if rise >= LIFT_RISE and self._now - self._last_lift >= LIFT_DEBOUNCE:
            self._last_lift = self._now
            return True
        return False
