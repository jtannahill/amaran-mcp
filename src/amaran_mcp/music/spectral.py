"""A per-second spectral summary for the section model.

Eight log-spaced band energies, each against the track's own recent level
(so a mastering choice does not look like a section change), plus spectral
flux, flatness and centroid. Richer than music mode's hand-built readings,
which proved too coarse for the model to find section boundaries.
"""

from __future__ import annotations

import math

import numpy as np

BANDS = ((30, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 11000))
REFERENCE_SECONDS = 30.0


class Spectral:
    WIDTH = len(BANDS) + 3

    def __init__(self, rate: int) -> None:
        self.rate = rate
        self._rows: list[np.ndarray] = []
        self._elapsed = 0.0
        self._last: np.ndarray | None = None
        self._ref: np.ndarray | None = None
        self._now = 0.0
        self._masks: dict[int, list[np.ndarray]] = {}

    def _bands(self, n: int) -> tuple[np.ndarray, list[np.ndarray]]:
        if n not in self._masks:
            freqs = np.fft.rfftfreq(n, 1 / self.rate)
            self._masks[n] = [freqs] + [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
        freqs, *masks = self._masks[n]
        return freqs, masks

    def feed(self, samples: np.ndarray) -> np.ndarray | None:
        """One block in; a finished second's row out (else None)."""

        n = len(samples)
        if n < 64:
            return None
        dt = n / self.rate
        self._now += dt
        mag = np.abs(np.fft.rfft(samples.astype(np.float64) * np.hanning(n)))
        freqs, masks = self._bands(n)
        power = mag ** 2 + 1e-12
        bands = np.array([math.log10(float(np.sum(power[m])) + 1e-12) for m in masks])
        flux = 0.0 if self._last is None or len(self._last) != len(mag) else float(np.sum(np.maximum(0, mag - self._last)) / (np.sum(mag) + 1e-12))
        self._last = mag
        audible = (freqs > 30) & (freqs < 11000)
        flatness = float(np.exp(np.mean(np.log(power[audible]))) / np.mean(power[audible]))
        centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)) / 11000
        self._rows.append(np.concatenate([bands, [flux, flatness, centroid]]))
        self._elapsed += dt
        if self._elapsed < 1.0:
            return None
        row = np.array(self._rows).mean(axis=0)
        self._rows, self._elapsed = [], self._elapsed - 1.0
        # Bands relative to their running average (a plain average until 30 s of history).
        if self._ref is None:
            self._ref = row[: len(BANDS)].copy()
        span = min(REFERENCE_SECONDS, max(self._now, 1.0))
        self._ref += (row[: len(BANDS)] - self._ref) * min(1.0, 1.0 / span)
        row[: len(BANDS)] = np.clip(row[: len(BANDS)] - self._ref, -3, 3)  # in decades
        return row.astype(np.float32)
