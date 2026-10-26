"""Vocal presence, tension and openness: the qualities beyond loudness.

Vocal: pitch is tracked on what changes against the sustained bed (the
spectrum minus its slow average), so pads cancel out and a voice remains; a
voice is harmonic and its pitch moves (vibrato, glides). A lead synth with
vibrato also reads as vocal; music mode only uses this to size the bloom.

Tension: noisiness and brightness above 3 kHz (risers, sweeps) and onset
density (snare rolls), each against its recent average.

Openness: spectral centroid within its recent range (a filter opening).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

FFT_SIZE_SECONDS = 0.085
F0_RANGE = (165.0, 1000.0)
HARMONICS = 6
BED_SECONDS = 3.0  # the sustained bed: slow average of the spectrum
VOCAL_RISE, VOCAL_FALL = 0.7, 1.5
MOD_WINDOW = 1.0  # seconds of pitch history for movement
MOD_MIN_CENTS, MOD_FULL_CENTS = 8.0, 38.0  # frame-to-frame pitch movement, RMS
MOD_JUMP_CENTS = 150.0  # bigger steps are note changes, not vibrato
REFERENCE_SECONDS = 30.0
TENSION_REFERENCE = 60.0  # longer than a build, so a build stands out against it
TENSION_RISE, TENSION_FALL = 6.0, 2.0  # tension accumulates, then lets go
HARMONIC_FULL = 0.25  # share of the changing spectrum at the pitch's harmonics
TENSION_BAND = (3000.0, 11000.0)
ONSET_WINDOW = 2.0
OPENNESS_SMOOTHING = 1.0


@dataclass
class Timbre:
    vocal: float  # 0..1
    tension: float  # 0..1
    openness: float  # 0..1
    level_db: float  # block level, dBFS


def _ease(value: float, target: float, dt: float, rise: float, fall: float) -> float:
    tau = rise if target > value else fall
    return value + (target - value) * (1 - math.exp(-dt / tau))


class TimbreTracker:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.size = 1 << int(round(math.log2(rate * FFT_SIZE_SECONDS)))
        self._window = np.hanning(self.size)
        self._freqs = np.fft.rfftfreq(self.size, 1 / rate)
        self._buffer = np.zeros(self.size, np.float32)
        self._bed: np.ndarray | None = None
        steps = int(12 * 16 * math.log2(F0_RANGE[1] / F0_RANGE[0]))
        self._f0_grid = F0_RANGE[0] * 2 ** (np.arange(steps) / (12 * 16))
        self._harmonics = np.arange(1, HARMONICS + 1)
        self._pitches: deque[float] = deque()
        self._vocal = 0.0
        self._tension = 0.0
        self._openness = 0.5
        self._refs: dict[str, float] = {}
        self._centroid_low: float | None = None
        self._centroid_high: float | None = None
        self._centroid = 0.0
        self._onsets: deque[float] = deque()
        self._last_flux = 0.0
        self._last_mag: np.ndarray | None = None
        self._now = 0.0

    def feed(self, samples: np.ndarray) -> Timbre:
        n = len(samples)
        dt = n / self.rate
        self._now += dt
        x = samples.astype(np.float32)
        self._buffer = x[-self.size :] if n >= self.size else np.concatenate([self._buffer[n:], x])
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
        level_db = 20 * math.log10(rms) if rms > 1e-9 else -120.0
        mag = np.abs(np.fft.rfft(self._buffer * self._window))
        self._update_vocal(mag, dt)
        self._update_tension(mag, dt)
        self._update_openness(mag, dt)
        return Timbre(self._vocal, self._tension, self._openness, level_db)

    # ── Vocal ────────────────────────────────────────────────────────────────

    def _update_vocal(self, mag: np.ndarray, dt: float) -> None:
        if self._bed is None:
            self._bed = mag.copy()
        residual = np.maximum(0.0, mag - self._bed)
        self._bed += (mag - self._bed) * (1 - math.exp(-dt / BED_SECONDS))
        band = (self._freqs >= F0_RANGE[0]) & (self._freqs <= 4000)
        band_energy = float(np.sum(mag[band] ** 2)) + 1e-12
        changing = float(np.sum(residual[band] ** 2)) + 1e-3 * band_energy
        compressed = np.log1p(residual * 100)
        grid = self._f0_grid[:, None] * self._harmonics[None, :]
        sums = np.interp(grid, self._freqs, compressed) @ (1 / self._harmonics)
        best = int(np.argmax(sums))
        # How much of what is changing sits on that pitch's harmonics.
        harmonic = np.interp(grid[best], self._freqs, residual)
        salience = float(np.sum(harmonic ** 2)) / changing
        self._pitches.append(math.log2(self._f0_grid[best]) * 1200)
        while len(self._pitches) > max(3, int(MOD_WINDOW / dt)):
            self._pitches.popleft()
        steps = np.abs(np.diff(np.array(self._pitches)))
        steps = steps[steps < MOD_JUMP_CENTS]
        movement = float(np.sqrt(np.mean(steps ** 2))) if len(steps) else 0.0
        moving = min(1.0, max(0.0, (movement - MOD_MIN_CENTS) / (MOD_FULL_CENTS - MOD_MIN_CENTS)))
        harmonicity = min(1.0, max(0.0, salience / HARMONIC_FULL))
        self._vocal = _ease(self._vocal, moving * harmonicity, dt, VOCAL_RISE, VOCAL_FALL)

    # ── Tension ──────────────────────────────────────────────────────────────

    def _relative(self, key: str, value: float, dt: float) -> float:
        ref = self._refs.get(key)
        # A plain running average until there is a full reference window of
        # history, so the first seconds do not set an artificially low bar.
        span = min(TENSION_REFERENCE, max(self._now, dt))
        ref = value if ref is None else ref + (value - ref) * min(1.0, dt / span)
        self._refs[key] = ref
        ratio = value / (ref + 1e-9)
        return min(1.0, max(0.0, (ratio - 1.0) / 2.0))  # three times the usual level is full

    def _update_tension(self, mag: np.ndarray, dt: float) -> None:
        power = mag ** 2 + 1e-12
        total = float(np.sum(power))
        band = (self._freqs >= TENSION_BAND[0]) & (self._freqs <= TENSION_BAND[1])
        high_share = float(np.sum(power[band])) / total
        flatness = float(np.exp(np.mean(np.log(power[band]))) / np.mean(power[band]))
        flux = 0.0
        if self._last_mag is not None:
            flux = float(np.sum(np.maximum(0.0, mag - self._last_mag)))
        self._last_mag = mag
        if flux > 1.5 * self._last_flux + 1e-6 and flux > 0:
            self._onsets.append(self._now)
        self._last_flux = flux
        while self._onsets and self._onsets[0] < self._now - ONSET_WINDOW:
            self._onsets.popleft()
        density = len(self._onsets) / ONSET_WINDOW
        parts = (
            self._relative("high", high_share, dt),
            self._relative("flat", flatness * high_share, dt),
            self._relative("onsets", density, dt),
        )
        # A build need not use every device; lean on the strongest one.
        target = 0.6 * max(parts) + 0.4 * float(np.mean(parts))
        self._tension = _ease(self._tension, target, dt, TENSION_RISE, TENSION_FALL)

    # ── Openness ─────────────────────────────────────────────────────────────

    def _update_openness(self, mag: np.ndarray, dt: float) -> None:
        weight = float(np.sum(mag)) + 1e-12
        centroid = float(np.sum(self._freqs * mag)) / weight
        self._centroid = _ease(self._centroid or centroid, centroid, dt, OPENNESS_SMOOTHING, OPENNESS_SMOOTHING)
        c = self._centroid
        decay = 1 - math.exp(-dt / REFERENCE_SECONDS)
        low = c if self._centroid_low is None else min(c, self._centroid_low + (c - self._centroid_low) * decay)
        high = c if self._centroid_high is None else max(c, self._centroid_high + (c - self._centroid_high) * decay)
        self._centroid_low, self._centroid_high = low, high
        self._openness = 0.5 if high - low < 1e-6 else (c - low) / (high - low)
