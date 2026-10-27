"""Tempo, bars and 8-bar phrases, so colour moves can land on the music's phrasing.

An onset envelope from the kick band (about 10 ms steps) is autocorrelated to
find the tempo between 100 and 150 BPM. Once confident, a beat grid is locked
and then only nudged, so bar and phrase boundaries stay steady. Bars are 4
beats and phrases 8 bars, counted from the first locked beat (there is no
downbeat detection; in four-on-the-floor music the grid is what matters).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

BPM_RANGE = (100.0, 150.0)
WINDOW_SECONDS = 8.0  # envelope history used for the estimate
ESTIMATE_EVERY = 0.5  # seconds between estimates
MIN_HISTORY = 4.0  # seconds before the first estimate
CONFIDENT = 0.25  # tuned on GiantSteps (417 previews): trance 97% within 4%, house 86%; 0.55 gave 75% / 33%
SPIKY = 0.5  # share of onset energy in the top 10% of frames: kicks are spiky, swells are not
KICK_BAND = (40.0, 200.0)
BEATS_PER_BAR = 4
BARS_PER_PHRASE = 8
PERIOD_SMOOTHING = 0.15  # how far each estimate moves a locked period
PHASE_SMOOTHING = 0.2  # how far each estimate moves a locked grid
PULSE_WINDOW = 2.0  # seconds: the short look that says whether the kick is in right now
KICK_PULSE = 0.5  # beat strength at the remembered tempo that means a kick
KICK_ONSETS = 0.3  # and at least this share of the groove's low-band onset energy
JUMP_WINDOW = 0.5  # seconds compared against the rest of the envelope for onset_jump
GROOVE_REFERENCE = 10.0  # seconds: how the groove's onset energy is remembered
ONSET_SMOOTHING = 0.05  # seconds; onsets are one-frame spikes, and beat lengths
# rarely fall on whole frames (138 BPM is 43.5 frames), so they are widened
# before correlating


@dataclass
class Beat:
    bpm: float | None  # None until confident
    confidence: float  # 0..1
    bar: bool  # a bar boundary fell in this block
    phrase: bool  # an 8-bar boundary fell in this block
    seconds_to_bar: float | None
    seconds_to_phrase: float | None
    # Whether the kick is in right now, from the last 2 s at the remembered
    # tempo: quick where bpm/confidence (8 s window) are slow, and not fooled
    # by a held sub-bass. Seen on a real episode.
    kick: bool = False
    onset_strength: float = 0.0  # low-band onset energy over 2 s / the groove's
    onset_jump: float = 0.0  # last 0.5 s against the 8 s before: a kick slamming back


class TempoTracker:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        self.hop = max(128, rate // 100)
        self.frame_rate = rate / self.hop
        self._window = np.hanning(self.hop * 2)
        freqs = np.fft.rfftfreq(self.hop * 2, 1 / rate)
        self._band = (freqs >= KICK_BAND[0]) & (freqs <= KICK_BAND[1])
        self._carry = np.zeros(0, np.float32)
        self._last_frame = np.zeros(self.hop, np.float32)
        self._last_energy = 0.0
        self._envelope: deque[float] = deque(maxlen=int(WINDOW_SECONDS * self.frame_rate))
        self._frames = 0  # envelope frames so far
        self._next_estimate = MIN_HISTORY
        self.confidence = 0.0
        self._period: float | None = None  # seconds per beat, once locked
        self._grid: float | None = None  # time of beat number 0
        self._beats_reported = -1  # last beat index reported as a boundary
        self._groove_onsets: float | None = None

    @property
    def locked(self) -> bool:
        """A tempo has been found at some point (it is remembered through breakdowns)."""

        return self._period is not None

    @property
    def now(self) -> float:
        return self._frames / self.frame_rate

    def feed(self, samples: np.ndarray) -> Beat:
        data = np.concatenate([self._carry, samples.astype(np.float32)])
        usable = len(data) // self.hop * self.hop
        for start in range(0, usable, self.hop):
            self._frame(data[start : start + self.hop])
        self._carry = data[usable:]
        if self.now >= self._next_estimate:
            self._next_estimate = self.now + ESTIMATE_EVERY
            self._estimate()
        return self._report(len(samples) / self.rate)

    def _frame(self, frame: np.ndarray) -> None:
        both = np.concatenate([self._last_frame, frame]) * self._window
        self._last_frame = frame
        energy = float(np.sum(np.abs(np.fft.rfft(both))[self._band] ** 2))
        level = math.log1p(energy * 1e4)
        self._envelope.append(max(0.0, level - self._last_energy))  # rising energy = onset
        self._last_energy = level
        self._frames += 1

    def _estimate(self) -> None:
        width = max(3, int(ONSET_SMOOTHING * self.frame_rate) | 1)
        kernel = np.hanning(width + 2)[1:-1]
        raw = np.array(self._envelope)
        total = float(np.sum(raw * raw))
        top = np.sort(raw)[-max(1, len(raw) // 10):]
        if total <= 1e-12 or float(np.sum(top * top)) / total < SPIKY:
            self.confidence = 0.0  # a smooth ripple (pads beating) is not a beat
            return
        env = np.convolve(raw, kernel / kernel.sum(), mode="same")
        env = env - env.mean()
        zero = float(np.dot(env, env))
        if zero <= 1e-9:
            self.confidence = 0.0
            return
        n = len(env)
        corr = np.correlate(env, env, mode="full")[n - 1 :] / zero
        low = int(self.frame_rate * 60 / BPM_RANGE[1])
        high = int(math.ceil(self.frame_rate * 60 / BPM_RANGE[0])) + 1
        # Comb: a true beat period also correlates at 2x and 3x its lag.
        scores = []
        for lag in range(low, min(high, n // 3)):
            scores.append((corr[lag] + corr[2 * lag] + corr[3 * lag]) / 3)
        if not scores:
            return
        scores = np.array(scores)
        best = int(np.argmax(scores))
        lag = low + best
        if 0 < best < len(scores) - 1:  # parabolic interpolation between lags
            a, b, c = scores[best - 1], scores[best], scores[best + 1]
            denom = a - 2 * b + c
            if denom:
                lag += 0.5 * (a - c) / denom
        self.confidence = float(max(0.0, min(1.0, scores.max())))
        if self.confidence < CONFIDENT:
            return
        period = lag / self.frame_rate
        phase = self._phase(np.array(self._envelope), period)
        if self._period is None or abs(period - self._period) / self._period > 0.03:
            self._period, self._grid = period, phase
            self._beats_reported = self._beat_index(self.now) - 1
        else:
            self._period += (period - self._period) * PERIOD_SMOOTHING
            # Move the grid a little toward the measured phase (shortest way).
            error = (phase - self._grid + self._period / 2) % self._period - self._period / 2
            self._grid += error * PHASE_SMOOTHING

    def _phase(self, env: np.ndarray, period: float) -> float:
        """Time of a beat: fold the recent envelope at the period, take the peak."""

        frames = period * self.frame_rate
        bins = max(8, int(round(frames)))
        folded = np.zeros(bins)
        start = self._frames - len(env)
        for i, value in enumerate(env):
            folded[int(((start + i) % frames) / frames * bins) % bins] += value
        offset = int(np.argmax(folded)) / bins * period
        return offset

    def _beat_index(self, t: float) -> int:
        return int(math.floor((t - self._grid) / self._period))

    def _kick(self, dt: float) -> tuple[bool, float]:
        frames = int(PULSE_WINDOW * self.frame_rate)
        if self._period is None or len(self._envelope) < frames:
            return False, 0.0
        raw = np.array(self._envelope)[-frames:]
        level = float(np.mean(raw))
        width = max(3, int(ONSET_SMOOTHING * self.frame_rate) | 1)
        kernel = np.hanning(width + 2)[1:-1]
        env = np.convolve(raw, kernel / kernel.sum(), mode="same")
        env = env - env.mean()
        energy = float(np.dot(env, env))
        lag = self._period * self.frame_rate
        whole = int(lag)
        pulse = 0.0
        if energy > 1e-12 and whole + 1 < len(env):
            a = float(np.dot(env[:-whole], env[whole:])) / energy
            b = float(np.dot(env[: -(whole + 1)], env[whole + 1 :])) / energy
            pulse = 2 * (a * (1 - (lag - whole)) + b * (lag - whole))  # one lag spans about half the window
        # Learn the groove's onset energy only while the kick is clearly in,
        # or a breakdown would teach it its own, lower level.
        if self.confidence >= CONFIDENT and pulse >= KICK_PULSE:
            ref = self._groove_onsets
            self._groove_onsets = level if ref is None else ref + (level - ref) * min(1.0, dt / GROOVE_REFERENCE)
        strength = level / self._groove_onsets if self._groove_onsets else 0.0
        # A kick's attack is sharp; sidechain pumping of a held sub rises
        # smoothly at the same rate (a real breakdown cue, seen in testing).
        total = float(np.sum(raw * raw))
        top = np.sort(raw)[-max(1, len(raw) // 10):]
        spiky = total > 1e-12 and float(np.sum(top * top)) / total >= SPIKY
        return pulse >= KICK_PULSE and strength >= KICK_ONSETS and spiky, strength

    def _jump(self) -> float:
        recent = int(JUMP_WINDOW * self.frame_rate)
        if len(self._envelope) < recent * 4:
            return 0.0
        env = np.array(self._envelope)
        before = float(np.mean(env[:-recent]))
        now = float(np.mean(env[-recent:]))
        return now / before if before > 1e-9 else (10.0 if now > 1e-6 else 0.0)

    def _report(self, dt: float) -> Beat:
        kick, strength = self._kick(dt)
        jump = self._jump()
        if self._period is None or self.confidence < CONFIDENT:
            return Beat(None, self.confidence, False, False, None, None, kick, strength, jump)
        index = self._beat_index(self.now)
        bar = phrase = False
        while self._beats_reported < index:
            self._beats_reported += 1
            if self._beats_reported % BEATS_PER_BAR == 0:
                bar = True
            if self._beats_reported % (BEATS_PER_BAR * BARS_PER_PHRASE) == 0:
                phrase = True
        position = (self.now - self._grid) / self._period  # beats since beat 0
        to_bar = (BEATS_PER_BAR - position % BEATS_PER_BAR) * self._period
        per_phrase = BEATS_PER_BAR * BARS_PER_PHRASE
        to_phrase = (per_phrase - position % per_phrase) * self._period
        return Beat(60 / self._period, self.confidence, bar, phrase, to_bar, to_phrase, kick, strength, jump)
