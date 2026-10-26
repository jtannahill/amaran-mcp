"""Turn audio blocks into slow features: energy, music or not, song section.

Everything is smoothed over seconds on purpose. Music mode should follow the
shape of a track (builds, breakdowns, drops), never individual kicks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

ENERGY_RISE = 1.5  # seconds
ENERGY_FALL = 4.0
REFERENCE_SECONDS = 30.0  # long-term level the features are relative to
SILENCE_FLOOR = 0.001  # about -60 dBFS
SILENCE_SECONDS = 10.0
BASS_BAND = (40.0, 150.0)
BASS_SHARE = 0.25  # bass amplitude / overall amplitude that counts as a beat
MUSIC_ONSET = 2.0  # seconds of sustained bass before it counts as music
MUSIC_GRACE = 3.0  # seconds without bass or sustained sound before it stops
STEADY_WINDOW = 2.0  # seconds over which sound must be sustained (not bursty)
STEADY_CV = 0.5  # max coefficient of variation of block level for "sustained"
BREAKDOWN_LEVEL = 0.4  # bass below this share of its reference
BREAKDOWN_SECONDS = 4.0
DROP_LEVEL = 0.8  # bass back above this share of its reference


@dataclass
class Features:
    energy: float  # 0..1, smoothed, 0.5 = the track's usual level
    bass: float  # 0..1, smoothed bass relative to its usual level
    is_music: bool
    section: str  # "silence" | "breakdown" | "playing"
    drop: bool  # True on exactly one update when a drop is detected
    digital_zero: bool  # the last block was exactly 0.0 throughout


class _Follower:
    """One-pole smoother with separate rise and fall times.

    With no start value it takes its first input as is; references use that,
    and are only ever fed values that are already smoothed.
    """

    def __init__(self, rise: float, fall: float, start: float | None = None) -> None:
        self.rise, self.fall = rise, fall
        self.value: float | None = start

    def update(self, x: float, dt: float) -> float:
        if self.value is None:
            self.value = x
        else:
            tau = self.rise if x > self.value else self.fall
            self.value += (x - self.value) * (1 - math.exp(-dt / tau))
        return self.value


class Analyzer:
    def __init__(self, rate: int) -> None:
        self.rate = rate
        # Start from zero: seeding from the first block would latch onto
        # whatever it caught, a kick peak or a gap.
        self._energy = _Follower(ENERGY_RISE, ENERGY_FALL, start=0.0)
        self._bass = _Follower(ENERGY_RISE, ENERGY_FALL, start=0.0)
        self._bass_share = _Follower(0.5, 0.5, start=0.0)
        self._level_ref = _Follower(REFERENCE_SECONDS, REFERENCE_SECONDS)
        self._bass_ref = _Follower(REFERENCE_SECONDS, REFERENCE_SECONDS)
        self._levels: deque[float] = deque()
        self._bass_time = 0.0  # continuous seconds with a beat
        self._no_music_time = 0.0
        self._quiet_time = 0.0
        self._low_bass_time = 0.0
        self.is_music = False
        self.section = "silence"

    def feed(self, samples: np.ndarray) -> Features:
        n = len(samples)
        dt = n / self.rate
        x = samples.astype(np.float64)
        level = float(np.sqrt(np.mean(x * x))) if n else 0.0
        bass = self._bass_amplitude(x)

        # Sustained versus bursty sound, over the last couple of seconds.
        self._levels.append(level)
        while len(self._levels) > max(1, int(STEADY_WINDOW / dt)):
            self._levels.popleft()
        levels = np.array(self._levels)
        mean = levels.mean()
        sustained = mean > SILENCE_FLOOR and levels.std() / mean < STEADY_CV

        share = self._bass_share.update(bass / level if level > SILENCE_FLOOR else 0.0, dt)
        self._bass_time = self._bass_time + dt if share > BASS_SHARE else 0.0
        if self._bass_time >= MUSIC_ONSET:
            self.is_music, self._no_music_time = True, 0.0
        elif self.is_music and not sustained and share <= BASS_SHARE:
            self._no_music_time += dt
            if self._no_music_time >= MUSIC_GRACE:
                self.is_music = False
        else:
            self._no_music_time = 0.0

        energy = self._energy.update(level, dt)
        bass_now = self._bass.update(bass, dt)
        drop = False
        if self.is_music:
            reference = self._level_ref.update(energy, dt)
            # The bass reference only learns while the beat is in: a breakdown
            # must not lower the bar the drop has to clear.
            if self.section != "breakdown" and bass_now > BREAKDOWN_LEVEL * (self._bass_ref.value or bass_now):
                self._bass_ref.update(bass_now, dt)
        else:
            reference = self._level_ref.value or level
        bass_ref = self._bass_ref.value or 0.0

        # Silence covers no sound at all and sound that is not music.
        quiet = level < SILENCE_FLOOR or not self.is_music
        self._quiet_time = self._quiet_time + dt if quiet else 0.0
        if self._quiet_time >= SILENCE_SECONDS:
            self.section = "silence"
        elif self.is_music:
            low = bass_ref > 0 and bass_now < BREAKDOWN_LEVEL * bass_ref
            self._low_bass_time = self._low_bass_time + dt if low else 0.0
            if self.section == "breakdown":
                if bass_now > DROP_LEVEL * bass_ref:
                    self.section, drop = "playing", True
            elif self._low_bass_time >= BREAKDOWN_SECONDS:
                self.section = "breakdown"
            else:
                self.section = "playing"

        def relative(value: float, ref: float) -> float:
            return min(1.0, max(0.0, 0.5 * value / ref)) if ref > 0 else 0.0

        return Features(
            energy=relative(energy, reference),
            bass=relative(bass_now, bass_ref),
            is_music=self.is_music,
            section=self.section,
            drop=drop,
            digital_zero=bool(n) and not np.any(samples),
        )

    def _bass_amplitude(self, x: np.ndarray) -> float:
        if len(x) < 16:
            return 0.0
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
        band = (freqs >= BASS_BAND[0]) & (freqs <= BASS_BAND[1])
        # Scaled so a full-scale sine in the band reads about its RMS.
        return float(np.sqrt(np.sum(spectrum[band] ** 2)) * 2 / np.sum(np.hanning(len(x))) / math.sqrt(2))
