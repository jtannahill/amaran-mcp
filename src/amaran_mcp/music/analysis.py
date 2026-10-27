"""Turn audio blocks into what music mode follows: energy, sections and events.

Everything is smoothed over seconds on purpose: music mode follows the shape
of a track (intro, groove, breakdown, build, climax), never single kicks.
Tempo, harmony and timbre come from their own trackers; this module combines
them into sections and the two events the light answers: the bloom (a chord
lift inside a breakdown) and the release (the kick returning after a build or
breakdown).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from . import sections as section_model
from .harmony import Harmony, HarmonyTracker
from .tempo import Beat, TempoTracker
from .timbre import Timbre, TimbreTracker

ENERGY_RISE = 1.5  # seconds
ENERGY_FALL = 4.0
REFERENCE_SECONDS = 30.0  # long-term level the features are relative to
SILENCE_FLOOR = 0.001  # about -60 dBFS
SILENCE_SECONDS = 10.0
BASS_BAND = (40.0, 150.0)
BASS_SHARE = 0.25  # bass amplitude / overall amplitude that counts as a beat
MUSIC_ONSET = 2.0  # seconds of sustained bass before it counts as music
TONAL_ONSET = 4.0  # seconds of sustained, chord-confident sound (an intro, pads)
MUSIC_GRACE = 3.0  # seconds without bass or sustained sound before it stops
STEADY_WINDOW = 2.0  # seconds over which sound must be sustained (not bursty)
STEADY_CV = 0.5  # max coefficient of variation of block level for "sustained"
KICK_LEVEL = 0.4  # bass above this share of its reference: the kick is in
KICK_PULSE = 0.3  # bass variation (std / mean over ~1 s) that means a kick, not a held sub
PULSE_WINDOW = 1.0
RELEASE_LEVEL = 0.8  # bass back above this share of its reference: release
RELEASE_JUMP = 3.0  # low-band onsets in the last 0.5 s against the 8 s before
RELEASE_PULSE = 0.45  # bass pulsing needed for a fast release (a swell pumps less)
KICK_ARRIVES = 2.0  # seconds of kick that end an intro
BREAKDOWN_BARS = 4  # kick gone this long after a groove
BUILD_TENSION = 0.5
BUILD_BARS = 2  # tension high and rising this long
CLIMAX_MIN_BARS = 8
CLIMAX_ENERGY = 0.55  # back at about the usual level for CLIMAX_MIN_BARS: groove again
FALLBACK_BAR = 1.8  # seconds per bar when the tempo is unknown (about 133 BPM)


@dataclass
class Features:
    energy: float  # 0..1, smoothed, 0.5 = the track's usual level
    bass: float  # 0..1, smoothed bass relative to its usual level
    is_music: bool
    digital_zero: bool  # the last block was exactly 0.0 throughout
    section: str  # "silence" | "intro" | "groove" | "breakdown" | "build" | "climax"
    section_seconds: float  # time in the current section
    release: bool  # the kick returned after a build or breakdown (one block)
    bloom: bool  # a chord lift inside a breakdown (one block)
    beat: Beat
    harmony: Harmony
    timbre: Timbre


def merge_events(earlier: Features, later: Features) -> Features:
    """The later features, keeping one-block events from the earlier ones."""

    return replace(
        later,
        release=earlier.release or later.release,
        bloom=earlier.bloom or later.bloom,
        beat=replace(later.beat, bar=earlier.beat.bar or later.beat.bar,
                     phrase=earlier.beat.phrase or later.beat.phrase),
        harmony=replace(later.harmony, change=earlier.harmony.change or later.harmony.change,
                        lift=earlier.harmony.lift or later.harmony.lift),
    )


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
    def __init__(self, rate: int, model: "section_model.Model | None" = None) -> None:
        self.rate = rate
        # A trained section model, when present, proposes sections once a
        # second; the state machine still decides what is musically allowed.
        self.model = model
        self._seconds = section_model.Seconds()
        self._history: deque = deque(maxlen=section_model.WINDOW_SECONDS)
        self._decoder = section_model.SectionDecoder()
        self._model_sure = False
        self.tempo = TempoTracker(rate)
        self.harmony = HarmonyTracker(rate)
        self.timbre = TimbreTracker(rate)
        # Start from zero: seeding from the first block would latch onto
        # whatever it caught, a kick peak or a gap.
        self._energy = _Follower(ENERGY_RISE, ENERGY_FALL, start=0.0)
        self._bass = _Follower(ENERGY_RISE, ENERGY_FALL, start=0.0)
        self._bass_share = _Follower(0.5, 0.5, start=0.0)
        self._level_ref = _Follower(REFERENCE_SECONDS, REFERENCE_SECONDS)
        self._bass_ref = _Follower(REFERENCE_SECONDS, REFERENCE_SECONDS)
        self._levels: deque[float] = deque()
        self._tensions: deque[tuple[float, float]] = deque()
        self._bass_blocks: deque[float] = deque()
        self._pulse_cv = 0.0
        self._bass_time = 0.0  # continuous seconds with a beat
        self._tonal_time = 0.0
        self._no_music_time = 0.0
        self._quiet_time = 0.0
        self._kick_time = 0.0
        self._no_kick_time = 0.0
        self._build_time = 0.0
        self._calm_time = 0.0
        self._bar_seconds: float | None = None  # last confident bar length
        self._now = 0.0
        self.is_music = False
        self.section = "silence"
        self._section_since = 0.0

    def _bar(self) -> float:
        return self._bar_seconds or FALLBACK_BAR

    def _enter(self, section: str) -> None:
        if section != self.section:
            self.section, self._section_since = section, self._now

    def feed(self, samples: np.ndarray, classes: dict[str, float] | None = None) -> Features:
        n = len(samples)
        dt = n / self.rate
        self._now += dt
        x = samples.astype(np.float64)
        level = float(np.sqrt(np.mean(x * x))) if n else 0.0
        bass = self._bass_amplitude(x)
        # A kick makes the bass pulse block to block; a breakdown's held
        # sub-bass does not. Seen on a real episode: level alone missed it.
        self._bass_blocks.append(bass)
        while len(self._bass_blocks) > max(2, int(PULSE_WINDOW / dt)):
            self._bass_blocks.popleft()
        recent = np.array(self._bass_blocks)
        self._pulse_cv = float(recent.std() / recent.mean()) if recent.mean() > 0 else 0.0
        pulsing = self._pulse_cv > KICK_PULSE
        beat = self.tempo.feed(samples)
        harmony = self.harmony.feed(samples)
        timbre = self.timbre.feed(samples, classes)
        if beat.bpm:
            self._bar_seconds = 4 * 60 / beat.bpm

        # Sustained versus bursty sound, over the last couple of seconds.
        self._levels.append(level)
        while len(self._levels) > max(1, int(STEADY_WINDOW / dt)):
            self._levels.popleft()
        levels = np.array(self._levels)
        mean = levels.mean()
        sustained = mean > SILENCE_FLOOR and levels.std() / mean < STEADY_CV

        share = self._bass_share.update(bass / level if level > SILENCE_FLOOR else 0.0, dt)
        self._bass_time = self._bass_time + dt if share > BASS_SHARE else 0.0
        tonal = sustained and harmony.confidence >= 0.5
        self._tonal_time = self._tonal_time + dt if tonal else 0.0
        if self._bass_time >= MUSIC_ONSET or self._tonal_time >= TONAL_ONSET:
            self.is_music, self._no_music_time = True, 0.0
        elif self.is_music and not sustained and share <= BASS_SHARE:
            self._no_music_time += dt
            if self._no_music_time >= MUSIC_GRACE:
                self.is_music = False
        else:
            self._no_music_time = 0.0

        energy = self._energy.update(level, dt)
        bass_now = self._bass.update(bass, dt)
        if self.is_music:
            reference = self._level_ref.update(energy, dt)
            # The bass reference only learns while the kick is in: a breakdown
            # must not lower the bar the release has to clear.
            kick_ref = self._bass_ref.value
            if self.section in ("groove", "climax", "intro") and (kick_ref is None or bass_now > KICK_LEVEL * kick_ref):
                if share > BASS_SHARE and pulsing:
                    self._bass_ref.update(bass_now, dt)
        else:
            reference = self._level_ref.value or level
        bass_ref = self._bass_ref.value or 0.0
        relative_energy = min(1.0, max(0.0, 0.5 * energy / reference)) if reference else 0.0

        release = self._sections(dt, level, bass_now, bass_ref, share, pulsing, relative_energy, timbre, beat)
        if self.model is not None:
            release = self._model_step(release, relative_energy, bass_now, bass_ref, beat, harmony, timbre, dt, n) or release
        bloom = harmony.lift and self.section == "breakdown"

        return Features(
            energy=relative_energy,
            bass=min(1.0, max(0.0, 0.5 * bass_now / bass_ref)) if bass_ref else 0.0,
            is_music=self.is_music,
            digital_zero=bool(n) and not np.any(samples),
            section=self.section,
            section_seconds=self._now - self._section_since,
            release=release,
            bloom=bloom,
            beat=beat,
            harmony=harmony,
            timbre=timbre,
        )

    def _model_step(self, release: bool, energy: float, bass_now: float, bass_ref: float, beat: Beat,
                    harmony: Harmony, timbre: Timbre, dt: float, n: int) -> bool:
        """Once a second, let the model move the section (within what is allowed)."""

        if self.section in ("silence", "intro") or not self.is_music:
            self._decoder.follow("groove")
            return False
        self._decoder.follow(self.section)
        tick = Features(energy, min(1.0, 0.5 * bass_now / bass_ref) if bass_ref else 0.0, self.is_music,
                        False, self.section, 0.0, release, False, beat, harmony, timbre)
        row = self._seconds.add(section_model.vector(tick), dt)
        if row is None:
            return False
        self._history.append(row)
        rows = np.array(self._history)
        proba = self.model.proba(section_model.window(rows, len(rows) - 1))
        self._model_sure = max(proba.values()) >= section_model.CONFIDENT
        proposed = self._decoder.step(proba)
        if proposed != self.section:
            self._enter(proposed)
            if self._decoder.released:
                self._calm_time = self._no_kick_time = 0.0
                return True
        return False

    def _sections(self, dt: float, level: float, bass_now: float, bass_ref: float, share: float,
                  pulsing: bool, energy: float, timbre: Timbre, beat: Beat) -> bool:
        """Advance the section state machine; True on a release."""

        # Silence covers no sound at all and sound that is not music.
        quiet = level < SILENCE_FLOOR or not self.is_music
        self._quiet_time = self._quiet_time + dt if quiet else 0.0
        if self._quiet_time >= SILENCE_SECONDS:
            self._enter("silence")
            return False
        if not self.is_music:
            return False

        if self.tempo.locked:
            # The short look at the remembered tempo: a held sub-bass is not a kick.
            kick = beat.kick
        else:
            kick = pulsing and share > BASS_SHARE and (bass_ref == 0 or bass_now >= KICK_LEVEL * bass_ref)
        self._kick_time = self._kick_time + dt if kick else 0.0
        self._no_kick_time = 0.0 if kick else self._no_kick_time + dt

        self._tensions.append((self._now, timbre.tension))
        while self._tensions and self._tensions[0][0] < self._now - BUILD_BARS * self._bar():
            self._tensions.popleft()
        rising = timbre.tension >= BUILD_TENSION and timbre.tension > self._tensions[0][1] + 0.02
        self._build_time = self._build_time + dt if rising else 0.0

        section = self.section
        if self.model is not None and self._model_sure and section in ("groove", "breakdown", "build", "climax"):
            # The model owns changes among these while it is sure (start-up,
            # silence and intro stay with the rules: SSME has no intro); the fast release
            # (a kick slamming back) still applies, it is quicker than a second.
            if section in ("breakdown", "build"):
                bass_back = bass_ref and bass_now > RELEASE_LEVEL * bass_ref
                if beat.onset_jump >= RELEASE_JUMP and self._pulse_cv >= RELEASE_PULSE and bass_back:
                    self._enter("climax")
                    self._calm_time = self._no_kick_time = 0.0
                    return True
            return False
        if section == "silence":
            self._enter("groove" if self._kick_time >= KICK_ARRIVES else "intro")
        elif section == "intro":
            if self._kick_time >= KICK_ARRIVES:
                self._enter("groove")
        elif section in ("breakdown", "build"):
            bass_back = bass_ref and bass_now > RELEASE_LEVEL * bass_ref
            # Within half a second of a real drop; a breakdown's bass swell
            # (seen on a real episode) jumps less and pumps more gently.
            slam = beat.onset_jump >= RELEASE_JUMP and self._pulse_cv >= RELEASE_PULSE and bass_back
            if slam or (kick and bass_back):
                self._enter("climax")
                # The kick is back by definition, even before the slower test
                # confirms it; otherwise the breakdown's no-kick time sends the
                # section straight back.
                self._calm_time = self._no_kick_time = 0.0
                return True
            if section == "breakdown" and self._build_time >= BUILD_BARS * self._bar():
                self._enter("build")
        elif section in ("groove", "climax"):
            if self._no_kick_time >= BREAKDOWN_BARS * self._bar():
                self._enter("breakdown")
            elif self._build_time >= BUILD_BARS * self._bar():
                self._enter("build")
            elif section == "climax":
                self._calm_time = self._calm_time + dt if energy <= CLIMAX_ENERGY else 0.0
                long_enough = self._now - self._section_since >= CLIMAX_MIN_BARS * self._bar()
                if long_enough and self._calm_time >= CLIMAX_MIN_BARS * self._bar():
                    self._enter("groove")
        return False

    def _bass_amplitude(self, x: np.ndarray) -> float:
        if len(x) < 16:
            return 0.0
        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        freqs = np.fft.rfftfreq(len(x), 1 / self.rate)
        band = (freqs >= BASS_BAND[0]) & (freqs <= BASS_BAND[1])
        # Scaled so a full-scale sine in the band reads about its RMS.
        return float(np.sqrt(np.sum(spectrum[band] ** 2)) * 2 / np.sum(np.hanning(len(x))) / math.sqrt(2))
