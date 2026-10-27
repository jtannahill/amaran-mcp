"""From audio features to one slow-moving colour, section by section.

  intro      restrained: the lower third of the brightness range
  groove     drifts through the mood's palette and breathes with the energy
  breakdown  pale for its first bars, then the harmony drives it: minor
             chords take the mood's cool stops, major the warm ones, each
             chord's wash starting on the bar
  bloom      a chord lift in a breakdown: the light warms, swells and turns
             luminous, more so with a vocal, then eases back
  build      gathers: the drift stops, the colour deepens, brightness holds
  release    opens out to the ceiling; the climax stays full

The slew limits at the end of step() guard music mode's anti-goals: whatever
the analysis says, the light can only change so fast, so it never flashes,
cuts or pulses on a beat.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .analysis import Features
from .moods import Mood

# Fastest allowed change, per second.
HUE_RATE = 30.0
SATURATION_RATE = 20.0
BRIGHTNESS_RATE = 12.0

BREAKDOWN_SATURATION = 40.0  # the paler opening of a breakdown (still a colour)
PALE_BARS = 2
FALLBACK_BAR = 1.8  # seconds per bar when the tempo is unknown
PALE_ANCHOR = 30.0  # stops below this saturation are for drifting, not for releases
DROP_HOLD = 3.0  # seconds the drift pauses while a release crossfades in
ENERGY_DRIFT_BOOST = 0.3  # up to 30% faster drift at full energy
CLIMAX_DRIFT = 1.3
BRIGHTNESS_LIMITS = (10.0, 100.0)  # where a shifted range is clamped
BREAKDOWN_LEVEL = 0.8  # breakdowns sit a little lower: quieter, more intimate
WANDER_DEGREES, WANDER_SECONDS = 6.0, 30.0  # slow drift around a held chord colour
BLOOM_RISE, BLOOM_FALL = 1.5, 1.5  # seconds (time constants)
BLOOM_HOLD = 4.0  # seconds a bloom lasts at least
BLOOM_MAX_PHRASES = 2
BLOOM_VOCAL_END = 0.3  # the bloom eases back once vocal presence falls below this
# Apple's classifier reads real sung vocals at about 0.25 to 0.45 (90th percentile 0.42
# on Miss You, 0.37 on Through It All) and instrumentals near 0.01 to 0.10, so the
# renderer rescales it: at or below VOCAL_FLOOR no voice, at VOCAL_FULL a full one.
# The section model keeps the raw reading it was trained on, and the fallback
# estimate (no classifier) is left on its own scale.
VOCAL_FLOOR, VOCAL_FULL = 0.08, 0.40
BLOOM_SATURATION = 6.0  # luminous through brightness, barely whiter
OPEN_SATURATION, OPEN_BRIGHTNESS = 6.0, 10.0  # at full openness


@dataclass
class Output:
    hue: float
    saturation: float
    brightness: float


def vocal_presence(vocal: float) -> float:
    """The classifier's vocal reading on the scale real singing actually reaches."""

    return max(0.0, min(1.0, (vocal - VOCAL_FLOOR) / (VOCAL_FULL - VOCAL_FLOOR)))


def _voice(f: Features) -> float:
    """How much voice there is: the classifier's reading rescaled; the fallback
    estimate (no classifier, so no singing reading) is already on its own scale."""

    return vocal_presence(f.timbre.vocal) if f.timbre.singing is not None else f.timbre.vocal


def _arc(a: float, b: float) -> float:
    """Signed shortest turn from hue a to hue b, in degrees."""

    return (b - a + 180) % 360 - 180


def _toward(current: float, target: float, limit: float) -> float:
    return current + max(-limit, min(limit, target - current))


def _ease(value: float, target: float, dt: float, tau: float) -> float:
    return value + (target - value) * (1 - math.exp(-dt / tau))


class Renderer:
    def __init__(self, mood: Mood, start_anchor: int = 0, brightness_offset: float = 0.0) -> None:
        self.mood = mood
        self.brightness_offset = brightness_offset
        self._position = float(start_anchor % len(mood.anchors))  # place along the palette
        self._hold = 0.0
        self._now = 0.0
        hue, saturation = mood.anchors[int(self._position)]
        self._out = Output(hue % 360, min(saturation * 0.6, mood.max_saturation), self._range()[0])
        self._chord: tuple[float, float] | None = None  # the breakdown colour now showing
        self._next_chord: tuple[float, float] | None = None
        self._next_chord_at = 0.0
        self._bloom = 0.0  # 0..1 envelope
        self._bloom_target = 0.0
        self._bloom_since = -math.inf
        self._build_brightness: float | None = None

    def _range(self) -> tuple[float, float]:
        low, high = (b + self.brightness_offset for b in self.mood.brightness)
        lo, hi = BRIGHTNESS_LIMITS
        return max(lo, min(hi, low)), max(lo, min(hi, high))

    def _palette_at(self, position: float) -> tuple[float, float]:
        n = len(self.mood.anchors)
        i = int(math.floor(position)) % n
        frac = position - math.floor(position)
        (h1, s1), (h2, s2) = self.mood.anchors[i], self.mood.anchors[(i + 1) % n]
        return (h1 + _arc(h1, h2) * frac) % 360, s1 + (s2 - s1) * frac

    def _drop_anchor(self) -> int:
        """The next stop with real colour: a release should never land on white."""

        n = len(self.mood.anchors)
        here = int(math.floor(self._position))
        for step in range(1, n + 1):
            index = (here + step) % n
            if self.mood.anchors[index][1] >= PALE_ANCHOR:
                return index
        return (here + 1) % n

    def current_target_hue_after_drop(self) -> float:
        return self.mood.anchors[self._drop_anchor()][0] % 360

    def begin(self, output: Output) -> None:
        """Start slewing from what the light shows now, instead of the floor."""

        self._out = Output(output.hue % 360, output.saturation, output.brightness)

    def jump_to_anchor(self, index: int) -> None:
        """Crossfade to a palette stop, e.g. when a new episode starts."""

        self._position = float(index % len(self.mood.anchors))
        self._hold = DROP_HOLD

    # ── Pieces ───────────────────────────────────────────────────────────────

    def _chord_colour(self, f: Features, family_override: str | None = None) -> tuple[float, float] | None:
        family = family_override or ("warm" if f.harmony.balance >= 0 else "cool")
        stops = getattr(self.mood, family) or self.mood.anchors
        if f.harmony.root is None:
            return None
        return stops[f.harmony.root % len(stops)]

    def _drift(self, f: Features, dt: float, speed_factor: float = 1.0) -> tuple[float, float]:
        n = len(self.mood.anchors)
        if self._hold > 0:
            self._hold = max(0.0, self._hold - dt)
        else:
            speed = n / self.mood.loop_seconds * (1 + ENERGY_DRIFT_BOOST * f.energy) * speed_factor
            self._position = (self._position + speed * dt) % n
        return self._palette_at(self._position)

    def _breakdown(self, f: Features, dt: float) -> tuple[float, float] | None:
        """The chord colour to show, once the pale opening is over."""

        bar = 4 * 60 / f.beat.bpm if f.beat.bpm else FALLBACK_BAR
        if f.section_seconds < PALE_BARS * bar:
            self._chord = None
            return None
        colour = self._chord_colour(f)
        if colour is None:
            return self._chord
        if self._chord is None:
            self._chord, self._next_chord = colour, None
        elif colour != self._chord and colour != self._next_chord:
            # A new chord: its wash starts on the next bar (now, without tempo).
            self._next_chord = colour
            self._next_chord_at = self._now + (f.beat.seconds_to_bar or 0.0)
        if self._next_chord is not None and self._now >= self._next_chord_at:
            self._chord, self._next_chord = self._next_chord, None
        return self._chord

    def _update_bloom(self, f: Features, dt: float) -> None:
        if f.bloom:
            self._bloom_target, self._bloom_since = 1.0, self._now
        elif self._bloom_target > 0:
            phrase = 32 * 60 / f.beat.bpm if f.beat.bpm else 8 * FALLBACK_BAR
            held = self._now - self._bloom_since
            vocal_gone = _voice(f) < BLOOM_VOCAL_END and held >= BLOOM_HOLD
            if vocal_gone or held >= BLOOM_MAX_PHRASES * phrase or f.section not in ("breakdown", "build"):
                self._bloom_target = 0.0
        tau = BLOOM_RISE if self._bloom_target > self._bloom else BLOOM_FALL
        self._bloom = _ease(self._bloom, self._bloom_target, dt, tau)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, f: Features, dt: float) -> Output:
        self._now += dt
        if f.release:
            self.jump_to_anchor(self._drop_anchor())
        low, high = self._range()
        section = f.section
        if section != "build":
            self._build_brightness = None
        self._update_bloom(f, dt)

        if section == "breakdown":
            hue, anchor_saturation = self._drift(f, dt, speed_factor=0.3)
            chord = self._breakdown(f, dt)
            brightness = low + f.energy * (high - low) * BREAKDOWN_LEVEL
            if chord is None:
                saturation = min(anchor_saturation, BREAKDOWN_SATURATION)
            else:
                wander = WANDER_DEGREES * math.sin(2 * math.pi * self._now / WANDER_SECONDS)
                hue, saturation = (chord[0] + wander) % 360, chord[1] * (0.85 + 0.15 * f.energy)
        elif section == "build":
            # Gathering: the drift stops as tension rises; colour deepens; hold.
            hue, anchor_saturation = self._drift(f, dt, speed_factor=max(0.0, 1 - f.timbre.tension / 0.8))
            if self._build_brightness is None:
                self._build_brightness = self._out.brightness
            brightness = self._build_brightness
            saturation = anchor_saturation + (self.mood.max_saturation - anchor_saturation) * f.timbre.tension
        elif section == "climax":
            hue, anchor_saturation = self._drift(f, dt, speed_factor=CLIMAX_DRIFT)
            brightness = high - (1 - f.energy) * (high - low) * 0.3
            saturation = max(anchor_saturation, self.mood.max_saturation * 0.9)
        else:  # intro, groove
            hue, anchor_saturation = self._drift(f, dt)
            span = (high - low) / 3 if section == "intro" else high - low
            brightness = low + f.energy * span
            saturation = anchor_saturation * (0.85 + 0.15 * f.energy)

        if self._bloom > 0.01:
            warm = self._chord_colour(f, "warm") or (self.mood.warm or self.mood.anchors)[0]
            hue = (hue + _arc(hue, warm[0]) * self._bloom) % 360
            headroom = high - brightness
            brightness += headroom * (0.3 + 0.3 * _voice(f)) * self._bloom
            saturation -= BLOOM_SATURATION * self._bloom

        opening = section in ("groove", "climax") or (section == "breakdown" and self._chord is not None)
        if opening:
            saturation -= OPEN_SATURATION * f.timbre.openness
            brightness += OPEN_BRIGHTNESS * f.timbre.openness
        if section == "intro":
            brightness = min(brightness, low + (high - low) / 3)

        brightness = max(low, min(high, brightness))
        saturation = max(0.0, min(saturation, self.mood.max_saturation))

        out = self._out
        out.hue = (out.hue + max(-HUE_RATE * dt, min(HUE_RATE * dt, _arc(out.hue, hue)))) % 360
        out.saturation = _toward(out.saturation, saturation, SATURATION_RATE * dt)
        out.brightness = _toward(out.brightness, brightness, BRIGHTNESS_RATE * dt)
        # A shifted range can move the bounds past the current output; the
        # slew limit still applies on the way back in.
        return Output(out.hue, out.saturation, out.brightness)
