"""From slow audio features to one slow-moving colour.

The slew limits at the end of step() are the guard for music mode's
anti-goals: whatever the analysis says, the light can only change so fast,
so it never flashes, cuts or pulses on a beat.
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

BREAKDOWN_SATURATION = 25.0
DROP_HOLD = 3.0  # seconds the drift pauses while a drop crossfades in
ENERGY_DRIFT_BOOST = 0.3  # up to 30% faster drift at full energy
BRIGHTNESS_LIMITS = (10.0, 100.0)  # where a shifted range is clamped


@dataclass
class Output:
    hue: float
    saturation: float
    brightness: float


def _arc(a: float, b: float) -> float:
    """Signed shortest turn from hue a to hue b, in degrees."""

    return (b - a + 180) % 360 - 180


def _toward(current: float, target: float, limit: float) -> float:
    return current + max(-limit, min(limit, target - current))


class Renderer:
    def __init__(self, mood: Mood, start_anchor: int = 0, brightness_offset: float = 0.0) -> None:
        self.mood = mood
        self.brightness_offset = brightness_offset
        self._position = float(start_anchor % len(mood.anchors))  # place along the palette
        self._hold = 0.0
        hue, saturation = mood.anchors[int(self._position)]
        self._out = Output(hue % 360, min(saturation * 0.6, mood.max_saturation), self._range()[0])

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

    def current_target_hue_after_drop(self) -> float:
        n = len(self.mood.anchors)
        return self.mood.anchors[(int(math.floor(self._position)) + 1) % n][0] % 360

    def begin(self, output: Output) -> None:
        """Start slewing from what the light shows now, instead of the floor."""

        self._out = Output(output.hue % 360, output.saturation, output.brightness)

    def jump_to_anchor(self, index: int) -> None:
        """Crossfade to a palette stop, e.g. when a new track starts."""

        self._position = float(index % len(self.mood.anchors))
        self._hold = DROP_HOLD

    def step(self, f: Features, dt: float) -> Output:
        n = len(self.mood.anchors)
        if f.drop:
            self.jump_to_anchor(int(math.floor(self._position)) + 1)
        if self._hold > 0:
            self._hold = max(0.0, self._hold - dt)
        else:
            speed = n / self.mood.loop_seconds * (1 + ENERGY_DRIFT_BOOST * f.energy)
            self._position = (self._position + speed * dt) % n

        hue, anchor_saturation = self._palette_at(self._position)
        low, high = self._range()
        brightness = low + f.energy * (high - low)
        saturation = min(anchor_saturation * (0.6 + 0.4 * f.energy), self.mood.max_saturation)
        if f.section == "breakdown":
            saturation = min(saturation, BREAKDOWN_SATURATION)

        out = self._out
        out.hue = (out.hue + max(-HUE_RATE * dt, min(HUE_RATE * dt, _arc(out.hue, hue)))) % 360
        out.saturation = _toward(out.saturation, saturation, SATURATION_RATE * dt)
        out.brightness = _toward(out.brightness, brightness, BRIGHTNESS_RATE * dt)
        # A shifted range can move the bounds past the current output; the
        # slew limit still applies on the way back in.
        return Output(out.hue, out.saturation, out.brightness)
