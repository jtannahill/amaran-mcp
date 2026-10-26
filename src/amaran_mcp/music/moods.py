"""Music mode moods: a palette, a brightness range and how fast it drifts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mood:
    name: str
    anchors: tuple[tuple[float, float], ...]  # (hue degrees, saturation %) palette stops
    brightness: tuple[float, float]  # floor, ceiling in %
    max_saturation: float
    loop_seconds: float  # one trip around the palette


MOODS: dict[str, Mood] = {
    "ethereal": Mood(
        "ethereal",
        anchors=((225, 70), (270, 60), (185, 45), (220, 12)),  # deep blue, violet, soft teal, misty white
        brightness=(25, 70),
        max_saturation=75,
        loop_seconds=75,
    ),
    "deep": Mood(
        "deep",
        anchors=((245, 80), (220, 85), (290, 65)),  # indigo, navy, magenta-violet
        brightness=(20, 55),
        max_saturation=85,
        loop_seconds=90,
    ),
    "dawn": Mood(
        "dawn",
        anchors=((265, 35), (25, 45), (45, 40), (35, 10)),  # lavender, peach, pale gold, warm white
        brightness=(30, 70),
        max_saturation=55,
        loop_seconds=60,
    ),
}
DEFAULT_MOOD = "ethereal"
