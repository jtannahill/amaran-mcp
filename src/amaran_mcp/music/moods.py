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
    # Chord colours in breakdowns: minor harmony uses the cool stops, major the
    # warm ones; the chord root picks the stop. Neighbouring stops sit 20 to 40
    # degrees apart so a chord's wash lands within about a bar.
    cool: tuple[tuple[float, float], ...] = ()
    warm: tuple[tuple[float, float], ...] = ()


MOODS: dict[str, Mood] = {
    "ethereal": Mood(
        "ethereal",
        anchors=((225, 70), (270, 60), (185, 45), (220, 12)),  # deep blue, violet, soft teal, misty white
        brightness=(25, 70),
        max_saturation=75,
        loop_seconds=75,
        cool=((215, 70), (240, 65), (265, 60)),  # deep blue, indigo, violet
        warm=((285, 45), (310, 42), (335, 40), (15, 35)),  # lavender, orchid, rose, peach
    ),
    "deep": Mood(
        "deep",
        anchors=((245, 80), (220, 85), (290, 65)),  # indigo, navy, magenta-violet
        brightness=(20, 55),
        max_saturation=85,
        loop_seconds=90,
        cool=((230, 85), (250, 82), (275, 75)),
        warm=((295, 70), (320, 65), (345, 60)),
    ),
    "dawn": Mood(
        "dawn",
        anchors=((265, 35), (25, 45), (45, 40), (35, 10)),  # lavender, peach, pale gold, warm white
        brightness=(30, 70),
        max_saturation=55,
        loop_seconds=60,
        cool=((250, 40), (270, 36), (290, 34)),
        warm=((15, 45), (35, 45), (50, 40)),  # peach, apricot, pale gold
    ),
}
DEFAULT_MOOD = "ethereal"
