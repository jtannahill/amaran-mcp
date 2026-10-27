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
        anchors=((225, 85), (270, 78), (190, 70), (300, 65)),  # deep blue, violet, teal, orchid
        brightness=(25, 70),
        max_saturation=90,
        loop_seconds=75,
        cool=((215, 85), (240, 82), (265, 78)),  # deep blue, indigo, violet
        warm=((290, 72), (310, 70), (335, 66), (15, 60)),  # lavender, orchid, rose, peach
    ),
    "deep": Mood(
        "deep",
        anchors=((245, 90), (220, 92), (290, 80)),  # indigo, navy, magenta-violet
        brightness=(20, 55),
        max_saturation=95,
        loop_seconds=90,
        cool=((230, 92), (250, 90), (275, 85)),
        warm=((295, 82), (320, 78), (345, 74)),
    ),
    "dawn": Mood(
        "dawn",
        anchors=((265, 62), (25, 72), (45, 66), (330, 60)),  # lavender, peach, gold, rose
        brightness=(30, 70),
        max_saturation=80,
        loop_seconds=60,
        cool=((250, 66), (270, 62), (290, 60)),
        warm=((15, 72), (35, 72), (50, 66)),  # peach, apricot, gold
    ),
}
DEFAULT_MOOD = "ethereal"
