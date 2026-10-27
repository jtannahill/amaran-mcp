"""Score section predictions against labels, per second.

  frame accuracy        share of seconds with the right section
  boundaries            per section: a predicted start within 2 s of a true start
  release F             predicted releases within 2 s of the true drops

Used by train_sections.py for the rules baseline and the model alike.
"""

from __future__ import annotations

from collections.abc import Sequence

TOLERANCE = 2.0
SCORED = ("breakdown", "build", "climax")


def starts(sections: Sequence[str], name: str) -> list[float]:
    return [float(i) for i, s in enumerate(sections) if s == name and (i == 0 or sections[i - 1] != name)]


def match(true: list[float], predicted: list[float], tolerance: float = TOLERANCE) -> dict[str, float]:
    used: set[int] = set()
    hits = 0
    for t in true:
        best = None
        for j, p in enumerate(predicted):
            if j not in used and abs(p - t) <= tolerance and (best is None or abs(p - t) < abs(predicted[best] - t)):
                best = j
        if best is not None:
            used.add(best)
            hits += 1
    precision = hits / len(predicted) if predicted else (1.0 if not true else 0.0)
    recall = hits / len(true) if true else (1.0 if not predicted else 0.0)
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f": f, "true": len(true), "predicted": len(predicted)}


def score(truth: Sequence[str], predicted: Sequence[str], true_releases: list[float],
          predicted_releases: list[float]) -> dict:
    n = min(len(truth), len(predicted))
    accuracy = sum(truth[i] == predicted[i] for i in range(n)) / n if n else 0.0
    return {
        "frame_accuracy": accuracy,
        "boundaries": {name: match(starts(truth[:n], name), starts(predicted[:n], name)) for name in SCORED},
        "release_f": match(true_releases, predicted_releases)["f"],
    }
