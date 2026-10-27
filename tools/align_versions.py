"""Carry a listening session's marks from one version of a track to another.

Usage: uv run python tools/align_versions.py CUT.wav FULL.wav --marks marks.json --start SECONDS --out full-marks.json

A radio-show cut (A State of Trance, Find Your Harmony) is the same master as
the full release with parts left out and DJ blends at the ends. Both are turned
into music mode's log-mel spectrogram (10 frames a second); each 8 s stretch of
the cut is matched to the place in the full version it sounds most like, and a
mark moves by the offset of the stretch it falls in. Marks in a stretch that
matches nothing (another track blended in) are dropped. The output is a marks
file in session format with times in the full version, for tools/demo_timeline.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

FPS = 10  # music mode's mel frames per second
MATCH = 0.5  # mean frame similarity that counts as the same music


def _normalise(features: np.ndarray) -> np.ndarray:
    z = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-6)


def _candidates(score: np.ndarray, keep: int = 6) -> list[tuple[int, float]]:
    """The best few places for a stretch (local peaks at least a window apart)."""

    order = np.argsort(score)[::-1]
    picked: list[tuple[int, float]] = []
    for j in order:
        if score[j] < MATCH or len(picked) == keep:
            break
        if all(abs(j - k) > 40 for k, _ in picked):
            picked.append((int(j), float(score[j])))
    return picked


def mapping(cut: np.ndarray, full: np.ndarray, window: int = 80, step: int = 20) -> list[tuple[float, float, float]]:
    """(start s, end s, offset s) for each stretch of the cut found in the full version.

    Trance repeats its sections, so a stretch can sound like several places. A cut
    leaves parts out but never reorders them, so of each stretch's candidate places
    the chain kept is the best-scoring one in which time only moves forward."""

    a, b = _normalise(cut), _normalise(full)
    similarity = a @ b.T
    places = similarity.shape[1] - window + 1
    starts, options = [], []
    for start in range(0, len(a) - window + 1, step):
        # The mean similarity along each diagonal: the stretch laid over every place in `full`.
        score = np.zeros(places)
        for k in range(window):
            score += similarity[start + k, k : k + places]
        starts.append(start)
        options.append([(j - start, value) for j, value in _candidates(score / window)])
    # Longest forward-moving chain by total similarity (stretches may be left out).
    tolerance = FPS  # a second of slack between neighbouring stretches
    best: list[list[tuple[float, tuple[int, int] | None]]] = []
    for w, opts in enumerate(options):
        row = []
        for offset, value in opts:
            prior, link = 0.0, None
            for pw in range(w):
                for pc, (poff, _) in enumerate(options[pw]):
                    if poff <= offset + tolerance and best[pw][pc][0] > prior:
                        prior, link = best[pw][pc][0], (pw, pc)
            row.append((prior + value, link))
        best.append(row)
    end = max(((total, (w, c)) for w, row in enumerate(best) for c, (total, _) in enumerate(row)), default=None)
    chain, at = [], end[1] if end else None
    while at is not None:
        w, c = at
        chain.append((starts[w] / FPS, (starts[w] + window) / FPS, options[w][c][0] / FPS))
        at = best[w][c][1]
    return chain[::-1]


def map_time(found: list[tuple[float, float, float]], t: float) -> float | None:
    """`t` in the cut, in the full version; None when no matched stretch covers it."""

    covering = [(abs((s + e) / 2 - t), offset) for s, e, offset in found if s <= t < e]
    if not covering:
        return None
    return t + min(covering)[1]


def main() -> None:
    from benchmark_tempo import RATE, decode

    from amaran_mcp.music.mel import mel_of

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cut", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("--marks", type=Path, required=True, help="the session's marks.json")
    parser.add_argument("--start", type=float, required=True, help="the cut's start in session seconds")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found = mapping(mel_of(decode(args.cut), RATE), mel_of(decode(args.full), RATE))
    print(f"{len(found)} stretches matched; offsets " + ", ".join(f"{s:.0f}s:{o:+.1f}" for s, _, o in found[::6]))
    moved = []
    for entry in json.loads(args.marks.read_text()):
        t = map_time(found, entry["t"] - args.start)
        if t is not None:
            moved.append({**entry, "t": round(t, 2)})
    args.out.write_text(json.dumps(moved, indent=1))
    print(f"{len(moved)} marks carried over")


if __name__ == "__main__":
    main()
