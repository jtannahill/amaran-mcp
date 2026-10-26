"""Benchmark music mode's tempo tracker on the GiantSteps tempo dataset.

Usage: uv run python tools/benchmark_tempo.py ~/datasets/giantsteps-tempo-dataset [--genres trance,psy-trance]

Scores the standard way (MIREX): Accuracy 1 = within 4% of the annotated
tempo; Accuracy 2 = within 4% allowing double, half, triple and third tempo.
Annotations are the 2018 crowdsourced ones (annotations_v2) where present,
else the original ones. The audio is not part of this repository.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys

import numpy as np

RATE = 24000
BLOCK = 1024
TOLERANCE = 0.04


def decode(path: Path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4")


def estimate(path: Path) -> float | None:
    """The tracker's settled tempo: the most common confident reading in the second half."""

    from amaran_mcp.music.tempo import TempoTracker

    samples = decode(path)
    tracker = TempoTracker(RATE)
    readings = []
    half = len(samples) // 2
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        beat = tracker.feed(samples[start : start + BLOCK])
        if start >= half and beat.bpm:
            readings.append(round(beat.bpm, 1))
    if not readings:
        return None
    return Counter(readings).most_common(1)[0][0]


def annotation(root: Path, stem: str) -> float | None:
    v2 = root / "annotations_v2" / "tempo" / f"{stem}.bpm"
    v1 = root / "annotations" / "tempo" / f"{stem}.bpm"
    for path in (v2, v1):
        if path.exists():
            text = path.read_text().split()
            if text:
                return float(text[0])
    return None


def score(estimated: float | None, truth: float) -> tuple[bool, bool]:
    if estimated is None:
        return False, False
    close = lambda target: abs(estimated - target) <= TOLERANCE * target
    acc1 = close(truth)
    acc2 = any(close(truth * k) for k in (1, 2, 0.5, 3, 1 / 3))
    return acc1, acc2


def run_one(args: tuple[Path, Path]) -> tuple[str, float | None]:
    root, mp3 = args
    try:
        return mp3.name, estimate(mp3)
    except Exception as err:  # a broken file should not stop the run
        print(f"{mp3.name}: {err}", file=sys.stderr)
        return mp3.name, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--genres", help="comma-separated genres to include (default: all)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    root = args.root.expanduser()
    wanted = set(args.genres.split(",")) if args.genres else None

    items = []
    for mp3 in sorted((root / "audio").glob("*.mp3")):
        stem = mp3.stem
        genre_file = root / "annotations" / "genre" / f"{stem}.genre"
        genre = genre_file.read_text().strip() if genre_file.exists() else "unknown"
        truth = annotation(root, stem)
        if truth is None or (wanted and genre not in wanted):
            continue
        items.append((mp3, genre, truth))

    with ProcessPoolExecutor(args.jobs) as pool:
        estimates = dict(pool.map(run_one, [(root, mp3) for mp3, _, _ in items]))

    by_genre: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    misses = []
    for mp3, genre, truth in items:
        est = estimates.get(mp3.name)
        acc1, acc2 = score(est, truth)
        in_range = 100 <= truth <= 150
        by_genre[genre].append((acc1, acc2, in_range))
        if in_range and not acc1:
            misses.append((genre, mp3.name, truth, est))

    print(f"{'genre':22} {'n':>4} {'acc1':>6} {'acc2':>6}   {'acc1 (100-150 BPM)':>18}")
    rows = sorted(by_genre.items(), key=lambda kv: -len(kv[1]))
    total = [r for _, results in rows for r in results]
    for genre, results in rows + [("ALL", total)]:
        n = len(results)
        a1 = sum(r[0] for r in results) / n
        a2 = sum(r[1] for r in results) / n
        ranged = [r for r in results if r[2]]
        a1r = sum(r[0] for r in ranged) / len(ranged) if ranged else float("nan")
        print(f"{genre:22} {n:4d} {a1:6.1%} {a2:6.1%}   {a1r:18.1%} (n={len(ranged)})")
    if misses:
        print("\nMisses inside 100-150 BPM (genre, file, truth, estimate):")
        for genre, name, truth, est in misses[:40]:
            print(f"  {genre:20} {name:22} {truth:6.1f} {est if est is not None else '-'}")


if __name__ == "__main__":
    main()
