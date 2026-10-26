"""Benchmark music mode's harmony tracking on the Beatport EDM Key Dataset.

Usage: uv run python tools/benchmark_key.py ~/datasets/beatport-edm-key [--genres trance,progressive-house]

Music mode follows chords, not keys, but both rest on the same chroma, so the
key of each excerpt is estimated from the HarmonyTracker's smoothed chroma
(Krumhansl-Kessler profiles) and scored the MIREX way: 1.0 correct, 0.5 a
fifth away, 0.3 relative major/minor, 0.2 parallel major/minor. Also reported:
how often the major/minor mode alone is right, which is what music mode's
cool/warm colour families depend on. The audio is not part of this repository.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import re
import sys

import numpy as np

from benchmark_tempo import BLOCK, RATE, decode

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def parse_key(text: str) -> tuple[int, bool] | None:
    match = re.match(r"\s*([A-Ga-g][#b]?)\s*(major|minor|maj|min)?", text)
    if not match:
        return None
    name = match.group(1).capitalize()
    name = FLATS.get(name, name)
    if name not in NAMES:
        return None
    return NAMES.index(name), (match.group(2) or "major").startswith("maj")


def key_of(chroma: np.ndarray) -> tuple[int, bool]:
    best, best_score = (0, True), -np.inf
    for root in range(12):
        for major, profile in ((True, MAJOR), (False, MINOR)):
            score = np.corrcoef(chroma, np.roll(profile, root))[0, 1]
            if score > best_score:
                best, best_score = (root, major), score
    return best


def estimate(path: Path) -> tuple[int, bool] | None:
    from amaran_mcp.music.harmony import HarmonyTracker

    samples = decode(path)
    tracker = HarmonyTracker(RATE)
    total = np.zeros(12)
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        tracker.feed(samples[start : start + BLOCK])
        total += tracker._chroma
    return key_of(total) if total.any() else None


def mirex(est: tuple[int, bool] | None, truth: tuple[int, bool]) -> float:
    if est is None:
        return 0.0
    (r, m), (tr, tm) = est, truth
    if (r, m) == (tr, tm):
        return 1.0
    if m == tm and (r - tr) % 12 in (5, 7):
        return 0.5
    relative = (tr + 9) % 12 if tm else (tr + 3) % 12
    if m != tm and r == relative:
        return 0.3
    if m != tm and r == tr:
        return 0.2
    return 0.0


def run_one(path: Path) -> tuple[str, tuple[int, bool] | None]:
    try:
        return path.stem, estimate(path)
    except Exception as err:  # a broken file should not stop the run
        print(f"{path.name}: {err}", file=sys.stderr)
        return path.stem, None


def genre_of(root: Path, stem: str) -> str:
    """Genre from the dataset's original metadata (some files are not valid JSON)."""

    track_id = stem.split()[0]
    for meta in (root / "metadata").rglob(f"{track_id} *.json"):
        if meta.name.startswith("._"):
            continue
        match = re.search(r'"genres"\s*:\s*\[\s*\{[^}]*?"name"\s*:\s*"([^"]+)"', meta.read_text(errors="ignore"), re.S)
        if match:
            return match.group(1).lower().replace(" / ", "-").replace(" ", "-")
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--genres", help="comma-separated genres to include (default: all)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    root = args.root.expanduser()
    wanted = set(args.genres.split(",")) if args.genres else None

    truths: dict[str, tuple[int, bool]] = {}
    for key_file in (root / "keys").rglob("*.txt"):
        if key_file.name.startswith("._"):
            continue
        parsed = parse_key(key_file.read_text(errors="ignore"))
        if parsed:
            truths[key_file.stem] = parsed
    audio = {p.stem: p for p in (root / "audio").rglob("*.mp3") if not p.name.startswith("._")}
    stems = sorted(set(truths) & set(audio))
    genres = {stem: genre_of(root, stem) for stem in stems}
    if wanted:
        stems = [s for s in stems if genres[s] in wanted]

    with ProcessPoolExecutor(args.jobs) as pool:
        estimates = dict(pool.map(run_one, [audio[s] for s in stems]))

    by_genre: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for stem in stems:
        est = estimates.get(stem)
        by_genre[genres[stem]].append((mirex(est, truths[stem]), est is not None and est[1] == truths[stem][1]))
    print(f"{'genre':24} {'n':>5} {'mirex':>7} {'exact':>7} {'mode':>7}")
    rows = sorted(by_genre.items(), key=lambda kv: -len(kv[1]))
    total = [r for _, results in rows for r in results]
    for genre, results in rows + [("ALL", total)]:
        n = len(results)
        print(f"{genre:24} {n:5d} {sum(r[0] for r in results) / n:7.1%} "
              f"{sum(r[0] == 1.0 for r in results) / n:7.1%} {sum(r[1] for r in results) / n:7.1%}")


if __name__ == "__main__":
    main()
