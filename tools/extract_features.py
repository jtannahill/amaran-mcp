"""Per-second feature sequences for training and scoring the section model.

Usage: uv run python tools/extract_features.py ~/datasets/ssme --out ~/datasets/features/ssme [--jobs 6]

For each track: decode to 24 kHz mono, run Apple's sound classifier on the
file (the capture helper's --classify-file), run music mode's Analyzer with
those classes, and save one .npz: vectors (seconds x FEATURES), the rules'
section and releases, the track's tempo, and the SSME labels per second.
Cached: tracks already extracted are skipped.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_tempo import BLOCK, RATE, decode  # noqa: E402
import ssme  # noqa: E402


def classes_for(path: Path) -> list[tuple[float, dict]]:
    from amaran_mcp.music.capture import helper_path

    out = subprocess.run([str(helper_path()), "--classify-file", str(path)], capture_output=True, text=True)
    rows = []
    for line in out.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rows.append((float(d.pop("t")), d))
    return rows


def analyse(path: Path) -> dict[str, np.ndarray]:
    from amaran_mcp.music import sections
    from amaran_mcp.music.analysis import Analyzer
    from amaran_mcp.music.spectral import Spectral

    samples = decode(path)
    classes = classes_for(path)
    analyzer = Analyzer(RATE)
    seconds = sections.Seconds()
    spectral = Spectral(RATE)
    spectral_row = None
    vectors, rules, releases, bpms = [], [], [], []
    k = 0
    current = None
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        t = (start + BLOCK) / RATE
        while k < len(classes) and classes[k][0] + 1.5 <= t:  # a window's result is known at its end
            current = classes[k][1]
            k += 1
        block = samples[start : start + BLOCK]
        f = analyzer.feed(block, current)
        latest = spectral.feed(block)  # every block, so both accumulators close the same second
        if latest is not None:
            spectral_row = latest
        if f.release:
            releases.append(t)
        if f.beat.bpm:
            bpms.append(f.beat.bpm)
        row = seconds.add(sections.vector(f), BLOCK / RATE)
        if row is not None:
            if spectral_row is None:  # the two accumulators close the same block
                spectral_row = np.zeros(Spectral.WIDTH, np.float32)
            vectors.append(np.concatenate([row, spectral_row]))
            spectral_row = None
            rules.append(f.section)
    return {
        "vectors": np.array(vectors, dtype=np.float32),
        "rules": np.array(rules),
        "rule_releases": np.array(releases, dtype=np.float32),
        "bpm": np.array(float(np.median(bpms)) if bpms else 0.0),
    }


def truth_for(mat: Path, seconds: int, bpm: float | None) -> np.ndarray:
    truth = np.array(["groove"] * seconds, dtype=object)
    for seg in ssme.labels(mat, float(seconds), bpm):
        truth[int(seg["start"]) : int(np.ceil(seg["end"]))] = seg["section"]
    return truth.astype(str)


def relabel(target: Path, mat: Path) -> None:
    """Recompute the labels of an extracted track (after a change to ssme.labels)."""

    with np.load(target, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["truth"] = truth_for(mat, len(data["vectors"]), float(data["bpm"]) or None)
    data["true_releases"] = np.array(ssme.releases(mat), dtype=np.float32)
    np.savez_compressed(target, **data)


def run_one(args: tuple[Path, Path, Path]) -> str:
    mp3, mat, out_dir = args
    target = out_dir / f"{mp3.stem}.npz"
    if target.exists():
        return f"{mp3.stem}: cached"
    try:
        data = analyse(mp3)
        duration = float(len(data["vectors"]))
        bpm = float(data["bpm"]) or None
        np.savez_compressed(target, **data, truth=truth_for(mat, int(duration), bpm),
                            true_releases=np.array(ssme.releases(mat), dtype=np.float32))
        return f"{mp3.stem}: {int(duration)} s, {bpm or 0:.0f} BPM"
    except Exception as err:  # one bad file should not stop the run
        return f"{mp3.stem}: FAILED {err}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--relabel", action="store_true", help="only recompute labels of extracted tracks")
    args = parser.parse_args()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = ssme.tracks(args.root.expanduser())[: args.limit]
    if args.relabel:
        for mp3, mat in pairs:
            target = out_dir / f"{mp3.stem}.npz"
            if target.exists():
                relabel(target, mat)
        print(f"relabelled {len(pairs)} tracks")
        return
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, message in enumerate(pool.map(run_one, [(mp3, mat, out_dir) for mp3, mat in pairs]), 1):
            print(f"[{i}/{len(pairs)}] {message}", flush=True)


if __name__ == "__main__":
    main()
