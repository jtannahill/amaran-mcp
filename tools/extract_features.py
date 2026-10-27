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

    samples = decode(path)
    classes = classes_for(path)
    analyzer = Analyzer(RATE)
    seconds = sections.Seconds()
    vectors, rules, releases, bpms = [], [], [], []
    k = 0
    current = None
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        t = (start + BLOCK) / RATE
        while k < len(classes) and classes[k][0] + 1.5 <= t:  # a window's result is known at its end
            current = classes[k][1]
            k += 1
        f = analyzer.feed(samples[start : start + BLOCK], current)
        if f.release:
            releases.append(t)
        if f.beat.bpm:
            bpms.append(f.beat.bpm)
        row = seconds.add(sections.vector(f), BLOCK / RATE)
        if row is not None:
            vectors.append(row)
            rules.append(f.section)
    return {
        "vectors": np.array(vectors, dtype=np.float32),
        "rules": np.array(rules),
        "rule_releases": np.array(releases, dtype=np.float32),
        "bpm": np.array(float(np.median(bpms)) if bpms else 0.0),
    }


def run_one(args: tuple[Path, Path, Path]) -> str:
    mp3, mat, out_dir = args
    target = out_dir / f"{mp3.stem}.npz"
    if target.exists():
        return f"{mp3.stem}: cached"
    try:
        data = analyse(mp3)
        duration = float(len(data["vectors"]))
        bpm = float(data["bpm"]) or None
        labels = ssme.labels(mat, duration, bpm)
        truth = np.array(["groove"] * int(duration), dtype=object)
        for seg in labels:
            truth[int(seg["start"]) : int(np.ceil(seg["end"]))] = seg["section"]
        np.savez_compressed(target, **data, truth=truth.astype(str),
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
    args = parser.parse_args()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = ssme.tracks(args.root.expanduser())[: args.limit]
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, message in enumerate(pool.map(run_one, [(mp3, mat, out_dir) for mp3, mat in pairs]), 1):
            print(f"[{i}/{len(pairs)}] {message}", flush=True)


if __name__ == "__main__":
    main()
