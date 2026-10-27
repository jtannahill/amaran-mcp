"""The whole live pipeline, as music mode runs it, on a sample of SSME tracks.

Usage: uv run python tools/live_check.py [--tracks 36] [--seed 7] [--rate 48000]

Held-out scores judge the section model's per-second calls; music mode also
runs the capture rules, the intro gate, the fast release and the state
machine. This runs all of it (the analyser with the shipped model, no
classifier) on random SSME tracks and reports, per track and overall: when
the intro ends, the share of time in intro, agreement with the experts per
second (intro and silence count as groove, which SSME has no label for),
blooms, and flickers (one second in another section, then back). --rate
decodes at another sample rate, as a Mac's output device delivers it live
(usually 48 kHz); the training rate is 24 kHz.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

MP3 = Path.home() / "datasets/ssme/mp3s_soundcloud_cc_event_detection"
LABELS = Path.home() / "datasets/features/ssme-v3"


def decode(path: Path, rate: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def run(stem: str, rate: int = 24000) -> tuple:
    from amaran_mcp.music.analysis import Analyzer
    from amaran_mcp.music.runner import load_model

    samples = decode(MP3 / f"{stem}.mp3", rate)
    analyzer = Analyzer(rate, model=load_model())
    block = 1024 * rate // 24000  # the same 43 ms blocks at any rate
    seconds, next_t, blooms = [], 1.0, 0
    for start in range(0, len(samples) - block, block):
        f = analyzer.feed(samples[start : start + block])
        blooms += bool(f.bloom)
        if (start + block) / rate >= next_t:
            next_t += 1
            seconds.append(f.section)
    with np.load(LABELS / f"{stem}.npz") as d:
        truth = list(d["truth"])
    n = min(len(truth), len(seconds))
    first_out = next((i for i, x in enumerate(seconds) if x not in ("silence", "intro")), len(seconds))
    intro_share = sum(x == "intro" for x in seconds[:n]) / n
    agree = float(np.mean([("groove" if x in ("intro", "silence") else x) == y for x, y in zip(seconds[:n], truth[:n])]))
    flips = sum(1 for i in range(2, len(seconds)) if seconds[i] == seconds[i - 2] != seconds[i - 1])
    return stem, first_out, intro_share, agree, blooms, flips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", type=int, default=36)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate", type=int, default=24000)
    args = parser.parse_args()
    stems = sorted(p.stem for p in LABELS.glob("*.npz"))
    random.Random(args.seed).shuffle(stems)
    with ProcessPoolExecutor(6) as pool:
        results = list(pool.map(partial(run, rate=args.rate), stems[: args.tracks]))
    for stem, first_out, share, agree, blooms, flips in results:
        print(f"{stem:>10}  out of intro {first_out:4d} s  intro {share:.2f}  agree {agree:.2f}  blooms {blooms}  flips {flips}")
    print(f"tracks {len(results)}  stuck in intro over half: {sum(r[2] > 0.5 for r in results)}  "
          f"out of intro within 60 s: {sum(r[1] <= 60 for r in results)}  "
          f"agreement {np.mean([r[3] for r in results]):.2f}  flickers {sum(r[5] for r in results)}  "
          f"with a bloom: {sum(r[4] > 0 for r in results)}")


if __name__ == "__main__":
    main()
