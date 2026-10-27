"""Log-mel spectrograms for training the spectrogram section model.

Usage: uv run python tools/extract_mel.py ~/datasets/ssme --labels ~/datasets/features/ssme-v2 --out ~/datasets/mel/ssme

For each track: decode to 24 kHz mono, compute music/mel.py's spectrogram
(10 frames a second, 64 mels), and save it (float16) with the per-second
labels and releases already produced by extract_features.py, so both models
are trained and scored against the same answers.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_tempo import RATE, decode  # noqa: E402
import ssme  # noqa: E402


def run_one(args: tuple[Path, Path, Path]) -> str:
    from amaran_mcp.music.mel import mel_of

    mp3, labels_dir, out_dir = args
    target = out_dir / f"{mp3.stem}.npz"
    source = labels_dir / f"{mp3.stem}.npz"
    if target.exists():
        return f"{mp3.stem}: cached"
    if not source.exists():
        return f"{mp3.stem}: no labels, skipped"
    try:
        mel = mel_of(decode(mp3), RATE)
        with np.load(source, allow_pickle=False) as d:
            np.savez_compressed(target, mel=mel.astype(np.float16), truth=d["truth"], rules=d["rules"],
                                rule_releases=d["rule_releases"], true_releases=d["true_releases"], bpm=d["bpm"])
        return f"{mp3.stem}: {len(mel)} frames"
    except Exception as err:  # one bad file should not stop the run
        return f"{mp3.stem}: FAILED {err}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = ssme.tracks(args.root.expanduser())
    with ProcessPoolExecutor(args.jobs) as pool:
        jobs = [(mp3, args.labels.expanduser(), out_dir) for mp3, _ in pairs]
        for i, message in enumerate(pool.map(run_one, jobs), 1):
            print(f"[{i}/{len(pairs)}] {message}", flush=True)


if __name__ == "__main__":
    main()
