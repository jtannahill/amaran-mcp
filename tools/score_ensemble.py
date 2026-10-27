"""Score blends of the trees and the spectrogram net from their held-out predictions.

Usage: uv run python tools/score_ensemble.py --labels ~/datasets/features/ssme-v3 \\
           --trees ~/datasets/heldout/trees --cnn ~/datasets/heldout/cnn

Both models' held-out probabilities (written with --save-heldout, same folds)
are blended per second, p = w * trees + (1 - w) * cnn, decoded with music
mode's SectionDecoder and scored with the pooled measures of train_sections.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import score_sections  # noqa: E402
import train_sections  # noqa: E402

from amaran_mcp.music import sections  # noqa: E402


def aligned(path: Path, classes: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as d:
        order = [list(d["classes"]).index(c) for c in classes]
        return d["proba"][:, order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    args = parser.parse_args()
    classes = ["breakdown", "build", "climax", "groove"]
    tracks = []
    for label in sorted(args.labels.glob("*.npz")):
        t, c = args.trees / label.name, args.cnn / label.name
        if not (t.exists() and c.exists()):
            continue
        with np.load(label, allow_pickle=False) as d:
            truth, releases = d["truth"], list(d["true_releases"])
        pt, pc = aligned(t, classes), aligned(c, classes)
        n = min(len(truth), len(pt), len(pc))
        tracks.append((truth[:n], releases, pt[:n], pc[:n]))
    print(f"{len(tracks)} tracks with both models' held-out predictions")
    for w in (1.0, 0.75, 0.5, 0.25, 0.0):
        results = []
        for truth, releases, pt, pc in tracks:
            decoder = sections.SectionDecoder()
            out, found = [], []
            for i, row in enumerate(w * pt + (1 - w) * pc):
                out.append(decoder.step(dict(zip(classes, row))))
                if decoder.released:
                    found.append(float(i))
            results.append(score_sections.score(truth, out, releases, found))
        r = train_sections.summarise(results)
        print(f"  trees weight {w:4.2f}: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
