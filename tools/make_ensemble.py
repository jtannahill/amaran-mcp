"""Combine the trees and the spectrogram net into one section model file.

Usage: uv run python tools/make_ensemble.py --trees TREES.npz --cnn CNN.npz --weight 0.5 --out OUT.npz

The blend weight comes from tools/score_ensemble.py on held-out predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from amaran_mcp.music import sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trees", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True, help="share of the trees in the blend")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.trees, allow_pickle=False) as t, np.load(args.cnn, allow_pickle=False) as c:
        trees, cnn = {k: t[k] for k in t.files}, {k: c[k] for k in c.files}
    if str(trees.get("kind")) != "trees" or str(cnn.get("kind")) != "cnn":
        raise SystemExit("expected a trees file and a cnn file")
    np.savez(args.out, **sections.ensemble_weights(trees, cnn, args.weight))
    model = sections.Model.load(args.out)
    print(f"wrote {args.out}: {model.kind}, input {model.input}, classes {model.classes}")


if __name__ == "__main__":
    main()
