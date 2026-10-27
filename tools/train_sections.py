"""Train music mode's section model, and only ship it if it beats the rules.

Usage:
  uv run python tools/train_sections.py --features ~/datasets/features/ssme
  uv run python tools/train_sections.py --features ~/datasets/features/ssme --owner   # local model, never committed

5-fold cross-validation split by track (a track is never in both training and
testing). For each fold the hand-written rules and the model are scored the
same way (tools/score_sections.py), with the model's per-second
probabilities turned into sections by the same SectionDecoder music mode
uses live. The weights are written only if the model beats the rules on
pooled frame accuracy and on build and breakdown boundaries (within 2 s), and
is no worse than the rules on the owner's labelled sessions when there are any.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import score_sections  # noqa: E402

from amaran_mcp.music import sections  # noqa: E402

PUBLISHED = Path(__file__).parent.parent / "src" / "amaran_mcp" / "music" / "sections.npz"
OWNER = Path.home() / ".config" / "amaran-mcp" / "sections.npz"
SCORE_AS = {"intro": "groove", "silence": "groove"}  # SSME has neither; the rules' versions count as groove


def export(clf, means: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray]:
    weights = {"means": means.astype(np.float64), "scales": scales.astype(np.float64),
               "classes": np.array([str(c) for c in clf.classes_])}
    for k, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights[f"W{k}"], weights[f"b{k}"] = W, b
    return weights


def load(directory: Path) -> list[dict]:
    tracks = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            vectors = d["vectors"]
            if len(vectors) < sections.WINDOW_SECONDS:
                continue
            X = np.array([sections.window(vectors, i) for i in range(len(vectors))])
            tracks.append({"name": path.stem, "X": X, "y": d["truth"][: len(X)], "rules": d["rules"][: len(X)],
                           "rule_releases": list(d["rule_releases"]), "true_releases": list(d["true_releases"])})
    return tracks


def decode(model: sections.Model, X: np.ndarray) -> tuple[list[str], list[float]]:
    decoder = sections.SectionDecoder()
    out, releases = [], []
    for i, x in enumerate(X):
        out.append(decoder.step(model.proba(x)))
        if decoder.released:
            releases.append(float(i))
    return out, releases


def summarise(results: list[dict]) -> dict[str, float]:
    return {
        "frame_accuracy": float(np.mean([r["frame_accuracy"] for r in results])),
        "breakdown_f": float(np.mean([r["boundaries"]["breakdown"]["f"] for r in results])),
        "build_f": float(np.mean([r["boundaries"]["build"]["f"] for r in results])),
        "release_f": float(np.mean([r["release_f"] for r in results])),
    }


def fit(tracks: list[dict], seed: int = 0):
    from sklearn.neural_network import MLPClassifier

    X = np.concatenate([t["X"] for t in tracks])
    y = np.concatenate([t["y"] for t in tracks])
    means, scales = X.mean(axis=0), X.std(axis=0) + 1e-6
    clf = MLPClassifier(hidden_layer_sizes=(32, 32), alpha=1e-3, max_iter=200, early_stopping=True,
                        random_state=seed).fit((X - means) / scales, y)
    return sections.Model(export(clf, means, scales)), export(clf, means, scales)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="directory of extracted .npz tracks")
    parser.add_argument("--owner", action="store_true", help="also train on the owner's sessions; write the local model")
    parser.add_argument("--owner-features", type=Path, default=Path.home() / ".config/amaran-mcp/features")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    public = load(args.features.expanduser())
    owner = load(args.owner_features) if args.owner_features.exists() else []
    print(f"{len(public)} public tracks, {len(owner)} owner sessions")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(public))
    folds = [order[k :: args.folds] for k in range(args.folds)]

    rules_results, model_results = [], []
    for k, held in enumerate(folds):
        held_set = set(held.tolist())
        train = [public[i] for i in range(len(public)) if i not in held_set] + (owner if args.owner else [])
        model, _ = fit(train, seed=k)
        for i in held:
            t = public[i]
            rules = [SCORE_AS.get(s, s) for s in t["rules"]]
            rules_results.append(score_sections.score(t["y"], rules, t["true_releases"], t["rule_releases"]))
            predicted, releases = decode(model, t["X"])
            model_results.append(score_sections.score(t["y"], predicted, t["true_releases"], releases))
        r, m = summarise(rules_results[-len(held):]), summarise(model_results[-len(held):])
        print(f"fold {k + 1}: rules {r}  model {m}")

    rules_all, model_all = summarise(rules_results), summarise(model_results)
    print(f"\nheld-out, pooled ({len(public)} tracks):")
    for key in rules_all:
        print(f"  {key:15} rules {rules_all[key]:.3f}   model {model_all[key]:.3f}")

    passes = (model_all["frame_accuracy"] > rules_all["frame_accuracy"]
              and model_all["breakdown_f"] > rules_all["breakdown_f"]
              and model_all["build_f"] > rules_all["build_f"])
    model, weights = fit(public + (owner if args.owner else []))
    if owner:
        o_rules = summarise([score_sections.score(t["y"], [SCORE_AS.get(s, s) for s in t["rules"]],
                                                  t["true_releases"], t["rule_releases"]) for t in owner])
        o_model = summarise([score_sections.score(t["y"], *decode(model, t["X"])[:1], t["true_releases"],
                                                  decode(model, t["X"])[1]) for t in owner])
        print(f"\nowner sessions: rules {o_rules}  model {o_model}")
        passes = passes and o_model["frame_accuracy"] >= o_rules["frame_accuracy"]
    if not passes:
        print("\nThe model does not beat the rules: weights NOT written.")
        raise SystemExit(1)
    out = OWNER if args.owner else PUBLISHED
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **weights)
    print(f"\nThe model beats the rules: wrote {out}")


if __name__ == "__main__":
    main()
