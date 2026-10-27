"""Train music mode's section model, and only ship it if it beats the rules.

Usage:
  uv run python tools/train_sections.py --features ~/datasets/features/ssme
  uv run python tools/train_sections.py --features ~/datasets/features/ssme --owner   # local model, never committed

5-fold cross-validation split by track (a track is never in both training and
testing). For each fold the hand-written rules and the model are scored the
same way (tools/score_sections.py), with the model's per-second
probabilities turned into sections by the same SectionDecoder music mode
uses live. The weights are written only if the model beats the rules on
pooled balanced accuracy (mean per-section recall; raw frame accuracy would
reward always answering groove, 58% of SSME seconds) and on build and
breakdown boundaries (within 2 s), and
is no worse than the rules on the owner's labelled sessions when there are any.
"""

from __future__ import annotations

import argparse
import json
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


def export_trees(clf) -> dict[str, np.ndarray]:
    """A fitted HistGradientBoostingClassifier as flat numpy arrays.

    Reads scikit-learn's fitted trees (_predictors, _baseline_prediction): pin
    the scikit-learn version this is trained with.
    """

    feature, threshold, left, right, leaf, value, roots, tree_class = [], [], [], [], [], [], [], []
    depth = 0
    offset = 0
    binary = len(clf.classes_) == 2
    for iteration in clf._predictors:
        for k, tree in enumerate(iteration):
            nodes = tree.nodes
            roots.append(offset)
            tree_class.append(1 if binary else k)
            feature.append(nodes["feature_idx"].astype(np.int64))
            threshold.append(nodes["num_threshold"].astype(np.float64))
            left.append(nodes["left"].astype(np.int64) + offset)
            right.append(nodes["right"].astype(np.int64) + offset)
            leaf.append(nodes["is_leaf"].astype(bool))
            value.append(nodes["value"].astype(np.float64))
            depth = max(depth, int(nodes["depth"].max()) + 1)
            offset += len(nodes)
    baseline = np.asarray(clf._baseline_prediction, dtype=np.float64).ravel()
    if binary:
        baseline = np.array([0.0, baseline[0]])
    return {"kind": np.array("trees"), "classes": np.array([str(c) for c in clf.classes_]),
            "baseline": baseline, "feature": np.concatenate(feature), "threshold": np.concatenate(threshold),
            "left": np.concatenate(left), "right": np.concatenate(right), "leaf": np.concatenate(leaf),
            "value": np.concatenate(value), "roots": np.array(roots), "tree_class": np.array(tree_class),
            "depth": np.array(depth)}


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
    probabilities = model.proba_batch(X)
    for i, row in enumerate(probabilities):
        out.append(decoder.step(dict(zip(model.classes, row))))
        if decoder.released:
            releases.append(float(i))
    return out, releases


def summarise(results: list[dict]) -> dict[str, float]:
    return {
        "balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in results])),
        "frame_accuracy": float(np.mean([r["frame_accuracy"] for r in results])),
        # Boundary and release scores pooled over all tracks (score_sections.pooled_f).
        "breakdown_f": score_sections.pooled_f([r["boundaries"]["breakdown"] for r in results]),
        "build_f": score_sections.pooled_f([r["boundaries"]["build"] for r in results]),
        "release_f": score_sections.pooled_f([r["releases"] for r in results]),
    }


def fit(tracks: list[dict], seed: int = 0):
    """Gradient-boosted trees, sections balanced (otherwise the model answers groove)."""

    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.concatenate([t["X"] for t in tracks])
    y = np.concatenate([t["y"] for t in tracks])
    clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, learning_rate=0.1,
                                         max_leaf_nodes=31, l2_regularization=1.0, random_state=seed).fit(X, y)
    weights = export_trees(clf)
    return sections.Model(weights), weights


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
    (args.features.expanduser() / "cv_results.json").write_text(
        json.dumps({**model_all, "rules": rules_all}, indent=1))
    print(f"\nheld-out, pooled ({len(public)} tracks):")
    for key in rules_all:
        print(f"  {key:15} rules {rules_all[key]:.3f}   model {model_all[key]:.3f}")

    passes = (model_all["balanced_accuracy"] > rules_all["balanced_accuracy"]
              and model_all["breakdown_f"] > rules_all["breakdown_f"]
              and model_all["build_f"] > rules_all["build_f"])
    model, weights = fit(public + (owner if args.owner else []))
    if owner:
        o_rules = summarise([score_sections.score(t["y"], [SCORE_AS.get(s, s) for s in t["rules"]],
                                                  t["true_releases"], t["rule_releases"]) for t in owner])
        o_model = summarise([score_sections.score(t["y"], *decode(model, t["X"])[:1], t["true_releases"],
                                                  decode(model, t["X"])[1]) for t in owner])
        print(f"\nowner sessions: rules {o_rules}  model {o_model}")
        passes = passes and o_model["balanced_accuracy"] >= o_rules["balanced_accuracy"]
    if not passes:
        print("\nThe model does not beat the rules: weights NOT written.")
        raise SystemExit(1)
    out = OWNER if args.owner else PUBLISHED
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **weights)
    print(f"\nThe model beats the rules: wrote {out}")


if __name__ == "__main__":
    main()
