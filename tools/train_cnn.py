"""Train the spectrogram section model (option 2) and ship it only if it is better.

Usage: uv run python tools/train_cnn.py --mel ~/datasets/mel/ssme [--epochs 6]

A small convolutional network reads 16 s of log-mel spectrogram (music/mel.py)
and says which section the last second is in. Trained with PyTorch on the
Mac's GPU (MPS); exported to numpy weights that music/sections.py runs live.
Cross-validated by track with the same folds and the same scoring as
train_sections.py; it replaces the shipped model only if it beats both the
hand-written rules (the ship gate) and the shipped trees on held-out tracks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import score_sections  # noqa: E402
import train_sections  # noqa: E402

from amaran_mcp.music import sections  # noqa: E402

FPS = 10


class SectionNet(nn.Module):
    def __init__(self, classes: list[str]) -> None:
        super().__init__()
        self.classes = classes
        self.convs = nn.ModuleList([nn.Conv2d(2, 16, 3, padding=1), nn.Conv2d(16, 32, 3, padding=1),
                                    nn.Conv2d(32, 48, 3, padding=1)])
        self.fc0 = nn.Linear(48 * (sections.MEL_FRAMES // 8), 64)
        self.fc1 = nn.Linear(64, len(classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = torch.max_pool2d(torch.relu(conv(x)), 2)
        x = x.mean(dim=3).flatten(1)  # average over frequency, keep time
        return self.fc1(torch.relu(self.fc0(x)))


def prepare(windows: np.ndarray, level_mean: float, level_scale: float) -> np.ndarray:
    return sections.prepare_mel(windows, level_mean, level_scale)


def export_cnn(net: SectionNet, level_mean: float, level_scale: float) -> dict[str, np.ndarray]:
    weights = {"kind": np.array("cnn"), "classes": np.array(net.classes),
               "level_mean": np.array(level_mean), "level_scale": np.array(level_scale)}
    for k, conv in enumerate(net.convs):
        weights[f"conv{k}_w"] = conv.weight.detach().cpu().numpy().astype(np.float64)
        weights[f"conv{k}_b"] = conv.bias.detach().cpu().numpy().astype(np.float64)
    for k, fc in enumerate((net.fc0, net.fc1)):
        weights[f"fc{k}_w"] = fc.weight.detach().cpu().numpy().astype(np.float64)
        weights[f"fc{k}_b"] = fc.bias.detach().cpu().numpy().astype(np.float64)
    return weights


def load(directory: Path) -> list[dict]:
    tracks = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as d:
            mel = d["mel"].astype(np.float32)
            seconds = min(len(d["truth"]), len(mel) // FPS)
            tracks.append({"name": path.stem, "mel": mel, "y": d["truth"][:seconds], "rules": d["rules"][:seconds],
                           "rule_releases": list(d["rule_releases"]), "true_releases": list(d["true_releases"])})
    return tracks


def window_at(mel: np.ndarray, second: int) -> np.ndarray:
    end = (second + 1) * FPS
    start = end - sections.MEL_FRAMES
    if start >= 0:
        return mel[start:end]
    return np.concatenate([np.repeat(mel[:1], -start, axis=0), mel[:end]])


def train(tracks: list[dict], classes: list[str], epochs: int, seed: int, device: str) -> tuple[SectionNet, float, float]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    level_mean = float(np.mean([t["mel"].mean() for t in tracks]))
    level_scale = float(np.mean([t["mel"].std() for t in tracks])) + 1e-6
    index = [(i, s) for i, t in enumerate(tracks) for s in range(len(t["y"]))]
    labels = np.array([classes.index(tracks[i]["y"][s]) for i, s in index])
    counts = np.bincount(labels, minlength=len(classes))
    weights = (1.0 / counts[labels]) / (1.0 / counts[labels]).sum()  # balanced sampling
    net = SectionNet(classes).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    per_epoch, batch = min(60000, len(index)), 128
    for epoch in range(epochs):
        net.train()
        picks = rng.choice(len(index), per_epoch, p=weights)
        total = 0.0
        for b in range(0, per_epoch, batch):
            chosen = picks[b : b + batch]
            windows = np.stack([window_at(tracks[index[j][0]]["mel"], index[j][1]) for j in chosen])
            x = torch.from_numpy(prepare(windows, level_mean, level_scale)).to(device)
            y = torch.from_numpy(labels[chosen]).to(device)
            optimiser.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(chosen)
        print(f"    epoch {epoch + 1}: loss {total / per_epoch:.3f}", flush=True)
    return net.eval().cpu(), level_mean, level_scale


def decode(model: sections.Model, track: dict) -> tuple[list[str], list[float]]:
    windows = np.stack([window_at(track["mel"], s) for s in range(len(track["y"]))])
    probabilities = np.concatenate([model.proba_batch(windows[i : i + 256]) for i in range(0, len(windows), 256)])
    decoder = sections.SectionDecoder()
    out, releases = [], []
    for i, row in enumerate(probabilities):
        out.append(decoder.step(dict(zip(model.classes, row))))
        if decoder.released:
            releases.append(float(i))
    return out, releases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mel", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees-results", type=Path, help="cv_results.json from train_sections.py")
    parser.add_argument("--save-heldout", type=Path, help="write each track's held-out probabilities here (for ensembles)")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tracks = load(args.mel.expanduser())
    classes = sorted({s for t in tracks for s in t["y"]})
    print(f"{len(tracks)} tracks, classes {classes}, device {device}")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tracks))
    folds = [order[k :: args.folds] for k in range(args.folds)]
    rules_results, model_results = [], []
    for k, held in enumerate(folds):
        held_set = set(held.tolist())
        print(f"fold {k + 1}/{len(folds)}", flush=True)
        net, level_mean, level_scale = train([tracks[i] for i in range(len(tracks)) if i not in held_set],
                                             classes, args.epochs, k, device)
        model = sections.Model(export_cnn(net, level_mean, level_scale))
        for i in held:
            t = tracks[i]
            if args.save_heldout:
                args.save_heldout.mkdir(parents=True, exist_ok=True)
                windows = np.stack([window_at(t["mel"], s) for s in range(len(t["y"]))])
                probs = np.concatenate([model.proba_batch(windows[j : j + 256]) for j in range(0, len(windows), 256)])
                np.savez(args.save_heldout / f"{t['name']}.npz", proba=probs, classes=np.array(model.classes))
            rules = [train_sections.SCORE_AS.get(s, s) for s in t["rules"]]
            rules_results.append(score_sections.score(t["y"], rules, t["true_releases"], t["rule_releases"]))
            predicted, releases = decode(model, t)
            model_results.append(score_sections.score(t["y"], predicted, t["true_releases"], releases))
    rules_all, model_all = train_sections.summarise(rules_results), train_sections.summarise(model_results)
    trees = json.loads(args.trees_results.read_text()) if args.trees_results and args.trees_results.exists() else None
    print(f"\nheld-out, pooled ({len(tracks)} tracks):")
    for key in rules_all:
        line = f"  {key:18} rules {rules_all[key]:.3f}   spectrogram net {model_all[key]:.3f}"
        if trees:
            line += f"   trees {trees[key]:.3f}"
        print(line)
    beats_rules = all(model_all[k] > rules_all[k] for k in ("balanced_accuracy", "breakdown_f", "build_f"))
    beats_trees = trees is None or all(model_all[k] >= trees[k] for k in ("balanced_accuracy", "breakdown_f", "build_f", "release_f"))
    if not (beats_rules and beats_trees):
        print("\nThe spectrogram net does not beat both the rules and the shipped trees: NOT shipped.")
        raise SystemExit(1)
    net, level_mean, level_scale = train(tracks, classes, args.epochs, 99, device)
    np.savez(train_sections.PUBLISHED, **export_cnn(net, level_mean, level_scale))
    print(f"\nThe spectrogram net is better: wrote {train_sections.PUBLISHED}")


if __name__ == "__main__":
    main()
