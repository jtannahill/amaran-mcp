"""A structural reading of a track, independent of music mode's section model.

Usage: uv run python tools/expert_reading.py TRACK.wav --classes classes.jsonl [--singer NAME] [--json]

For tracks no person has marked. It reads the recording the way an annotator
would mark one, from signals the section model never sees in this form:

  whether the kick is in    how strongly 40 to 150 Hz pulses at 100 to 160 BPM, over 4 s
  how loud it is            energy against the track's own range
  when someone sings        Apple's classifier (singing), sustained for 2 s

Sections come from the kick and the energy: without the kick it is a
breakdown; when the kick returns, a build while the energy still climbs, then
the drop; with the kick at the track's top energy a drop, below it a groove;
the start is an intro and everything after the last drop the outro.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

FPS = 10
KERNEL = 80  # frames each side: 8 s
MIN_GAP = 120  # frames between changes: 12 s


def novelty(mel: np.ndarray) -> np.ndarray:
    z = (mel - mel.mean(axis=0)) / (mel.std(axis=0) + 1e-6)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-6
    s = z @ z.T
    k = KERNEL
    sign = np.ones((2 * k, 2 * k))
    sign[:k, k:] = sign[k:, :k] = -1
    taper = np.outer(np.hanning(2 * k), np.hanning(2 * k))
    kernel = sign * taper
    n = len(z)
    out = np.zeros(n)
    for i in range(k, n - k):
        out[i] = float(np.sum(s[i - k : i + k, i - k : i + k] * kernel))
    return out


def peaks(curve: np.ndarray, gap: int, floor: float) -> list[int]:
    order = np.argsort(curve)[::-1]
    picked: list[int] = []
    for i in order:
        if curve[i] < floor:
            break
        if all(abs(i - j) >= gap for j in picked):
            picked.append(int(i))
    return sorted(picked)


def _runs(flags: np.ndarray, shortest: int) -> list[list]:
    """[start, end, value] runs of a boolean series; runs shorter than `shortest` join their neighbour."""

    runs: list[list] = []
    for i, v in enumerate(flags):
        if runs and runs[-1][2] == v:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, bool(v)])
    merged = True
    while merged and len(runs) > 1:
        merged = False
        for k, (a, b, v) in enumerate(runs):
            if b - a < shortest:
                j = k - 1 if k > 0 else k + 1
                lo, hi = min(runs[j][0], a), max(runs[j][1], b)
                runs[j] = [lo, hi, runs[j][2]]
                del runs[k]
                merged = True
                break
        out: list[list] = []
        for r in runs:
            if out and out[-1][2] == r[2]:
                out[-1][1] = r[1]
            else:
                out.append(r)
        runs = out
    return runs


def _snap(t: int, candidates: list[int], reach: int) -> int:
    near = [c for c in candidates if abs(c - t) <= reach]
    return min(near, key=lambda c: abs(c - t)) if near else t


def beat_strength(samples: np.ndarray, rate: int) -> np.ndarray:
    """How strongly the low band (40 to 150 Hz) pulses at a dance tempo (100 to 160 BPM),
    over 4 s windows, at 10 frames a second: 0 for no kick, near 1 for a steady one."""

    fps, nfft = 100, 2048
    hop = rate // fps
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(samples, (0, nfft)), nfft)[::hop]
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    band = (freqs >= 40) & (freqs <= 150)
    low = np.empty(len(frames))
    for i in range(0, len(frames), 2048):  # in blocks, to keep memory small
        chunk = frames[i : i + 2048] * np.hanning(nfft)
        low[i : i + 2048] = np.log1p(np.abs(np.fft.rfft(chunk, axis=1))[:, band].sum(axis=1))
    env = np.maximum(0, np.diff(low, prepend=low[0]))
    w, lo, hi = 4 * fps, int(fps * 60 / 160), int(fps * 60 / 100)
    out = np.zeros(len(env) // 10 + 1)
    for k, start in enumerate(range(0, len(env), 10)):
        seg = env[max(0, start - w // 2) : start + w // 2]
        if len(seg) <= hi + 1:
            continue
        seg = seg - seg.mean()
        denom = float(seg @ seg) + 1e-9
        out[k] = max(float(seg[:-lag] @ seg[lag:]) / denom for lag in range(lo, hi + 1))
    return out


def reading(mel: np.ndarray, beat: np.ndarray, singing: np.ndarray) -> dict:
    n = len(mel)
    beat = np.resize(beat, n)
    energy = np.convolve(mel.mean(axis=1), np.ones(80) / 80, mode="same")  # 8 s
    span_e = np.percentile(energy, 95) - np.percentile(energy, 5)
    audible = energy > np.percentile(energy, 5) + 0.25 * span_e
    kick = (np.convolve(beat, np.ones(20) / 20, mode="same") >= 0.35) & audible
    runs = _runs(kick, shortest=80)  # sections last at least 8 s
    top = np.percentile(energy[kick], 90) if kick.any() else energy.max()
    sections: list[list] = []
    after_break = False
    for a, b, has_kick in runs:
        if not has_kick:
            label = "Breakdown" if audible[a:b].mean() > 0.5 and 0 < a and b < n else "Quiet"
            sections.append([a, b, label])
            after_break = label == "Breakdown"
            continue
        seg = energy[a:b]
        if after_break:
            # The kick is back: a build while the energy still climbs, then the drop.
            peak = int(np.argmax(seg >= 0.97 * seg.max()))
            if peak > 80:
                sections.append([a, a + peak, "Build"])
                a = a + peak
                seg = energy[a:b]
        for c, d, hot in _runs(seg >= 0.9 * top, shortest=80):
            sections.append([a + c, a + d, "Drop" if hot else "Groove"])
        after_break = False
    if sections and sections[0][2] in ("Quiet", "Groove"):
        sections[0][2] = "Intro"
    # A groove climbing straight into a drop is a build.
    for i in range(len(sections) - 1):
        a, b, label = sections[i]
        if label == "Groove" and sections[i + 1][2] == "Drop" and energy[b - 1] > energy[a] + 0.15 * span_e:
            sections[i][2] = "Build"
    last_drop = max((i for i, sec in enumerate(sections) if sec[2] == "Drop"), default=None)
    if last_drop is not None:
        for sec in sections[last_drop + 1 :]:
            sec[2] = "Outro"
    spans: list[list] = []
    for a, b, label in sections:
        label = "Intro" if label == "Quiet" and not spans else label
        if spans and spans[-1][2] == label:
            spans[-1][1] = b
        else:
            spans.append([a, b, label])
    points = []
    sung = np.convolve(singing > 0.25, np.ones(20) / 20, mode="same") > 0.8
    quiet = 300
    for i, on in enumerate(sung):
        if on and quiet >= 300:  # first entry, or back after 30 s without
            points.append([round(i / FPS - 1.0, 1), "voice"])
        quiet = 0 if on else quiet + 1
    drops = [sp for sp in spans if sp[2] == "Drop"]
    if drops:
        a, b = drops[-1][0], drops[-1][1]
        points.append([round((a + int(np.argmax(energy[a:b]))) / FPS, 1), "Peak"])
    return {"spans": [[round(a / FPS, 1), round(b / FPS, 1), lab] for a, b, lab in spans], "points": points}


def main() -> None:
    from benchmark_tempo import RATE, decode

    from amaran_mcp.music.mel import mel_of

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track", type=Path)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--singer", default="The voice")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    samples = decode(args.track)
    mel = mel_of(samples, RATE)
    beat = beat_strength(samples, RATE)
    classes = [json.loads(line) for line in args.classes.read_text().splitlines()]
    times = np.array([c["t"] for c in classes])
    values = np.array([max(c["singing"], c["choir_singing"]) for c in classes])
    singing = np.interp(np.arange(len(mel)) / FPS, times, values)
    out = reading(mel, beat, singing)
    first = True
    named = []
    for t, label in out["points"]:
        if label == "voice":
            label, first = (f"{args.singer} comes in" if first else f"{args.singer} again"), False
        named.append([t, label])
    out["points"] = sorted(named)
    if args.json:
        print(json.dumps(out))
        return
    for a, b, label in out["spans"]:
        print(f"{int(a // 60)}:{int(a % 60):02d} to {int(b // 60)}:{int(b % 60):02d}  {label}")
    for t, label in out["points"]:
        print(f"{int(t // 60)}:{int(t % 60):02d}  {label}")


if __name__ == "__main__":
    main()
