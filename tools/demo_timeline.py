"""Colours for the demo page: music mode run offline over one track.

Usage: uv run python tools/demo_timeline.py TRACK --out docs/visionary.json
           [--classes classes.jsonl] [--marks marks.json --start SECONDS]

Runs the same pipeline as music mode (Apple's sound classifier, the analyser
with the shipped section model, and every mood's renderer) over a track and
writes one frame every 0.1 s: the section, chord, vocal, energy, bloom and
release, and each mood's colour. Only these derived values are written, never
audio. Expert labels come from an SSME .mat file beside the track when there
is one; a listening session's marks (marks.json from a recording, with the
song's start in session seconds) can be attached instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_tempo import RATE, decode  # noqa: E402

from amaran_mcp.music.analysis import Analyzer, merge_events  # noqa: E402
from amaran_mcp.music.capture import helper_path  # noqa: E402
from amaran_mcp.music.moods import MOODS  # noqa: E402
from amaran_mcp.music.render import Output, Renderer  # noqa: E402
from amaran_mcp.music.runner import load_model  # noqa: E402

BLOCK = 1024
STEP = 0.1  # seconds per frame
MARK_NAMES = {"build": "Build", "vocal": "Vocal", "lift": "Lift", "drop": "Climax", "breakdown": "Breakdown"}


def classify(track: Path) -> list[dict]:
    out = subprocess.run([str(helper_path()), "--classify-file", str(track)], capture_output=True, text=True, check=True)
    return [json.loads(line) for line in out.stdout.splitlines()]


def timeline(samples: np.ndarray, classes: list[dict]) -> tuple[list, dict]:
    analyzer = Analyzer(RATE, model=load_model())
    renderers = {name: Renderer(mood) for name, mood in MOODS.items()}
    for name, r in renderers.items():
        r.begin(Output(MOODS[name].anchors[0][0], 0.0, MOODS[name].brightness[0]))
    frames, colours = [], {name: [] for name in MOODS}
    next_t, ci, pending = STEP, 0, None
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        t = (start + BLOCK) / RATE
        while ci + 1 < len(classes) and classes[ci + 1]["t"] <= t:
            ci += 1
        current = {k: v for k, v in classes[ci].items() if k != "t"} if classes else None
        f = analyzer.feed(samples[start : start + BLOCK], current)
        outs = {name: r.step(f, BLOCK / RATE) for name, r in renderers.items()}
        # Keep one-block events (bloom, release) until the next frame.
        pending = f if pending is None else merge_events(pending, f)
        if t >= next_t:
            next_t += STEP
            for name, o in outs.items():
                colours[name].append([round(o.hue), round(o.saturation), round(o.brightness)])
            frames.append([round(t, 1), pending.section, pending.harmony.chord or "", round(pending.timbre.vocal, 2),
                           round(pending.energy, 2), int(bool(pending.bloom)), int(bool(pending.release))])
            pending = None
    return frames, colours


def experts(track: Path) -> dict | None:
    mat = track.with_suffix(".mat")
    if not mat.exists():
        return None
    import scipy.io

    m = scipy.io.loadmat(mat)

    def times(key: str) -> list[float]:
        return [round(float(x), 1) for x in np.ravel(m.get(key, []))]

    return {"drop": times("drop"), "build": list(zip(times("build_start"), times("build_end"))),
            "break": list(zip(times("break_start"), times("break_end")))}


def marks(path: Path, start: float, duration: float) -> list[list]:
    """A session's marks in song time; the same mark called twice within 4 s counts once,
    even with another mark between."""

    out: list[list] = []
    for entry in json.loads(path.read_text()):
        t = round(entry["t"] - start, 1)
        note = entry.get("note") or ""
        label = MARK_NAMES.get(entry["kind"])
        if entry["kind"] == "build" and note == "crescendo":
            label = "Crescendo"
        elif entry["kind"] == "other":
            label = {"top of the climax": "Top of the climax", "groove: steady beat energy": "Groove"}.get(note)
        if label is None or not 0 <= t <= duration:
            continue
        if any(kept == label and t - at < 4 for at, kept in out):
            continue
        out.append([t, label])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=Path, help="classifier output to reuse (JSON lines)")
    parser.add_argument("--marks", type=Path, help="a recording's marks.json")
    parser.add_argument("--start", type=float, default=0.0, help="the song's start in session seconds")
    parser.add_argument("--keep", type=Path, help="an earlier timeline whose hand-made rows (expert, marks_raw) carry over")
    args = parser.parse_args()
    samples = decode(args.track)
    classes = ([json.loads(line) for line in args.classes.read_text().splitlines()] if args.classes
               else classify(args.track))
    frames, colours = timeline(samples, classes)
    duration = round(len(samples) / RATE, 1)
    data = {"fields": ["t", "section", "chord", "vocal", "energy", "bloom", "release"],
            "frames": frames, "colours": colours, "duration": duration}
    if args.marks:
        data["marks"] = marks(args.marks, args.start, duration)
    else:
        data["experts"] = experts(args.track) or {"drop": [], "build": [], "break": []}
    if args.marks:
        data["marks_raw"] = data.pop("marks")
    if args.keep:
        earlier = json.loads(args.keep.read_text())
        for key in ("expert", "marks_raw"):
            if key in earlier and key not in data:
                data[key] = earlier[key]
        if "expert" in data:
            data.pop("experts", None)
    args.out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"{len(frames)} frames, {args.out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
