"""The Socially Significant Music Events dataset (Yadati et al., IEEE TMM; osf.io/eydxk).

402 full Creative Commons EDM tracks from SoundCloud, each with a .mat file of
expert-labelled events: drop times, and build and break start/end times. The
timed-comment fields (*_tc) are listeners' comments and are not used.

  unpack(root)          join the split zip once and extract it
  labels(mat, ...)      the events as music mode sections
  releases(mat)         the drop times (scored as release events)
  tracks(root)          (mp3, mat) pairs
"""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess

import numpy as np

log = logging.getLogger(__name__)

ARCHIVE = "mp3s_soundcloud_cc_event_detection"
CLIMAX_BARS = 32  # a drop opens a climax of about this long (SSME marks only the drop point)
FALLBACK_CLIMAX = 30.0


def unpack(root: Path) -> Path:
    """Extract the dataset beside its archive parts (once). Returns the track folder."""

    folder = root / ARCHIVE
    if folder.exists() and any(folder.glob("*.mp3")):
        return folder
    joined = root / f"{ARCHIVE}-joined.zip"
    if not joined.exists():
        subprocess.run(["zip", "-q", "-s", "0", str(root / f"{ARCHIVE}.zip"), "--out", str(joined)], check=True)
    subprocess.run(["unzip", "-q", "-o", str(joined), "-d", str(root)], check=True)
    return folder


def _times(data: dict, key: str) -> list[float]:
    value = data.get(key)
    if value is None:
        return []
    return [float(v) for v in np.asarray(value, dtype=float).ravel() if np.isfinite(v)]


def _load(mat_path: Path) -> dict:
    from scipy.io import loadmat  # dev dependency

    return {k: v for k, v in loadmat(mat_path).items() if not k.startswith("__")}


def releases(mat_path: Path) -> list[float]:
    return sorted(_times(_load(mat_path), "drop"))


def labels(mat_path: Path, duration: float, bpm: float | None = None) -> list[dict]:
    """Sections covering the whole track: build, breakdown, climax, and groove between."""

    data = _load(mat_path)
    events: list[tuple[float, float, str]] = []
    for kind, section in (("build", "build"), ("break", "breakdown")):
        starts, ends = _times(data, f"{kind}_start"), _times(data, f"{kind}_end")
        if len(starts) != len(ends):
            log.warning("%s: %d %s starts but %d ends; skipping its %ss", mat_path.name, len(starts), kind,
                        len(ends), kind)
            continue
        events += [(s, e, section) for s, e in zip(starts, ends) if e > s]
    climax = CLIMAX_BARS * 4 * 60 / bpm if bpm else FALLBACK_CLIMAX
    for drop in _times(data, "drop"):
        events.append((drop, min(duration, drop + climax), "climax"))

    # Lay events on a timeline; later starts cut earlier sections short, gaps are groove.
    events.sort()
    out: list[dict] = []
    cursor = 0.0
    for start, end, section in events:
        start, end = max(start, cursor), min(end, duration)
        if end <= start:
            continue
        if start > cursor:
            out.append({"start": cursor, "end": start, "section": "groove"})
        out.append({"start": start, "end": end, "section": section})
        cursor = end
    if cursor < duration:
        out.append({"start": cursor, "end": duration, "section": "groove"})
    return out


def tracks(root: Path) -> list[tuple[Path, Path]]:
    folder = unpack(root)
    pairs = []
    for mp3 in sorted(folder.glob("*.mp3")):
        mat = mp3.with_suffix(".mat")
        if mat.exists():
            pairs.append((mp3, mat))
    return pairs
