"""Record short clips of what this Mac plays, and replay them through the analysis.

For tuning music mode on real episodes: record a breakdown once, then replay
it as often as needed. Clips stay in ~/.config/amaran-mcp/clips on this Mac.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any
import wave

import numpy as np

from ..config import CONFIG_DIR
from .analysis import Analyzer, merge_events
from .capture import Source, SystemAudio

CLIPS_DIR = CONFIG_DIR / "clips"
BLOCK = 1024


async def record(seconds: float, name: str, source: Source | None = None) -> Path:
    """Record `seconds` of audio to a 16-bit mono WAV in CLIPS_DIR."""

    source = source or SystemAudio()
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "clip"
    path = CLIPS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{safe}.wav"
    chunks: list[np.ndarray] = []
    total = 0
    try:
        async for block in source.blocks():
            chunks.append(block)
            total += len(block)
            if total >= seconds * source.rate:
                break
    finally:
        await source.close()
    samples = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(source.rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as clip:
        rate, channels, width = clip.getframerate(), clip.getnchannels(), clip.getsampwidth()
        raw = clip.readframes(clip.getnframes())
    if width != 2:
        raise ValueError(f"{path.name}: only 16-bit WAV clips are supported")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def replay(path: Path, every: float = 1.0) -> list[dict[str, Any]]:
    """Run a clip through the analyzer; one row of features per `every` seconds."""

    samples, rate = load(path)
    analyzer = Analyzer(rate)
    rows: list[dict[str, Any]] = []
    next_row = every
    pending = None
    for start in range(0, len(samples) - BLOCK + 1, BLOCK):
        features = analyzer.feed(samples[start : start + BLOCK])
        # Keep one-block events (bloom, release, lift) between rows.
        pending = features if pending is None else merge_events(pending, features)
        t = (start + BLOCK) / rate
        if t >= next_row:
            next_row += every
            rows.append({"t": round(t, 2), **asdict(pending)})
            pending = None
    return rows
